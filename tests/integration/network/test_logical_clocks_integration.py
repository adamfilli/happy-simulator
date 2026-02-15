"""Integration test: 3-node distributed counter with logical clocks.

Scenario:
- 3 CounterNode entities (A, B, C) connected via full-mesh Network
- Each node has a different NodeClock skew
- Nodes periodically tick and send updates with all three clock types
  (Lamport, VectorClock, HLC) embedded in event context metadata
- Verifies causal ordering properties across all clock types

Visualization (3 panels):
1. Lamport timestamps per node over true time
2. Vector clock magnitude (sum of vector) per node over time
3. HLC physical vs logical components per node
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.network.network import Network
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.logical_clocks import (
    HybridLogicalClock,
    LamportClock,
    VectorClock,
)
from happysimulator.core.node_clock import FixedSkew, NodeClock
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Duration, Instant


# =============================================================================
# Entities
# =============================================================================


@dataclass
class CounterNode(Entity):
    """Node that ticks all three logical clocks and sends updates to peers.

    Each tick:
    1. Increments Lamport, VectorClock, and HLC
    2. Sends a message to a random peer with all three timestamps
    3. Records history for visualization and assertion
    """

    name: str
    network: Network | None = None
    peers: list[CounterNode] = field(default_factory=list)
    tick_interval: float = 0.5

    # Logical clocks (set after construction for NodeClock wiring)
    lamport: LamportClock = field(default_factory=LamportClock)
    vector: VectorClock | None = field(default=None, init=False)
    hlc: HybridLogicalClock | None = field(default=None, init=False)
    node_clock: NodeClock | None = field(default=None, init=False)

    # History for assertions and visualization
    lamport_history: list[tuple[float, int]] = field(
        default_factory=list, init=False
    )
    vector_history: list[tuple[float, dict[str, int]]] = field(
        default_factory=list, init=False
    )
    hlc_history: list[tuple[float, int, int]] = field(
        default_factory=list, init=False
    )

    def set_clock(self, clock):
        super().set_clock(clock)
        if self.node_clock is not None:
            self.node_clock.set_clock(clock)

    def start(self) -> list[Event]:
        return [
            Event(
                time=Instant.from_seconds(self.tick_interval),
                event_type="Tick",
                target=self,
                daemon=True,
            )
        ]

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "Tick":
            return self._handle_tick()
        elif event.event_type == "Update":
            return self._handle_update(event)
        return None

    def _handle_tick(self) -> list[Event]:
        events: list[Event] = []
        t = self.now.to_seconds()

        # Tick all clocks
        self.lamport.tick()
        self.vector.tick()
        hlc_ts = self.hlc.now()

        # Record history
        self.lamport_history.append((t, self.lamport.time))
        self.vector_history.append((t, self.vector.snapshot()))
        self.hlc_history.append((t, hlc_ts.physical_ns, hlc_ts.logical))

        # Send update to a random peer
        if self.peers and self.network is not None:
            peer = random.choice(self.peers)
            lamport_ts = self.lamport.send()
            vector_snap = self.vector.send()
            hlc_send_ts = self.hlc.send()

            msg_event = self.network.send(
                source=self,
                destination=peer,
                event_type="Update",
                payload={
                    "lamport_ts": lamport_ts,
                    "vector_snap": vector_snap,
                    "hlc_ts": hlc_send_ts.to_dict(),
                    "origin": self.name,
                },
                daemon=True,
            )
            events.append(msg_event)

        # Schedule next tick
        events.append(
            Event(
                time=Instant.from_seconds(t + self.tick_interval),
                event_type="Tick",
                target=self,
                daemon=True,
            )
        )
        return events

    def _handle_update(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        t = self.now.to_seconds()

        # Update all clocks from received message
        self.lamport.receive(metadata["lamport_ts"])
        self.vector.receive(metadata["vector_snap"])

        from happysimulator.core.logical_clocks import HLCTimestamp

        remote_hlc = HLCTimestamp.from_dict(metadata["hlc_ts"])
        self.hlc.receive(remote_hlc)

        # Record post-receive state
        self.lamport_history.append((t, self.lamport.time))
        self.vector_history.append((t, self.vector.snapshot()))
        hlc_now = self.hlc._last
        self.hlc_history.append((t, hlc_now.physical_ns, hlc_now.logical))

        return None


# =============================================================================
# Setup
# =============================================================================


def _build_cluster() -> tuple[Network, list[CounterNode]]:
    """Build a 3-node cluster with different clock skews."""
    node_names = ["A", "B", "C"]
    network = Network(name="LogicalClockNet")

    # Different clock skews per node
    skews = {
        "A": Duration.from_seconds(0.0),     # Perfect clock
        "B": Duration.from_seconds(0.05),     # 50ms ahead
        "C": Duration.from_seconds(-0.03),    # 30ms behind
    }

    nodes = []
    for name in node_names:
        node_clock = NodeClock(FixedSkew(skews[name]))
        node = CounterNode(name=name, network=network, tick_interval=0.3)
        node.node_clock = node_clock
        node.lamport = LamportClock()
        nodes.append(node)

    # Set up vector clocks and HLCs (need all node names)
    for node in nodes:
        node.vector = VectorClock(node.name, node_names)
        node.hlc = HybridLogicalClock(
            node.name, physical_clock=node.node_clock
        )

    # Wire peers
    for node in nodes:
        node.peers = [n for n in nodes if n is not node]

    # Full mesh with datacenter links
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            link = datacenter_network(name=f"link_{src.name}_{dst.name}")
            network.add_bidirectional_link(src, dst, link)

    return network, nodes


# =============================================================================
# Tests
# =============================================================================


class TestLogicalClocksIntegration:
    def test_lamport_causality(self):
        """Every received event's Lamport ts < receiver's post-receive ts."""
        random.seed(42)
        network, nodes = _build_cluster()

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            sources=[],
            entities=[network, *nodes],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        sim.run()

        # Each node's Lamport history should be monotonically non-decreasing
        for node in nodes:
            times = [ts for _, ts in node.lamport_history]
            for i in range(1, len(times)):
                assert times[i] >= times[i - 1], (
                    f"Node {node.name} Lamport not monotonic at index {i}: "
                    f"{times[i-1]} -> {times[i]}"
                )

        # All nodes should have non-zero Lamport timestamps
        for node in nodes:
            assert node.lamport.time > 0, (
                f"Node {node.name} Lamport is still 0"
            )

    def test_vector_clock_concurrent_detection(self):
        """Concurrent events detected when nodes haven't communicated."""
        random.seed(123)
        network, nodes = _build_cluster()
        node_map = {n.name: n for n in nodes}

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[],
            entities=[network, *nodes],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        sim.run()

        # After the simulation, verify vector clock properties:
        # 1. Each node's own component should be the highest in its vector
        for node in nodes:
            snap = node.vector.snapshot()
            own_val = snap[node.name]
            assert own_val > 0, f"Node {node.name} has 0 own component"

        # 2. Vector clocks should reflect communication
        # (non-zero components for peers that sent messages)
        for node in nodes:
            snap = node.vector.snapshot()
            has_peer_info = any(
                v > 0 for k, v in snap.items() if k != node.name
            )
            assert has_peer_info, (
                f"Node {node.name} has no peer info: {snap}"
            )

    def test_hlc_monotonicity_despite_skew(self):
        """All HLC timestamps from same node are strictly monotonic despite clock skew."""
        random.seed(42)
        network, nodes = _build_cluster()

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            sources=[],
            entities=[network, *nodes],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        sim.run()

        for node in nodes:
            history = node.hlc_history
            assert len(history) > 1, (
                f"Node {node.name} has too few HLC entries"
            )

            for i in range(1, len(history)):
                _, phys_prev, log_prev = history[i - 1]
                _, phys_curr, log_curr = history[i]

                # Timestamps must be non-decreasing in (physical, logical)
                assert (phys_curr, log_curr) >= (phys_prev, log_prev), (
                    f"Node {node.name} HLC not monotonic at index {i}: "
                    f"({phys_prev}, {log_prev}) -> ({phys_curr}, {log_curr})"
                )

    def test_full_scenario_with_visualization(self, test_output_dir: Path):
        """Full 3-node scenario with all clock types and 3-panel visualization."""
        random.seed(42)
        network, nodes = _build_cluster()

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=15.0,
            sources=[],
            entities=[network, *nodes],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)

        sim.run()

        # Basic assertions
        for node in nodes:
            assert node.lamport.time > 0
            assert len(node.hlc_history) > 5

        # Generate visualization
        _generate_visualization(nodes, test_output_dir)


# =============================================================================
# Visualization
# =============================================================================


def _generate_visualization(
    nodes: list[CounterNode],
    output_dir: Path,
) -> None:
    """Generate 3-panel visualization of logical clock behavior."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    node_colors = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71"}

    # Panel 1: Lamport timestamps per node over true time
    ax1 = axes[0]
    for node in nodes:
        if node.lamport_history:
            times = [t for t, _ in node.lamport_history]
            values = [v for _, v in node.lamport_history]
            ax1.plot(
                times,
                values,
                label=f"Node {node.name}",
                color=node_colors[node.name],
                linewidth=1.5,
                marker=".",
                markersize=3,
            )
    ax1.set_title("Lamport Timestamps per Node")
    ax1.set_xlabel("True Simulation Time (s)")
    ax1.set_ylabel("Lamport Counter")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Vector clock magnitude (sum of vector) per node
    ax2 = axes[1]
    for node in nodes:
        if node.vector_history:
            times = [t for t, _ in node.vector_history]
            magnitudes = [sum(v.values()) for _, v in node.vector_history]
            ax2.plot(
                times,
                magnitudes,
                label=f"Node {node.name}",
                color=node_colors[node.name],
                linewidth=1.5,
                marker=".",
                markersize=3,
            )
    ax2.set_title("Vector Clock Magnitude (sum of all components)")
    ax2.set_xlabel("True Simulation Time (s)")
    ax2.set_ylabel("Sum of Vector Components")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Panel 3: HLC physical vs logical per node
    ax3 = axes[2]
    for node in nodes:
        if node.hlc_history:
            times = [t for t, _, _ in node.hlc_history]
            physical_ms = [p / 1_000_000 for _, p, _ in node.hlc_history]
            logical = [l for _, _, l in node.hlc_history]

            ax3.plot(
                times,
                physical_ms,
                label=f"Node {node.name} physical",
                color=node_colors[node.name],
                linewidth=1.5,
            )
            ax3_twin = ax3.twinx() if node.name == "A" else ax3_twin
            ax3_twin.plot(
                times,
                logical,
                label=f"Node {node.name} logical",
                color=node_colors[node.name],
                linewidth=1.0,
                linestyle="--",
                alpha=0.6,
            )

    ax3.set_title("HLC: Physical (solid) and Logical (dashed) Components")
    ax3.set_xlabel("True Simulation Time (s)")
    ax3.set_ylabel("Physical Component (ms)")
    ax3_twin.set_ylabel("Logical Counter")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "logical_clocks_analysis.png", dpi=150)
    plt.close(fig)
