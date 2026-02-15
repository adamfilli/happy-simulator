"""Integration test: 5-node gossip cluster with network partition.

Scenario:
- 5 nodes (A-E) form a full-mesh cluster with datacenter-speed links
- Each node periodically gossips its counter value to a random peer
- On receiving gossip, a node merges by taking the max value
- A network partition splits {A, B} from {C, D, E}
- After healing, the cluster reconverges

Visualization (3 panels):
1. Node state convergence over time
2. Traffic heatmap (per-pair packet counts)
3. Partition timeline with annotations
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import pytest

from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.network.network import LinkStats, Network, Partition
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.data import Data


# =============================================================================
# Entities
# =============================================================================


@dataclass
class GossipNode(Entity):
    """Node that periodically gossips its counter to a random peer.

    On receiving gossip, merges by taking max(local, remote).
    Self-schedules periodic gossip ticks as daemon events.
    """

    name: str
    network: Network | None = None
    peers: list[GossipNode] = field(default_factory=list)
    gossip_interval: float = 0.5  # seconds between gossip rounds
    increment: int = 1  # how much to increment per tick

    counter: int = field(default=0, init=False)
    _history: list[tuple[float, int]] = field(default_factory=list, init=False)

    def start(self) -> list[Event]:
        """Schedule the first gossip tick."""
        tick = Event(
            time=Instant.from_seconds(self.gossip_interval),
            event_type="GossipTick",
            target=self,
            daemon=True,
        )
        return [tick]

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "GossipTick":
            return self._handle_tick()
        elif event.event_type == "Gossip":
            return self._handle_gossip(event)
        return None

    def _handle_tick(self) -> list[Event]:
        events: list[Event] = []

        # Increment own counter
        self.counter += self.increment
        self._record()

        # Pick a random peer and gossip
        if self.peers and self.network is not None:
            peer = random.choice(self.peers)
            gossip_event = self.network.send(
                source=self,
                destination=peer,
                event_type="Gossip",
                payload={"value": self.counter, "origin": self.name},
                daemon=True,
            )
            events.append(gossip_event)

        # Schedule next tick
        next_tick = Event(
            time=Instant.from_seconds(
                self.now.to_seconds() + self.gossip_interval
            ),
            event_type="GossipTick",
            target=self,
            daemon=True,
        )
        events.append(next_tick)

        return events

    def _handle_gossip(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        remote_value = metadata.get("value", 0)
        # Merge: take max
        if remote_value > self.counter:
            self.counter = remote_value
            self._record()
        return None

    def _record(self) -> None:
        t = self.now.to_seconds() if self._clock is not None else 0.0
        self._history.append((t, self.counter))


@dataclass
class TrafficSampler(Entity):
    """Periodically samples network.traffic_matrix() into Data series."""

    name: str = "TrafficSampler"
    network: Network | None = None
    interval: float = 1.0
    _samples: list[tuple[float, list[LinkStats]]] = field(
        default_factory=list, init=False
    )

    def start(self) -> list[Event]:
        return [
            Event(
                time=Instant.from_seconds(self.interval),
                event_type="SampleTraffic",
                target=self,
                daemon=True,
            )
        ]

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type != "SampleTraffic":
            return None

        if self.network is not None:
            t = self.now.to_seconds()
            self._samples.append((t, self.network.traffic_matrix()))

        # Schedule next sample
        return [
            Event(
                time=Instant.from_seconds(
                    self.now.to_seconds() + self.interval
                ),
                event_type="SampleTraffic",
                target=self,
                daemon=True,
            )
        ]


# =============================================================================
# Test Scenarios
# =============================================================================


def _build_cluster(
    gossip_interval: float = 0.5,
) -> tuple[Network, list[GossipNode], TrafficSampler]:
    """Build a 5-node full-mesh cluster with datacenter links."""
    network = Network(name="ClusterNetwork")
    node_names = ["A", "B", "C", "D", "E"]
    nodes = [
        GossipNode(name=n, network=network, gossip_interval=gossip_interval)
        for n in node_names
    ]

    # Wire peers (everyone except self)
    for node in nodes:
        node.peers = [n for n in nodes if n is not node]

    # Full mesh with datacenter-speed links
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1 :]:
            link = datacenter_network(name=f"link_{src.name}_{dst.name}")
            network.add_bidirectional_link(src, dst, link)

    sampler = TrafficSampler(name="TrafficSampler", network=network)

    return network, nodes, sampler


class TestGossipCluster:
    """Integration tests for 5-node gossip cluster with network partition."""

    def test_pre_partition_convergence(self):
        """All 5 nodes converge to the same max counter before partition."""
        random.seed(42)

        network, nodes, sampler = _build_cluster(gossip_interval=0.2)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            sources=[],
            entities=[network, *nodes, sampler],
        )

        # Start gossip ticks for each node
        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        for evt in sampler.start():
            sim.schedule(evt)

        sim.run()

        # All nodes should have converged to the same max value
        values = [n.counter for n in nodes]
        assert len(set(values)) == 1, f"Nodes did not converge: {values}"
        assert values[0] > 0

    def test_partition_causes_divergence(self):
        """Partition splits cluster into groups that diverge."""
        random.seed(42)

        network, nodes, sampler = _build_cluster(gossip_interval=0.3)
        node_map = {n.name: n for n in nodes}

        # Give CDE group a faster increment so divergence is guaranteed
        node_map["C"].increment = 3
        node_map["D"].increment = 3
        node_map["E"].increment = 3

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            sources=[],
            entities=[network, *nodes, sampler],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        for evt in sampler.start():
            sim.schedule(evt)

        # Phase 1: Let cluster converge (0-5s)
        sim.control.add_breakpoint(
            __import__(
                "happysimulator.core.control.breakpoints",
                fromlist=["TimeBreakpoint"],
            ).TimeBreakpoint(time=Instant.from_seconds(5.0))
        )
        sim.run()

        values_at_5 = {n.name: n.counter for n in nodes}
        max_val_at_5 = max(values_at_5.values())
        min_val_at_5 = min(values_at_5.values())
        # All nodes should be close (within one gossip round of max)
        assert max_val_at_5 - min_val_at_5 <= 3, (
            f"Nodes not converged at t=5: {values_at_5}"
        )

        # Phase 2: Create partition {A,B} vs {C,D,E}
        group_ab = [node_map["A"], node_map["B"]]
        group_cde = [node_map["C"], node_map["D"], node_map["E"]]
        partition_handle = network.partition(group_ab, group_cde)

        assert partition_handle.is_active

        # Continue to t=20s
        sim.control.add_breakpoint(
            __import__(
                "happysimulator.core.control.breakpoints",
                fromlist=["TimeBreakpoint"],
            ).TimeBreakpoint(time=Instant.from_seconds(20.0))
        )
        sim.control.resume()

        # During partition: CDE group (increment=3) races ahead of AB (increment=1)
        ab_max = max(node_map["A"].counter, node_map["B"].counter)
        cde_max = max(
            node_map["C"].counter,
            node_map["D"].counter,
            node_map["E"].counter,
        )

        # CDE should be significantly ahead since they increment 3x faster
        assert cde_max > ab_max, (
            f"CDE group ({cde_max}) should be ahead of AB group ({ab_max})"
        )

        # Phase 3: Heal partition and let reconverge
        partition_handle.heal()
        assert not partition_handle.is_active

        sim.control.resume()  # Run to end (t=30)

        # Post-heal: all nodes should reconverge to the max (CDE's higher value)
        final_values = [n.counter for n in nodes]
        final_max = max(final_values)
        final_min = min(final_values)
        # All nodes should be close to max (within a few gossip rounds)
        assert final_max - final_min <= 6, (
            f"Nodes did not reconverge: {final_values}"
        )
        assert final_max >= max(ab_max, cde_max)

    def test_traffic_stats_reflect_partition(self):
        """Cross-partition pairs have no traffic during partition."""
        random.seed(42)

        network, nodes, sampler = _build_cluster(gossip_interval=0.3)
        node_map = {n.name: n for n in nodes}

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=15.0,
            sources=[],
            entities=[network, *nodes, sampler],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        for evt in sampler.start():
            sim.schedule(evt)

        # Partition immediately
        group_ab = [node_map["A"], node_map["B"]]
        group_cde = [node_map["C"], node_map["D"], node_map["E"]]
        network.partition(group_ab, group_cde)

        sim.run()

        # Verify cross-partition links have 0 packets
        matrix = network.traffic_matrix()
        cross_partition_pairs = set()
        for a_name in ["A", "B"]:
            for c_name in ["C", "D", "E"]:
                cross_partition_pairs.add((a_name, c_name))
                cross_partition_pairs.add((c_name, a_name))

        for stats in matrix:
            if (stats.source, stats.destination) in cross_partition_pairs:
                assert stats.packets_sent == 0, (
                    f"Cross-partition pair {stats.source}->{stats.destination} "
                    f"should have 0 packets, got {stats.packets_sent}"
                )

        # Verify intra-group links have traffic
        intra_pairs_seen = set()
        for stats in matrix:
            pair = (stats.source, stats.destination)
            if pair not in cross_partition_pairs and stats.packets_sent > 0:
                intra_pairs_seen.add(pair)

        assert len(intra_pairs_seen) > 0, "No intra-group traffic observed"

    def test_selective_heal_preserves_other_partition(self):
        """Healing one partition leaves other partitions intact."""
        random.seed(42)

        network, nodes, _ = _build_cluster(gossip_interval=0.5)
        node_map = {n.name: n for n in nodes}

        # Create two separate partitions
        p1 = network.partition([node_map["A"]], [node_map["C"]])
        p2 = network.partition([node_map["B"]], [node_map["D"]])

        assert p1.is_active
        assert p2.is_active

        # Heal only p1
        p1.heal()

        assert not p1.is_active
        assert p2.is_active

        # A <-> C should be healed
        assert not network.is_partitioned("A", "C")
        # B <-> D should still be partitioned
        assert network.is_partitioned("B", "D")

    def test_partition_drops_are_counted(self):
        """Network counts dropped events due to partition."""
        random.seed(42)

        network, nodes, sampler = _build_cluster(gossip_interval=0.3)
        node_map = {n.name: n for n in nodes}

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            sources=[],
            entities=[network, *nodes, sampler],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        for evt in sampler.start():
            sim.schedule(evt)

        # Create partition
        group_ab = [node_map["A"], node_map["B"]]
        group_cde = [node_map["C"], node_map["D"], node_map["E"]]
        network.partition(group_ab, group_cde)

        sim.run()

        # Should have some routed events and some dropped
        assert network.events_routed > 0
        assert network.events_dropped_partition > 0

    def test_full_scenario_with_visualization(self, test_output_dir: Path):
        """Full gossip scenario with partition, heal, and 3-panel plot."""
        random.seed(42)

        network, nodes, sampler = _build_cluster(gossip_interval=0.3)
        node_map = {n.name: n for n in nodes}

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            sources=[],
            entities=[network, *nodes, sampler],
        )

        for node in nodes:
            for evt in node.start():
                sim.schedule(evt)
        for evt in sampler.start():
            sim.schedule(evt)

        # Phase 1: warm-up (0-5s)
        from happysimulator.core.control.breakpoints import TimeBreakpoint

        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(5.0))
        )
        sim.run()

        # Record pre-partition stats
        pre_partition_matrix = network.traffic_matrix()

        # Phase 2: partition at t=5 {A,B} vs {C,D,E}
        partition_start = 5.0
        group_ab = [node_map["A"], node_map["B"]]
        group_cde = [node_map["C"], node_map["D"], node_map["E"]]
        partition_handle = network.partition(group_ab, group_cde)

        sim.control.add_breakpoint(
            TimeBreakpoint(time=Instant.from_seconds(20.0))
        )
        sim.control.resume()

        # Phase 3: heal at t=20
        partition_end = 20.0
        partition_handle.heal()

        # Run to completion
        sim.control.resume()

        # Assertions: all nodes reconverge
        final_values = [n.counter for n in nodes]
        assert len(set(final_values)) == 1, (
            f"Nodes did not reconverge: {final_values}"
        )

        # Generate visualization
        _generate_cluster_visualization(
            nodes=nodes,
            network=network,
            sampler=sampler,
            partition_start=partition_start,
            partition_end=partition_end,
            output_dir=test_output_dir,
        )


def _generate_cluster_visualization(
    nodes: list[GossipNode],
    network: Network,
    sampler: TrafficSampler,
    partition_start: float,
    partition_end: float,
    output_dir: Path,
) -> None:
    """Generate 3-panel visualization of the gossip cluster scenario."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    node_colors = {
        "A": "#e74c3c",
        "B": "#e67e22",
        "C": "#2ecc71",
        "D": "#3498db",
        "E": "#9b59b6",
    }

    # Panel 1: Node state convergence
    ax1 = axes[0]
    for node in nodes:
        if node._history:
            times = [t for t, _ in node._history]
            values = [v for _, v in node._history]
            ax1.plot(
                times,
                values,
                label=f"Node {node.name}",
                color=node_colors[node.name],
                linewidth=1.5,
            )

    ax1.axvspan(
        partition_start,
        partition_end,
        alpha=0.15,
        color="red",
        label="Partition active",
    )
    ax1.axvline(partition_start, color="red", linestyle="--", alpha=0.7)
    ax1.axvline(partition_end, color="green", linestyle="--", alpha=0.7)
    ax1.set_title("Node State Convergence")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Counter Value")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Traffic heatmap (per-pair packet counts)
    ax2 = axes[1]
    node_names = [n.name for n in nodes]
    n = len(node_names)
    matrix_data = [[0] * n for _ in range(n)]

    final_matrix = network.traffic_matrix()
    name_to_idx = {name: i for i, name in enumerate(node_names)}
    for stats in final_matrix:
        if stats.source in name_to_idx and stats.destination in name_to_idx:
            i = name_to_idx[stats.source]
            j = name_to_idx[stats.destination]
            matrix_data[i][j] = stats.packets_sent

    im = ax2.imshow(matrix_data, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(node_names)
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(node_names)
    ax2.set_xlabel("Destination")
    ax2.set_ylabel("Source")
    ax2.set_title("Traffic Matrix (packets sent)")
    fig.colorbar(im, ax=ax2, label="Packets")

    # Annotate cells with counts
    for i in range(n):
        for j in range(n):
            val = matrix_data[i][j]
            color = "white" if val > max(max(row) for row in matrix_data) * 0.6 else "black"
            ax2.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    # Panel 3: Partition timeline
    ax3 = axes[2]
    ax3.barh(
        0,
        partition_end - partition_start,
        left=partition_start,
        height=0.4,
        color="red",
        alpha=0.6,
        label="Partition: {A,B} <-> {C,D,E}",
    )
    ax3.set_xlim(0, 30)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_yticks([])
    ax3.set_xlabel("Time (s)")
    ax3.set_title("Partition Timeline")

    ax3.annotate(
        "Partition start",
        xy=(partition_start, 0),
        xytext=(partition_start - 2, 0.3),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )
    ax3.annotate(
        "Heal",
        xy=(partition_end, 0),
        xytext=(partition_end + 1, 0.3),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9,
        color="green",
    )

    # Add stats annotations
    ax3.text(
        0.02,
        -0.35,
        f"Events routed: {network.events_routed}  |  "
        f"Dropped (partition): {network.events_dropped_partition}",
        transform=ax3.transAxes,
        fontsize=9,
        verticalalignment="top",
    )

    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(output_dir / "gossip_cluster_analysis.png", dpi=150)
    plt.close(fig)
