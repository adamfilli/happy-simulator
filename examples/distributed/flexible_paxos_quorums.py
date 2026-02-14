"""Flexible Paxos with asymmetric quorum configurations.

This example compares standard majority quorums against asymmetric quorums
in Flexible Paxos. The key insight is that Phase 1 (prepare) and Phase 2
(accept) quorums need only satisfy Q1 + Q2 > N, not both be majorities.

Configurations compared (N=5):
- Standard: Q1=3, Q2=3 (classic majority)
- Fast writes: Q1=4, Q2=2 (larger prepare quorum, smaller accept quorum)

The fast-writes configuration commits with fewer acks per write at the
cost of requiring more responses during leader election (Phase 1).

## Architecture Diagram

```
+------------------------------------------------------------------+
|            FLEXIBLE PAXOS QUORUM COMPARISON                       |
+------------------------------------------------------------------+

    Standard Paxos (Q1=3, Q2=3):
    +-------+     +-------+     +-------+     +-------+     +-------+
    |Node 1 |<--->|Node 2 |<--->|Node 3 |<--->|Node 4 |<--->|Node 5 |
    +-------+     +-------+     +-------+     +-------+     +-------+
    Phase 1: need 3 promises   Phase 2: need 3 accepts

    Fast-Write Paxos (Q1=4, Q2=2):
    +-------+     +-------+     +-------+     +-------+     +-------+
    |Node 1 |<--->|Node 2 |<--->|Node 3 |<--->|Node 4 |<--->|Node 5 |
    +-------+     +-------+     +-------+     +-------+     +-------+
    Phase 1: need 4 promises   Phase 2: need 2 accepts (FASTER!)
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.components.network.network import Network
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.consensus.flexible_paxos import (
    FlexiblePaxosNode,
    FlexiblePaxosStats,
)
from happysimulator.components.consensus.raft_state_machine import KVStateMachine


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class ClusterResult:
    """Result for a single Flexible Paxos cluster."""
    label: str
    nodes: list[FlexiblePaxosNode]
    state_machines: dict[str, KVStateMachine]
    network: Network
    phase1_quorum: int
    phase2_quorum: int
    commands_submitted: int


@dataclass
class SimulationResult:
    """Results from the Flexible Paxos quorum comparison."""
    standard: ClusterResult
    fast_writes: ClusterResult
    duration_s: float


# =============================================================================
# Cluster Setup Helper
# =============================================================================


def _run_cluster(
    label: str,
    num_nodes: int,
    phase1_quorum: int,
    phase2_quorum: int,
    num_commands: int,
    duration_s: float,
    seed: int,
) -> ClusterResult:
    """Run a single Flexible Paxos cluster simulation.

    Args:
        label: Label for this cluster configuration.
        num_nodes: Number of nodes in the cluster.
        phase1_quorum: Phase 1 quorum size.
        phase2_quorum: Phase 2 quorum size.
        num_commands: Number of commands to submit.
        duration_s: Simulation duration.
        seed: Random seed.

    Returns:
        ClusterResult with all node states.
    """
    random.seed(seed)

    network = Network(name=f"{label}-net")
    state_machines: dict[str, KVStateMachine] = {}
    nodes: list[FlexiblePaxosNode] = []

    for i in range(1, num_nodes + 1):
        name = f"{label}-node-{i}"
        sm = KVStateMachine()
        state_machines[name] = sm
        node = FlexiblePaxosNode(
            name=name,
            network=network,
            state_machine=sm,
            phase1_quorum=phase1_quorum,
            phase2_quorum=phase2_quorum,
            heartbeat_interval=1.0,
        )
        nodes.append(node)

    for node in nodes:
        node.set_peers(nodes)

    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            network.add_bidirectional_link(
                a, b, datacenter_network(f"link-{a.name}-{b.name}")
            )

    # Start node-1 as the initial leader candidate
    def start_leader(event: Event):
        return nodes[0].start()

    start_evt = Event.once(
        time=Instant.from_seconds(0.01),
        event_type="StartLeader",
        fn=start_leader,
    )

    # Submit commands once leader is established
    def submit_commands(event: Event):
        leader = None
        for n in nodes:
            if n.is_leader:
                leader = n
                break

        if leader is None:
            retry = Event.once(
                time=event.time + 0.5,
                event_type="RetrySubmit",
                fn=submit_commands,
            )
            return [retry]

        events = []
        for i in range(num_commands):
            cmd = {"op": "set", "key": f"key-{i}", "value": i}

            def make_submit_fn(ld: FlexiblePaxosNode, command: dict):
                def fn(e: Event):
                    ld.submit(command)
                    # Trigger replication of the newly assigned slot
                    slot = ld.log.last_index
                    return ld._replicate_slot(slot)
                return fn

            evt = Event.once(
                time=event.time + 0.05 * i,
                event_type="SubmitCommand",
                fn=make_submit_fn(leader, cmd),
            )
            events.append(evt)
        return events

    submit_trigger = Event.once(
        time=Instant.from_seconds(2.0),
        event_type="SubmitCommands",
        fn=submit_commands,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s),
        entities=[network] + nodes,
    )
    sim.schedule(start_evt)
    sim.schedule(submit_trigger)
    sim.run()

    return ClusterResult(
        label=label,
        nodes=nodes,
        state_machines=state_machines,
        network=network,
        phase1_quorum=phase1_quorum,
        phase2_quorum=phase2_quorum,
        commands_submitted=num_commands,
    )


# =============================================================================
# Main Simulation
# =============================================================================


def run(args=None) -> SimulationResult:
    """Run Flexible Paxos comparison between standard and fast-write quorums.

    Args:
        args: Optional argparse namespace with simulation parameters.

    Returns:
        SimulationResult with both cluster results.
    """
    duration_s = getattr(args, "duration", 10.0) if args else 10.0
    num_commands = getattr(args, "commands", 10) if args else 10
    seed = getattr(args, "seed", 42) if args else 42

    effective_seed = seed if seed is not None and seed >= 0 else 42

    # Standard majority quorums: Q1=3, Q2=3 for N=5
    standard = _run_cluster(
        label="standard",
        num_nodes=5,
        phase1_quorum=3,
        phase2_quorum=3,
        num_commands=num_commands,
        duration_s=duration_s,
        seed=effective_seed,
    )

    # Fast writes: Q1=4, Q2=2 for N=5
    fast_writes = _run_cluster(
        label="fast-write",
        num_nodes=5,
        phase1_quorum=4,
        phase2_quorum=2,
        num_commands=num_commands,
        duration_s=duration_s,
        seed=effective_seed,
    )

    return SimulationResult(
        standard=standard,
        fast_writes=fast_writes,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def _print_cluster_summary(cluster: ClusterResult) -> None:
    """Print summary for a single cluster configuration."""
    print(f"\n  Configuration: {cluster.label}")
    print(f"  Quorums: Q1={cluster.phase1_quorum}, Q2={cluster.phase2_quorum} (N=5)")
    print(f"  Constraint: {cluster.phase1_quorum} + {cluster.phase2_quorum} = "
          f"{cluster.phase1_quorum + cluster.phase2_quorum} > 5")
    print(f"  Commands submitted: {cluster.commands_submitted}")

    print(f"\n  {'Node':<20} {'Leader':<8} {'Ballot':<8} {'Log Len':<9} "
          f"{'Committed':<11}")
    print(f"  {'-' * 56}")

    for node in cluster.nodes:
        s = node.stats
        leader_str = "YES" if s.is_leader else "no"
        print(f"  {node.name:<20} {leader_str:<8} {s.current_ballot:<8} "
              f"{s.log_length:<9} {s.commands_committed:<11}")

    total_committed = sum(n.stats.commands_committed for n in cluster.nodes)
    print(f"\n  Total committed (all nodes): {total_committed}")
    print(f"  Network events routed: {cluster.network.events_routed}")


def print_summary(result: SimulationResult) -> None:
    """Print comparison of Flexible Paxos quorum configurations."""
    print("\n" + "=" * 70)
    print("FLEXIBLE PAXOS QUORUM COMPARISON RESULTS")
    print("=" * 70)

    print(f"\nDuration: {result.duration_s}s")

    _print_cluster_summary(result.standard)
    _print_cluster_summary(result.fast_writes)

    # Comparison
    std_committed = sum(n.stats.commands_committed for n in result.standard.nodes)
    fw_committed = sum(n.stats.commands_committed for n in result.fast_writes.nodes)

    print(f"\n  {'Comparison':-^56}")
    print(f"  {'Metric':<30} {'Standard':<14} {'Fast-Write':<14}")
    print(f"  {'-' * 56}")
    print(f"  {'Phase 1 Quorum (Q1)':<30} {result.standard.phase1_quorum:<14} "
          f"{result.fast_writes.phase1_quorum:<14}")
    print(f"  {'Phase 2 Quorum (Q2)':<30} {result.standard.phase2_quorum:<14} "
          f"{result.fast_writes.phase2_quorum:<14}")
    print(f"  {'Total commits (all nodes)':<30} {std_committed:<14} {fw_committed:<14}")
    print(f"  {'Network messages':<30} {result.standard.network.events_routed:<14} "
          f"{result.fast_writes.network.events_routed:<14}")

    print(f"\n  Key insight: Fast-write configuration (Q2=2) can commit with")
    print(f"  fewer Phase 2 acks, potentially reducing write latency. The trade-off")
    print(f"  is that leader election (Phase 1) requires more responses (Q1=4).")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate comparison visualization of the two quorum configurations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    configs = [result.standard, result.fast_writes]
    labels = ["Standard\n(Q1=3, Q2=3)", "Fast-Write\n(Q1=4, Q2=2)"]
    colors = ["steelblue", "seagreen"]

    # Chart 1: Quorum sizes comparison
    ax = axes[0]
    x = range(len(labels))
    q1_sizes = [c.phase1_quorum for c in configs]
    q2_sizes = [c.phase2_quorum for c in configs]
    width = 0.35
    ax.bar([i - width / 2 for i in x], q1_sizes, width,
           label="Q1 (Prepare)", color="steelblue")
    ax.bar([i + width / 2 for i in x], q2_sizes, width,
           label="Q2 (Accept)", color="seagreen")
    ax.axhline(y=5, color="red", linestyle="--", alpha=0.5, label="N (cluster size)")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Quorum Size")
    ax.set_title("Quorum Sizes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 2: Total commits per configuration
    ax = axes[1]
    total_commits = [sum(n.stats.commands_committed for n in c.nodes) for c in configs]
    bars = ax.bar(labels, total_commits, color=colors)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Total Commits (all nodes)")
    ax.set_title("Commands Committed")
    for bar, val in zip(bars, total_commits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 3: Network messages
    ax = axes[2]
    messages = [c.network.events_routed for c in configs]
    bars = ax.bar(labels, messages, color=colors)
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Network Messages")
    ax.set_title("Total Network Messages")
    for bar, val in zip(bars, messages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Flexible Paxos: Standard vs. Fast-Write Quorums",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "flexible_paxos_quorums.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'flexible_paxos_quorums.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Flexible Paxos quorum comparison simulation"
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration (s)")
    parser.add_argument("--commands", type=int, default=10,
                        help="Number of commands to submit")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/flexible_paxos",
                        help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = None

    print("Running Flexible Paxos quorum comparison...")
    print(f"  Duration: {args.duration}s")
    print(f"  Commands: {args.commands}")
    print(f"  Random seed: {args.seed if args.seed is not None else 'random'}")

    result = run(args)
    print_summary(result)

    if not args.no_viz:
        try:
            import matplotlib
            matplotlib.use("Agg")
            output_dir = Path(args.output)
            visualize_results(result, output_dir)
            print(f"\nVisualizations saved to: {output_dir.absolute()}")
        except ImportError:
            print("\nSkipping visualization (matplotlib not installed)")
