"""SWIM membership protocol with failure detection via network partitions.

This example demonstrates the SWIM protocol for membership management:
1. A 5-node cluster starts with all nodes alive
2. Nodes periodically probe each other via ping/ack
3. A network partition isolates 2 nodes from the rest
4. The remaining 3 nodes detect the failure (suspect -> dead)
5. The partition is healed and membership stats are printed

## Architecture Diagram

```
+------------------------------------------------------------------+
|               SWIM MEMBERSHIP PROTOCOL SIMULATION                 |
+------------------------------------------------------------------+

    Phase 1: Healthy cluster (all probes succeed)
    +---------+     +---------+     +---------+
    | Node 1  |<--->| Node 2  |<--->| Node 3  |
    +---------+     +---------+     +---------+
         ^               ^               ^
         |          +---------+          |
         +--------->| Node 4  |<---------+
         |          +---------+          |
         |          +---------+          |
         +--------->| Node 5  |<---------+
                    +---------+

    Phase 2: Partition isolates nodes 4,5
    +---------+     +---------+     +---------+
    | Node 1  |<--->| Node 2  |<--->| Node 3  |
    +---------+     +---------+     +---------+
         X               X               X
         |          +---------+          |
         X    X     | Node 4  |     X    X
         |          +---------+          |
         |          +---------+          |
         X    X     | Node 5  |     X    X
                    +---------+

    Phase 3: Nodes 1-3 detect 4,5 as SUSPECT then DEAD
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.components.network.network import Network, Partition
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.consensus.membership import (
    MembershipProtocol,
    MembershipStats,
    MemberState,
)


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the SWIM membership simulation."""
    protocols: list[MembershipProtocol]
    network: Network
    partition_time_s: float
    heal_time_s: float
    partitioned_nodes: list[str]
    duration_s: float


# =============================================================================
# Main Simulation
# =============================================================================


def run(args=None) -> SimulationResult:
    """Run SWIM membership simulation with a network partition.

    Args:
        args: Optional argparse namespace with simulation parameters.

    Returns:
        SimulationResult with all protocol states and statistics.
    """
    duration_s = getattr(args, "duration", 30.0) if args else 30.0
    partition_time = getattr(args, "partition_time", 8.0) if args else 8.0
    heal_time = getattr(args, "heal_time", 22.0) if args else 22.0
    seed = getattr(args, "seed", 42) if args else 42

    if seed is not None and seed >= 0:
        random.seed(seed)

    # Create network
    network = Network(name="swim-net")

    # Create membership protocol instances (one per node)
    protocols: list[MembershipProtocol] = []
    for i in range(1, 6):
        proto = MembershipProtocol(
            name=f"node-{i}",
            network=network,
            probe_interval=1.0,
            suspicion_timeout=5.0,
            indirect_probe_count=2,
            phi_threshold=8.0,
        )
        protocols.append(proto)

    # Register members with each other
    for proto in protocols:
        for other in protocols:
            if other.name != proto.name:
                proto.add_member(other)

    # Create datacenter links between all pairs
    for i, a in enumerate(protocols):
        for b in protocols[i + 1:]:
            network.add_bidirectional_link(
                a, b, datacenter_network(f"link-{a.name}-{b.name}")
            )

    # Schedule protocol starts
    start_events: list[Event] = []
    for proto in protocols:

        def make_start_fn(p: MembershipProtocol):
            def fn(event: Event):
                return p.start()
            return fn

        evt = Event.once(
            time=Instant.from_seconds(0.01),
            event_type="StartProtocol",
            fn=make_start_fn(proto),
        )
        start_events.append(evt)

    # Schedule network partition at partition_time
    partitioned_names = ["node-4", "node-5"]
    group_a = [p for p in protocols if p.name in ("node-1", "node-2", "node-3")]
    group_b = [p for p in protocols if p.name in partitioned_names]

    partition_handle: list[Partition] = []  # mutable container for closure

    def create_partition(event: Event):
        p = network.partition(group_a, group_b)
        partition_handle.append(p)
        return None

    partition_evt = Event.once(
        time=Instant.from_seconds(partition_time),
        event_type="CreatePartition",
        fn=create_partition,
    )

    # Schedule partition heal
    def heal_partition(event: Event):
        if partition_handle:
            partition_handle[0].heal()
        return None

    heal_evt = Event.once(
        time=Instant.from_seconds(heal_time),
        event_type="HealPartition",
        fn=heal_partition,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s,
        entities=[network] + protocols,
    )

    for evt in start_events:
        sim.schedule(evt)
    sim.schedule(partition_evt)
    sim.schedule(heal_evt)

    sim.run()

    return SimulationResult(
        protocols=protocols,
        network=network,
        partition_time_s=partition_time,
        heal_time_s=heal_time,
        partitioned_nodes=partitioned_names,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary of the SWIM membership simulation."""
    print("\n" + "=" * 70)
    print("SWIM MEMBERSHIP PROTOCOL SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nCluster size: {len(result.protocols)} nodes")
    print(f"Partition at: t={result.partition_time_s}s (isolated {result.partitioned_nodes})")
    print(f"Healed at:    t={result.heal_time_s}s")
    print(f"Duration:     {result.duration_s}s")

    print(f"\nMembership View per Node:")
    print(f"  {'Observer':<10} {'Alive':<30} {'Suspect':<20} {'Dead':<20}")
    print(f"  {'-' * 80}")

    for proto in result.protocols:
        alive = proto.alive_members
        suspect = proto.suspected_members
        dead = proto.dead_members
        alive_str = ", ".join(sorted(alive)) if alive else "-"
        suspect_str = ", ".join(sorted(suspect)) if suspect else "-"
        dead_str = ", ".join(sorted(dead)) if dead else "-"
        print(f"  {proto.name:<10} {alive_str:<30} {suspect_str:<20} {dead_str:<20}")

    print(f"\nProtocol Statistics:")
    print(f"  {'Node':<10} {'Probes':<8} {'Indirect':<10} {'Acks':<8} "
          f"{'Updates':<10} {'Alive':<7} {'Suspect':<9} {'Dead':<6}")
    print(f"  {'-' * 68}")

    for proto in result.protocols:
        s = proto.stats
        print(f"  {proto.name:<10} {s.probes_sent:<8} {s.indirect_probes_sent:<10} "
              f"{s.acks_received:<8} {s.updates_disseminated:<10} "
              f"{s.alive_count:<7} {s.suspect_count:<9} {s.dead_count:<6}")

    # Check that nodes in the majority partition detected the minority as failed
    majority_nodes = [p for p in result.protocols
                      if p.name not in result.partitioned_nodes]
    detected_count = 0
    for p in majority_nodes:
        for partitioned_name in result.partitioned_nodes:
            state = p.get_member_state(partitioned_name)
            if state in (MemberState.SUSPECT, MemberState.DEAD):
                detected_count += 1

    total_expected = len(majority_nodes) * len(result.partitioned_nodes)
    print(f"\nFailure Detection:")
    print(f"  Partitioned nodes detected as suspect/dead: "
          f"{detected_count}/{total_expected}")

    print(f"\nNetwork Statistics:")
    print(f"  Events routed:  {result.network.events_routed}")
    print(f"  Dropped (partition): {result.network.events_dropped_partition}")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualization of SWIM membership statistics."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    node_names = [p.name for p in result.protocols]

    # Stacked bar chart: member states per observer
    ax = axes[0]
    alive_counts = [p.stats.alive_count for p in result.protocols]
    suspect_counts = [p.stats.suspect_count for p in result.protocols]
    dead_counts = [p.stats.dead_count for p in result.protocols]

    x = range(len(node_names))
    ax.bar(x, alive_counts, label="Alive", color="seagreen")
    ax.bar(x, suspect_counts, bottom=alive_counts, label="Suspect", color="gold")
    ax.bar(x, dead_counts,
           bottom=[a + s for a, s in zip(alive_counts, suspect_counts)],
           label="Dead", color="indianred")
    ax.set_xlabel("Observer Node")
    ax.set_ylabel("Member Count")
    ax.set_title("Membership View per Node")
    ax.set_xticks(x)
    ax.set_xticklabels(node_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Bar chart: probe and ack statistics
    ax = axes[1]
    probes = [p.stats.probes_sent for p in result.protocols]
    indirect = [p.stats.indirect_probes_sent for p in result.protocols]
    acks = [p.stats.acks_received for p in result.protocols]

    width = 0.25
    ax.bar([i - width for i in x], probes, width, label="Probes", color="steelblue")
    ax.bar(x, indirect, width, label="Indirect Probes", color="gold")
    ax.bar([i + width for i in x], acks, width, label="Acks", color="seagreen")
    ax.set_xlabel("Node")
    ax.set_ylabel("Count")
    ax.set_title("Protocol Message Statistics")
    ax.set_xticks(x)
    ax.set_xticklabels(node_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("SWIM Membership Protocol", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "swim_membership.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'swim_membership.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SWIM membership protocol simulation")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration (s)")
    parser.add_argument("--partition-time", type=float, default=8.0,
                        help="Time to create partition (s)")
    parser.add_argument("--heal-time", type=float, default=22.0,
                        help="Time to heal partition (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/swim", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    if args.seed == -1:
        args.seed = None

    print("Running SWIM membership protocol simulation...")
    print(f"  Duration: {args.duration}s")
    print(f"  Partition at: {args.partition_time}s, heal at: {args.heal_time}s")
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
