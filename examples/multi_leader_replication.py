"""Multi-leader replication with conflict detection and anti-entropy.

Demonstrates concurrent writes from different regions, conflict detection
via vector clocks, resolution via LWW, and convergence after partition
healing through anti-entropy synchronization.

## Architecture

```
  Writer-East ──► Leader-East ◄──► Network ◄──► Leader-West ◄── Writer-West
                     │                              │
                  KVStore                        KVStore
                (local data)                  (local data)
                     │          anti-entropy        │
                     └──────────────────────────────┘
```

## Phases

1. **Normal** (0-15s): Both leaders accept writes, replicate to each other.
2. **Partition** (15-30s): Network partition — writes diverge.
3. **Heal + Anti-Entropy** (30-45s): Partition healed, anti-entropy repairs.

## Key Observations

- During normal operation, writes converge quickly through replication.
- During partition, each leader diverges — same keys get different values.
- After healing, anti-entropy detects divergent keys via MerkleTree
  comparison and reconciles using the configured ConflictResolver.
- Conflict count increases during partition, then stabilizes after heal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Entity,
    Event,
    Instant,
    Network,
    Simulation,
    SimFuture,
    Source,
    datacenter_network,
)
from happysimulator.components.datastore import KVStore
from happysimulator.components.replication.conflict_resolver import LastWriterWins
from happysimulator.components.replication.multi_leader import LeaderNode


# =============================================================================
# Writer entity
# =============================================================================


class RegionalWriter(Entity):
    """Sends keyed writes to a local leader."""

    def __init__(self, name: str, leader: LeaderNode):
        super().__init__(name)
        self.leader = leader
        self._count = 0

    @property
    def write_count(self) -> int:
        return self._count

    def handle_event(self, event: Event) -> list[Event]:
        self._count += 1
        key = f"user-{self._count % 20}"
        value = f"{self.name}:{self._count}"

        write_event = Event(
            time=self.now,
            event_type="Write",
            target=self.leader,
            context={"metadata": {"key": key, "value": value}},
        )
        return [write_event]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class MultiLeaderResult:
    """Results from the multi-leader simulation."""

    leaders: list[LeaderNode]
    writers: list[RegionalWriter]
    partition_start: float
    partition_end: float


def run_simulation(
    *,
    duration_s: float = 45.0,
    write_rate: float = 20.0,
    anti_entropy_interval: float = 2.0,
    partition_start: float = 15.0,
    partition_end: float = 30.0,
    seed: int = 42,
) -> MultiLeaderResult:
    """Run the multi-leader replication simulation."""
    random.seed(seed)

    # Stores and network
    network = Network(name="net")

    # Create leaders
    leaders = [
        LeaderNode(
            f"leader-{region}",
            store=KVStore(f"store-{region}", write_latency=0.001, read_latency=0.001),
            network=network,
            conflict_resolver=LastWriterWins(),
            anti_entropy_interval=anti_entropy_interval,
        )
        for region in ["east", "west"]
    ]
    for leader in leaders:
        leader.add_peers([l for l in leaders if l is not leader])

    # Network links
    network.add_bidirectional_link(
        leaders[0], leaders[1], datacenter_network("cross-region"),
    )

    # Writers
    writers = [
        RegionalWriter(f"writer-{region}", leader)
        for region, leader in zip(["east", "west"], leaders)
    ]

    # Sources
    sources = [
        Source.constant(
            rate=write_rate,
            target=writer,
            event_type="NewWrite",
            stop_after=duration_s,
        )
        for writer in writers
    ]

    # Anti-entropy bootstrap events
    ae_events = []
    for leader in leaders:
        ae_events.append(Event(
            time=Instant.from_seconds(anti_entropy_interval),
            event_type="AntiEntropy",
            target=leader,
            daemon=True,
        ))

    # Partition/heal events
    from happysimulator import CallbackEntity

    partition_handle = None

    def do_partition(event: Event) -> None:
        nonlocal partition_handle
        partition_handle = network.partition([leaders[0]], [leaders[1]])

    def do_heal(event: Event) -> None:
        if partition_handle is not None:
            partition_handle.heal()

    partition_entity = CallbackEntity("partition_ctrl", fn=do_partition)
    heal_entity = CallbackEntity("heal_ctrl", fn=do_heal)

    partition_event = Event(
        time=Instant.from_seconds(partition_start),
        event_type="Partition",
        target=partition_entity,
    )
    heal_event = Event(
        time=Instant.from_seconds(partition_end),
        event_type="Heal",
        target=heal_entity,
    )

    # Assemble all entities
    all_entities: list = [*leaders, *writers, network, partition_entity, heal_entity]
    for l in leaders:
        all_entities.append(l.store)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 5.0),
        sources=sources,
        entities=all_entities,
    )
    for ae in ae_events:
        sim.schedule(ae)
    sim.schedule([partition_event, heal_event])
    sim.run()

    return MultiLeaderResult(
        leaders=leaders,
        writers=writers,
        partition_start=partition_start,
        partition_end=partition_end,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: MultiLeaderResult) -> None:
    """Print per-leader statistics."""
    print("\n" + "=" * 70)
    print("MULTI-LEADER REPLICATION — CONFLICT & CONVERGENCE")
    print("=" * 70)

    print(f"\nPartition window: {result.partition_start}s - {result.partition_end}s")

    for writer in result.writers:
        print(f"  {writer.name}: {writer.write_count} writes")

    print(f"\nPer-Leader Statistics:")
    header = (
        f"  {'Leader':<15s} {'Writes':>7s} {'ReplSent':>9s} {'ReplRecv':>9s} "
        f"{'Conflicts':>10s} {'AEsyncs':>8s} {'AErepairs':>10s} {'Keys':>5s}"
    )
    print(header)
    print(f"  {'-' * 75}")

    for leader in result.leaders:
        s = leader.stats
        print(
            f"  {leader.name:<15s} "
            f"{s.writes:>7d} "
            f"{s.replications_sent:>9d} "
            f"{s.replications_received:>9d} "
            f"{s.conflicts_detected:>10d} "
            f"{s.anti_entropy_syncs:>8d} "
            f"{s.anti_entropy_keys_repaired:>10d} "
            f"{leader.store.size:>5d}"
        )

    # Check convergence
    keys_0 = set(result.leaders[0].store.keys())
    keys_1 = set(result.leaders[1].store.keys())
    common = keys_0 & keys_1
    divergent = 0
    for key in common:
        if result.leaders[0].store.get_sync(key) != result.leaders[1].store.get_sync(key):
            divergent += 1

    print(f"\nConvergence:")
    print(f"  Keys on leader-east: {len(keys_0)}")
    print(f"  Keys on leader-west: {len(keys_1)}")
    print(f"  Common keys:         {len(common)}")
    print(f"  Divergent values:    {divergent}")
    print("=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: MultiLeaderResult, output_dir: Path) -> None:
    """Generate convergence visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Show per-leader stats as bar chart
    labels = [l.name for l in result.leaders]
    writes = [l.stats.writes for l in result.leaders]
    conflicts = [l.stats.conflicts_detected for l in result.leaders]
    repairs = [l.stats.anti_entropy_keys_repaired for l in result.leaders]

    x = range(len(labels))
    w = 0.25
    ax.bar([i - w for i in x], writes, w, label="Writes", color="steelblue")
    ax.bar(x, conflicts, w, label="Conflicts", color="coral")
    ax.bar([i + w for i in x], repairs, w, label="AE Repairs", color="seagreen")

    ax.set_xlabel("Leader Node")
    ax.set_ylabel("Count")
    ax.set_title("Multi-Leader Replication Statistics")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = output_dir / "multi_leader_replication.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-leader replication demo")
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--ae-interval", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/multi_leader")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print("Running multi-leader replication simulation...")
    print(f"  Duration: {args.duration}s | Write rate: {args.rate} req/s/leader")
    print(f"  Anti-entropy interval: {args.ae_interval}s")
    print(f"  Phases: Normal [0-15s], Partition [15-30s], Heal [30-45s]")

    result = run_simulation(
        duration_s=args.duration,
        write_rate=args.rate,
        anti_entropy_interval=args.ae_interval,
        seed=args.seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
