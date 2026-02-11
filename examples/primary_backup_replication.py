"""Primary-backup replication with sync/semi-sync/async modes.

Demonstrates write latency tradeoffs across replication modes and shows
how ASYNC mode risks data loss during network partitions.

## Architecture

```
  Source ──► PrimaryNode ──► Network ──► BackupNode-1
  (writes)       │                  └──► BackupNode-2
                 │
              KVStore                 KVStore x2
           (authoritative)          (replicated)
```

## Key Observations

- ASYNC: Lowest write latency, but writes may be lost if primary fails
  before replication completes.
- SEMI_SYNC: Moderate latency — waits for one backup ack.
- SYNC: Highest latency — waits for all backup acks, strongest durability.
- During a network partition, ASYNC continues accepting writes while
  SYNC blocks until the partition heals.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    Instant,
    Network,
    Probe,
    Simulation,
    SimFuture,
    Source,
    datacenter_network,
)
from happysimulator.components.datastore import KVStore
from happysimulator.components.replication.primary_backup import (
    BackupNode,
    PrimaryNode,
    ReplicationMode,
)


# =============================================================================
# Writer entity (generates write events from source traffic)
# =============================================================================


class Writer(Entity):
    """Converts source traffic into keyed writes to a PrimaryNode."""

    def __init__(self, name: str, primary: PrimaryNode, network: Network):
        super().__init__(name)
        self.primary = primary
        self.network = network
        self._write_count = 0
        self.latencies: list[tuple[float, float]] = []  # (time, latency_s)

    @property
    def write_count(self) -> int:
        return self._write_count

    def handle_event(self, event: Event) -> Generator[float | SimFuture | tuple[float, list[Event]], None, list[Event] | None]:
        self._write_count += 1
        key = f"key-{self._write_count % 100}"
        value = self._write_count

        reply_future = SimFuture()
        write_event = Event(
            time=self.now,
            event_type="Write",
            target=self.primary,
            context={"metadata": {
                "key": key, "value": value, "reply_future": reply_future,
            }},
        )
        start = self.now
        yield 0.0, [write_event]

        # Wait for write acknowledgment
        yield reply_future
        latency = (self.now - start).to_seconds()
        self.latencies.append((self.now.to_seconds(), latency))
        return None


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class ModeResult:
    """Results for one replication mode run."""

    mode: ReplicationMode
    primary: PrimaryNode
    backups: list[BackupNode]
    writer: Writer
    primary_store: KVStore
    backup_stores: list[KVStore]


def run_mode(
    mode: ReplicationMode,
    *,
    duration_s: float = 30.0,
    write_rate: float = 50.0,
    seed: int = 42,
) -> ModeResult:
    """Run a primary-backup simulation with the given mode."""
    random.seed(seed)

    # Stores
    primary_store = KVStore("ps", write_latency=0.001, read_latency=0.001)
    backup_stores = [
        KVStore(f"bs{i}", write_latency=0.001, read_latency=0.001)
        for i in range(2)
    ]

    # Network
    network = Network(name="net")

    # Nodes
    primary = PrimaryNode(
        "primary", store=primary_store, backups=[],
        network=network, mode=mode,
    )
    backups = [
        BackupNode(f"backup-{i}", store=backup_stores[i],
                   network=network, primary=primary)
        for i in range(2)
    ]
    primary._backups = backups
    primary._backup_lag = {b.name: 0 for b in backups}

    # Link topology
    for backup in backups:
        network.add_bidirectional_link(primary, backup, datacenter_network(f"link-{backup.name}"))

    # Writer
    writer = Writer("writer", primary=primary, network=network)

    # Source
    source = Source.constant(
        rate=write_rate,
        target=writer,
        event_type="NewWrite",
        stop_after=duration_s,
    )

    # Run
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 5.0),
        sources=[source],
        entities=[writer, primary, *backups, network, primary_store, *backup_stores],
    )
    sim.run()

    return ModeResult(
        mode=mode,
        primary=primary,
        backups=backups,
        writer=writer,
        primary_store=primary_store,
        backup_stores=backup_stores,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: list[ModeResult]) -> None:
    """Print comparison table across modes."""
    print("\n" + "=" * 70)
    print("PRIMARY-BACKUP REPLICATION — MODE COMPARISON")
    print("=" * 70)

    header = f"  {'Mode':<12s} {'Writes':>7s} {'Primary':>8s} {'Backup0':>8s} {'Backup1':>8s} {'AvgLat(ms)':>11s} {'p99Lat(ms)':>11s}"
    print(header)
    print(f"  {'-' * 66}")

    for r in results:
        lats = sorted(lat for _, lat in r.writer.latencies) if r.writer.latencies else [0]
        avg_ms = sum(lats) / len(lats) * 1000
        p99_ms = lats[int(len(lats) * 0.99)] * 1000 if len(lats) > 1 else avg_ms

        ps_keys = r.primary_store.size
        bs0_keys = r.backup_stores[0].size
        bs1_keys = r.backup_stores[1].size

        print(
            f"  {r.mode.value:<12s} "
            f"{r.writer.write_count:>7d} "
            f"{ps_keys:>8d} "
            f"{bs0_keys:>8d} "
            f"{bs1_keys:>8d} "
            f"{avg_ms:>10.2f} "
            f"{p99_ms:>10.2f}"
        )

    print("=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: list[ModeResult], output_dir: Path) -> None:
    """Generate a comparison chart of write latencies across modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4), sharey=True)
    if len(results) == 1:
        axes = [axes]

    colors = {"async": "steelblue", "semi_sync": "coral", "sync": "seagreen"}

    for ax, r in zip(axes, results):
        if r.writer.latencies:
            times = [t for t, _ in r.writer.latencies]
            lats = [l * 1000 for _, l in r.writer.latencies]
            color = colors.get(r.mode.value, "gray")
            ax.scatter(times, lats, s=2, alpha=0.3, color=color)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{r.mode.value.upper()}")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Write Latency (ms)")
    fig.suptitle("Primary-Backup Replication — Write Latency by Mode", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    path = output_dir / "primary_backup_replication.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Primary-backup replication demo")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/primary_backup")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running primary-backup replication simulation...")
    print(f"  Duration: {args.duration}s | Write rate: {args.rate} req/s")

    results = []
    for mode in [ReplicationMode.ASYNC, ReplicationMode.SEMI_SYNC, ReplicationMode.SYNC]:
        print(f"  Running {mode.value}...")
        r = run_mode(mode, duration_s=args.duration, write_rate=args.rate, seed=args.seed)
        results.append(r)

    print_summary(results)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(results, output_dir)
