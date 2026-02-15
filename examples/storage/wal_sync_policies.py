"""WAL sync policy comparison: throughput vs durability tradeoff.

This example demonstrates how different sync policies for a Write-Ahead Log
affect throughput and durability guarantees:

1. SyncEveryWrite: Maximum durability, lowest throughput. Every write is
   fsynced before the caller continues.
2. SyncPeriodic(0.01): Sync every 10ms. Writes accumulate in an OS buffer
   and are flushed on a timer. A crash can lose up to 10ms of writes.
3. SyncOnBatch(10): Sync after every 10 writes. Groups I/O for efficiency
   while bounding the data-at-risk window.

## Architecture Diagram

```
+----------------------------------------------------------------------+
|                  WAL SYNC POLICY COMPARISON                           |
+----------------------------------------------------------------------+

    Source (constant rate)
        |
        v
    WALWorker Entity
        |
        |  yield from wal.append(key, value)
        v
    WriteAheadLog (with sync policy)
        |
        v
    LatencyTracker (measures write latency)

    Repeated 3x with different sync policies:
        - SyncEveryWrite
        - SyncPeriodic(0.01)
        - SyncOnBatch(10)
```

## Key Insight

SyncEveryWrite pays the fsync latency on every write (~1ms per sync),
while batched and periodic policies amortize the fsync cost across
multiple writes. The tradeoff: batched policies can lose recent writes
if the process crashes between syncs.
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
    LatencyTracker,
    Probe,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.storage import (
    SyncEveryWrite,
    SyncOnBatch,
    SyncPeriodic,
    WriteAheadLog,
)


# =============================================================================
# WAL Worker Entity
# =============================================================================


class WALWorker(Entity):
    """Worker that receives write events and appends to a WAL.

    Uses the generator-based ``wal.append()`` to model I/O latency
    (both write latency and conditional sync latency). Forwards a
    completion event to a downstream tracker after each append.

    Attributes:
        writes_completed: Number of successfully appended writes.
    """

    def __init__(
        self,
        name: str,
        *,
        wal: WriteAheadLog,
        downstream: Entity,
    ) -> None:
        super().__init__(name)
        self.wal = wal
        self.downstream = downstream
        self.writes_completed: int = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Append the write to the WAL, yielding I/O latency."""
        key = event.context.get("key", f"k{self.writes_completed}")
        value = event.context.get("value", f"v{self.writes_completed}")

        _seq = yield from self.wal.append(key, value)
        self.writes_completed += 1

        return [
            self.forward(event, self.downstream, event_type="WriteComplete")
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class PolicyResult:
    """Results from a single sync-policy simulation run."""

    policy_name: str
    tracker: LatencyTracker
    wal: WriteAheadLog
    worker: WALWorker
    summary: SimulationSummary


@dataclass
class ComparisonResult:
    """Combined results from all three policy runs."""

    results: list[PolicyResult]
    duration_s: float
    write_rate: float


def run_single_policy(
    *,
    policy_name: str,
    wal: WriteAheadLog,
    duration_s: float,
    write_rate: float,
    seed: int | None,
) -> PolicyResult:
    """Run a simulation with one WAL sync policy.

    Args:
        policy_name: Human-readable policy label.
        wal: Configured WriteAheadLog entity.
        duration_s: Simulation duration in seconds.
        write_rate: Writes per second to generate.
        seed: Random seed for reproducibility.

    Returns:
        PolicyResult containing metrics for this policy.
    """
    if seed is not None:
        random.seed(seed)

    tracker = LatencyTracker(name=f"Tracker_{policy_name}")
    worker = WALWorker(name=f"Worker_{policy_name}", wal=wal, downstream=tracker)

    source = Source.constant(
        rate=write_rate,
        target=worker,
        event_type="Write",
        stop_after=Instant.from_seconds(duration_s),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[wal, worker, tracker],
    )
    summary = sim.run()

    return PolicyResult(
        policy_name=policy_name,
        tracker=tracker,
        wal=wal,
        worker=worker,
        summary=summary,
    )


def run_wal_sync_comparison(
    *,
    duration_s: float = 10.0,
    write_rate: float = 500.0,
    seed: int | None = 42,
) -> ComparisonResult:
    """Run three simulations comparing WAL sync policies.

    Args:
        duration_s: How long each simulation runs.
        write_rate: Writes per second.
        seed: Base random seed (each run uses seed, seed+1, seed+2).

    Returns:
        ComparisonResult with metrics for all three policies.
    """
    policies = [
        ("SyncEveryWrite", SyncEveryWrite()),
        ("SyncPeriodic(10ms)", SyncPeriodic(interval_s=0.01)),
        ("SyncOnBatch(10)", SyncOnBatch(batch_size=10)),
    ]

    results: list[PolicyResult] = []
    for i, (name, policy) in enumerate(policies):
        wal = WriteAheadLog(
            f"WAL_{name}",
            sync_policy=policy,
            write_latency=0.0001,   # 100us per append
            sync_latency=0.001,     # 1ms per fsync
        )
        policy_seed = seed + i if seed is not None else None
        result = run_single_policy(
            policy_name=name,
            wal=wal,
            duration_s=duration_s,
            write_rate=write_rate,
            seed=policy_seed,
        )
        results.append(result)

    return ComparisonResult(
        results=results,
        duration_s=duration_s,
        write_rate=write_rate,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(comparison: ComparisonResult) -> None:
    """Print a comparison table of throughput and sync counts."""
    print("\n" + "=" * 78)
    print("WAL SYNC POLICY COMPARISON")
    print("=" * 78)

    print(f"\nConfiguration:")
    print(f"  Duration:       {comparison.duration_s}s")
    print(f"  Write rate:     {comparison.write_rate} writes/s")
    print(f"  Write latency:  0.1ms per append")
    print(f"  Sync latency:   1.0ms per fsync")

    print(f"\n{'Policy':<22} {'Writes':>8} {'Syncs':>8} {'Sync Lat (s)':>14} {'Avg Lat (ms)':>14} {'p99 Lat (ms)':>14}")
    print("-" * 78)

    for r in comparison.results:
        stats = r.wal.stats
        avg_lat = r.tracker.mean_latency() * 1000 if r.tracker.count > 0 else 0.0
        p99_lat = r.tracker.p99() * 1000 if r.tracker.count > 0 else 0.0

        print(
            f"{r.policy_name:<22} "
            f"{stats.writes:>8} "
            f"{stats.syncs:>8} "
            f"{stats.total_sync_latency_s:>14.4f} "
            f"{avg_lat:>14.3f} "
            f"{p99_lat:>14.3f}"
        )

    print()

    # Interpretation
    every_write = comparison.results[0]
    periodic = comparison.results[1]
    batch = comparison.results[2]

    ew_syncs = every_write.wal.stats.syncs
    p_syncs = periodic.wal.stats.syncs
    b_syncs = batch.wal.stats.syncs

    if ew_syncs > 0 and p_syncs > 0:
        print("Observations:")
        print(f"  - SyncEveryWrite performed {ew_syncs} syncs (one per write)")
        if p_syncs > 0:
            print(f"  - SyncPeriodic reduced syncs by {(1 - p_syncs / ew_syncs) * 100:.0f}% ({p_syncs} syncs)")
        if b_syncs > 0:
            print(f"  - SyncOnBatch reduced syncs by {(1 - b_syncs / ew_syncs) * 100:.0f}% ({b_syncs} syncs)")

        ew_lat = every_write.tracker.mean_latency() * 1000
        p_lat = periodic.tracker.mean_latency() * 1000
        b_lat = batch.tracker.mean_latency() * 1000
        if ew_lat > 0:
            print(f"  - Periodic policy latency is {ew_lat / p_lat:.1f}x lower than every-write" if p_lat > 0 else "")
            print(f"  - Batch policy latency is {ew_lat / b_lat:.1f}x lower than every-write" if b_lat > 0 else "")

    print("\n" + "=" * 78)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(comparison: ComparisonResult, output_dir: Path) -> None:
    """Generate bar chart visualization comparing the three policies."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    names = [r.policy_name for r in comparison.results]
    syncs = [r.wal.stats.syncs for r in comparison.results]
    writes = [r.wal.stats.writes for r in comparison.results]
    avg_latencies = [
        r.tracker.mean_latency() * 1000 if r.tracker.count > 0 else 0.0
        for r in comparison.results
    ]
    p99_latencies = [
        r.tracker.p99() * 1000 if r.tracker.count > 0 else 0.0
        for r in comparison.results
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    # Chart 1: Total syncs
    ax = axes[0]
    bars = ax.bar(names, syncs, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, syncs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of fsyncs")
    ax.set_title("Sync Operations")
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 2: Average write latency
    ax = axes[1]
    bars = ax.bar(names, avg_latencies, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, avg_latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Average Latency (ms)")
    ax.set_title("Write Latency (avg)")
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 3: p99 write latency
    ax = axes[2]
    bars = ax.bar(names, p99_latencies, color=colors, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, p99_latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("Write Latency (p99)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"WAL Sync Policy Comparison ({comparison.write_rate:.0f} writes/s, {comparison.duration_s:.0f}s)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "wal_sync_policies.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'wal_sync_policies.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare WAL sync policies: throughput vs durability tradeoff"
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--rate", type=float, default=500.0, help="Write rate (writes/s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/wal_sync_policies", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running WAL sync policy comparison...")
    print(f"  Duration: {args.duration}s")
    print(f"  Write rate: {args.rate} writes/s")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    comparison = run_wal_sync_comparison(
        duration_s=args.duration,
        write_rate=args.rate,
        seed=seed,
    )

    print_summary(comparison)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(comparison, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
