"""Memtable write buffer filling and flushing to SSTable.

This example demonstrates how a Memtable accumulates writes in memory and
flushes to immutable SSTables when the size threshold is reached. This is
the core write path of an LSM-tree storage engine.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                  MEMTABLE FLUSH SIMULATION                             |
+-----------------------------------------------------------------------+

    Source (constant rate)
        |
        v
    MemtableWorker Entity
        |
        |  yield from memtable.put(key, value)
        |  if full -> memtable.flush() -> SSTable
        v
    Memtable (in-memory sorted buffer)
        |
        |  flush when size >= threshold
        v
    SSTable (immutable, on-disk)
        |
        v
    LatencyTracker (measures write latency)
```

## Write Path

1. Source generates write events at a constant rate
2. MemtableWorker receives each event and puts into the Memtable
3. When the Memtable reaches its size threshold, the worker flushes it
4. flush() freezes the contents into an immutable SSTable and clears the buffer
5. The cycle repeats: the memtable refills from empty after each flush

## Key Insight

The memtable size follows a sawtooth pattern: it grows linearly as writes
arrive, then drops to zero on flush. The flush frequency depends on the
write rate and size threshold. Larger thresholds mean fewer but bigger
flushes, while smaller thresholds flush more often with less data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from happysimulator.components.storage import Memtable, SSTable

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# MemtableWorker Entity
# =============================================================================


class MemtableWorker(Entity):
    """Worker that puts writes into a Memtable and flushes when full.

    Uses the generator-based ``memtable.put()`` to model write latency.
    When the memtable reports it is full, the worker flushes it to an
    SSTable and records the flush event.

    Attributes:
        writes_completed: Number of successfully written keys.
        flush_times: Simulation times at which flushes occurred.
        sstables: List of SSTables produced by flushes.
    """

    def __init__(
        self,
        name: str,
        *,
        memtable: Memtable,
        downstream: Entity,
    ) -> None:
        super().__init__(name)
        self.memtable = memtable
        self.downstream = downstream
        self.writes_completed: int = 0
        self.flush_times: list[float] = []
        self.sstables: list[SSTable] = []

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Write to the memtable, flushing when full."""
        key = f"key_{self.writes_completed:06d}"
        value = f"value_{self.writes_completed}"

        is_full = yield from self.memtable.put(key, value)
        self.writes_completed += 1

        if is_full:
            sstable = self.memtable.flush()
            self.sstables.append(sstable)
            self.flush_times.append(self.now.to_seconds())

        return [self.forward(event, self.downstream, event_type="WriteComplete")]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the memtable flush simulation."""

    tracker: LatencyTracker
    memtable: Memtable
    worker: MemtableWorker
    memtable_size_data: Data
    summary: SimulationSummary
    duration_s: float
    write_rate: float
    size_threshold: int


def run_memtable_flush_simulation(
    *,
    duration_s: float = 10.0,
    write_rate: float = 1000.0,
    size_threshold: int = 500,
    probe_interval_s: float = 0.01,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the memtable flush simulation.

    Args:
        duration_s: Simulation duration in seconds.
        write_rate: Writes per second.
        size_threshold: Memtable size before triggering a flush.
        probe_interval_s: How often to sample memtable size.
        seed: Random seed for reproducibility.

    Returns:
        SimulationResult with all metrics for analysis.
    """
    if seed is not None:
        random.seed(seed)

    memtable = Memtable(
        "memtable",
        size_threshold=size_threshold,
        write_latency=0.00001,  # 10us per put
        read_latency=0.000005,  # 5us per get
    )

    tracker = LatencyTracker(name="WriteTracker")
    worker = MemtableWorker(
        name="MemtableWorker",
        memtable=memtable,
        downstream=tracker,
    )

    # Probe memtable size over time for visualization

    size_probe, memtable_size_data = Probe.on(memtable, "size", interval=probe_interval_s)

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
        entities=[memtable, worker, tracker],
        probes=[size_probe],
    )
    summary = sim.run()

    return SimulationResult(
        tracker=tracker,
        memtable=memtable,
        worker=worker,
        memtable_size_data=memtable_size_data,
        summary=summary,
        duration_s=duration_s,
        write_rate=write_rate,
        size_threshold=size_threshold,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary statistics for the memtable flush simulation."""
    print("\n" + "=" * 70)
    print("MEMTABLE FLUSH SIMULATION RESULTS")
    print("=" * 70)

    stats = result.memtable.stats

    print("\nConfiguration:")
    print(f"  Duration:        {result.duration_s}s")
    print(f"  Write rate:      {result.write_rate} writes/s")
    print(f"  Size threshold:  {result.size_threshold} entries")

    print("\nMemtable Statistics:")
    print(f"  Total writes:       {stats.writes:,}")
    print(f"  Total flushes:      {stats.flushes}")
    print(f"  Current size:       {stats.current_size}")
    print(f"  Bytes written:      {stats.total_bytes_written:,}")

    num_flushes = len(result.worker.flush_times)
    print("\nFlush Behavior:")
    print(f"  Flushes observed:   {num_flushes}")
    if num_flushes > 0:
        expected_flushes = int(result.write_rate * result.duration_s / result.size_threshold)
        print(f"  Expected flushes:   ~{expected_flushes}")
        avg_interval = result.duration_s / num_flushes
        print(f"  Avg flush interval: {avg_interval:.3f}s")
        print(f"  Expected interval:  {result.size_threshold / result.write_rate:.3f}s")

    if result.worker.sstables:
        sizes = [sst.key_count for sst in result.worker.sstables]
        print("\nSSTable Outputs:")
        print(f"  Total SSTables:    {len(sizes)}")
        print(f"  Avg keys/SSTable:  {sum(sizes) / len(sizes):.0f}")
        print(f"  Min keys:          {min(sizes)}")
        print(f"  Max keys:          {max(sizes)}")

    if result.tracker.count > 0:
        print("\nWrite Latency:")
        print(f"  Completed writes:  {result.tracker.count:,}")
        print(f"  Avg latency:       {result.tracker.mean_latency() * 1000:.4f} ms")
        print(f"  p99 latency:       {result.tracker.p99() * 1000:.4f} ms")

    print(f"\n{result.summary}")
    print("=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate time series visualization of memtable size."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Chart 1: Memtable size over time (sawtooth pattern)
    ax = axes[0]
    size_times = result.memtable_size_data.times()
    size_values = result.memtable_size_data.raw_values()

    ax.plot(size_times, size_values, "b-", linewidth=1.5, label="Memtable Size")
    ax.axhline(
        y=result.size_threshold,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Flush Threshold ({result.size_threshold})",
    )

    # Mark flush events
    for ft in result.worker.flush_times:
        ax.axvline(x=ft, color="green", linestyle=":", alpha=0.4)

    if result.worker.flush_times:
        # Add one labeled flush line for the legend
        ax.axvline(
            x=result.worker.flush_times[0],
            color="green",
            linestyle=":",
            alpha=0.4,
            label=f"Flush ({len(result.worker.flush_times)} total)",
        )

    ax.set_ylabel("Entries")
    ax.set_title("Memtable Size Over Time (Sawtooth Pattern)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Chart 2: Write latency over time (bucketed)
    ax = axes[1]
    if result.tracker.count > 0:
        latency_buckets = result.tracker.data.bucket(window_s=0.1)
        bucket_times = latency_buckets.times()
        bucket_means = latency_buckets.means()
        bucket_p99s = latency_buckets.p99s()

        # Convert to microseconds for readability
        means_us = [v * 1_000_000 for v in bucket_means]
        p99s_us = [v * 1_000_000 for v in bucket_p99s]

        ax.plot(bucket_times, means_us, "b-", linewidth=1.5, label="Avg")
        ax.plot(bucket_times, p99s_us, "r-", linewidth=1, alpha=0.7, label="p99")
        ax.set_ylabel("Write Latency (us)")
        ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "No latency data", transform=ax.transAxes, ha="center", va="center")

    ax.set_xlabel("Time (s)")
    ax.set_title("Write Latency Over Time")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Memtable Flush Simulation "
        f"({result.write_rate:.0f} writes/s, threshold={result.size_threshold})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "memtable_flush.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'memtable_flush.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Memtable write buffer filling and flushing to SSTable"
    )
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration (s)")
    parser.add_argument("--rate", type=float, default=1000.0, help="Write rate (writes/s)")
    parser.add_argument("--threshold", type=int, default=500, help="Memtable size threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument(
        "--output", type=str, default="output/memtable_flush", help="Output directory"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running memtable flush simulation...")
    print(f"  Duration:  {args.duration}s")
    print(f"  Write rate: {args.rate} writes/s")
    print(f"  Threshold: {args.threshold} entries")
    print(f"  Seed:      {seed if seed is not None else 'random'}")

    result = run_memtable_flush_simulation(
        duration_s=args.duration,
        write_rate=args.rate,
        size_threshold=args.threshold,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
