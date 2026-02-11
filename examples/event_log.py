"""Event log: append-only partitioned log with retention.

Demonstrates a partitioned event log where multiple writers append
records, showing partition distribution, high-watermark progression,
and retention effects.

## Architecture

```
  Writer-0 ──┐
  Writer-1 ──┼──► EventLog (4 partitions) ──► [retention daemon]
  Writer-2 ──┘
```

## Key Observations

- Records distribute across partitions via hash-based sharding.
- High watermarks advance independently per partition.
- Retention daemon periodically sweeps expired records.
- Append latency is consistent regardless of partition count.
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
    Simulation,
    SimFuture,
    Source,
)
from happysimulator.components.streaming.event_log import (
    EventLog,
    TimeRetention,
)


# =============================================================================
# Client entities
# =============================================================================


class LogWriter(Entity):
    """Appends keyed records to the event log."""

    def __init__(self, name: str, log: EventLog, key_space: int = 100):
        super().__init__(name)
        self.log = log
        self._key_space = key_space
        self._count = 0
        self.latencies: list[tuple[float, float]] = []

    def handle_event(self, event: Event) -> Generator[float | SimFuture | tuple[float, list[Event]], None, None]:
        self._count += 1
        key = f"key-{self._count % self._key_space}"

        reply = SimFuture()
        append_event = Event(
            time=self.now,
            event_type="Append",
            target=self.log,
            context={"key": key, "value": {"seq": self._count, "writer": self.name}, "reply_future": reply},
        )
        start = self.now
        yield 0.0, [append_event]
        yield reply
        self.latencies.append((self.now.to_seconds(), (self.now - start).to_seconds()))


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class LogResult:
    """Results from an event log run."""

    log: EventLog
    writers: list[LogWriter]
    retention: bool


def run_event_log(
    num_writers: int = 3,
    num_partitions: int = 4,
    *,
    duration_s: float = 20.0,
    write_rate: float = 50.0,
    retention_s: float | None = None,
    seed: int = 42,
) -> LogResult:
    """Run an event log simulation."""
    random.seed(seed)

    retention = TimeRetention(max_age_s=retention_s) if retention_s else None
    log = EventLog(
        name="event-log",
        num_partitions=num_partitions,
        retention_policy=retention,
        retention_check_interval=2.0,
    )

    writers = [LogWriter(f"writer-{i}", log) for i in range(num_writers)]
    sources = [
        Source.constant(
            rate=write_rate, target=w,
            event_type="NewRecord", stop_after=duration_s,
        )
        for w in writers
    ]

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 5.0),
        sources=sources,
        entities=[log, *writers],
    )
    sim.run()

    return LogResult(log=log, writers=writers, retention=retention is not None)


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: list[LogResult]) -> None:
    """Print event log simulation results."""
    print("\n" + "=" * 70)
    print("EVENT LOG — PARTITION DISTRIBUTION & RETENTION")
    print("=" * 70)

    for r in results:
        log = r.log
        print(f"\n  Config: partitions={log.num_partitions}, retention={'yes' if r.retention else 'no'}")
        print(f"  Total appended: {log.stats.records_appended:,}")
        print(f"  Total expired:  {log.stats.records_expired:,}")
        print(f"  Current total:  {log.total_records:,}")
        print(f"  Avg append latency: {log.stats.avg_append_latency * 1000:.2f} ms")

        print(f"\n  {'Partition':>10s} {'Appends':>8s} {'Current':>8s} {'HW':>6s}")
        print(f"  {'-' * 36}")
        for p in log.partitions:
            appends = log.stats.per_partition_appends.get(p.id, 0)
            print(f"  {p.id:>10d} {appends:>8d} {len(p.records):>8d} {p.high_watermark:>6d}")

        # Writer latencies
        all_lats = []
        for w in r.writers:
            all_lats.extend(l for _, l in w.latencies)
        if all_lats:
            all_lats.sort()
            avg_ms = sum(all_lats) / len(all_lats) * 1000
            p99_ms = all_lats[int(len(all_lats) * 0.99)] * 1000
            print(f"\n  Writer latency: avg={avg_ms:.2f}ms  p99={p99_ms:.2f}ms")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: list[LogResult], output_dir: Path) -> None:
    """Generate partition distribution charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        log = r.log
        pids = [p.id for p in log.partitions]
        counts = [len(p.records) for p in log.partitions]
        ax.bar(pids, counts, color="steelblue", alpha=0.8)
        ax.set_xlabel("Partition")
        ax.set_ylabel("Records")
        title = f"Partitions={log.num_partitions}"
        if r.retention:
            title += " (retention)"
        ax.set_title(title)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Event Log — Partition Distribution")
    fig.tight_layout()

    path = output_dir / "event_log.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Event log demo")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/event_log")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print("Running event log simulation...")

    results = []

    # Without retention
    print("  Without retention...")
    r1 = run_event_log(duration_s=args.duration, write_rate=args.rate, seed=args.seed)
    results.append(r1)

    # With retention
    print("  With time retention (5s)...")
    r2 = run_event_log(
        duration_s=args.duration, write_rate=args.rate, seed=args.seed,
        retention_s=5.0,
    )
    results.append(r2)

    print_summary(results)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(results, output_dir)
