"""Stream processor: windowed aggregation with late event handling.

Demonstrates stateful stream processing with tumbling windows,
out-of-order event times, and watermark-driven window closure.

## Architecture

```
  Source ──► StreamProcessor ──► ResultSink
               │
               └── (late events) ──► LateSink
```

## Key Observations

- Tumbling windows group events by event time, not arrival time.
- Watermark advancement triggers window emission.
- Late events (event_time < watermark - allowed_lateness) are routed
  to a side output or dropped based on policy.
- Aggregate function is applied to all records in a window at emission.
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
from happysimulator.components.streaming.stream_processor import (
    LateEventPolicy,
    StreamProcessor,
    TumblingWindow,
)


# =============================================================================
# Client entities
# =============================================================================


class EventEmitter(Entity):
    """Emits Process events with potentially out-of-order event times."""

    def __init__(self, name: str, processor: Entity, jitter_s: float = 2.0):
        super().__init__(name)
        self.processor = processor
        self._jitter_s = jitter_s
        self._count = 0

    def handle_event(self, event: Event) -> Generator[float | tuple[float, list[Event]], None, None]:
        self._count += 1
        # Simulate out-of-order: event_time = now + random jitter
        base_time = self.now.to_seconds()
        jitter = random.uniform(-self._jitter_s, self._jitter_s * 0.2)
        event_time_s = max(0.0, base_time + jitter)

        key = f"group-{self._count % 3}"
        value = random.randint(1, 100)

        yield 0.0, [Event(
            time=self.now,
            event_type="Process",
            target=self.processor,
            context={
                "key": key,
                "value": value,
                "event_time_s": event_time_s,
            },
        )]


class ResultCollector(Entity):
    """Collects window results and late events."""

    def __init__(self, name: str):
        super().__init__(name)
        self.results: list[dict] = []

    def handle_event(self, event: Event) -> None:
        self.results.append({
            "type": event.event_type,
            "time": self.now.to_seconds(),
            **event.context,
        })


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class ProcessorResult:
    """Results from a stream processor run."""

    window_size_s: float
    allowed_lateness_s: float
    processor: StreamProcessor
    result_sink: ResultCollector
    late_sink: ResultCollector


def run_stream_processor(
    *,
    window_size_s: float = 5.0,
    allowed_lateness_s: float = 1.0,
    duration_s: float = 30.0,
    event_rate: float = 50.0,
    jitter_s: float = 3.0,
    seed: int = 42,
) -> ProcessorResult:
    """Run a stream processor simulation."""
    random.seed(seed)

    result_sink = ResultCollector("results")
    late_sink = ResultCollector("late-events")

    processor = StreamProcessor(
        name="windowed-agg",
        window_type=TumblingWindow(size_s=window_size_s),
        aggregate_fn=lambda records: {
            "sum": sum(records),
            "count": len(records),
            "avg": sum(records) / len(records) if records else 0,
        },
        downstream=result_sink,
        allowed_lateness_s=allowed_lateness_s,
        late_event_policy=LateEventPolicy.SIDE_OUTPUT,
        side_output=late_sink,
        watermark_interval_s=1.0,
    )

    emitter = EventEmitter("emitter", processor, jitter_s=jitter_s)

    source = Source.constant(
        rate=event_rate, target=emitter,
        event_type="Emit", stop_after=duration_s,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 10.0),
        sources=[source],
        entities=[processor, result_sink, late_sink, emitter],
    )
    sim.run()

    return ProcessorResult(
        window_size_s=window_size_s,
        allowed_lateness_s=allowed_lateness_s,
        processor=processor,
        result_sink=result_sink,
        late_sink=late_sink,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: list[ProcessorResult]) -> None:
    """Print stream processor results."""
    print("\n" + "=" * 70)
    print("STREAM PROCESSOR — WINDOWED AGGREGATION")
    print("=" * 70)

    for r in results:
        proc = r.processor
        print(f"\n  Window: {r.window_size_s}s tumbling, lateness={r.allowed_lateness_s}s")
        print(f"  Events processed:    {proc.stats.events_processed:,}")
        print(f"  Windows emitted:     {proc.stats.windows_emitted}")
        print(f"  Late events total:   {proc.stats.late_events}")
        print(f"    Dropped:           {proc.stats.late_events_dropped}")
        print(f"    Side output:       {proc.stats.late_events_side_output}")
        print(f"  Watermark:           {proc.watermark_s:.2f}s")

        window_results = [
            r_entry for r_entry in r.result_sink.results
            if r_entry["type"] == "WindowResult"
        ]
        if window_results:
            print(f"\n  {'Key':>10s} {'Window':>15s} {'Count':>6s} {'Sum':>8s} {'Avg':>8s}")
            print(f"  {'-' * 51}")
            for wr in window_results[:15]:  # Show first 15
                agg = wr["result"]
                ws = f"[{wr['window_start']:.0f},{wr['window_end']:.0f})"
                print(
                    f"  {wr['key']:>10s} {ws:>15s} "
                    f"{agg['count']:>6d} {agg['sum']:>8d} {agg['avg']:>8.1f}"
                )
            if len(window_results) > 15:
                print(f"  ... and {len(window_results) - 15} more windows")

    print("\n" + "=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: list[ProcessorResult], output_dir: Path) -> None:
    """Generate window emission timeline chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

        # Window emissions over time
        window_results = [
            entry for entry in r.result_sink.results
            if entry["type"] == "WindowResult"
        ]
        if window_results:
            times = [wr["time"] for wr in window_results]
            counts = [wr["record_count"] for wr in window_results]
            ax1.bar(times, counts, width=0.3, color="steelblue", alpha=0.8)
        ax1.set_xlabel("Emission Time (s)")
        ax1.set_ylabel("Records per Window")
        ax1.set_title("Window Emissions")
        ax1.grid(True, alpha=0.2)

        # Late events over time
        late_events = [
            entry for entry in r.late_sink.results
            if entry["type"] == "LateEvent"
        ]
        if late_events:
            late_times = [le["time"] for le in late_events]
            late_event_times = [le["event_time_s"] for le in late_events]
            ax2.scatter(late_times, late_event_times, s=10, color="coral", alpha=0.6)
        ax2.set_xlabel("Arrival Time (s)")
        ax2.set_ylabel("Event Time (s)")
        ax2.set_title(f"Late Events (total: {len(late_events)})")
        ax2.grid(True, alpha=0.2)

        fig.suptitle(f"Stream Processor — Window={r.window_size_s}s, Lateness={r.allowed_lateness_s}s")
        fig.tight_layout()

        path = output_dir / "stream_processor.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stream processor demo")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--lateness", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/stream_processor")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print("Running stream processor simulation...")

    results = [
        run_stream_processor(
            window_size_s=args.window,
            allowed_lateness_s=args.lateness,
            duration_s=args.duration,
            event_rate=args.rate,
            jitter_s=args.jitter,
            seed=args.seed,
        ),
    ]

    print_summary(results)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(results, output_dir)
