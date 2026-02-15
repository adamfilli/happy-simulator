"""Pure event loop speed benchmark.

Minimal M/M/1 queue with near-zero service time to maximize event throughput.
No probes, trackers, or instrumentation — just raw pop-invoke-push speed.
"""

from __future__ import annotations

import random
import time
import tracemalloc

from happysimulator import (
    Event,
    Instant,
    QueuedResource,
    Simulation,
    Sink,
    Source,
)
from happysimulator.components.queue_policy import FIFOQueue
from tests.perf.runner import BenchmarkResult

BASE_EVENT_COUNT = 500_000
WARMUP_EVENTS = 1_000


class _MinimalServer(QueuedResource):
    """Server with near-zero service time for throughput testing."""

    def __init__(self, name: str, downstream):
        super().__init__(name, policy=FIFOQueue())
        self._downstream = downstream

    def handle_queued_event(self, event: Event):
        yield 0.0
        return [
            Event(time=self.now, event_type="Done", target=self._downstream, context=event.context)
        ]


def _build_and_run(event_count: int) -> tuple[int, float]:
    """Build a simulation and run it, returning (events_processed, wall_clock)."""
    random.seed(42)
    rate = event_count * 10  # Very high rate to finish quickly
    duration_s = event_count / rate

    sink = Sink("Sink")
    server = _MinimalServer("Server", downstream=sink)
    source = Source.constant(rate=rate, target=server, name="Source", stop_after=duration_s)

    sim = Simulation(
        end_time=Instant.from_seconds(duration_s + 0.001),
        sources=[source],
        entities=[server, sink],
    )

    start = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - start
    return summary.total_events_processed, wall


def run(*, scale: float = 1.0) -> BenchmarkResult:
    """Run the throughput benchmark."""
    # Warm-up
    _build_and_run(WARMUP_EVENTS)

    # Measured run
    event_count = int(BASE_EVENT_COUNT * scale)
    tracemalloc.reset_peak()
    events_processed, wall = _build_and_run(event_count)
    _, peak = tracemalloc.get_traced_memory()

    eps = events_processed / wall if wall > 0 else 0.0
    return BenchmarkResult(
        name="throughput",
        events_processed=events_processed,
        wall_clock_s=wall,
        events_per_second=eps,
        peak_memory_mb=peak / (1024 * 1024),
    )
