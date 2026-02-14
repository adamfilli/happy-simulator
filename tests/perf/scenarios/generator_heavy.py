"""Generator yield overhead benchmark.

Each handle_event yields 5 times, creating 5 ProcessContinuation objects
per event. Measures the cost of generator-based multi-step processing.
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

BASE_EVENT_COUNT = 100_000
WARMUP_EVENTS = 500
YIELDS_PER_EVENT = 5


class _MultiYieldServer(QueuedResource):
    """Server that yields multiple times per event."""

    def __init__(self, name: str, downstream):
        super().__init__(name, policy=FIFOQueue())
        self._downstream = downstream

    def handle_queued_event(self, event: Event):
        for _ in range(YIELDS_PER_EVENT):
            yield 0.0
        return [Event(time=self.now, event_type="Done", target=self._downstream, context=event.context)]


def _build_and_run(event_count: int) -> tuple[int, float]:
    random.seed(42)
    rate = event_count * 10
    duration_s = event_count / rate

    sink = Sink("Sink")
    server = _MultiYieldServer("Server", downstream=sink)
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
    """Run the generator-heavy benchmark."""
    _build_and_run(WARMUP_EVENTS)

    event_count = int(BASE_EVENT_COUNT * scale)
    tracemalloc.reset_peak()
    events_processed, wall = _build_and_run(event_count)
    _, peak = tracemalloc.get_traced_memory()

    eps = events_processed / wall if wall > 0 else 0.0
    return BenchmarkResult(
        name="generator_heavy",
        events_processed=events_processed,
        wall_clock_s=wall,
        events_per_second=eps,
        peak_memory_mb=peak / (1024 * 1024),
        extra={"yields_per_event": YIELDS_PER_EVENT},
    )
