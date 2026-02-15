"""Instrumentation overhead benchmark.

Same workload as throughput but with a LatencyTracker on the sink and a
Probe sampling queue depth. Measures the cost of Data.add_stat() and
probe event generation on top of the base event loop.
"""

from __future__ import annotations

import random
import time
import tracemalloc

from happysimulator import (
    Data,
    Event,
    Instant,
    LatencyTracker,
    Probe,
    QueuedResource,
    Simulation,
    Source,
)
from happysimulator.components.queue_policy import FIFOQueue
from tests.perf.runner import BenchmarkResult

BASE_EVENT_COUNT = 200_000
WARMUP_EVENTS = 500
PROBE_INTERVAL = 0.01  # seconds


class _MinimalServer(QueuedResource):
    """Server with near-zero service time."""

    def __init__(self, name: str, downstream):
        super().__init__(name, policy=FIFOQueue())
        self._downstream = downstream

    def handle_queued_event(self, event: Event):
        yield 0.0
        return [
            Event(time=self.now, event_type="Done", target=self._downstream, context=event.context)
        ]


def _build_and_run(event_count: int) -> tuple[int, float]:
    random.seed(42)
    rate = event_count * 10
    duration_s = event_count / rate

    tracker = LatencyTracker("Tracker")
    server = _MinimalServer("Server", downstream=tracker)

    depth_data = Data()
    probe = Probe(target=server, metric="depth", data=depth_data, interval=PROBE_INTERVAL)

    source = Source.constant(rate=rate, target=server, name="Source", stop_after=duration_s)

    sim = Simulation(
        end_time=Instant.from_seconds(duration_s + 0.001),
        sources=[source],
        entities=[server, tracker],
        probes=[probe],
    )

    start = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - start
    return summary.total_events_processed, wall


def run(*, scale: float = 1.0) -> BenchmarkResult:
    """Run the instrumented benchmark."""
    _build_and_run(WARMUP_EVENTS)

    event_count = int(BASE_EVENT_COUNT * scale)
    tracemalloc.reset_peak()
    events_processed, wall = _build_and_run(event_count)
    _, peak = tracemalloc.get_traced_memory()

    eps = events_processed / wall if wall > 0 else 0.0
    return BenchmarkResult(
        name="instrumented",
        events_processed=events_processed,
        wall_clock_s=wall,
        events_per_second=eps,
        peak_memory_mb=peak / (1024 * 1024),
        extra={"probe_interval_s": PROBE_INTERVAL},
    )
