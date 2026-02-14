"""Cancelled event bloat benchmark.

An entity schedules timeout events and cancels ~80% of them before they fire.
Measures the impact of cancelled events remaining in the heap (lazy deletion).
"""

from __future__ import annotations

import random
import time
import tracemalloc

from happysimulator import Entity, Event, Instant, Simulation, Sink, Source
from tests.perf.runner import BenchmarkResult

BASE_EVENT_COUNT = 100_000
WARMUP_EVENTS = 500
CANCEL_RATIO = 0.80
# Short timeout so cancelled events are encountered quickly during processing
TIMEOUT_DELAY_S = 0.001


class _CancellingServer(Entity):
    """Server that schedules a timeout for each request and cancels most of them."""

    def __init__(self, name: str, downstream: Entity, cancel_ratio: float):
        super().__init__(name)
        self._downstream = downstream
        self._cancel_ratio = cancel_ratio
        self._rng = random.Random(42)
        self._total_processed = 0
        self._total_cancelled = 0

    def handle_event(self, event: Event):
        # Schedule a timeout slightly in the future
        timeout = Event(
            time=self.now + TIMEOUT_DELAY_S,
            event_type="Timeout",
            target=self._downstream,
            context={"source": "timeout"},
        )

        yield 0.0

        # Cancel the timeout with high probability (simulating a successful response)
        if self._rng.random() < self._cancel_ratio:
            timeout.cancel()
            self._total_cancelled += 1

        self._total_processed += 1
        return [
            timeout,
            Event(time=self.now, event_type="Done", target=self._downstream, context=event.context),
        ]


def _build_and_run(event_count: int) -> tuple[int, float, int, int]:
    """Returns (events_processed, wall_clock, total_cancelled, final_heap_size)."""
    random.seed(42)
    rate = event_count * 10
    duration_s = event_count / rate

    sink = Sink("Sink")
    server = _CancellingServer("Server", downstream=sink, cancel_ratio=CANCEL_RATIO)
    source = Source.constant(rate=rate, target=server, name="Source", stop_after=duration_s)

    sim = Simulation(
        end_time=Instant.from_seconds(duration_s + TIMEOUT_DELAY_S + 0.1),
        sources=[source],
        entities=[server, sink],
    )

    start = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - start

    return (
        summary.total_events_processed,
        wall,
        server._total_cancelled,
        summary.events_cancelled,
    )


def run(*, scale: float = 1.0) -> BenchmarkResult:
    """Run the cancellation benchmark."""
    _build_and_run(WARMUP_EVENTS)

    event_count = int(BASE_EVENT_COUNT * scale)
    tracemalloc.reset_peak()
    events_processed, wall, total_cancelled, events_cancelled = _build_and_run(event_count)
    _, peak = tracemalloc.get_traced_memory()

    eps = events_processed / wall if wall > 0 else 0.0
    actual_cancelled_ratio = total_cancelled / event_count if event_count > 0 else 0.0

    return BenchmarkResult(
        name="cancellation",
        events_processed=events_processed,
        wall_clock_s=wall,
        events_per_second=eps,
        peak_memory_mb=peak / (1024 * 1024),
        extra={
            "cancelled_ratio": round(actual_cancelled_ratio, 2),
            "events_cancelled": events_cancelled,
        },
    )
