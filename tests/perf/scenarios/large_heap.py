"""Large heap scaling benchmark.

Schedules many future events spanning a wide time range, then processes them all.
Tests EventHeap push/pop performance when the heap is large.
"""

from __future__ import annotations

import random
import time
import tracemalloc

from happysimulator import Event, Instant, NullEntity, Simulation
from tests.perf.runner import BenchmarkResult

BASE_EVENT_COUNT = 100_000
WARMUP_EVENTS = 1_000


def _build_and_run(event_count: int) -> tuple[int, float, int]:
    """Schedule N events across a wide time range and process them all.

    Returns (events_processed, wall_clock, peak_heap_size).
    """
    random.seed(42)
    target = NullEntity()

    # Create events spread across a wide time range
    events = [
        Event(
            time=Instant.from_seconds(random.uniform(0.0, 1000.0)),
            event_type="Work",
            target=target,
        )
        for _ in range(event_count)
    ]

    sim = Simulation(
        end_time=Instant.from_seconds(1001.0),
        entities=[target],
    )
    sim.schedule(events)
    peak_heap = sim._event_heap.size()

    start = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - start

    return summary.total_events_processed, wall, peak_heap


def run(*, scale: float = 1.0) -> BenchmarkResult:
    """Run the large heap benchmark."""
    _build_and_run(WARMUP_EVENTS)

    event_count = int(BASE_EVENT_COUNT * scale)
    tracemalloc.reset_peak()
    events_processed, wall, peak_heap = _build_and_run(event_count)
    _, peak = tracemalloc.get_traced_memory()

    eps = events_processed / wall if wall > 0 else 0.0
    return BenchmarkResult(
        name="large_heap",
        events_processed=events_processed,
        wall_clock_s=wall,
        events_per_second=eps,
        peak_memory_mb=peak / (1024 * 1024),
        extra={"peak_heap_size": peak_heap},
    )
