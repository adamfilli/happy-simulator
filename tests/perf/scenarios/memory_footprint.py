"""Event object memory footprint benchmark.

Creates N Event objects in a list and measures total memory via tracemalloc.
Directly quantifies per-event memory cost (context dict, trace, on_complete, etc.).
"""

from __future__ import annotations

import time
import tracemalloc

from happysimulator import Event, Instant, NullEntity
from tests.perf.runner import BenchmarkResult

BASE_EVENT_COUNT = 100_000


def run(*, scale: float = 1.0) -> BenchmarkResult:
    """Run the memory footprint benchmark."""
    event_count = int(BASE_EVENT_COUNT * scale)
    target = NullEntity()

    tracemalloc.reset_peak()
    snapshot_before = tracemalloc.take_snapshot()

    start = time.perf_counter()
    events = [
        Event(
            time=Instant.from_seconds(i * 0.001),
            event_type="Request",
            target=target,
        )
        for i in range(event_count)
    ]
    wall = time.perf_counter() - start

    snapshot_after = tracemalloc.take_snapshot()
    _, peak = tracemalloc.get_traced_memory()

    # Compute memory difference attributable to the event list
    stats = snapshot_after.compare_to(snapshot_before, "filename")
    event_memory = sum(s.size_diff for s in stats if s.size_diff > 0)
    bytes_per_event = event_memory / event_count if event_count > 0 else 0

    # Keep reference alive until measurement is done
    _ = len(events)

    return BenchmarkResult(
        name="memory_footprint",
        events_processed=event_count,
        wall_clock_s=wall,
        events_per_second=0.0,  # Not meaningful for this scenario
        peak_memory_mb=peak / (1024 * 1024),
        extra={
            "bytes_per_event": round(bytes_per_event, 1),
            "total_memory_mb": round(event_memory / (1024 * 1024), 2),
        },
    )
