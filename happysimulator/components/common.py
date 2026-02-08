"""Common reusable entities for simulations.

Provides ready-made Sink and Counter entities that eliminate the need
to write boilerplate event-collecting classes in every example or test.
"""

from __future__ import annotations

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class Sink(Entity):
    """Event collector with latency tracking.

    Computes per-event latency from ``context["created_at"]``.  When that
    key is missing the latency is recorded as 0.

    Attributes:
        events_received: Total number of events handled.
        latencies_s: Per-event latency in seconds (same order as arrival).
        completion_times: Simulation ``Instant`` when each event arrived.
    """

    def __init__(self, name: str = "Sink"):
        super().__init__(name)
        self.events_received: int = 0
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.events_received += 1

        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()
        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)

        return []

    def average_latency(self) -> float:
        """Mean latency across all received events."""
        if not self.latencies_s:
            return 0.0
        return sum(self.latencies_s) / len(self.latencies_s)

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return ``(completion_times_s, latencies_s)`` for plotting."""
        return (
            [t.to_seconds() for t in self.completion_times],
            list(self.latencies_s),
        )

    def latency_stats(self) -> dict:
        """Summary statistics over all recorded latencies.

        Returns a dict with keys: count, avg, min, max, p50, p99.
        """
        n = len(self.latencies_s)
        if n == 0:
            return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0, "p99": 0.0}

        sorted_vals = sorted(self.latencies_s)
        return {
            "count": n,
            "avg": sum(sorted_vals) / n,
            "min": sorted_vals[0],
            "max": sorted_vals[-1],
            "p50": _percentile(sorted_vals, 0.50),
            "p99": _percentile(sorted_vals, 0.99),
        }


class Counter(Entity):
    """Simple event counter that tallies events by type.

    Attributes:
        total: Total events received.
        by_type: Mapping from ``event_type`` string to count.
    """

    def __init__(self, name: str = "Counter"):
        super().__init__(name)
        self.total: int = 0
        self.by_type: dict[str, int] = {}

    def handle_event(self, event: Event) -> None:
        self.total += 1
        et = event.event_type
        self.by_type[et] = self.by_type.get(et, 0) + 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Linearly-interpolated percentile from a pre-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac
