"""Built-in collector entities for common metrics.

LatencyTracker and ThroughputTracker eliminate the boilerplate sink
entities that every simulation example reimplements.
"""

from __future__ import annotations

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.instrumentation.data import Data, BucketedData


class LatencyTracker(Entity):
    """Records end-to-end latency from event context['created_at'].

    Drop-in replacement for the custom LatencyTrackingSink that
    every example reimplements. Uses the 'created_at' field that
    Event.__post_init__ sets automatically.

    Stores (completion_time_s, latency_s) pairs in self.data.
    """

    def __init__(self, name: str = "LatencyTracker") -> None:
        super().__init__(name)
        self.data = Data()
        self.count: int = 0

    def handle_event(self, event: Event) -> list[Event]:
        created_at = event.context.get("created_at", event.time)
        latency = event.time - created_at
        latency_s = latency.to_seconds()
        self.data.add_stat(latency_s, event.time)
        self.count += 1
        return []

    def p50(self) -> float:
        """50th percentile latency in seconds."""
        return self.data.percentile(0.50)

    def p99(self) -> float:
        """99th percentile latency in seconds."""
        return self.data.percentile(0.99)

    def mean_latency(self) -> float:
        """Mean latency in seconds."""
        return self.data.mean()

    def summary(self, window_s: float = 1.0) -> BucketedData:
        """Bucket latencies by time window."""
        return self.data.bucket(window_s)


class ThroughputTracker(Entity):
    """Counts events per time window for throughput analysis.

    Records one sample per event received. Use .throughput() to get
    events-per-window bucketed by time.
    """

    def __init__(self, name: str = "ThroughputTracker") -> None:
        super().__init__(name)
        self.data = Data()
        self.count: int = 0

    def handle_event(self, event: Event) -> list[Event]:
        self.data.add_stat(1.0, event.time)
        self.count += 1
        return []

    def throughput(self, window_s: float = 1.0) -> BucketedData:
        """Returns events-per-window bucketed by time.

        The 'sum' field in each bucket equals the event count for that window.
        """
        return self.data.bucket(window_s)
