from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Generator, List

import pytest

from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.probe import Probe
from happysimulator.core.entity import Entity
from happysimulator.components.queue import Queue
from happysimulator.components.queue_driver import QueueDriver
from happysimulator.components.queue_policy import FIFOQueue, LIFOQueue, QueuePolicy
from happysimulator.core.event import Event
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass(frozen=True)
class LinearRampProfile(Profile):

    """Linear ramp from start_rate to end_rate over t_end_s seconds."""

    t_end_s: float
    start_rate: float
    end_rate: float

    def get_rate(self, time: Instant) -> float:
        t = max(0.0, min(time.to_seconds(), self.t_end_s))
        if self.t_end_s <= 0:
            return float(self.end_rate)
        frac = t / self.t_end_s
        return float(self.start_rate + frac * (self.end_rate - self.start_rate))


class RequestProvider(EventProvider):

    """Generates request events targeting the queue.

    Optionally stops emitting events after a cutoff time. This lets tests generate
    load for a bounded window, then let the system drain without injecting more work.
    """

    def __init__(self, queue: Entity, *, stop_after: Instant | None = None):
        self._queue = queue
        self._stop_after = stop_after
        self.generated_requests: int = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated_requests += 1
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._queue,
                context={
                    "created_at": time,
                    "request_id": self.generated_requests,
                },
            )
        ]


class LatencyTrackingSink(Entity):

    """Sink that records end-to-end latency using the event context."""

    def __init__(self, name: str):
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
        if not self.latencies_s:
            return 0.0
        return sum(self.latencies_s) / len(self.latencies_s)

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)


@dataclass
class ConcurrencyLimitedServer(Entity):

    """A server with configurable concurrency and fixed service time."""

    name: str = "Server"
    service_time_s: float = 0.5
    concurrency: int = 1
    downstream: Entity | None = None

    _in_flight: int = field(default=0, init=False)
    stats_processed: int = field(default=0, init=False)

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_event(self, event: Event) -> Generator[Instant, None, list[Event]]:
        self._in_flight += 1
        yield self.service_time_s, None
        self._in_flight -= 1
        self.stats_processed += 1

        if self.downstream is None:
            return []

        # Use current simulation time (self.now) as the completion time.
        completed = Event(
            time=self.now,
            event_type="Completed",
            target=self.downstream,
            context=event.context,
        )
        return [completed]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _percentile_sorted(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


@dataclass(frozen=True)
class QueueLatencyScenarioResult:
    sink: LatencyTrackingSink
    server: ConcurrencyLimitedServer
    queue_depth_data: Data
    requests_generated: int


def run_queue_latency_scenario(
    *,
    profile: Profile,
    service_time_s: float,
    server_concurrency: int,
    queue_policy: QueuePolicy,
    test_output_dir: Path | None = None,
    duration_s: float = 30.0,
    drain_s: float = 0.0,
    probe_interval_s: float = 0.1,
    bucket_size_s: float = 1.0,
) -> QueueLatencyScenarioResult:
    if server_concurrency <= 0:
        raise ValueError("server_concurrency must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if drain_s < 0:
        raise ValueError("drain_s must be >= 0")
    if probe_interval_s <= 0:
        raise ValueError("probe_interval_s must be > 0")
    if bucket_size_s <= 0:
        raise ValueError("bucket_size_s must be > 0")

    sink = LatencyTrackingSink(name="Sink")
    server = ConcurrencyLimitedServer(
        service_time_s=service_time_s,
        concurrency=server_concurrency,
        downstream=sink,
    )
    driver = QueueDriver(name="Driver", queue=None, target=server)
    queue = Queue(name="RequestQueue", egress=driver, policy=queue_policy)
    driver.queue = queue

    queue_depth_data = Data()
    queue_depth_probe = Probe(
        target=queue,
        metric="depth",
        data=queue_depth_data,
        interval=probe_interval_s,
        start_time=Instant.Epoch,
    )

    stop_after = Instant.from_seconds(duration_s)
    provider = RequestProvider(queue, stop_after=stop_after)
    arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="RequestSource", event_provider=provider, arrival_time_provider=arrival)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + drain_s),
        sources=[source],
        entities=[queue, driver, server, sink],
        probes=[queue_depth_probe],
    )
    sim.run()

    times_s, latencies_s = sink.latency_time_series_seconds()

    buckets: dict[int, list[float]] = defaultdict(list)
    for t_s, latency_s in zip(times_s, latencies_s, strict=False):
        bucket = int(math.floor(t_s / bucket_size_s))
        buckets[bucket].append(latency_s)

    bucket_times_s: list[float] = []
    bucket_avg_s: list[float] = []
    bucket_p0_s: list[float] = []
    bucket_p50_s: list[float] = []
    bucket_p99_s: list[float] = []
    bucket_p100_s: list[float] = []
    for bucket in sorted(buckets.keys()):
        vals_sorted = sorted(buckets[bucket])
        bucket_start = bucket * bucket_size_s

        bucket_times_s.append(bucket_start)
        bucket_avg_s.append(sum(vals_sorted) / len(vals_sorted))
        bucket_p0_s.append(_percentile_sorted(vals_sorted, 0.0))
        bucket_p50_s.append(_percentile_sorted(vals_sorted, 0.50))
        bucket_p99_s.append(_percentile_sorted(vals_sorted, 0.99))
        bucket_p100_s.append(_percentile_sorted(vals_sorted, 1.0))

    queue_depth_times_s = [t for (t, _v) in queue_depth_data.values]
    queue_depth_values = [int(v) for (_t, v) in queue_depth_data.values]

    if test_output_dir is not None:
        _write_csv(
            test_output_dir / "latency_events.csv",
            header=[
                "index",
                "completion_time_s",
                "latency_s",
            ],
            rows=[[i, t, l] for i, (t, l) in enumerate(zip(times_s, latencies_s, strict=False))],
        )

        _write_csv(
            test_output_dir / "latency_timeseries.csv",
            header=[
                "bucket_start_s",
                "bucket_size_s",
                "avg_latency_s",
                "p0_latency_s",
                "p50_latency_s",
                "p99_latency_s",
                "p100_latency_s",
            ],
            rows=[
                [t, bucket_size_s, a, p0, p50, p99, p100]
                for (t, a, p0, p50, p99, p100) in zip(
                    bucket_times_s,
                    bucket_avg_s,
                    bucket_p0_s,
                    bucket_p50_s,
                    bucket_p99_s,
                    bucket_p100_s,
                    strict=False,
                )
            ],
        )

        _write_csv(
            test_output_dir / "queue_depth_timeseries.csv",
            header=["index", "time_s", "queue_depth"],
            rows=[[i, t, d] for i, (t, d) in enumerate(zip(queue_depth_times_s, queue_depth_values, strict=False))],
        )

        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(bucket_times_s, bucket_avg_s, label="avg")
        ax.plot(bucket_times_s, bucket_p0_s, label="p0")
        ax.plot(bucket_times_s, bucket_p50_s, label="p50")
        ax.plot(bucket_times_s, bucket_p99_s, label="p99")
        ax.plot(bucket_times_s, bucket_p100_s, label="p100")
        ax.set_title("Queue end-to-end latency over time (bucketed)")
        ax.set_xlabel("completion time (s)")
        ax.set_ylabel("latency (s)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(test_output_dir / "latency_timeseries.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(queue_depth_times_s, queue_depth_values)
        ax.set_title("Queue depth over time")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("queue depth")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(test_output_dir / "queue_depth_timeseries.png", dpi=150)
        plt.close(fig)

    return QueueLatencyScenarioResult(
        sink=sink,
        server=server,
        queue_depth_data=queue_depth_data,
        requests_generated=provider.generated_requests,
    )


def test_queue_latency_with_ramp_source(test_output_dir: Path) -> None:

    """Ramp load above capacity should increase average end-to-end latency."""

    duration_s = 30.0
    drain_s = 10.0
    service_time_s = 0.2  # capacity ~5 req/s
    server_concurrency = 1
    queue_policy = FIFOQueue()

    # Ramp from below capacity to well above capacity.
    profile = LinearRampProfile(t_end_s=duration_s, start_rate=0.5, end_rate=10.0)

    result = run_queue_latency_scenario(
        profile=profile,
        service_time_s=service_time_s,
        server_concurrency=server_concurrency,
        queue_policy=queue_policy,
        test_output_dir=test_output_dir,
        duration_s=duration_s,
        drain_s=drain_s,
        probe_interval_s=0.1,
        bucket_size_s=1.0,
    )

    assert result.requests_generated > 0
    assert result.sink.events_received == result.requests_generated
    assert result.server.stats_processed >= result.sink.events_received

    # Under overload, queueing delay should push average latency above service time.
    avg_latency = result.sink.average_latency()
    assert avg_latency > service_time_s


def test_queue_latency_with_ramp_source_lifo(test_output_dir: Path) -> None:

    """Same scenario as FIFO test, but with a LIFO queue policy."""

    duration_s = 30.0
    drain_s = 10.0
    service_time_s = 0.2  # capacity ~5 req/s
    server_concurrency = 1
    queue_policy = LIFOQueue()

    profile = LinearRampProfile(t_end_s=duration_s, start_rate=0.5, end_rate=10.0)

    result = run_queue_latency_scenario(
        profile=profile,
        service_time_s=service_time_s,
        server_concurrency=server_concurrency,
        queue_policy=queue_policy,
        test_output_dir=test_output_dir,
        duration_s=duration_s,
        drain_s=drain_s,
        probe_interval_s=0.1,
        bucket_size_s=1.0,
    )

    assert result.requests_generated > 0
    assert result.sink.events_received == result.requests_generated
    assert result.server.stats_processed >= result.sink.events_received

    avg_latency = result.sink.average_latency()
    assert avg_latency > service_time_s


def test_queue_latency_fifo_and_lifo_have_same_average_when_drained() -> None:

    """FIFO vs LIFO averages only differ under censoring; draining removes bias."""

    duration_s = 30.0
    drain_s = 10.0
    service_time_s = 0.2
    server_concurrency = 1
    profile = LinearRampProfile(t_end_s=duration_s, start_rate=0.5, end_rate=10.0)

    fifo = run_queue_latency_scenario(
        profile=profile,
        service_time_s=service_time_s,
        server_concurrency=server_concurrency,
        queue_policy=FIFOQueue(),
        test_output_dir=None,
        duration_s=duration_s,
        drain_s=drain_s,
        probe_interval_s=0.1,
        bucket_size_s=1.0,
    )
    lifo = run_queue_latency_scenario(
        profile=profile,
        service_time_s=service_time_s,
        server_concurrency=server_concurrency,
        queue_policy=LIFOQueue(),
        test_output_dir=None,
        duration_s=duration_s,
        drain_s=drain_s,
        probe_interval_s=0.1,
        bucket_size_s=1.0,
    )

    assert fifo.requests_generated > 0
    assert lifo.requests_generated > 0
    assert fifo.sink.events_received == fifo.requests_generated
    assert lifo.sink.events_received == lifo.requests_generated

    assert lifo.sink.average_latency() == pytest.approx(fifo.sink.average_latency(), abs=1e-6)
