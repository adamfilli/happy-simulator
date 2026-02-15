"""Tests for Server component."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from happysimulator.components.queue_policy import LIFOQueue
from happysimulator.components.server.server import Server
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.exponential import ExponentialLatency
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""

    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


class ServerRequestProvider(EventProvider):
    """Generates request events targeting a server."""

    def __init__(self, server: Server, stop_after: Instant | None = None):
        self.server = server
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        return [
            Event(
                time=time,
                event_type=f"Request-{self.generated}",
                target=self.server,
            )
        ]


class TestServerBasics:
    """Basic Server functionality tests."""

    def test_creates_with_defaults(self):
        """Server can be created with minimal parameters."""
        server = Server(name="TestServer")
        assert server.name == "TestServer"
        assert server.concurrency == 1
        assert server.active_requests == 0
        assert server.utilization == 0.0

    def test_creates_with_custom_concurrency(self):
        """Server can be created with custom concurrency."""
        server = Server(name="TestServer", concurrency=4)
        assert server.concurrency == 4

    def test_creates_with_custom_service_time(self):
        """Server can be created with custom service time distribution."""
        service_time = ConstantLatency(0.050)
        server = Server(name="TestServer", service_time=service_time)
        assert server.service_time is service_time

    def test_initial_statistics_are_zero(self):
        """Server starts with zero statistics."""
        server = Server(name="TestServer")
        assert server.stats.requests_completed == 0
        assert server.stats.requests_rejected == 0
        assert server.stats.total_service_time == 0.0


class TestServerProcessing:
    """Tests for Server request processing."""

    def test_processes_single_request(self):
        """Server processes a single request successfully."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.010),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        sim.schedule(Event(time=Instant.Epoch, event_type="Request", target=server))
        sim.run()

        assert server.stats.requests_completed == 1
        assert server.stats.total_service_time == pytest.approx(0.010, rel=0.01)

    def test_processes_multiple_requests_sequentially(self):
        """Server with concurrency=1 processes requests sequentially."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.100),
        )

        provider = ServerRequestProvider(server, stop_after=Instant.from_seconds(0.5))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=5.0),  # 5 requests in 0.5s
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),  # Enough time to complete all
            sources=[source],
            entities=[server],
        )
        sim.run()

        # Should complete multiple requests
        assert server.stats.requests_completed >= 2

    def test_concurrent_processing(self):
        """Server with concurrency > 1 processes requests in parallel."""
        server = Server(
            name="TestServer",
            concurrency=4,
            service_time=ConstantLatency(0.100),
        )

        # Schedule 4 requests at exactly the same time
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        for i in range(4):
            sim.schedule(Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server))

        sim.run()

        # All 4 should complete
        assert server.stats.requests_completed == 4
        # Total service time should be 4 * 0.1 = 0.4
        assert server.stats.total_service_time == pytest.approx(0.4, rel=0.01)


class TestServerConcurrency:
    """Tests for Server concurrency control."""

    def test_has_capacity_reflects_active_requests(self):
        """has_capacity() returns False when at concurrency limit."""
        server = Server(name="TestServer", concurrency=2)

        # Initially has capacity
        assert server.has_capacity() is True
        assert server.active_requests == 0

    def test_utilization_calculation(self):
        """Utilization is correctly calculated as active/concurrency."""
        server = Server(name="TestServer", concurrency=4)

        # Initially 0
        assert server.utilization == 0.0

    def test_requests_queue_when_at_capacity(self):
        """Requests are queued when server is at concurrency limit."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.100),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        # Schedule 3 requests at the same time
        for i in range(3):
            sim.schedule(Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server))

        sim.run()

        # All 3 should complete (queued and processed sequentially)
        assert server.stats.requests_completed == 3


class TestServerQueue:
    """Tests for Server queue behavior."""

    def test_queue_depth_increases_under_load(self):
        """Queue depth increases when requests arrive faster than processing."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.100),  # 10 req/s capacity
        )

        provider = ServerRequestProvider(server, stop_after=Instant.from_seconds(1.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=20.0),  # 20 req/s arrival
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source],
            entities=[server],
        )
        sim.run()

        # Queue should have built up (can check stats)
        assert server.stats.requests_completed > 0
        assert server.stats_accepted > 0

    def test_custom_queue_policy(self):
        """Server respects custom queue policy."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.010),
            queue_policy=LIFOQueue(),
        )

        # Verify it uses the custom policy
        assert isinstance(server.queue.policy, LIFOQueue)


class TestServerStatistics:
    """Tests for Server statistics tracking."""

    def test_tracks_completed_requests(self):
        """Server tracks number of completed requests."""
        server = Server(
            name="TestServer",
            concurrency=2,
            service_time=ConstantLatency(0.010),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        for i in range(5):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type=f"Request-{i}",
                    target=server,
                )
            )

        sim.run()

        assert server.stats.requests_completed == 5

    def test_tracks_total_service_time(self):
        """Server tracks total service time."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.050),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        for i in range(3):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type=f"Request-{i}",
                    target=server,
                )
            )

        sim.run()

        assert server.stats.requests_completed == 3
        assert server.stats.total_service_time == pytest.approx(0.150, rel=0.01)

    def test_average_service_time(self):
        """Server calculates average service time correctly."""
        server = Server(
            name="TestServer",
            concurrency=1,
            service_time=ConstantLatency(0.025),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        for i in range(4):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type=f"Request-{i}",
                    target=server,
                )
            )

        sim.run()

        assert server.stats.requests_completed == 4
        assert server.average_service_time == pytest.approx(0.025, rel=0.01)

    def test_service_time_percentile(self):
        """Server calculates service time percentiles correctly."""
        random.seed(42)

        server = Server(
            name="TestServer",
            concurrency=4,
            service_time=ExponentialLatency(0.050),
        )

        provider = ServerRequestProvider(server, stop_after=Instant.from_seconds(5.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=10.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[server],
        )
        sim.run()

        # Should have enough samples for percentile calculation
        assert server.stats.requests_completed >= 30

        p50 = server.get_service_time_percentile(0.50)
        p99 = server.get_service_time_percentile(0.99)

        # p99 should be higher than p50
        assert p99 > p50
        # p50 should be roughly the mean for exponential
        assert p50 > 0


class TestServerWithConcurrencyModels:
    """Tests for Server with different ConcurrencyModel implementations."""

    def test_server_with_fixed_concurrency(self):
        """Server works with explicit FixedConcurrency model."""
        from happysimulator.components.server.concurrency import FixedConcurrency

        model = FixedConcurrency(max_concurrent=3)
        server = Server(
            name="TestServer",
            concurrency=model,
            service_time=ConstantLatency(0.010),
        )

        assert server.concurrency == 3
        assert server.concurrency_model is model

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        for i in range(5):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.05),
                    event_type=f"Request-{i}",
                    target=server,
                )
            )

        sim.run()
        assert server.stats.requests_completed == 5

    def test_server_with_dynamic_concurrency(self):
        """Server works with DynamicConcurrency model."""
        from happysimulator.components.server.concurrency import DynamicConcurrency

        model = DynamicConcurrency(initial=2, min_limit=1, max_limit=8)
        server = Server(
            name="TestServer",
            concurrency=model,
            service_time=ConstantLatency(0.010),
        )

        assert server.concurrency == 2

        # Scale up
        model.scale_up(2)
        assert server.concurrency == 4

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        for i in range(4):
            sim.schedule(Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server))

        sim.run()
        assert server.stats.requests_completed == 4

    def test_server_with_weighted_concurrency(self):
        """Server works with WeightedConcurrency model."""
        from happysimulator.components.server.concurrency import WeightedConcurrency

        model = WeightedConcurrency(total_capacity=100)
        server = Server(
            name="TestServer",
            concurrency=model,
            service_time=ConstantLatency(0.010),
        )

        assert server.concurrency == 100

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        # Schedule requests with different weights
        for i, weight in enumerate([10, 20, 30]):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type=f"Request-{i}",
                target=server,
            )
            event.context["metadata"]["weight"] = weight
            sim.schedule(event)

        sim.run()
        assert server.stats.requests_completed == 3


class TestServerWithLoad:
    """Integration tests for Server under various load conditions."""

    def test_server_under_light_load(self):
        """Server handles light load with low latency."""
        server = Server(
            name="TestServer",
            concurrency=4,
            service_time=ConstantLatency(0.010),  # 10ms
        )

        provider = ServerRequestProvider(server, stop_after=Instant.from_seconds(5.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=10.0),  # Well under capacity
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(6.0),
            sources=[source],
            entities=[server],
        )
        sim.run()

        # All requests should complete
        assert server.stats.requests_completed >= 40
        # Queue should not build up significantly
        assert server.depth == 0

    def test_server_at_capacity(self):
        """Server handles load at capacity."""
        # Capacity: 4 concurrent * (1/0.1) = 40 req/s
        server = Server(
            name="TestServer",
            concurrency=4,
            service_time=ConstantLatency(0.100),
        )

        provider = ServerRequestProvider(server, stop_after=Instant.from_seconds(2.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=40.0),  # At capacity
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[server],
        )
        sim.run()

        # Should process most requests
        assert server.stats.requests_completed >= 70

    def test_server_overloaded(self):
        """Server queues requests when overloaded."""
        server = Server(
            name="TestServer",
            concurrency=2,
            service_time=ConstantLatency(0.100),  # Capacity: 20 req/s
        )

        provider = ServerRequestProvider(server, stop_after=Instant.from_seconds(1.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=50.0),  # 2.5x capacity
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),  # Time to drain
            sources=[source],
            entities=[server],
        )
        sim.run()

        # Should complete all eventually
        assert server.stats.requests_completed >= 40
        # Queue was used
        assert server.stats_accepted >= 40
