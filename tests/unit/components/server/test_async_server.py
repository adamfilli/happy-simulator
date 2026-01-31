"""Tests for AsyncServer component."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Generator, List

import pytest

from happysimulator.components.server.async_server import AsyncServer, AsyncServerStats
from happysimulator.core.entity import Entity
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


class RequestProvider(EventProvider):
    """Generates request events targeting an async server."""

    def __init__(self, server: AsyncServer, stop_after: Instant | None = None):
        self.server = server
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
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


class TestAsyncServerBasics:
    """Basic AsyncServer functionality tests."""

    def test_creates_with_defaults(self):
        """AsyncServer can be created with minimal parameters."""
        server = AsyncServer(name="TestServer")
        assert server.name == "TestServer"
        assert server.max_connections == 10000
        assert server.active_connections == 0
        assert server.peak_connections == 0
        assert server.utilization == 0.0

    def test_creates_with_custom_max_connections(self):
        """AsyncServer can be created with custom max_connections."""
        server = AsyncServer(name="TestServer", max_connections=100)
        assert server.max_connections == 100

    def test_rejects_zero_max_connections(self):
        """AsyncServer rejects max_connections < 1."""
        with pytest.raises(ValueError):
            AsyncServer(name="TestServer", max_connections=0)

        with pytest.raises(ValueError):
            AsyncServer(name="TestServer", max_connections=-1)

    def test_creates_with_custom_cpu_distribution(self):
        """AsyncServer can be created with custom CPU work distribution."""
        cpu_work = ConstantLatency(0.005)
        server = AsyncServer(name="TestServer", cpu_work_distribution=cpu_work)
        # The distribution is used internally

    def test_initial_statistics_are_zero(self):
        """AsyncServer starts with zero statistics."""
        server = AsyncServer(name="TestServer")
        assert server.stats.requests_completed == 0
        assert server.stats.requests_rejected == 0
        assert server.stats.total_cpu_time == 0.0


class TestAsyncServerProcessing:
    """Tests for AsyncServer request processing."""

    def test_processes_single_request(self):
        """AsyncServer processes a single request successfully."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.010),
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
        assert server.stats.total_cpu_time == pytest.approx(0.010, rel=0.01)

    def test_processes_multiple_requests(self):
        """AsyncServer processes multiple requests."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.010),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        # Schedule 5 requests spread out
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
        assert server.stats.total_cpu_time == pytest.approx(0.050, rel=0.01)

    def test_cpu_work_serialized(self):
        """CPU work is serialized even with concurrent connections."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.100),  # 100ms CPU each
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server],
        )

        # Schedule 3 requests at the same time
        for i in range(3):
            sim.schedule(
                Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server)
            )

        sim.run()

        # All should complete (serialized CPU processing)
        assert server.stats.requests_completed == 3
        # Total CPU time should be 3 * 0.1 = 0.3s
        assert server.stats.total_cpu_time == pytest.approx(0.300, rel=0.01)


class TestAsyncServerConnections:
    """Tests for AsyncServer connection management."""

    def test_tracks_active_connections(self):
        """AsyncServer tracks active connection count."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.100),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        # Schedule multiple requests at same time
        for i in range(5):
            sim.schedule(
                Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server)
            )

        sim.run()

        # All completed
        assert server.stats.requests_completed == 5
        # Peak should have been recorded
        assert server.peak_connections >= 1

    def test_rejects_connections_at_limit(self):
        """AsyncServer rejects connections when at max_connections."""
        server = AsyncServer(
            name="TestServer",
            max_connections=2,
            cpu_work_distribution=ConstantLatency(0.500),  # Long CPU work
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server],
        )

        # Schedule 5 requests at the same time (only 2 can connect)
        for i in range(5):
            sim.schedule(
                Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server)
            )

        sim.run()

        # Only 2 should complete (at max connections)
        assert server.stats.requests_completed == 2
        assert server.stats.requests_rejected == 3

    def test_has_capacity_reflects_state(self):
        """has_capacity() returns correct state."""
        server = AsyncServer(name="TestServer", max_connections=2)

        # Initially has capacity
        assert server.has_capacity() is True
        assert server.active_connections == 0

    def test_peak_connections_tracked(self):
        """Peak connections is correctly tracked."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.200),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[server],
        )

        # Schedule 10 requests at the same time
        for i in range(10):
            sim.schedule(
                Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server)
            )

        sim.run()

        # Peak should be 10 (all arrived simultaneously)
        assert server.peak_connections == 10
        assert server.stats.requests_completed == 10


class TestAsyncServerCPUQueue:
    """Tests for AsyncServer CPU queue behavior."""

    def test_cpu_queue_builds_under_load(self):
        """CPU queue builds up when requests arrive faster than CPU processing."""
        server = AsyncServer(
            name="TestServer",
            max_connections=1000,
            cpu_work_distribution=ConstantLatency(0.100),  # 100ms CPU each
        )

        provider = RequestProvider(server, stop_after=Instant.from_seconds(0.5))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=50.0),  # 50 req/s, CPU handles 10/s
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

        # Should complete all requests eventually
        assert server.stats.requests_completed > 0


class TestAsyncServerStatistics:
    """Tests for AsyncServer statistics tracking."""

    def test_tracks_cpu_time(self):
        """AsyncServer tracks total CPU time."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.025),
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
        assert server.stats.total_cpu_time == pytest.approx(0.100, rel=0.01)

    def test_average_cpu_time(self):
        """AsyncServer calculates average CPU time correctly."""
        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.020),
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
        assert server.average_cpu_time == pytest.approx(0.020, rel=0.01)

    def test_cpu_time_percentile(self):
        """AsyncServer calculates CPU time percentiles correctly."""
        random.seed(42)

        server = AsyncServer(
            name="TestServer",
            max_connections=1000,
            cpu_work_distribution=ExponentialLatency(0.020),
        )

        provider = RequestProvider(server, stop_after=Instant.from_seconds(3.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=20.0),
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

        # Should have enough samples
        assert server.stats.requests_completed >= 30

        p50 = server.get_cpu_time_percentile(0.50)
        p99 = server.get_cpu_time_percentile(0.99)

        assert p99 >= p50
        assert p50 > 0


class TestAsyncServerWithIOHandler:
    """Tests for AsyncServer with I/O handler."""

    def test_io_handler_called_after_cpu(self):
        """I/O handler is called after CPU work completes."""
        io_calls = []

        def io_handler(event: Event):
            io_calls.append(event.event_type)
            return None

        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.010),
            io_handler=io_handler,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server],
        )

        sim.schedule(Event(time=Instant.Epoch, event_type="Request-1", target=server))
        sim.run()

        assert server.stats.requests_completed == 1
        assert len(io_calls) == 1
        assert io_calls[0] == "Request-1"

    def test_io_handler_with_generator(self):
        """I/O handler can be a generator that yields delays."""
        io_times = []

        def io_handler(event: Event) -> Generator[float, None, None]:
            io_times.append(0.050)
            yield 0.050  # 50ms I/O wait

        server = AsyncServer(
            name="TestServer",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.010),
            io_handler=io_handler,
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
        assert len(io_times) == 1


class TestAsyncServerWithLoad:
    """Integration tests for AsyncServer under various load conditions."""

    def test_server_under_light_load(self):
        """AsyncServer handles light load efficiently."""
        server = AsyncServer(
            name="TestServer",
            max_connections=1000,
            cpu_work_distribution=ConstantLatency(0.010),  # 10ms CPU
        )

        provider = RequestProvider(server, stop_after=Instant.from_seconds(5.0))
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
        assert server.stats.requests_rejected == 0

    def test_server_at_cpu_capacity(self):
        """AsyncServer handles load at CPU capacity."""
        # CPU capacity: 1 / 0.010 = 100 req/s
        server = AsyncServer(
            name="TestServer",
            max_connections=10000,
            cpu_work_distribution=ConstantLatency(0.010),
        )

        provider = RequestProvider(server, stop_after=Instant.from_seconds(2.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=100.0),  # At CPU capacity
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

        # Should process requests (CPU is bottleneck)
        assert server.stats.requests_completed >= 150

    def test_server_connection_limited(self):
        """AsyncServer rejects requests when connection limited."""
        server = AsyncServer(
            name="TestServer",
            max_connections=10,  # Low connection limit
            cpu_work_distribution=ConstantLatency(0.100),  # 100ms CPU
        )

        provider = RequestProvider(server, stop_after=Instant.from_seconds(1.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=100.0),  # High arrival rate
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

        # Some should be rejected
        assert server.stats.requests_rejected > 0
        # But some should complete
        assert server.stats.requests_completed > 0


class TestAsyncServerHighConcurrency:
    """Tests for AsyncServer high concurrency scenarios."""

    def test_many_concurrent_connections(self):
        """AsyncServer handles many concurrent connections."""
        server = AsyncServer(
            name="TestServer",
            max_connections=10000,
            cpu_work_distribution=ConstantLatency(0.001),  # 1ms CPU
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server],
        )

        # Schedule 100 requests at the same time
        for i in range(100):
            sim.schedule(
                Event(time=Instant.Epoch, event_type=f"Request-{i}", target=server)
            )

        sim.run()

        # All should complete
        assert server.stats.requests_completed == 100
        # Peak should be 100
        assert server.peak_connections == 100

    def test_utilization_tracking(self):
        """AsyncServer utilization is calculated correctly."""
        server = AsyncServer(name="TestServer", max_connections=100)

        # Initially 0
        assert server.utilization == 0.0
