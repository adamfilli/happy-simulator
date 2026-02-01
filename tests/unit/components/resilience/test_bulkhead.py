"""Tests for Bulkhead component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.resilience import Bulkhead
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass
class SlowServer(Entity):
    """Server with configurable response time."""

    name: str
    response_time: float = 0.100

    requests_received: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)
    peak_connections: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        self.active_connections += 1
        if self.active_connections > self.peak_connections:
            self.peak_connections = self.active_connections
        yield self.response_time
        self.active_connections -= 1


class TestBulkheadCreation:
    """Tests for Bulkhead creation."""

    def test_creates_with_defaults(self):
        """Bulkhead can be created with minimal parameters."""
        server = SlowServer(name="server")
        bh = Bulkhead(name="bh", target=server, max_concurrent=10)

        assert bh.name == "bh"
        assert bh.target is server
        assert bh.max_concurrent == 10
        assert bh.max_wait_queue == 0
        assert bh.max_wait_time is None

    def test_creates_with_queue_options(self):
        """Bulkhead can be created with queue options."""
        server = SlowServer(name="server")
        bh = Bulkhead(
            name="bh",
            target=server,
            max_concurrent=5,
            max_wait_queue=20,
            max_wait_time=2.0,
        )

        assert bh.max_concurrent == 5
        assert bh.max_wait_queue == 20
        assert bh.max_wait_time == 2.0

    def test_rejects_invalid_max_concurrent(self):
        """Bulkhead rejects max_concurrent < 1."""
        server = SlowServer(name="server")

        with pytest.raises(ValueError):
            Bulkhead(name="bh", target=server, max_concurrent=0)

    def test_rejects_invalid_max_wait_queue(self):
        """Bulkhead rejects max_wait_queue < 0."""
        server = SlowServer(name="server")

        with pytest.raises(ValueError):
            Bulkhead(name="bh", target=server, max_concurrent=5, max_wait_queue=-1)

    def test_rejects_invalid_max_wait_time(self):
        """Bulkhead rejects max_wait_time <= 0."""
        server = SlowServer(name="server")

        with pytest.raises(ValueError):
            Bulkhead(name="bh", target=server, max_concurrent=5, max_wait_time=0)

        with pytest.raises(ValueError):
            Bulkhead(name="bh", target=server, max_concurrent=5, max_wait_time=-1)

    def test_initial_statistics_are_zero(self):
        """Bulkhead starts with zero statistics."""
        server = SlowServer(name="server")
        bh = Bulkhead(name="bh", target=server, max_concurrent=10)

        assert bh.stats.total_requests == 0
        assert bh.stats.accepted_requests == 0
        assert bh.stats.rejected_requests == 0
        assert bh.stats.queued_requests == 0


class TestBulkheadConcurrencyLimit:
    """Tests for Bulkhead concurrency limiting."""

    def test_allows_requests_within_limit(self):
        """Bulkhead allows requests within concurrency limit."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=5)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, bh],
        )

        # Send 3 concurrent requests (within limit of 5)
        for i in range(3):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert server.requests_received == 3
        assert bh.stats.accepted_requests == 3
        assert bh.stats.rejected_requests == 0

    def test_rejects_when_at_capacity_no_queue(self):
        """Bulkhead rejects when at capacity with no queue."""
        server = SlowServer(name="server", response_time=0.200)
        bh = Bulkhead(name="bh", target=server, max_concurrent=2, max_wait_queue=0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, bh],
        )

        # Send 5 concurrent requests (limit is 2, no queue)
        for i in range(5):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Only 2 should be accepted, 3 rejected
        assert server.requests_received == 2
        assert bh.stats.accepted_requests == 2
        assert bh.stats.rejected_requests == 3

    def test_limits_concurrent_connections(self):
        """Bulkhead limits peak concurrent connections."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=3)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, bh],
        )

        # Send many concurrent requests
        for i in range(10):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Peak connections should be limited
        assert server.peak_connections <= 3


class TestBulkheadQueueing:
    """Tests for Bulkhead wait queue."""

    def test_queues_when_at_capacity(self):
        """Bulkhead queues requests when at capacity."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=2, max_wait_queue=5)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, bh],
        )

        # Send 5 concurrent requests (2 active, 3 queued)
        for i in range(5):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # All should eventually be processed
        assert server.requests_received == 5
        assert bh.stats.accepted_requests == 5
        assert bh.stats.rejected_requests == 0
        assert bh.stats.queued_requests == 3

    def test_rejects_when_queue_full(self):
        """Bulkhead rejects when queue is full."""
        server = SlowServer(name="server", response_time=0.200)
        bh = Bulkhead(name="bh", target=server, max_concurrent=2, max_wait_queue=2)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, bh],
        )

        # Send 6 requests (2 active, 2 queued, 2 rejected)
        for i in range(6):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert bh.stats.accepted_requests == 4
        assert bh.stats.rejected_requests == 2
        assert bh.stats.queued_requests == 2

    def test_queue_timeout(self):
        """Bulkhead times out queued requests."""
        server = SlowServer(name="server", response_time=0.500)
        bh = Bulkhead(
            name="bh",
            target=server,
            max_concurrent=1,
            max_wait_queue=5,
            max_wait_time=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, bh],
        )

        # Send 3 requests (1 active, 2 queued)
        for i in range(3):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Queued requests should timeout before first completes
        assert bh.stats.timed_out_requests >= 1


class TestBulkheadProperties:
    """Tests for Bulkhead properties."""

    def test_active_count(self):
        """active_count reflects current active requests."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=10)

        assert bh.active_count == 0

    def test_queue_depth(self):
        """queue_depth reflects current queue size."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=10, max_wait_queue=20)

        assert bh.queue_depth == 0

    def test_available_permits(self):
        """available_permits shows remaining capacity."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=10)

        assert bh.available_permits == 10

    def test_peak_tracking(self):
        """Bulkhead tracks peak concurrent and queue depth."""
        server = SlowServer(name="server", response_time=0.100)
        bh = Bulkhead(name="bh", target=server, max_concurrent=3, max_wait_queue=10)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, bh],
        )

        # Send 8 concurrent requests
        for i in range(8):
            request = Event(
                time=Instant.Epoch,
                event_type="request",
                target=bh,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert bh.stats.peak_concurrent == 3
        assert bh.stats.peak_queue_depth == 5
