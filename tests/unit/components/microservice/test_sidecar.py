"""Tests for Sidecar component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.microservice import Sidecar
from happysimulator.components.rate_limiter import TokenBucketPolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class FastServer(Entity):
    """Server that completes quickly."""

    name: str
    response_time: float = 0.001

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class SlowServer(Entity):
    """Server that takes a long time (for timeout testing)."""

    name: str
    response_time: float = 100.0

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


class TestSidecarCreation:
    """Tests for Sidecar creation."""

    def test_creates_with_defaults(self):
        server = FastServer(name="server")
        sidecar = Sidecar(name="proxy", target=server)

        assert sidecar.name == "proxy"
        assert sidecar.target is server
        assert sidecar.circuit_state == "closed"

    def test_initial_stats_are_zero(self):
        server = FastServer(name="server")
        sidecar = Sidecar(name="proxy", target=server)

        assert sidecar.stats.total_requests == 0
        assert sidecar.stats.successful_requests == 0
        assert sidecar.stats.failed_requests == 0
        assert sidecar.stats.retries == 0
        assert sidecar.stats.rate_limited == 0
        assert sidecar.stats.circuit_broken == 0
        assert sidecar.stats.timed_out == 0

    def test_rejects_invalid_circuit_failure_threshold(self):
        server = FastServer(name="server")
        with pytest.raises(ValueError):
            Sidecar(name="x", target=server, circuit_failure_threshold=0)

    def test_rejects_invalid_circuit_timeout(self):
        server = FastServer(name="server")
        with pytest.raises(ValueError):
            Sidecar(name="x", target=server, circuit_timeout=0)

    def test_rejects_invalid_request_timeout(self):
        server = FastServer(name="server")
        with pytest.raises(ValueError):
            Sidecar(name="x", target=server, request_timeout=0)

    def test_rejects_negative_max_retries(self):
        server = FastServer(name="server")
        with pytest.raises(ValueError):
            Sidecar(name="x", target=server, max_retries=-1)


class TestSidecarForwarding:
    """Tests for Sidecar request forwarding."""

    def test_forwards_requests_to_target(self):
        server = FastServer(name="server")
        sidecar = Sidecar(name="proxy", target=server, request_timeout=1.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, sidecar],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=sidecar,
        )
        sim.schedule(request)
        sim.run()

        assert server.requests_received == 1
        assert sidecar.stats.total_requests == 1
        assert sidecar.stats.successful_requests == 1

    def test_forwards_multiple_requests(self):
        server = FastServer(name="server")
        sidecar = Sidecar(name="proxy", target=server, request_timeout=1.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, sidecar],
        )

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=sidecar,
            )
            sim.schedule(event)

        sim.run()

        assert server.requests_received == 5
        assert sidecar.stats.successful_requests == 5


class TestSidecarRateLimiting:
    """Tests for Sidecar rate limiting."""

    def test_rate_limits_excess_requests(self):
        server = FastServer(name="server")
        policy = TokenBucketPolicy(capacity=2, refill_rate=1.0)
        sidecar = Sidecar(
            name="proxy",
            target=server,
            rate_limit_policy=policy,
            request_timeout=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, sidecar],
        )

        # Send 5 requests at t=0 — only 2 should pass (bucket capacity)
        for _ in range(5):
            event = Event(
                time=Instant.Epoch,
                event_type="request",
                target=sidecar,
            )
            sim.schedule(event)

        sim.run()

        assert sidecar.stats.rate_limited == 3
        assert server.requests_received == 2

    def test_no_rate_limit_when_policy_is_none(self):
        server = FastServer(name="server")
        sidecar = Sidecar(name="proxy", target=server, request_timeout=1.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, sidecar],
        )

        for _ in range(10):
            event = Event(
                time=Instant.Epoch,
                event_type="request",
                target=sidecar,
            )
            sim.schedule(event)

        sim.run()

        assert sidecar.stats.rate_limited == 0
        assert server.requests_received == 10


class TestSidecarCircuitBreaker:
    """Tests for Sidecar circuit breaker behavior."""

    def test_opens_circuit_after_failures(self):
        """Circuit opens after failure_threshold consecutive timeouts."""
        server = SlowServer(name="server")
        sidecar = Sidecar(
            name="proxy",
            target=server,
            request_timeout=0.01,
            max_retries=0,
            circuit_failure_threshold=3,
            circuit_timeout=1000.0,  # Large enough to stay OPEN after sim ends
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[server, sidecar],
        )

        # Send enough requests to trigger circuit open
        for i in range(5):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=sidecar,
            )
            sim.schedule(event)

        sim.run()

        assert sidecar.stats.failed_requests >= 3
        assert sidecar.circuit_state == "open"
        assert sidecar.stats.circuit_broken > 0


class TestSidecarTimeout:
    """Tests for Sidecar timeout and retry behavior."""

    def test_times_out_slow_requests(self):
        server = SlowServer(name="server")
        sidecar = Sidecar(
            name="proxy",
            target=server,
            request_timeout=0.05,
            max_retries=0,
            circuit_failure_threshold=100,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, sidecar],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=sidecar,
        )
        sim.schedule(event)
        sim.run()

        assert sidecar.stats.timed_out == 1
        assert sidecar.stats.failed_requests == 1

    def test_retries_on_timeout(self):
        server = SlowServer(name="server")
        sidecar = Sidecar(
            name="proxy",
            target=server,
            request_timeout=0.01,
            max_retries=2,
            retry_base_delay=0.01,
            circuit_failure_threshold=100,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[server, sidecar],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=sidecar,
        )
        sim.schedule(event)
        sim.run()

        # Should have retried twice (max_retries=2) then failed
        assert sidecar.stats.retries == 2
        assert sidecar.stats.timed_out == 3  # original + 2 retries
        assert sidecar.stats.failed_requests == 1
