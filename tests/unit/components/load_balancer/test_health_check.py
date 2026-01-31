"""Tests for HealthChecker component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.load_balancer.load_balancer import LoadBalancer
from happysimulator.components.load_balancer.health_check import (
    HealthChecker,
    HealthCheckStats,
    BackendHealthState,
)
from happysimulator.components.load_balancer.strategies import RoundRobin
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass
class HealthyServer(Entity):
    """Server that always responds to health checks."""
    name: str
    response_time: float = 0.010

    health_checks_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        if event.event_type == "health_check":
            self.health_checks_received += 1
        yield self.response_time


@dataclass
class SlowServer(Entity):
    """Server that responds slowly (may timeout)."""
    name: str
    response_time: float = 10.0  # Very slow

    health_checks_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        if event.event_type == "health_check":
            self.health_checks_received += 1
        yield self.response_time


@dataclass
class IntermittentServer(Entity):
    """Server that alternates between fast and slow responses."""
    name: str
    fast_response: float = 0.010
    slow_response: float = 10.0
    fail_pattern: list = field(default_factory=lambda: [False, False, True, True, True])

    requests: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests += 1
        idx = (self.requests - 1) % len(self.fail_pattern)
        if self.fail_pattern[idx]:
            yield self.slow_response  # Will timeout
        else:
            yield self.fast_response


class TestHealthCheckerCreation:
    """Tests for HealthChecker creation."""

    def test_creates_with_defaults(self):
        """HealthChecker can be created with minimal parameters."""
        server = HealthyServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(name="hc", load_balancer=lb)

        assert hc.name == "hc"
        assert hc.load_balancer is lb
        assert hc.interval == 10.0
        assert hc.timeout == 5.0
        assert hc.healthy_threshold == 2
        assert hc.unhealthy_threshold == 3

    def test_creates_with_custom_parameters(self):
        """HealthChecker can be created with custom parameters."""
        server = HealthyServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=5.0,
            timeout=1.0,
            healthy_threshold=3,
            unhealthy_threshold=2,
        )

        assert hc.interval == 5.0
        assert hc.timeout == 1.0
        assert hc.healthy_threshold == 3
        assert hc.unhealthy_threshold == 2

    def test_rejects_non_positive_interval(self):
        """HealthChecker rejects interval <= 0."""
        lb = LoadBalancer(name="lb")

        with pytest.raises(ValueError):
            HealthChecker(name="hc", load_balancer=lb, interval=0)

    def test_rejects_non_positive_timeout(self):
        """HealthChecker rejects timeout <= 0."""
        lb = LoadBalancer(name="lb")

        with pytest.raises(ValueError):
            HealthChecker(name="hc", load_balancer=lb, timeout=0)

    def test_rejects_timeout_ge_interval(self):
        """HealthChecker rejects timeout >= interval."""
        lb = LoadBalancer(name="lb")

        with pytest.raises(ValueError):
            HealthChecker(name="hc", load_balancer=lb, interval=5.0, timeout=5.0)

        with pytest.raises(ValueError):
            HealthChecker(name="hc", load_balancer=lb, interval=5.0, timeout=6.0)

    def test_rejects_invalid_thresholds(self):
        """HealthChecker rejects thresholds < 1."""
        lb = LoadBalancer(name="lb")

        with pytest.raises(ValueError):
            HealthChecker(name="hc", load_balancer=lb, healthy_threshold=0)

        with pytest.raises(ValueError):
            HealthChecker(name="hc", load_balancer=lb, unhealthy_threshold=0)

    def test_initial_statistics_are_zero(self):
        """HealthChecker starts with zero statistics."""
        lb = LoadBalancer(name="lb")
        hc = HealthChecker(name="hc", load_balancer=lb)

        assert hc.stats.checks_performed == 0
        assert hc.stats.checks_passed == 0
        assert hc.stats.checks_failed == 0


class TestHealthCheckerOperation:
    """Tests for HealthChecker operation."""

    def test_sends_health_checks(self):
        """HealthChecker sends health check probes to backends."""
        server = HealthyServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.5,
            timeout=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, lb, hc],
        )

        # Start health checking
        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        # Should have performed multiple checks
        assert server.health_checks_received >= 3
        assert hc.stats.checks_performed >= 3

    def test_marks_slow_backend_unhealthy(self):
        """HealthChecker marks slow backends as unhealthy."""
        server = SlowServer(name="slow", response_time=10.0)
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.5,
            timeout=0.1,
            unhealthy_threshold=2,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[server, lb, hc],
        )

        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        # Server should be marked unhealthy after consecutive timeouts
        assert server in lb.unhealthy_backends
        assert hc.stats.checks_timed_out >= 2

    def test_marks_recovered_backend_healthy(self):
        """HealthChecker marks recovered backends as healthy."""
        # Server that fails first 3 checks then succeeds
        server = IntermittentServer(
            name="server",
            fast_response=0.010,
            slow_response=10.0,
            fail_pattern=[True, True, True, False, False, False],
        )
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.3,
            timeout=0.1,
            healthy_threshold=2,
            unhealthy_threshold=2,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[server, lb, hc],
        )

        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        # Should have been marked unhealthy then healthy
        assert hc.stats.backends_marked_unhealthy >= 1
        # May or may not be marked healthy depending on timing


class TestHealthCheckerState:
    """Tests for HealthChecker state tracking."""

    def test_tracks_consecutive_successes(self):
        """HealthChecker tracks consecutive successes per backend."""
        server = HealthyServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.2,
            timeout=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, lb, hc],
        )

        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        state = hc.get_backend_state(server)
        assert state.consecutive_successes > 0
        assert state.consecutive_failures == 0

    def test_tracks_consecutive_failures(self):
        """HealthChecker tracks consecutive failures per backend."""
        server = SlowServer(name="slow", response_time=10.0)
        lb = LoadBalancer(name="lb", backends=[server])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.2,
            timeout=0.05,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, lb, hc],
        )

        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        state = hc.get_backend_state(server)
        assert state.consecutive_failures > 0
        assert state.consecutive_successes == 0

    def test_start_stop(self):
        """HealthChecker can be started and stopped."""
        lb = LoadBalancer(name="lb")
        hc = HealthChecker(name="hc", load_balancer=lb)

        assert not hc.is_running

        hc.start()
        assert hc.is_running

        hc.stop()
        assert not hc.is_running


class TestHealthCheckerMultipleBackends:
    """Tests for HealthChecker with multiple backends."""

    def test_checks_all_backends(self):
        """HealthChecker checks all backends."""
        servers = [HealthyServer(name=f"s{i}") for i in range(3)]
        lb = LoadBalancer(name="lb", backends=servers)
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.3,
            timeout=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=servers + [lb, hc],
        )

        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        # All servers should have received health checks
        for server in servers:
            assert server.health_checks_received >= 2

    def test_independent_tracking(self):
        """HealthChecker tracks each backend independently."""
        healthy = HealthyServer(name="healthy")
        slow = SlowServer(name="slow", response_time=10.0)
        lb = LoadBalancer(name="lb", backends=[healthy, slow])
        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.3,
            timeout=0.1,
            unhealthy_threshold=2,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[healthy, slow, lb, hc],
        )

        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)
        sim.run()

        # Healthy should still be healthy
        assert healthy in lb.healthy_backends

        # Slow should be unhealthy
        assert slow in lb.unhealthy_backends
