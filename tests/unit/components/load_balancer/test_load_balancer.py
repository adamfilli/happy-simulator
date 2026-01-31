"""Tests for LoadBalancer component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.load_balancer.load_balancer import (
    LoadBalancer,
    LoadBalancerStats,
    BackendInfo,
)
from happysimulator.components.load_balancer.strategies import (
    RoundRobin,
    LeastConnections,
    Random,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass
class MockServer(Entity):
    """Mock server that processes requests."""
    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        self.active_connections += 1
        yield self.response_time
        self.active_connections -= 1


@dataclass
class FailingServer(Entity):
    """Server that fails after N requests."""
    name: str
    fail_after: int = 3
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        if self.requests_received > self.fail_after:
            raise RuntimeError("Server failed")
        yield self.response_time


class TestLoadBalancerCreation:
    """Tests for LoadBalancer creation."""

    def test_creates_with_defaults(self):
        """LoadBalancer can be created with minimal parameters."""
        lb = LoadBalancer(name="lb")

        assert lb.name == "lb"
        assert isinstance(lb.strategy, RoundRobin)
        assert lb.backend_count == 0
        assert lb.healthy_count == 0

    def test_creates_with_backends(self):
        """LoadBalancer can be created with initial backends."""
        servers = [MockServer(name=f"s{i}") for i in range(3)]
        lb = LoadBalancer(name="lb", backends=servers)

        assert lb.backend_count == 3
        assert lb.healthy_count == 3

    def test_creates_with_custom_strategy(self):
        """LoadBalancer can be created with custom strategy."""
        strategy = LeastConnections()
        lb = LoadBalancer(name="lb", strategy=strategy)

        assert lb.strategy is strategy

    def test_rejects_invalid_on_no_backend(self):
        """LoadBalancer rejects invalid on_no_backend value."""
        with pytest.raises(ValueError):
            LoadBalancer(name="lb", on_no_backend="invalid")

    def test_initial_statistics_are_zero(self):
        """LoadBalancer starts with zero statistics."""
        lb = LoadBalancer(name="lb")

        assert lb.stats.requests_received == 0
        assert lb.stats.requests_forwarded == 0
        assert lb.stats.requests_failed == 0
        assert lb.stats.no_backend_available == 0


class TestLoadBalancerBackendManagement:
    """Tests for backend management."""

    def test_add_backend(self):
        """add_backend adds a backend to the pool."""
        lb = LoadBalancer(name="lb")
        server = MockServer(name="server")

        lb.add_backend(server)

        assert lb.backend_count == 1
        assert server in lb.all_backends
        assert server in lb.healthy_backends

    def test_add_backend_with_weight(self):
        """add_backend accepts weight parameter."""
        lb = LoadBalancer(name="lb")
        server = MockServer(name="server")

        lb.add_backend(server, weight=5)

        info = lb.get_backend_info(server)
        assert info.weight == 5

    def test_add_backend_rejects_invalid_weight(self):
        """add_backend rejects weight < 1."""
        lb = LoadBalancer(name="lb")
        server = MockServer(name="server")

        with pytest.raises(ValueError):
            lb.add_backend(server, weight=0)

    def test_remove_backend(self):
        """remove_backend removes a backend from the pool."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        lb.remove_backend(server)

        assert lb.backend_count == 0
        assert server not in lb.all_backends

    def test_mark_unhealthy(self):
        """mark_unhealthy excludes backend from healthy pool."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        lb.mark_unhealthy(server)

        assert server in lb.all_backends
        assert server not in lb.healthy_backends
        assert server in lb.unhealthy_backends

    def test_mark_healthy(self):
        """mark_healthy includes backend in healthy pool."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        lb.mark_unhealthy(server)
        lb.mark_healthy(server)

        assert server in lb.healthy_backends
        assert server not in lb.unhealthy_backends

    def test_record_success(self):
        """record_success updates backend tracking."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        lb.record_success(server)

        info = lb.get_backend_info(server)
        assert info.consecutive_successes == 1
        assert info.consecutive_failures == 0

    def test_record_failure(self):
        """record_failure updates backend tracking."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        lb.record_failure(server)

        info = lb.get_backend_info(server)
        assert info.consecutive_failures == 1
        assert info.total_failures == 1


class TestLoadBalancerRouting:
    """Tests for request routing."""

    def test_routes_request_to_backend(self):
        """LoadBalancer routes requests to backends."""
        servers = [MockServer(name=f"s{i}") for i in range(3)]
        lb = LoadBalancer(name="lb", backends=servers, strategy=RoundRobin())

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=servers + [lb],
        )

        # Send request to load balancer
        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=lb,
            context={"metadata": {"payload": "test"}},
        )
        sim.schedule(request)
        sim.run()

        # One server should have received the request
        total_received = sum(s.requests_received for s in servers)
        assert total_received == 1
        assert lb.stats.requests_forwarded == 1

    def test_distributes_with_round_robin(self):
        """LoadBalancer distributes with round-robin strategy."""
        servers = [MockServer(name=f"s{i}") for i in range(3)]
        lb = LoadBalancer(name="lb", backends=servers, strategy=RoundRobin())

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=servers + [lb],
        )

        # Send 6 requests
        for i in range(6):
            request = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=lb,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Each server should have received 2 requests
        for server in servers:
            assert server.requests_received == 2

    def test_skips_unhealthy_backends(self):
        """LoadBalancer skips unhealthy backends when routing."""
        servers = [MockServer(name=f"s{i}") for i in range(3)]
        lb = LoadBalancer(name="lb", backends=servers, strategy=RoundRobin())

        # Mark one as unhealthy
        lb.mark_unhealthy(servers[1])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=servers + [lb],
        )

        # Send 4 requests
        for i in range(4):
            request = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=lb,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Unhealthy server should not receive requests
        assert servers[1].requests_received == 0
        assert servers[0].requests_received == 2
        assert servers[2].requests_received == 2

    def test_rejects_when_no_backends(self):
        """LoadBalancer rejects when no backends available."""
        lb = LoadBalancer(name="lb", on_no_backend="reject")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[lb],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=lb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert lb.stats.no_backend_available == 1
        assert lb.stats.requests_failed == 1


class TestLoadBalancerWithLeastConnections:
    """Tests for LoadBalancer with LeastConnections strategy."""

    def test_routes_to_least_loaded(self):
        """LeastConnections routes to backend with fewest connections."""
        servers = [
            MockServer(name="idle", response_time=0.100),
            MockServer(name="busy", response_time=0.100),
        ]
        lb = LoadBalancer(name="lb", backends=servers, strategy=LeastConnections())

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(0.5),
            sources=[],
            entities=servers + [lb],
        )

        # Make "busy" server busy first
        servers[1].active_connections = 5

        # Send a request - should go to idle server
        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=lb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert servers[0].requests_received == 1  # idle got it
        assert servers[1].requests_received == 0  # busy didn't


class TestLoadBalancerStatistics:
    """Tests for LoadBalancer statistics."""

    def test_tracks_requests_received(self):
        """LoadBalancer tracks total requests received."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, lb],
        )

        for i in range(5):
            request = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=lb,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert lb.stats.requests_received == 5
        assert lb.stats.requests_forwarded == 5

    def test_tracks_backend_health_changes(self):
        """LoadBalancer tracks health status changes."""
        server = MockServer(name="server")
        lb = LoadBalancer(name="lb", backends=[server])

        lb.mark_unhealthy(server)
        lb.mark_healthy(server)
        lb.mark_unhealthy(server)

        assert lb.stats.backends_marked_unhealthy == 2
        assert lb.stats.backends_marked_healthy == 1
