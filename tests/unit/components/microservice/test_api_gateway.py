"""Tests for APIGateway component."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.microservice import (
    APIGateway,
    RouteConfig,
)
from happysimulator.components.rate_limiter import TokenBucketPolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class BackendService(Entity):
    """Simple backend that counts requests."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


class TestAPIGatewayCreation:
    """Tests for APIGateway creation."""

    def test_creates_with_routes(self):
        backend = BackendService(name="backend")
        gw = APIGateway(
            name="gw",
            routes={"/api/users": RouteConfig(name="users", backends=[backend])},
        )

        assert gw.name == "gw"
        assert "/api/users" in gw.routes

    def test_initial_stats_are_zero(self):
        backend = BackendService(name="backend")
        gw = APIGateway(
            name="gw",
            routes={"/api/users": RouteConfig(name="users", backends=[backend])},
        )

        assert gw.stats.total_requests == 0
        assert gw.stats.requests_routed == 0
        assert gw.stats.requests_rejected_auth == 0
        assert gw.stats.requests_rejected_rate_limit == 0
        assert gw.stats.requests_no_route == 0

    def test_rejects_empty_routes(self):
        with pytest.raises(ValueError):
            APIGateway(name="gw", routes={})

    def test_rejects_negative_auth_latency(self):
        backend = BackendService(name="backend")
        with pytest.raises(ValueError):
            APIGateway(
                name="gw",
                routes={"/x": RouteConfig(name="x", backends=[backend])},
                auth_latency=-1,
            )

    def test_rejects_invalid_auth_failure_rate(self):
        backend = BackendService(name="backend")
        with pytest.raises(ValueError):
            APIGateway(
                name="gw",
                routes={"/x": RouteConfig(name="x", backends=[backend])},
                auth_failure_rate=1.5,
            )


class TestAPIGatewayRouting:
    """Tests for request routing."""

    def test_routes_to_correct_backend(self):
        users_backend = BackendService(name="users_svc")
        orders_backend = BackendService(name="orders_svc")

        gw = APIGateway(
            name="gw",
            routes={
                "/api/users": RouteConfig(
                    name="users", backends=[users_backend], auth_required=False
                ),
                "/api/orders": RouteConfig(
                    name="orders", backends=[orders_backend], auth_required=False
                ),
            },
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[users_backend, orders_backend, gw],
        )

        # Route to users
        event1 = Event(
            time=Instant.Epoch,
            event_type="request",
            target=gw,
            context={"metadata": {"route": "/api/users"}},
        )
        sim.schedule(event1)

        # Route to orders
        event2 = Event(
            time=Instant.from_seconds(0.1),
            event_type="request",
            target=gw,
            context={"metadata": {"route": "/api/orders"}},
        )
        sim.schedule(event2)

        sim.run()

        assert users_backend.requests_received == 1
        assert orders_backend.requests_received == 1
        assert gw.stats.requests_routed == 2

    def test_rejects_unknown_route(self):
        backend = BackendService(name="backend")
        gw = APIGateway(
            name="gw",
            routes={
                "/api/users": RouteConfig(name="users", backends=[backend], auth_required=False)
            },
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[backend, gw],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=gw,
            context={"metadata": {"route": "/api/unknown"}},
        )
        sim.schedule(event)
        sim.run()

        assert gw.stats.requests_no_route == 1
        assert backend.requests_received == 0

    def test_round_robin_across_backends(self):
        b1 = BackendService(name="b1")
        b2 = BackendService(name="b2")

        gw = APIGateway(
            name="gw",
            routes={"/api": RouteConfig(name="api", backends=[b1, b2], auth_required=False)},
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[b1, b2, gw],
        )

        for i in range(4):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=gw,
                context={"metadata": {"route": "/api"}},
            )
            sim.schedule(event)

        sim.run()

        assert b1.requests_received == 2
        assert b2.requests_received == 2

    def test_custom_route_extractor(self):
        backend = BackendService(name="backend")

        gw = APIGateway(
            name="gw",
            routes={"users": RouteConfig(name="users", backends=[backend], auth_required=False)},
            route_extractor=lambda e: e.context.get("metadata", {}).get("service"),
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[backend, gw],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=gw,
            context={"metadata": {"service": "users"}},
        )
        sim.schedule(event)
        sim.run()

        assert backend.requests_received == 1

    def test_no_backends_for_route(self):
        gw = APIGateway(
            name="gw",
            routes={"/api": RouteConfig(name="api", backends=[], auth_required=False)},
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[gw],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=gw,
            context={"metadata": {"route": "/api"}},
        )
        sim.schedule(event)
        sim.run()

        assert gw.stats.requests_no_backend == 1


class TestAPIGatewayRateLimiting:
    """Tests for per-route rate limiting."""

    def test_rate_limits_per_route(self):
        backend = BackendService(name="backend")
        policy = TokenBucketPolicy(capacity=2, refill_rate=1.0)

        gw = APIGateway(
            name="gw",
            routes={
                "/api": RouteConfig(
                    name="api",
                    backends=[backend],
                    rate_limit_policy=policy,
                    auth_required=False,
                ),
            },
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[backend, gw],
        )

        # Send 5 requests at t=0 — only 2 should pass
        for _ in range(5):
            event = Event(
                time=Instant.Epoch,
                event_type="request",
                target=gw,
                context={"metadata": {"route": "/api"}},
            )
            sim.schedule(event)

        sim.run()

        assert gw.stats.requests_rejected_rate_limit == 3
        assert backend.requests_received == 2

    def test_independent_route_rate_limits(self):
        b1 = BackendService(name="b1")
        b2 = BackendService(name="b2")

        gw = APIGateway(
            name="gw",
            routes={
                "/hot": RouteConfig(
                    name="hot",
                    backends=[b1],
                    rate_limit_policy=TokenBucketPolicy(capacity=1, refill_rate=0.1),
                    auth_required=False,
                ),
                "/cold": RouteConfig(
                    name="cold",
                    backends=[b2],
                    auth_required=False,
                ),
            },
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[b1, b2, gw],
        )

        # Blast /hot — most get rate limited
        for _ in range(5):
            event = Event(
                time=Instant.Epoch,
                event_type="request",
                target=gw,
                context={"metadata": {"route": "/hot"}},
            )
            sim.schedule(event)

        # /cold is unaffected
        for _ in range(5):
            event = Event(
                time=Instant.Epoch,
                event_type="request",
                target=gw,
                context={"metadata": {"route": "/cold"}},
            )
            sim.schedule(event)

        sim.run()

        assert b1.requests_received == 1  # Only 1 passes rate limit
        assert b2.requests_received == 5  # All pass (no rate limit)


class TestAPIGatewayAuth:
    """Tests for auth simulation."""

    def test_auth_rejection(self):
        random.seed(42)
        backend = BackendService(name="backend")

        gw = APIGateway(
            name="gw",
            routes={"/api": RouteConfig(name="api", backends=[backend], auth_required=True)},
            auth_latency=0.001,
            auth_failure_rate=1.0,  # All auth fails
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[backend, gw],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=gw,
            context={"metadata": {"route": "/api"}},
        )
        sim.schedule(event)
        sim.run()

        assert gw.stats.requests_rejected_auth == 1
        assert backend.requests_received == 0

    def test_auth_not_required_skips_check(self):
        backend = BackendService(name="backend")

        gw = APIGateway(
            name="gw",
            routes={"/api": RouteConfig(name="api", backends=[backend], auth_required=False)},
            auth_latency=0.001,
            auth_failure_rate=1.0,  # Would fail if checked
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[backend, gw],
        )

        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=gw,
            context={"metadata": {"route": "/api"}},
        )
        sim.schedule(event)
        sim.run()

        # Auth not required, so request passes despite 100% failure rate
        assert backend.requests_received == 1

    def test_tracks_per_route_requests(self):
        b1 = BackendService(name="b1")
        b2 = BackendService(name="b2")

        gw = APIGateway(
            name="gw",
            routes={
                "/a": RouteConfig(name="a", backends=[b1], auth_required=False),
                "/b": RouteConfig(name="b", backends=[b2], auth_required=False),
            },
            auth_latency=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[b1, b2, gw],
        )

        for _ in range(3):
            sim.schedule(
                Event(
                    time=Instant.Epoch,
                    event_type="request",
                    target=gw,
                    context={"metadata": {"route": "/a"}},
                )
            )
        for _ in range(2):
            sim.schedule(
                Event(
                    time=Instant.Epoch,
                    event_type="request",
                    target=gw,
                    context={"metadata": {"route": "/b"}},
                )
            )

        sim.run()

        assert gw.stats.per_route_requests["/a"] == 3
        assert gw.stats.per_route_requests["/b"] == 2
