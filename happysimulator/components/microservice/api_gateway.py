"""API gateway with per-route rate limiting and backend routing.

Provides request routing, per-route rate limiting, auth simulation,
and round-robin backend selection for simulating API gateway bottlenecks.

Example:
    from happysimulator.components.microservice import APIGateway, RouteConfig
    from happysimulator.components.rate_limiter import TokenBucketPolicy

    gateway = APIGateway(
        name="gateway",
        routes={
            "/api/users": RouteConfig(
                name="users",
                backends=[user_svc_1, user_svc_2],
                rate_limit_policy=TokenBucketPolicy(capacity=100, refill_rate=10),
            ),
            "/api/orders": RouteConfig(
                name="orders",
                backends=[order_svc],
                auth_required=True,
                timeout=5.0,
            ),
        },
    )
"""

import logging
import random
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from happysimulator.components.rate_limiter.policy import RateLimiterPolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass
class RouteConfig:
    """Configuration for a single API gateway route.

    Args:
        name: Human-readable route name.
        backends: List of backend entities for round-robin selection.
        rate_limit_policy: Optional per-route rate limiter.
        auth_required: Whether this route requires authentication.
        timeout: Optional per-route request timeout in seconds.
    """

    name: str
    backends: list[Entity] = field(default_factory=list)
    rate_limit_policy: RateLimiterPolicy | None = None
    auth_required: bool = True
    timeout: float | None = None


@dataclass(frozen=True)
class APIGatewayStats:
    """Statistics tracked by APIGateway."""

    total_requests: int = 0
    requests_routed: int = 0
    requests_rejected_auth: int = 0
    requests_rejected_rate_limit: int = 0
    requests_no_route: int = 0
    requests_no_backend: int = 0
    per_route_requests: dict[str, int] = field(default_factory=dict)


class APIGateway(Entity):
    """API gateway with per-route rate limiting and backend routing.

    Incoming requests are routed based on a route key extracted from
    the event. Each route can have its own rate limit policy, auth
    requirement, and pool of backends.

    Request flow:
    1. Extract route from event (via ``route_extractor``)
    2. Auth check (simulated latency + probabilistic failure)
    3. Per-route rate limit check
    4. Select backend via round-robin
    5. Forward request to backend

    Attributes:
        name: Gateway identifier.
        stats: Accumulated statistics.
    """

    def __init__(
        self,
        name: str,
        routes: dict[str, RouteConfig],
        auth_latency: float = 0.001,
        auth_failure_rate: float = 0.0,
        route_extractor: Callable[[Event], str | None] | None = None,
    ):
        """Initialize the API gateway.

        Args:
            name: Gateway identifier.
            routes: Mapping from route key to RouteConfig.
            auth_latency: Simulated auth check latency in seconds.
            auth_failure_rate: Probability of auth failure (0.0 to 1.0).
            route_extractor: Function to extract route key from event.
                           Defaults to reading ``metadata.route``.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if not routes:
            raise ValueError("At least one route must be configured")
        if auth_latency < 0:
            raise ValueError(f"auth_latency must be >= 0, got {auth_latency}")
        if not 0.0 <= auth_failure_rate <= 1.0:
            raise ValueError(f"auth_failure_rate must be in [0, 1], got {auth_failure_rate}")

        self._routes = dict(routes)
        self._auth_latency = auth_latency
        self._auth_failure_rate = auth_failure_rate
        self._route_extractor = route_extractor or self._default_route_extractor

        # Per-route round-robin indices
        self._route_indices: dict[str, int] = dict.fromkeys(routes, 0)

        # Request tracking for timeout
        self._in_flight: dict[int, dict] = {}
        self._next_request_id = 0

        self._total_requests = 0
        self._requests_routed = 0
        self._requests_rejected_auth = 0
        self._requests_rejected_rate_limit = 0
        self._requests_no_route = 0
        self._requests_no_backend = 0
        self._per_route_requests: dict[str, int] = {}

        logger.debug(
            "[%s] APIGateway initialized with %d routes: %s",
            name,
            len(routes),
            list(routes.keys()),
        )

    @property
    def stats(self) -> APIGatewayStats:
        """Return a frozen snapshot of current statistics."""
        return APIGatewayStats(
            total_requests=self._total_requests,
            requests_routed=self._requests_routed,
            requests_rejected_auth=self._requests_rejected_auth,
            requests_rejected_rate_limit=self._requests_rejected_rate_limit,
            requests_no_route=self._requests_no_route,
            requests_no_backend=self._requests_no_backend,
            per_route_requests=dict(self._per_route_requests),
        )

    @property
    def routes(self) -> dict[str, RouteConfig]:
        """The configured routes."""
        return dict(self._routes)

    @staticmethod
    def _default_route_extractor(event: Event) -> str | None:
        """Default: extract route from metadata.route."""
        return event.context.get("metadata", {}).get("route")

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        """Route incoming requests through the gateway pipeline.

        Args:
            event: The incoming event.

        Returns:
            Events to schedule, generator for async auth, or None.
        """
        if event.event_type == "_gw_timeout":
            return self._handle_timeout(event)

        if event.event_type == "_gw_response":
            return self._handle_response(event)

        return self._handle_request(event)

    def _handle_request(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        """Process request through auth, rate limit, and routing."""
        self._total_requests += 1

        # 1. Extract route
        route_key = self._route_extractor(event)
        if route_key is None or route_key not in self._routes:
            self._requests_no_route += 1
            logger.debug("[%s] No route for key=%s", self.name, route_key)
            return None

        route = self._routes[route_key]

        # Track per-route requests
        self._per_route_requests[route_key] = self._per_route_requests.get(route_key, 0) + 1

        # 2. Auth check (needs generator for latency simulation)
        if route.auth_required:
            return self._handle_request_with_auth(event, route_key, route)

        # 3. Rate limit + route (no auth needed)
        return self._rate_limit_and_route(event, route_key, route)

    def _handle_request_with_auth(
        self, event: Event, route_key: str, route: RouteConfig
    ) -> Generator[float, None, list[Event]]:
        """Handle request with auth check (generator for latency)."""
        # Simulate auth latency
        if self._auth_latency > 0:
            yield self._auth_latency

        # Check auth failure
        if self._auth_failure_rate > 0 and random.random() < self._auth_failure_rate:
            self._requests_rejected_auth += 1
            logger.debug("[%s] Auth rejected for route=%s", self.name, route_key)
            return []

        # Continue to rate limit and routing
        result = self._rate_limit_and_route(event, route_key, route)
        return result if result is not None else []

    def _rate_limit_and_route(
        self, event: Event, route_key: str, route: RouteConfig
    ) -> list[Event] | None:
        """Apply per-route rate limiting and forward to backend."""
        # Rate limit check
        if route.rate_limit_policy is not None and not route.rate_limit_policy.try_acquire(
            self.now
        ):
            self._requests_rejected_rate_limit += 1
            logger.debug("[%s] Rate limited on route=%s", self.name, route_key)
            return None

        # Select backend
        if not route.backends:
            self._requests_no_backend += 1
            logger.debug("[%s] No backends for route=%s", self.name, route_key)
            return None

        backend = self._select_backend(route_key, route)

        # Forward
        return self._forward_request(event, route_key, backend, route.timeout)

    def _select_backend(self, route_key: str, route: RouteConfig) -> Entity:
        """Round-robin backend selection."""
        idx = self._route_indices[route_key]
        backend = route.backends[idx % len(route.backends)]
        self._route_indices[route_key] = idx + 1
        return backend

    def _forward_request(
        self,
        event: Event,
        route_key: str,
        backend: Entity,
        timeout: float | None,
    ) -> list[Event]:
        """Forward request to selected backend."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._requests_routed += 1

        self._in_flight[request_id] = {
            "start_time": self.now,
            "route_key": route_key,
            "completed": False,
        }

        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=backend,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_gw_request_id": request_id,
                    "_gw_name": self.name,
                    "_gw_route": route_key,
                },
            },
        )

        # Completion hook
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_gw_response",
                target=self,
                context={"metadata": {"request_id": request_id}},
            )

        forwarded.add_completion_hook(on_complete)

        # Copy original completion hooks
        for hook in event.on_complete:
            forwarded.add_completion_hook(hook)

        events: list[Event] = [forwarded]

        # Schedule timeout if configured
        if timeout is not None:
            timeout_event = Event(
                time=self.now + Duration.from_seconds(timeout),
                event_type="_gw_timeout",
                target=self,
                context={"metadata": {"request_id": request_id}},
            )
            events.append(timeout_event)

        logger.debug(
            "[%s] Routed to %s via route=%s (request_id=%d)",
            self.name,
            backend.name,
            route_key,
            request_id,
        )

        return events

    def _handle_response(self, event: Event) -> None:
        """Handle response from backend."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            return
        request_info = self._in_flight[request_id]
        if request_info["completed"]:
            return

        request_info["completed"] = True
        del self._in_flight[request_id]

    def _handle_timeout(self, event: Event) -> None:
        """Handle request timeout."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            return
        request_info = self._in_flight[request_id]
        if request_info["completed"]:
            return

        request_info["completed"] = True
        del self._in_flight[request_id]

        logger.debug("[%s] Request %d timed out", self.name, request_id)
