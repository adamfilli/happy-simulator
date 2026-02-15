"""Load balancer for distributing requests across backend servers.

Provides a configurable load balancer that routes incoming requests to
backend servers using pluggable strategies. Supports health checking
and dynamic backend management.

Example:
    from happysimulator.components.load_balancer import LoadBalancer, RoundRobin

    lb = LoadBalancer(
        name="api_lb",
        backends=[server1, server2, server3],
        strategy=RoundRobin(),
    )

    # Route a request
    request = Event(time=now, event_type="request", target=lb, ...)
    sim.schedule(request)
"""

import logging
from dataclasses import dataclass
from typing import Generator

from happysimulator.components.load_balancer.strategies import (
    LoadBalancingStrategy,
    RoundRobin,
    LeastResponseTime,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information tracked for each backend."""

    backend: Entity
    weight: int = 1
    is_healthy: bool = True
    consecutive_successes: int = 0
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0


@dataclass(frozen=True)
class LoadBalancerStats:
    """Statistics tracked by LoadBalancer."""

    requests_received: int = 0
    requests_forwarded: int = 0
    requests_failed: int = 0
    no_backend_available: int = 0
    backends_marked_unhealthy: int = 0
    backends_marked_healthy: int = 0


class LoadBalancer(Entity):
    """Distributes requests across multiple backend servers.

    The load balancer receives incoming requests and forwards them to
    one of the configured backends based on the selected strategy.
    It tracks backend health and can exclude unhealthy backends from
    the routing pool.

    Attributes:
        name: Load balancer identifier for logging.
        backends: List of backend entities to route to.
        strategy: Algorithm for selecting backends.
    """

    def __init__(
        self,
        name: str,
        backends: list[Entity] | None = None,
        strategy: LoadBalancingStrategy | None = None,
        on_no_backend: str = "reject",  # "reject" or "queue"
    ):
        """Initialize the load balancer.

        Args:
            name: Load balancer identifier.
            backends: Initial list of backend entities.
            strategy: Load balancing strategy (default RoundRobin).
            on_no_backend: Action when no backend available.

        Raises:
            ValueError: If on_no_backend is invalid.
        """
        super().__init__(name)

        if on_no_backend not in ("reject", "queue"):
            raise ValueError(f"on_no_backend must be 'reject' or 'queue', got {on_no_backend}")

        self._strategy = strategy or RoundRobin()
        self._on_no_backend = on_no_backend

        # Backend tracking
        self._backends: dict[str, BackendInfo] = {}
        if backends:
            for backend in backends:
                self.add_backend(backend)

        # In-flight request tracking for response time measurement
        self._in_flight: dict[tuple[int, str], dict] = {}  # (request_id, backend_name) -> info
        self._next_request_id = 0

        # Statistics
        self._requests_received = 0
        self._requests_forwarded = 0
        self._requests_failed = 0
        self._no_backend_available = 0
        self._backends_marked_unhealthy = 0
        self._backends_marked_healthy = 0

        logger.debug(
            "[%s] LoadBalancer initialized: backends=%d, strategy=%s",
            name,
            len(self._backends),
            type(self._strategy).__name__,
        )

    @property
    def stats(self) -> LoadBalancerStats:
        """Return a frozen snapshot of load balancer statistics."""
        return LoadBalancerStats(
            requests_received=self._requests_received,
            requests_forwarded=self._requests_forwarded,
            requests_failed=self._requests_failed,
            no_backend_available=self._no_backend_available,
            backends_marked_unhealthy=self._backends_marked_unhealthy,
            backends_marked_healthy=self._backends_marked_healthy,
        )

    @property
    def strategy(self) -> LoadBalancingStrategy:
        """The load balancing strategy."""
        return self._strategy

    @property
    def all_backends(self) -> list[Entity]:
        """All registered backends (healthy and unhealthy)."""
        return [info.backend for info in self._backends.values()]

    @property
    def healthy_backends(self) -> list[Entity]:
        """Only healthy backends available for routing."""
        return [
            info.backend
            for info in self._backends.values()
            if info.is_healthy
        ]

    @property
    def unhealthy_backends(self) -> list[Entity]:
        """Backends currently marked as unhealthy."""
        return [
            info.backend
            for info in self._backends.values()
            if not info.is_healthy
        ]

    @property
    def backend_count(self) -> int:
        """Total number of registered backends."""
        return len(self._backends)

    @property
    def healthy_count(self) -> int:
        """Number of healthy backends."""
        return len(self.healthy_backends)

    def get_backend_info_by_name(self, name: str) -> "BackendInfo | None":
        """Get backend info by backend name."""
        return self._backends.get(name)

    def add_backend(self, backend: Entity, weight: int = 1) -> None:
        """Add a backend to the load balancer.

        Args:
            backend: The backend entity to add.
            weight: Weight for weighted strategies (default 1).

        Raises:
            ValueError: If weight is less than 1.
        """
        if weight < 1:
            raise ValueError(f"weight must be >= 1, got {weight}")

        if backend.name in self._backends:
            logger.warning(
                "[%s] Backend %s already registered, updating weight",
                self.name,
                backend.name,
            )
            self._backends[backend.name].weight = weight
            return

        self._backends[backend.name] = BackendInfo(
            backend=backend,
            weight=weight,
        )

        # Update strategy weights if applicable
        if hasattr(self._strategy, 'set_weight'):
            self._strategy.set_weight(backend, weight)

        # Add to consistent hash ring if applicable
        if hasattr(self._strategy, 'add_backend'):
            self._strategy.add_backend(backend)

        logger.debug(
            "[%s] Added backend: %s (weight=%d)",
            self.name,
            backend.name,
            weight,
        )

    def remove_backend(self, backend: Entity) -> None:
        """Remove a backend from the load balancer.

        Args:
            backend: The backend entity to remove.
        """
        if backend.name not in self._backends:
            logger.warning(
                "[%s] Attempted to remove unknown backend: %s",
                self.name,
                backend.name,
            )
            return

        del self._backends[backend.name]

        # Remove from consistent hash ring if applicable
        if hasattr(self._strategy, 'remove_backend'):
            self._strategy.remove_backend(backend)

        logger.debug(
            "[%s] Removed backend: %s",
            self.name,
            backend.name,
        )

    def mark_unhealthy(self, backend: Entity) -> None:
        """Mark a backend as unhealthy.

        Unhealthy backends are excluded from the routing pool until
        marked healthy again.

        Args:
            backend: The backend to mark unhealthy.
        """
        if backend.name not in self._backends:
            logger.warning(
                "[%s] Cannot mark unknown backend unhealthy: %s",
                self.name,
                backend.name,
            )
            return

        info = self._backends[backend.name]
        if info.is_healthy:
            info.is_healthy = False
            info.consecutive_successes = 0
            self._backends_marked_unhealthy += 1
            logger.info(
                "[%s] Marked backend unhealthy: %s",
                self.name,
                backend.name,
            )

    def mark_healthy(self, backend: Entity) -> None:
        """Mark a backend as healthy.

        Healthy backends are included in the routing pool.

        Args:
            backend: The backend to mark healthy.
        """
        if backend.name not in self._backends:
            logger.warning(
                "[%s] Cannot mark unknown backend healthy: %s",
                self.name,
                backend.name,
            )
            return

        info = self._backends[backend.name]
        if not info.is_healthy:
            info.is_healthy = True
            info.consecutive_failures = 0
            self._backends_marked_healthy += 1
            logger.info(
                "[%s] Marked backend healthy: %s",
                self.name,
                backend.name,
            )

    def get_backend_info(self, backend: Entity) -> BackendInfo | None:
        """Get tracking information for a backend."""
        return self._backends.get(backend.name)

    def record_success(self, backend: Entity) -> None:
        """Record a successful request to a backend.

        Args:
            backend: The backend that handled the request.
        """
        if backend.name in self._backends:
            info = self._backends[backend.name]
            info.consecutive_successes += 1
            info.consecutive_failures = 0

    def record_failure(self, backend: Entity) -> None:
        """Record a failed request to a backend.

        Args:
            backend: The backend that failed.
        """
        if backend.name in self._backends:
            info = self._backends[backend.name]
            info.consecutive_failures += 1
            info.consecutive_successes = 0
            info.total_failures += 1

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle incoming events.

        Routes requests to backends and handles responses.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule.
        """
        event_type = event.event_type

        if event_type == "_lb_response":
            return self._handle_response(event)

        # Forward request to a backend
        return self._forward_request(event)

    def _forward_request(self, event: Event) -> list[Event] | None:
        """Forward a request to a selected backend."""
        self._requests_received += 1

        # Select a backend
        backends = self.healthy_backends
        if not backends:
            self._no_backend_available += 1
            logger.warning(
                "[%s] No healthy backends available",
                self.name,
            )

            if self._on_no_backend == "reject":
                self._requests_failed += 1
                # Could invoke failure callback if present
                return None

            # Queue mode - would need to implement queuing
            return None

        backend = self._strategy.select(backends, event)
        if backend is None:
            self._no_backend_available += 1
            self._requests_failed += 1
            return None

        # Track the request
        self._next_request_id += 1
        request_id = self._next_request_id
        flight_key = (request_id, backend.name)

        self._in_flight[flight_key] = {
            "start_time": self.now,
            "backend": backend,
            "original_event": event,
        }

        # Update backend stats
        if backend.name in self._backends:
            self._backends[backend.name].total_requests += 1

        self._requests_forwarded += 1

        logger.debug(
            "[%s] Forwarding request %d to %s",
            self.name,
            request_id,
            backend.name,
        )

        # Create forwarded event
        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=backend,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_lb_request_id": request_id,
                    "_lb_name": self.name,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_lb_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "backend_name": backend.name,
                    },
                },
            )

        forwarded.add_completion_hook(on_complete)

        # Copy completion hooks from original event
        for hook in event.on_complete:
            forwarded.add_completion_hook(hook)

        return [forwarded]

    def _handle_response(self, event: Event) -> None:
        """Handle a response from a backend."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        backend_name = metadata.get("backend_name")

        flight_key = (request_id, backend_name)
        if flight_key not in self._in_flight:
            logger.debug(
                "[%s] Response for unknown request: id=%s, backend=%s",
                self.name,
                request_id,
                backend_name,
            )
            return

        request_info = self._in_flight.pop(flight_key)
        start_time = request_info["start_time"]
        backend = request_info["backend"]

        # Calculate response time
        response_time = (self.now - start_time).to_seconds()

        # Record success
        self.record_success(backend)

        # Update response time tracking if strategy supports it
        if isinstance(self._strategy, LeastResponseTime):
            self._strategy.record_response_time(backend, response_time)

        logger.debug(
            "[%s] Response received: request=%d, backend=%s, time=%.4fs",
            self.name,
            request_id,
            backend_name,
            response_time,
        )

        return None
