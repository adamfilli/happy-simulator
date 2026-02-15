"""Health checking for load balancer backends.

Provides periodic health checks to detect and remove unhealthy backends
from the load balancer pool, and restore them when they recover.

Example:
    from happysimulator.components.load_balancer import (
        LoadBalancer, HealthChecker, RoundRobin
    )

    lb = LoadBalancer(name="lb", backends=servers, strategy=RoundRobin())

    health_checker = HealthChecker(
        name="health_check",
        load_balancer=lb,
        interval=5.0,        # Check every 5 seconds
        timeout=1.0,         # 1 second timeout per check
        healthy_threshold=2,  # 2 successes to mark healthy
        unhealthy_threshold=3,  # 3 failures to mark unhealthy
    )

    # Start health checking
    sim.schedule(health_checker.start())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.components.load_balancer.load_balancer import LoadBalancer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HealthCheckStats:
    """Statistics tracked by HealthChecker."""

    checks_performed: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    checks_timed_out: int = 0
    backends_marked_healthy: int = 0
    backends_marked_unhealthy: int = 0


@dataclass
class BackendHealthState:
    """Health state tracking for a backend."""

    consecutive_successes: int = 0
    consecutive_failures: int = 0
    last_check_time: Instant | None = None
    last_check_passed: bool | None = None
    is_checking: bool = False


class HealthChecker(Entity):
    """Periodically checks backend health.

    Sends health check probes to backends and tracks consecutive
    successes/failures to determine health status. Updates the
    load balancer when backends become healthy or unhealthy.

    Attributes:
        name: Health checker identifier for logging.
        load_balancer: The load balancer to update.
        interval: Time between health checks in seconds.
        timeout: Maximum time to wait for health check response.
        healthy_threshold: Consecutive successes to mark healthy.
        unhealthy_threshold: Consecutive failures to mark unhealthy.
    """

    def __init__(
        self,
        name: str,
        load_balancer: LoadBalancer,
        interval: float = 10.0,
        timeout: float = 5.0,
        healthy_threshold: int = 2,
        unhealthy_threshold: int = 3,
        check_event_type: str = "health_check",
    ):
        """Initialize the health checker.

        Args:
            name: Health checker identifier.
            load_balancer: Load balancer to manage.
            interval: Seconds between checks (default 10).
            timeout: Seconds before check times out (default 5).
            healthy_threshold: Successes to mark healthy (default 2).
            unhealthy_threshold: Failures to mark unhealthy (default 3).
            check_event_type: Event type for health check probes.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if timeout >= interval:
            raise ValueError(f"timeout ({timeout}) must be < interval ({interval})")
        if healthy_threshold < 1:
            raise ValueError(f"healthy_threshold must be >= 1, got {healthy_threshold}")
        if unhealthy_threshold < 1:
            raise ValueError(f"unhealthy_threshold must be >= 1, got {unhealthy_threshold}")

        self._load_balancer = load_balancer
        self._interval = interval
        self._timeout = timeout
        self._healthy_threshold = healthy_threshold
        self._unhealthy_threshold = unhealthy_threshold
        self._check_event_type = check_event_type

        # Track health state per backend
        self._backend_states: dict[str, BackendHealthState] = {}

        # Track in-flight health checks
        self._pending_checks: dict[str, int] = {}  # backend_name -> check_id
        self._next_check_id = 0

        # Statistics
        self._checks_performed = 0
        self._checks_passed = 0
        self._checks_failed = 0
        self._checks_timed_out = 0
        self._backends_marked_healthy = 0
        self._backends_marked_unhealthy = 0

        # Running state
        self._is_running = False

        logger.debug(
            "[%s] HealthChecker initialized: lb=%s, interval=%.1fs, "
            "timeout=%.1fs, healthy=%d, unhealthy=%d",
            name,
            load_balancer.name,
            interval,
            timeout,
            healthy_threshold,
            unhealthy_threshold,
        )

    @property
    def stats(self) -> HealthCheckStats:
        """Return a frozen snapshot of health check statistics."""
        return HealthCheckStats(
            checks_performed=self._checks_performed,
            checks_passed=self._checks_passed,
            checks_failed=self._checks_failed,
            checks_timed_out=self._checks_timed_out,
            backends_marked_healthy=self._backends_marked_healthy,
            backends_marked_unhealthy=self._backends_marked_unhealthy,
        )

    @property
    def load_balancer(self) -> LoadBalancer:
        """The load balancer being monitored."""
        return self._load_balancer

    @property
    def interval(self) -> float:
        """Seconds between health checks."""
        return self._interval

    @property
    def timeout(self) -> float:
        """Seconds before a check times out."""
        return self._timeout

    @property
    def healthy_threshold(self) -> int:
        """Consecutive successes to mark healthy."""
        return self._healthy_threshold

    @property
    def unhealthy_threshold(self) -> int:
        """Consecutive failures to mark unhealthy."""
        return self._unhealthy_threshold

    @property
    def is_running(self) -> bool:
        """Whether health checking is active."""
        return self._is_running

    def start(self) -> Event:
        """Start periodic health checking.

        Returns an event that begins the health check cycle.
        Schedule this event to start health checking.

        Returns:
            Event to schedule.
        """
        self._is_running = True
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_health_check_cycle",
            target=self,
            context={},
        )

    def stop(self) -> None:
        """Stop periodic health checking."""
        self._is_running = False
        logger.info("[%s] Health checking stopped", self.name)

    def get_backend_state(self, backend: Entity) -> BackendHealthState:
        """Get the health state for a backend."""
        if backend.name not in self._backend_states:
            self._backend_states[backend.name] = BackendHealthState()
        return self._backend_states[backend.name]

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle health checker events.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule.
        """
        event_type = event.event_type

        if event_type == "_health_check_cycle":
            return self._run_check_cycle(event)

        if event_type == "_health_check_timeout":
            return self._handle_timeout(event)

        if event_type == "_health_check_response":
            return self._handle_response(event)

        return None

    def _run_check_cycle(self, event: Event) -> list[Event]:
        """Run a health check cycle for all backends."""
        if not self._is_running:
            return []

        result_events = []

        # Check all backends
        for backend in self._load_balancer.all_backends:
            check_events = self._check_backend(backend)
            result_events.extend(check_events)

        # Schedule next cycle
        next_cycle = Event(
            time=self.now + Duration.from_seconds(self._interval),
            event_type="_health_check_cycle",
            target=self,
            context={},
        )
        result_events.append(next_cycle)

        logger.debug(
            "[%s] Health check cycle: checking %d backends",
            self.name,
            len(self._load_balancer.all_backends),
        )

        return result_events

    def _check_backend(self, backend: Entity) -> list[Event]:
        """Send a health check probe to a backend."""
        state = self.get_backend_state(backend)

        # Skip if already checking
        if state.is_checking:
            return []

        state.is_checking = True
        state.last_check_time = self.now

        self._next_check_id += 1
        check_id = self._next_check_id
        self._pending_checks[backend.name] = check_id

        self._checks_performed += 1

        logger.debug(
            "[%s] Sending health check to %s (id=%d)",
            self.name,
            backend.name,
            check_id,
        )

        # Create health check probe
        probe = Event(
            time=self.now,
            event_type=self._check_event_type,
            target=backend,
            context={
                "metadata": {
                    "_health_check_id": check_id,
                    "_health_checker": self.name,
                    "_backend_name": backend.name,
                },
            },
        )

        # Add completion hook for response
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_health_check_response",
                target=self,
                context={
                    "metadata": {
                        "check_id": check_id,
                        "backend_name": backend.name,
                    },
                },
            )

        probe.add_completion_hook(on_complete)

        # Schedule timeout
        timeout_event = Event(
            time=self.now + Duration.from_seconds(self._timeout),
            event_type="_health_check_timeout",
            target=self,
            context={
                "metadata": {
                    "check_id": check_id,
                    "backend_name": backend.name,
                },
            },
        )

        return [probe, timeout_event]

    def _handle_response(self, event: Event) -> None:
        """Handle a health check response (success)."""
        metadata = event.context.get("metadata", {})
        check_id = metadata.get("check_id")
        backend_name = metadata.get("backend_name")

        # Check if this is still the pending check
        if self._pending_checks.get(backend_name) != check_id:
            # Stale response (already timed out or newer check in progress)
            return

        del self._pending_checks[backend_name]

        state = self.get_backend_state_by_name(backend_name)
        if state is None:
            return

        state.is_checking = False
        state.last_check_passed = True
        state.consecutive_successes += 1
        state.consecutive_failures = 0

        self._checks_passed += 1

        logger.debug(
            "[%s] Health check passed: %s (consecutive=%d)",
            self.name,
            backend_name,
            state.consecutive_successes,
        )

        # Check if backend should be marked healthy
        backend_info = self._load_balancer.get_backend_info_by_name(backend_name)
        if (
            backend_info
            and not backend_info.is_healthy
            and state.consecutive_successes >= self._healthy_threshold
        ):
            backend = backend_info.backend
            self._load_balancer.mark_healthy(backend)
            self._backends_marked_healthy += 1
            logger.info(
                "[%s] Backend marked healthy: %s",
                self.name,
                backend_name,
            )

    def _handle_timeout(self, event: Event) -> None:
        """Handle a health check timeout (failure)."""
        metadata = event.context.get("metadata", {})
        check_id = metadata.get("check_id")
        backend_name = metadata.get("backend_name")

        # Check if this is still the pending check
        if self._pending_checks.get(backend_name) != check_id:
            # Already responded or newer check in progress
            return

        del self._pending_checks[backend_name]

        state = self.get_backend_state_by_name(backend_name)
        if state is None:
            return

        state.is_checking = False
        state.last_check_passed = False
        state.consecutive_failures += 1
        state.consecutive_successes = 0

        self._checks_failed += 1
        self._checks_timed_out += 1

        logger.debug(
            "[%s] Health check timeout: %s (consecutive=%d)",
            self.name,
            backend_name,
            state.consecutive_failures,
        )

        # Check if backend should be marked unhealthy
        backend_info = self._load_balancer.get_backend_info_by_name(backend_name)
        if (
            backend_info
            and backend_info.is_healthy
            and state.consecutive_failures >= self._unhealthy_threshold
        ):
            backend = backend_info.backend
            self._load_balancer.mark_unhealthy(backend)
            self._backends_marked_unhealthy += 1
            logger.info(
                "[%s] Backend marked unhealthy: %s",
                self.name,
                backend_name,
            )

    def get_backend_state_by_name(self, name: str) -> BackendHealthState | None:
        """Get health state by backend name."""
        return self._backend_states.get(name)
