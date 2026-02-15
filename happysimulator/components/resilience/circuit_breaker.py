"""Circuit breaker pattern implementation.

Provides protection against cascading failures by failing fast when
a downstream service is unhealthy, and automatically recovering when
the service becomes healthy again.

Example:
    from happysimulator.components.resilience import CircuitBreaker

    cb = CircuitBreaker(
        name="api_breaker",
        target=backend_server,
        failure_threshold=5,
        success_threshold=2,
        timeout=30.0,
    )

    # Requests go through the circuit breaker
    request = Event(time=now, event_type="request", target=cb, ...)
    sim.schedule(request)
"""

import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing fast, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing recovery with limited requests


@dataclass(frozen=True)
class CircuitBreakerStats:
    """Frozen snapshot of CircuitBreaker statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rejected_requests: int = 0
    state_changes: int = 0
    times_opened: int = 0
    times_closed: int = 0


class CircuitBreaker(Entity):
    """Implements the circuit breaker pattern.

    The circuit breaker monitors requests to a target service and
    tracks failures. When failures exceed a threshold, the circuit
    opens and subsequent requests fail fast without calling the target.
    After a timeout, the circuit enters half-open state to test if
    the target has recovered.

    States:
        CLOSED: Normal operation. Requests forwarded to target.
                Failures tracked. Opens after failure_threshold failures.
        OPEN: Failing fast. Requests rejected immediately.
              Transitions to HALF_OPEN after timeout expires.
        HALF_OPEN: Testing recovery. Limited requests allowed through.
                   success_threshold successes -> CLOSED
                   Any failure -> OPEN

    Attributes:
        name: Circuit breaker identifier.
        target: The service being protected.
        state: Current circuit state.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 30.0,
        half_open_max_requests: int = 1,
        failure_predicate: Callable[[Event], bool] | None = None,
        on_state_change: Callable[[CircuitState, CircuitState], None] | None = None,
    ):
        """Initialize the circuit breaker.

        Args:
            name: Circuit breaker identifier.
            target: The downstream entity to protect.
            failure_threshold: Consecutive failures before opening circuit.
            success_threshold: Consecutive successes in half-open to close.
            timeout: Seconds in open state before transitioning to half-open.
            half_open_max_requests: Max concurrent requests in half-open state.
            failure_predicate: Optional function to determine if response is failure.
                               If None, only exceptions count as failures.
            on_state_change: Optional callback when state changes.

        Raises:
            ValueError: If thresholds or timeout are invalid.
        """
        super().__init__(name)

        if failure_threshold < 1:
            raise ValueError(f"failure_threshold must be >= 1, got {failure_threshold}")
        if success_threshold < 1:
            raise ValueError(f"success_threshold must be >= 1, got {success_threshold}")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        if half_open_max_requests < 1:
            raise ValueError(f"half_open_max_requests must be >= 1, got {half_open_max_requests}")

        self._target = target
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout = timeout
        self._half_open_max_requests = half_open_max_requests
        self._failure_predicate = failure_predicate
        self._on_state_change = on_state_change

        # State tracking
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Instant | None = None
        self._half_open_in_flight = 0

        # Request tracking for response correlation
        self._in_flight: dict[int, dict] = {}
        self._next_request_id = 0

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._rejected_requests = 0
        self._state_changes = 0
        self._times_opened = 0
        self._times_closed = 0

        logger.debug(
            "[%s] CircuitBreaker initialized: target=%s, failure_threshold=%d, timeout=%.1fs",
            name,
            target.name,
            failure_threshold,
            timeout,
        )

    @property
    def target(self) -> Entity:
        """The protected target entity."""
        return self._target

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            self._transition_to(CircuitState.HALF_OPEN)
        return self._state

    @property
    def failure_threshold(self) -> int:
        """Number of failures before opening."""
        return self._failure_threshold

    @property
    def success_threshold(self) -> int:
        """Number of successes to close from half-open."""
        return self._success_threshold

    @property
    def timeout(self) -> float:
        """Seconds before transitioning from open to half-open."""
        return self._timeout

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        return self._failure_count

    @property
    def success_count(self) -> int:
        """Current consecutive success count (in half-open)."""
        return self._success_count

    @property
    def stats(self) -> CircuitBreakerStats:
        """Frozen snapshot of circuit breaker statistics."""
        return CircuitBreakerStats(
            total_requests=self._total_requests,
            successful_requests=self._successful_requests,
            failed_requests=self._failed_requests,
            rejected_requests=self._rejected_requests,
            state_changes=self._state_changes,
            times_opened=self._times_opened,
            times_closed=self._times_closed,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to target."""
        super().set_clock(clock)
        self._target.set_clock(clock)

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        # Can't check time without a clock or recorded failure time
        if self._clock is None or self._last_failure_time is None:
            return False
        elapsed = (self.now - self._last_failure_time).to_seconds()
        return elapsed >= self._timeout

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        self._state_changes += 1

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._times_closed += 1
        elif new_state == CircuitState.OPEN:
            self._success_count = 0
            # Only record failure time if clock is available
            if self._clock is not None:
                self._last_failure_time = self.now
            self._times_opened += 1
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._half_open_in_flight = 0

        logger.info(
            "[%s] State transition: %s -> %s",
            self.name,
            old_state.value,
            new_state.value,
        )

        if self._on_state_change:
            self._on_state_change(old_state, new_state)

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle incoming events.

        Routes requests through the circuit breaker logic and handles
        responses from the target.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule, or None if rejected.
        """
        event_type = event.event_type

        if event_type == "_cb_response":
            return self._handle_response(event)

        # Check current state (this also handles OPEN -> HALF_OPEN transition)
        current_state = self.state

        self._total_requests += 1

        if current_state == CircuitState.OPEN:
            # Fail fast
            self._rejected_requests += 1
            logger.debug("[%s] Request rejected (circuit OPEN)", self.name)
            return None

        if current_state == CircuitState.HALF_OPEN:
            # Only allow limited requests through
            if self._half_open_in_flight >= self._half_open_max_requests:
                self._rejected_requests += 1
                logger.debug("[%s] Request rejected (half-open limit reached)", self.name)
                return None
            self._half_open_in_flight += 1

        # Forward request to target
        return self._forward_request(event)

    def _forward_request(self, event: Event) -> list[Event]:
        """Forward a request to the target."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._in_flight[request_id] = {
            "start_time": self.now,
            "original_event": event,
            "state_when_sent": self._state,
        }

        logger.debug(
            "[%s] Forwarding request %d to %s (state=%s)",
            self.name,
            request_id,
            self._target.name,
            self._state.value,
        )

        # Create forwarded event
        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=self._target,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_cb_request_id": request_id,
                    "_cb_name": self.name,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_cb_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "success": True,
                    },
                },
            )

        forwarded.add_completion_hook(on_complete)

        # Copy completion hooks from original event
        for hook in event.on_complete:
            forwarded.add_completion_hook(hook)

        return [forwarded]

    def _handle_response(self, event: Event) -> None:
        """Handle a response from the target."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        success = metadata.get("success", True)

        if request_id not in self._in_flight:
            logger.debug(
                "[%s] Response for unknown request: id=%s",
                self.name,
                request_id,
            )
            return

        request_info = self._in_flight.pop(request_id)
        state_when_sent = request_info["state_when_sent"]

        # Check if response indicates failure
        if self._failure_predicate:
            original_event = request_info["original_event"]
            success = not self._failure_predicate(original_event)

        if success:
            self._on_success(state_when_sent)
        else:
            self._on_failure(state_when_sent)

    def _on_success(self, state_when_sent: CircuitState) -> None:
        """Record a successful request."""
        self._successful_requests += 1

        if state_when_sent == CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            self._success_count += 1

            if self._success_count >= self._success_threshold:
                self._transition_to(CircuitState.CLOSED)
        elif state_when_sent == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _on_failure(self, state_when_sent: CircuitState) -> None:
        """Record a failed request."""
        self._failed_requests += 1

        if state_when_sent == CircuitState.HALF_OPEN:
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            # Any failure in half-open opens the circuit
            self._transition_to(CircuitState.OPEN)
        elif state_when_sent == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._transition_to(CircuitState.OPEN)

    def record_success(self) -> None:
        """Manually record a success (for external failure detection)."""
        self._on_success(self._state)

    def record_failure(self) -> None:
        """Manually record a failure (for external failure detection)."""
        self._on_failure(self._state)

    def force_open(self) -> None:
        """Force the circuit to open."""
        self._transition_to(CircuitState.OPEN)

    def force_close(self) -> None:
        """Force the circuit to close."""
        self._transition_to(CircuitState.CLOSED)

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_in_flight = 0
        self._in_flight.clear()
