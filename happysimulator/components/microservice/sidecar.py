"""Service mesh sidecar proxy with integrated resilience patterns.

Combines rate limiting, circuit breaking, timeout, and retry logic into
a single Entity that proxies requests to a target service. This avoids
the complexity of chaining separate resilience entities.

Example:
    from happysimulator.components.microservice import Sidecar
    from happysimulator.components.rate_limiter import TokenBucketPolicy

    sidecar = Sidecar(
        name="svc_proxy",
        target=backend,
        rate_limit_policy=TokenBucketPolicy(capacity=100, refill_rate=10),
        circuit_failure_threshold=5,
        request_timeout=5.0,
        max_retries=3,
    )
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum

from happysimulator.components.rate_limiter.policy import RateLimiterPolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


class _CircuitState(Enum):
    """Internal circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(frozen=True)
class SidecarStats:
    """Statistics tracked by Sidecar."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    retries: int = 0
    rate_limited: int = 0
    circuit_broken: int = 0
    timed_out: int = 0


class Sidecar(Entity):
    """Service mesh sidecar proxy with integrated resilience.

    Inlines rate limiting, circuit breaking, timeout, and retry logic
    so that users only need to register a single entity. Requests flow
    through each layer in order:

    1. Rate limit check (if policy provided)
    2. Circuit breaker check
    3. Forward to target with timeout
    4. On failure/timeout: retry with exponential backoff

    Attributes:
        name: Sidecar identifier.
        stats: Accumulated statistics.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        rate_limit_policy: RateLimiterPolicy | None = None,
        rate_limit_queue_capacity: int = 1000,
        circuit_failure_threshold: int = 5,
        circuit_success_threshold: int = 2,
        circuit_timeout: float = 30.0,
        request_timeout: float = 5.0,
        max_retries: int = 3,
        retry_base_delay: float = 0.1,
    ):
        """Initialize the sidecar proxy.

        Args:
            name: Sidecar identifier.
            target: Downstream service to proxy requests to.
            rate_limit_policy: Optional rate limiter policy. None disables rate limiting.
            rate_limit_queue_capacity: Max queued events when rate limited.
            circuit_failure_threshold: Consecutive failures before opening circuit.
            circuit_success_threshold: Consecutive successes in half-open to close.
            circuit_timeout: Seconds before transitioning open to half-open.
            request_timeout: Per-request timeout in seconds.
            max_retries: Maximum retry attempts (0 = no retries).
            retry_base_delay: Base delay for exponential backoff.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if circuit_failure_threshold < 1:
            raise ValueError(
                f"circuit_failure_threshold must be >= 1, got {circuit_failure_threshold}"
            )
        if circuit_success_threshold < 1:
            raise ValueError(
                f"circuit_success_threshold must be >= 1, got {circuit_success_threshold}"
            )
        if circuit_timeout <= 0:
            raise ValueError(f"circuit_timeout must be > 0, got {circuit_timeout}")
        if request_timeout <= 0:
            raise ValueError(f"request_timeout must be > 0, got {request_timeout}")
        if max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {max_retries}")
        if retry_base_delay < 0:
            raise ValueError(f"retry_base_delay must be >= 0, got {retry_base_delay}")

        self._target = target
        self._rate_limit_policy = rate_limit_policy
        self._rate_limit_queue_capacity = rate_limit_queue_capacity
        self._circuit_failure_threshold = circuit_failure_threshold
        self._circuit_success_threshold = circuit_success_threshold
        self._circuit_timeout = circuit_timeout
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

        # Circuit breaker state
        self._circuit_state = _CircuitState.CLOSED
        self._circuit_failure_count = 0
        self._circuit_success_count = 0
        self._last_failure_time: Instant | None = None

        # Request tracking
        self._in_flight: dict[int, dict] = {}
        self._next_request_id = 0

        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._retries = 0
        self._rate_limited = 0
        self._circuit_broken = 0
        self._timed_out = 0

        logger.debug(
            "[%s] Sidecar initialized: target=%s, timeout=%.1fs, max_retries=%d",
            name,
            target.name,
            request_timeout,
            max_retries,
        )

    @property
    def stats(self) -> SidecarStats:
        """Return a frozen snapshot of current statistics."""
        return SidecarStats(
            total_requests=self._total_requests,
            successful_requests=self._successful_requests,
            failed_requests=self._failed_requests,
            retries=self._retries,
            rate_limited=self._rate_limited,
            circuit_broken=self._circuit_broken,
            timed_out=self._timed_out,
        )

    @property
    def target(self) -> Entity:
        """The proxied target entity."""
        return self._target

    @property
    def circuit_state(self) -> str:
        """Current circuit breaker state as a string."""
        self._check_circuit_timeout()
        return self._circuit_state.value

    def _check_circuit_timeout(self) -> None:
        """Transition from OPEN to HALF_OPEN if timeout has elapsed."""
        if self._circuit_state != _CircuitState.OPEN:
            return
        if self._clock is None or self._last_failure_time is None:
            return
        elapsed = (self.now - self._last_failure_time).to_seconds()
        if elapsed >= self._circuit_timeout:
            self._circuit_state = _CircuitState.HALF_OPEN
            self._circuit_success_count = 0
            logger.info("[%s] Circuit: OPEN -> HALF_OPEN", self.name)

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        """Route incoming events through the sidecar pipeline.

        Args:
            event: The incoming event.

        Returns:
            Events to schedule, generator for async processing, or None.
        """
        if event.event_type == "_sc_response":
            return self._handle_response(event)

        if event.event_type == "_sc_timeout":
            return self._handle_timeout(event)

        return self._handle_request(event)

    def _handle_request(self, event: Event) -> list[Event] | None:
        """Apply rate limiting and circuit breaking, then forward."""
        self._total_requests += 1

        # Check if this is a retry (carries attempt number in metadata)
        attempt = event.context.get("metadata", {}).get("_sc_retry_attempt", 0)

        # 1. Rate limit check
        if self._rate_limit_policy is not None and not self._rate_limit_policy.try_acquire(
            self.now
        ):
            self._rate_limited += 1
            logger.debug("[%s] Request rate limited", self.name)
            return None

        # 2. Circuit breaker check
        self._check_circuit_timeout()
        if self._circuit_state == _CircuitState.OPEN:
            self._circuit_broken += 1
            logger.debug("[%s] Request circuit broken", self.name)
            return None

        # 3. Forward with timeout
        return self._forward_request(event, attempt=attempt)

    def _forward_request(self, event: Event, *, attempt: int) -> list[Event]:
        """Forward to target with timeout tracking."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._in_flight[request_id] = {
            "start_time": self.now,
            "original_event": event,
            "attempt": attempt,
            "completed": False,
        }

        # Create forwarded event
        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=self._target,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_sc_request_id": request_id,
                    "_sc_name": self.name,
                },
            },
        )

        # Completion hook
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_sc_response",
                target=self,
                context={"metadata": {"request_id": request_id, "success": True}},
            )

        forwarded.add_completion_hook(on_complete)

        # Copy original completion hooks (only on first attempt)
        if attempt == 0:
            for hook in event.on_complete:
                forwarded.add_completion_hook(hook)

        # Schedule timeout
        timeout_event = Event(
            time=self.now + Duration.from_seconds(self._request_timeout),
            event_type="_sc_timeout",
            target=self,
            context={"metadata": {"request_id": request_id}},
        )

        return [forwarded, timeout_event]

    def _handle_response(self, event: Event) -> list[Event] | None:
        """Handle response from target — record success."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            return None

        request_info = self._in_flight[request_id]
        if request_info["completed"]:
            return None

        request_info["completed"] = True
        del self._in_flight[request_id]

        self._successful_requests += 1
        self._record_circuit_success()

        return None

    def _handle_timeout(self, event: Event) -> list[Event] | None:
        """Handle timeout — retry or record failure."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            return None

        request_info = self._in_flight[request_id]
        if request_info["completed"]:
            return None

        request_info["completed"] = True
        del self._in_flight[request_id]

        self._timed_out += 1
        attempt = request_info["attempt"]

        # Retry if attempts remaining
        if attempt < self._max_retries:
            self._retries += 1
            delay = self._retry_base_delay * (2**attempt)

            # Schedule retry after backoff delay
            retry_event = Event(
                time=self.now + Duration.from_seconds(delay),
                event_type=request_info["original_event"].event_type,
                target=self,
                context=request_info["original_event"].context.copy(),
            )
            # Mark as retry so _handle_request processes it
            retry_event.context.setdefault("metadata", {})["_sc_retry_attempt"] = attempt + 1
            return [retry_event]

        # All retries exhausted
        self._failed_requests += 1
        self._record_circuit_failure()
        return None

    def _record_circuit_success(self) -> None:
        """Record success for circuit breaker logic."""
        if self._circuit_state == _CircuitState.HALF_OPEN:
            self._circuit_success_count += 1
            if self._circuit_success_count >= self._circuit_success_threshold:
                self._circuit_state = _CircuitState.CLOSED
                self._circuit_failure_count = 0
                logger.info("[%s] Circuit: HALF_OPEN -> CLOSED", self.name)
        elif self._circuit_state == _CircuitState.CLOSED:
            self._circuit_failure_count = 0

    def _record_circuit_failure(self) -> None:
        """Record failure for circuit breaker logic."""
        if self._circuit_state == _CircuitState.HALF_OPEN:
            self._circuit_state = _CircuitState.OPEN
            self._last_failure_time = self.now
            logger.info("[%s] Circuit: HALF_OPEN -> OPEN", self.name)
        elif self._circuit_state == _CircuitState.CLOSED:
            self._circuit_failure_count += 1
            if self._circuit_failure_count >= self._circuit_failure_threshold:
                self._circuit_state = _CircuitState.OPEN
                self._last_failure_time = self.now
                logger.info("[%s] Circuit: CLOSED -> OPEN", self.name)
