"""Adaptive rate limiter entity.

Implements AIMD (Additive Increase, Multiplicative Decrease) rate limiting.
The rate limit adjusts dynamically based on downstream health signals:
- On success: gradually increase the limit
- On failure/timeout: rapidly decrease the limit

This creates a self-tuning rate limiter that discovers the sustainable
throughput of downstream services without manual configuration.

Example:
    from happysimulator.components.rate_limiter import AdaptiveRateLimiter

    limiter = AdaptiveRateLimiter(
        name="adaptive_limiter",
        downstream=server,
        initial_rate=100.0,
        min_rate=10.0,
        max_rate=1000.0,
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant, Duration

logger = logging.getLogger(__name__)


class RateAdjustmentReason(Enum):
    """Reason for rate adjustment."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    THROTTLED = "throttled"


@dataclass
class AdaptiveRateLimiterStats:
    """Statistics tracked by AdaptiveRateLimiter."""

    requests_received: int = 0
    requests_forwarded: int = 0
    requests_dropped: int = 0
    successes: int = 0
    failures: int = 0
    timeouts: int = 0
    rate_increases: int = 0
    rate_decreases: int = 0


@dataclass
class RateSnapshot:
    """A snapshot of the rate at a point in time."""

    time: Instant
    rate: float
    reason: RateAdjustmentReason | None = None


class AdaptiveRateLimiter(Entity):
    """An adaptive rate limiter using AIMD algorithm.

    AIMD (Additive Increase, Multiplicative Decrease):
    - On success: rate = min(rate + increase_step, max_rate)
    - On failure: rate = max(rate * decrease_factor, min_rate)

    The limiter uses a token bucket internally with the adaptive rate
    as the refill rate. This allows controlled bursting while respecting
    the current rate limit.

    Feedback is provided via the `record_success()` and `record_failure()`
    methods, which should be called by the downstream handler or a wrapper.

    Attributes:
        current_rate: Current rate limit (requests per second).
        min_rate: Minimum allowed rate.
        max_rate: Maximum allowed rate.
        downstream: The entity to forward accepted requests to.
        stats: Statistics including rate adjustments.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        initial_rate: float = 100.0,
        min_rate: float = 1.0,
        max_rate: float = 10000.0,
        increase_step: float | None = None,
        decrease_factor: float = 0.5,
        window_size: float = 1.0,
        failure_predicate: Callable[[Event], bool] | None = None,
    ):
        """Initialize the adaptive rate limiter.

        Args:
            name: Entity name for identification.
            downstream: Entity to forward accepted requests to.
            initial_rate: Starting rate limit (requests per second).
            min_rate: Minimum rate limit.
            max_rate: Maximum rate limit.
            increase_step: Additive increase on success (default: initial_rate * 0.1).
            decrease_factor: Multiplicative factor on failure (0.5 = halve rate).
            window_size: Time window for rate limiting in seconds.
            failure_predicate: Optional function to detect failures from response events.

        Raises:
            ValueError: If rate parameters are invalid.
        """
        if min_rate <= 0:
            raise ValueError(f"min_rate must be > 0, got {min_rate}")
        if max_rate < min_rate:
            raise ValueError(f"max_rate must be >= min_rate, got {max_rate} < {min_rate}")
        if initial_rate < min_rate or initial_rate > max_rate:
            raise ValueError(
                f"initial_rate must be in [{min_rate}, {max_rate}], got {initial_rate}"
            )
        if decrease_factor <= 0 or decrease_factor >= 1:
            raise ValueError(f"decrease_factor must be in (0, 1), got {decrease_factor}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        super().__init__(name)

        self._downstream = downstream
        self._min_rate = min_rate
        self._max_rate = max_rate
        self._current_rate = initial_rate
        self._increase_step = increase_step if increase_step is not None else initial_rate * 0.1
        self._decrease_factor = decrease_factor
        self._window_size = window_size
        self._failure_predicate = failure_predicate

        # Token bucket state (using current_rate as refill rate)
        self._tokens = initial_rate * window_size  # Start with full bucket
        self._bucket_capacity = max_rate * window_size
        self._last_refill_time: Instant | None = None

        # Statistics
        self.stats = AdaptiveRateLimiterStats()

        # Rate history for visualization
        self.rate_history: list[RateSnapshot] = []

        # Time series for visualization
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []

    @property
    def downstream(self) -> Entity:
        """The entity receiving forwarded requests."""
        return self._downstream

    @property
    def current_rate(self) -> float:
        """Current rate limit (requests per second)."""
        return self._current_rate

    @property
    def min_rate(self) -> float:
        """Minimum allowed rate."""
        return self._min_rate

    @property
    def max_rate(self) -> float:
        """Maximum allowed rate."""
        return self._max_rate

    @property
    def tokens(self) -> float:
        """Current token count."""
        return self._tokens

    def _refill(self, now: Instant) -> None:
        """Refill tokens based on elapsed time and current rate."""
        if self._last_refill_time is None:
            self._last_refill_time = now
            return

        elapsed_seconds = (now - self._last_refill_time).to_seconds()
        if elapsed_seconds <= 0:
            return

        tokens_to_add = elapsed_seconds * self._current_rate
        max_tokens = self._current_rate * self._window_size
        self._tokens = min(max_tokens, self._tokens + tokens_to_add)
        self._last_refill_time = now

    def record_success(self, now: Instant | None = None) -> None:
        """Record a successful request, potentially increasing rate.

        Call this when a forwarded request completes successfully.

        Args:
            now: Current time (uses clock if not provided).
        """
        self.stats.successes += 1

        if now is None and self._clock:
            now = self._clock.now

        old_rate = self._current_rate
        self._current_rate = min(self._max_rate, self._current_rate + self._increase_step)

        if self._current_rate > old_rate:
            self.stats.rate_increases += 1
            if now:
                self.rate_history.append(
                    RateSnapshot(time=now, rate=self._current_rate, reason=RateAdjustmentReason.SUCCESS)
                )

            logger.debug(
                "[%s] Rate increased: %.2f -> %.2f (success)",
                self.name,
                old_rate,
                self._current_rate,
            )

    def record_failure(
        self, now: Instant | None = None, reason: RateAdjustmentReason = RateAdjustmentReason.FAILURE
    ) -> None:
        """Record a failed request, decreasing rate.

        Call this when a forwarded request fails or times out.

        Args:
            now: Current time (uses clock if not provided).
            reason: The reason for failure.
        """
        if reason == RateAdjustmentReason.TIMEOUT:
            self.stats.timeouts += 1
        else:
            self.stats.failures += 1

        if now is None and self._clock:
            now = self._clock.now

        old_rate = self._current_rate
        self._current_rate = max(self._min_rate, self._current_rate * self._decrease_factor)

        if self._current_rate < old_rate:
            self.stats.rate_decreases += 1
            if now:
                self.rate_history.append(
                    RateSnapshot(time=now, rate=self._current_rate, reason=reason)
                )

            logger.debug(
                "[%s] Rate decreased: %.2f -> %.2f (%s)",
                self.name,
                old_rate,
                self._current_rate,
                reason.value,
            )

    def handle_event(self, event: Event) -> list[Event]:
        """Handle an incoming request event.

        Uses token bucket with adaptive refill rate.

        Args:
            event: The incoming request event.

        Returns:
            List containing a forwarded event to downstream, or empty if dropped.
        """
        now = event.time

        # Record received
        self.stats.requests_received += 1
        self.received_times.append(now)

        # Refill tokens at current rate
        self._refill(now)

        if self._tokens >= 1.0:
            # Consume a token and forward
            self._tokens -= 1.0
            self.stats.requests_forwarded += 1
            self.forwarded_times.append(now)

            logger.debug(
                "[%.3f][%s] Forwarded request; rate=%.2f, tokens=%.2f",
                now.to_seconds(),
                self.name,
                self._current_rate,
                self._tokens,
            )

            # Create forwarding event to downstream entity
            forward_event = Event(
                time=now,
                event_type=f"forward::{event.event_type}",
                target=self._downstream,
                context=event.context.copy(),
            )
            return [forward_event]

        # No tokens available - drop and record as throttled
        self.stats.requests_dropped += 1
        self.dropped_times.append(now)

        logger.debug(
            "[%.3f][%s] Dropped request; rate=%.2f, tokens=%.2f",
            now.to_seconds(),
            self.name,
            self._current_rate,
            self._tokens,
        )
        return []

    def handle_response(self, event: Event) -> list[Event]:
        """Handle a response event from downstream.

        If a failure_predicate was configured, this can be used to
        automatically detect failures and adjust the rate.

        Args:
            event: The response event.

        Returns:
            Empty list (responses are not forwarded).
        """
        if self._failure_predicate is not None:
            if self._failure_predicate(event):
                self.record_failure(event.time)
            else:
                self.record_success(event.time)

        return []
