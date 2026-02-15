"""Rate limiter policies — pure algorithms decoupled from simulation entities.

Each policy implements `try_acquire` and `time_until_available` using an
`Instant` parameter for the current time. Policies are plain classes (not
Entities) and carry no simulation state beyond the algorithm itself.

Available policies:
- TokenBucketPolicy: Classic token bucket with burst capacity
- LeakyBucketPolicy: Strict fixed output rate (no bursting)
- SlidingWindowPolicy: Sliding window log algorithm
- FixedWindowPolicy: Fixed time window counter
- AdaptivePolicy: Token bucket with AIMD rate adjustment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@runtime_checkable
class RateLimiterPolicy(Protocol):
    """Protocol for rate limiter algorithms.

    Implementations decide whether a request should be allowed at a given
    point in time and can report how long until capacity is next available.
    """

    def try_acquire(self, now: Instant) -> bool:
        """Attempt to acquire one unit of capacity.

        Args:
            now: Current simulation time.

        Returns:
            True if the request is granted, False if denied.
        """
        ...

    def time_until_available(self, now: Instant) -> Duration:
        """Duration until next capacity is available.

        Invariant: if this returns Duration.ZERO, a subsequent call to
        ``try_acquire(now)`` MUST succeed. Implementations must return
        at least ``Duration(1)`` (1 ns) when the computed wait truncates
        to zero due to floating-point precision but capacity is not yet
        available.

        Args:
            now: Current simulation time.

        Returns:
            Duration.ZERO if capacity is available now, otherwise the
            duration until the next unit of capacity becomes available.
        """
        ...


class TokenBucketPolicy:
    """Token bucket rate limiter policy.

    Tokens refill at a constant rate up to a maximum capacity. Each
    request consumes one token. Allows controlled bursting up to the
    bucket capacity.

    Args:
        capacity: Maximum tokens the bucket can hold.
        refill_rate: Tokens added per second.
        initial_tokens: Starting token count (defaults to capacity).
    """

    def __init__(
        self,
        capacity: float = 10.0,
        refill_rate: float = 1.0,
        initial_tokens: float | None = None,
    ):
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)
        self._tokens = self._capacity if initial_tokens is None else float(initial_tokens)
        self._last_refill_time: Instant | None = None

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def refill_rate(self) -> float:
        return self._refill_rate

    @property
    def tokens(self) -> float:
        return self._tokens

    def _refill(self, now: Instant) -> None:
        if self._last_refill_time is None:
            self._last_refill_time = now
            return
        elapsed = (now - self._last_refill_time).to_seconds()
        if elapsed <= 0:
            return
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill_time = now

    def try_acquire(self, now: Instant) -> bool:
        self._refill(now)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def time_until_available(self, now: Instant) -> Duration:
        self._refill(now)
        if self._tokens >= 1.0:
            return Duration.ZERO
        deficit = 1.0 - self._tokens
        wait = Duration.from_seconds(deficit / self._refill_rate)
        # Guard: if FP truncation yields zero but tokens < 1.0, ensure progress
        if wait == Duration.ZERO:
            return Duration(1)
        return wait


class LeakyBucketPolicy:
    """Leaky bucket rate limiter policy.

    Enforces a strict fixed output rate with no bursting. Tracks whether
    sufficient time has elapsed since the last leak to allow a new request.

    Args:
        leak_rate: Requests per second leak rate.
    """

    def __init__(self, leak_rate: float = 1.0):
        self._leak_rate = float(leak_rate)
        self._leak_interval = 1.0 / leak_rate if leak_rate > 0 else float("inf")
        self._last_leak_time: Instant | None = None

    @property
    def leak_rate(self) -> float:
        return self._leak_rate

    def try_acquire(self, now: Instant) -> bool:
        if self._last_leak_time is None:
            self._last_leak_time = now
            return True
        elapsed = (now - self._last_leak_time).to_seconds()
        if elapsed >= self._leak_interval:
            self._last_leak_time = now
            return True
        return False

    def time_until_available(self, now: Instant) -> Duration:
        if self._last_leak_time is None:
            return Duration.ZERO
        elapsed = (now - self._last_leak_time).to_seconds()
        remaining = self._leak_interval - elapsed
        if remaining <= 0:
            return Duration.ZERO
        wait = Duration.from_seconds(remaining)
        # Guard: if FP truncation yields zero but interval hasn't elapsed, ensure progress
        if wait == Duration.ZERO:
            return Duration(1)
        return wait


class SlidingWindowPolicy:
    """Sliding window log rate limiter policy.

    Maintains a log of request timestamps and limits the number of
    requests within a rolling time window. Avoids the boundary burst
    problem of fixed windows.

    Args:
        window_size_seconds: Size of the sliding window in seconds.
        max_requests: Maximum requests allowed within the window.
    """

    def __init__(self, window_size_seconds: float = 1.0, max_requests: int = 10):
        self._window_size = float(window_size_seconds)
        self._max_requests = int(max_requests)
        self._request_log: list[Instant] = []

    @property
    def window_size_seconds(self) -> float:
        return self._window_size

    @property
    def max_requests(self) -> int:
        return self._max_requests

    def _prune(self, now: Instant) -> None:
        cutoff = now - self._window_size
        while self._request_log and self._request_log[0] < cutoff:
            self._request_log.pop(0)

    def try_acquire(self, now: Instant) -> bool:
        self._prune(now)
        if len(self._request_log) < self._max_requests:
            self._request_log.append(now)
            return True
        return False

    def time_until_available(self, now: Instant) -> Duration:
        self._prune(now)
        if len(self._request_log) < self._max_requests:
            return Duration.ZERO
        # Oldest entry must expire before a new slot opens
        oldest = self._request_log[0]
        expires_at = oldest + self._window_size
        remaining = (expires_at - now).to_seconds()
        wait = Duration.from_seconds(remaining)
        # Guard: if FP truncation yields zero but window is full, ensure progress
        if wait == Duration.ZERO:
            return Duration(1)
        return wait


class FixedWindowPolicy:
    """Fixed window rate limiter policy.

    Divides time into discrete windows and limits requests per window.
    Simple and O(1) space, but susceptible to boundary bursts.

    Args:
        requests_per_window: Maximum requests allowed per window.
        window_size: Size of each window in seconds.

    Raises:
        ValueError: If parameters are invalid.
    """

    def __init__(self, requests_per_window: int, window_size: float = 1.0):
        if requests_per_window < 1:
            raise ValueError(f"requests_per_window must be >= 1, got {requests_per_window}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        self._requests_per_window = requests_per_window
        self._window_size = window_size
        self._current_window_start: Instant | None = None
        self._current_window_count: int = 0

    @property
    def requests_per_window(self) -> int:
        return self._requests_per_window

    @property
    def window_size(self) -> float:
        return self._window_size

    def _get_window_start(self, now: Instant) -> Instant:
        now_s = now.to_seconds()
        return Instant.from_seconds((now_s // self._window_size) * self._window_size)

    def _maybe_reset(self, now: Instant) -> None:
        ws = self._get_window_start(now)
        if self._current_window_start is None or ws > self._current_window_start:
            self._current_window_start = ws
            self._current_window_count = 0

    def try_acquire(self, now: Instant) -> bool:
        self._maybe_reset(now)
        if self._current_window_count < self._requests_per_window:
            self._current_window_count += 1
            return True
        return False

    def time_until_available(self, now: Instant) -> Duration:
        self._maybe_reset(now)
        if self._current_window_count < self._requests_per_window:
            return Duration.ZERO
        # Wait until next window starts
        assert self._current_window_start is not None
        next_window = self._current_window_start + self._window_size
        remaining = (next_window - now).to_seconds()
        if remaining <= 0:
            return Duration.ZERO
        wait = Duration.from_seconds(remaining)
        # Guard: if FP truncation yields zero but window is exhausted, ensure progress
        if wait == Duration.ZERO:
            return Duration(1)
        return wait


class RateAdjustmentReason(Enum):
    """Reason for adaptive rate adjustment."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    THROTTLED = "throttled"


@dataclass
class RateSnapshot:
    """A snapshot of the adaptive rate at a point in time."""

    time: Instant
    rate: float
    reason: RateAdjustmentReason | None = None


class AdaptivePolicy:
    """Adaptive rate limiter policy using AIMD algorithm.

    Uses a token bucket internally with an adaptive refill rate. On
    success the rate increases additively; on failure it decreases
    multiplicatively. Exposes ``record_success`` / ``record_failure``
    for feedback.

    Args:
        initial_rate: Starting rate (requests per second).
        min_rate: Minimum allowed rate.
        max_rate: Maximum allowed rate.
        increase_step: Additive increase on success (default: initial_rate * 0.1).
        decrease_factor: Multiplicative factor on failure (0 < f < 1).
        window_size: Token bucket window size in seconds.

    Raises:
        ValueError: If parameters are invalid.
    """

    def __init__(
        self,
        initial_rate: float = 100.0,
        min_rate: float = 1.0,
        max_rate: float = 10000.0,
        increase_step: float | None = None,
        decrease_factor: float = 0.5,
        window_size: float = 1.0,
    ):
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

        self._min_rate = min_rate
        self._max_rate = max_rate
        self._current_rate = initial_rate
        self._increase_step = increase_step if increase_step is not None else initial_rate * 0.1
        self._decrease_factor = decrease_factor
        self._window_size = window_size

        # Internal token bucket
        self._tokens = initial_rate * window_size
        self._last_refill_time: Instant | None = None

        # Rate history for visualization
        self.rate_history: list[RateSnapshot] = []

        # Counters
        self.successes: int = 0
        self.failures: int = 0
        self.timeouts: int = 0
        self.rate_increases: int = 0
        self.rate_decreases: int = 0

    @property
    def current_rate(self) -> float:
        return self._current_rate

    @property
    def min_rate(self) -> float:
        return self._min_rate

    @property
    def max_rate(self) -> float:
        return self._max_rate

    @property
    def tokens(self) -> float:
        return self._tokens

    def _refill(self, now: Instant) -> None:
        if self._last_refill_time is None:
            self._last_refill_time = now
            return
        elapsed = (now - self._last_refill_time).to_seconds()
        if elapsed <= 0:
            return
        max_tokens = self._current_rate * self._window_size
        self._tokens = min(max_tokens, self._tokens + elapsed * self._current_rate)
        self._last_refill_time = now

    def try_acquire(self, now: Instant) -> bool:
        self._refill(now)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def time_until_available(self, now: Instant) -> Duration:
        self._refill(now)
        if self._tokens >= 1.0:
            return Duration.ZERO
        deficit = 1.0 - self._tokens
        wait_seconds = deficit / self._current_rate if self._current_rate > 0 else float("inf")
        wait = Duration.from_seconds(wait_seconds)
        # Guard: if FP truncation yields zero but tokens < 1.0, ensure progress
        if wait == Duration.ZERO:
            return Duration(1)
        return wait

    def record_success(self, now: Instant) -> None:
        """Record a successful request, potentially increasing rate."""
        self.successes += 1
        old_rate = self._current_rate
        self._current_rate = min(self._max_rate, self._current_rate + self._increase_step)
        if self._current_rate > old_rate:
            self.rate_increases += 1
            self.rate_history.append(
                RateSnapshot(time=now, rate=self._current_rate, reason=RateAdjustmentReason.SUCCESS)
            )

    def record_failure(
        self, now: Instant, reason: RateAdjustmentReason = RateAdjustmentReason.FAILURE
    ) -> None:
        """Record a failed request, decreasing rate."""
        if reason == RateAdjustmentReason.TIMEOUT:
            self.timeouts += 1
        else:
            self.failures += 1
        old_rate = self._current_rate
        self._current_rate = max(self._min_rate, self._current_rate * self._decrease_factor)
        if self._current_rate < old_rate:
            self.rate_decreases += 1
            self.rate_history.append(RateSnapshot(time=now, rate=self._current_rate, reason=reason))
