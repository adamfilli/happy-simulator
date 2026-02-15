"""Retry policies for client request handling.

Provides pluggable retry strategies that determine when and how to retry
failed requests. Includes common patterns like fixed delay, exponential
backoff, and AWS-style decorrelated jitter.

Example:
    # Exponential backoff with jitter
    policy = ExponentialBackoff(
        max_attempts=3,
        initial_delay=0.1,
        max_delay=10.0,
        multiplier=2.0,
        jitter=0.1,
    )

    # Check if should retry
    if policy.should_retry(attempt=2, error=TimeoutError()):
        delay = policy.get_delay(attempt=2)
        # Wait delay seconds before retry
"""

import logging
import random
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class RetryPolicy(Protocol):
    """Protocol for retry behavior strategies.

    Implementations determine whether to retry a failed request and
    how long to wait before the next attempt.
    """

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Determine if another retry attempt should be made.

        Args:
            attempt: The attempt number (1 = first attempt, 2 = first retry, etc.).
            error: The error that caused the failure, if any.

        Returns:
            True if another attempt should be made.
        """
        ...

    def get_delay(self, attempt: int) -> float:
        """Get the delay before the next retry attempt.

        Args:
            attempt: The attempt number (1 = first attempt, 2 = first retry, etc.).

        Returns:
            Delay in seconds before the next attempt.
        """
        ...


class NoRetry:
    """Never retry failed requests.

    Use when requests should fail immediately without retry attempts.
    This is the default policy when no retry is configured.
    """

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Never retry.

        Args:
            attempt: The attempt number (ignored).
            error: The error (ignored).

        Returns:
            Always False.
        """
        return False

    def get_delay(self, attempt: int) -> float:
        """No delay since we never retry.

        Args:
            attempt: The attempt number (ignored).

        Returns:
            Always 0.
        """
        return 0.0


class FixedRetry:
    """Retry with a fixed delay between attempts.

    Simple retry strategy that waits the same amount of time between
    each retry attempt. Good for transient failures where the delay
    doesn't need to increase.

    Attributes:
        max_attempts: Maximum number of total attempts (including initial).
        delay: Fixed delay in seconds between attempts.
    """

    def __init__(self, max_attempts: int, delay: float):
        """Initialize fixed retry policy.

        Args:
            max_attempts: Maximum total attempts (must be >= 1).
            delay: Delay in seconds between attempts (must be >= 0).

        Raises:
            ValueError: If max_attempts < 1 or delay < 0.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}")

        self._max_attempts = max_attempts
        self._delay = delay

        logger.debug(
            "FixedRetry created: max_attempts=%d, delay=%.3fs",
            max_attempts,
            delay,
        )

    @property
    def max_attempts(self) -> int:
        """Maximum number of total attempts."""
        return self._max_attempts

    @property
    def delay(self) -> float:
        """Fixed delay between attempts."""
        return self._delay

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Check if another retry should be attempted.

        Args:
            attempt: Current attempt number (1-based).
            error: The error that occurred (ignored for this policy).

        Returns:
            True if attempt < max_attempts.
        """
        return attempt < self._max_attempts

    def get_delay(self, attempt: int) -> float:
        """Get the fixed delay.

        Args:
            attempt: Current attempt number (ignored).

        Returns:
            The configured fixed delay.
        """
        return self._delay


class ExponentialBackoff:
    """Exponential backoff with optional jitter.

    Increases delay exponentially between retries, with optional random
    jitter to prevent thundering herd problems. This is the recommended
    retry strategy for most distributed systems.

    The delay formula is:
        delay = min(initial_delay * (multiplier ^ (attempt - 1)), max_delay)
        delay += random(0, jitter)

    Attributes:
        max_attempts: Maximum number of total attempts.
        initial_delay: Delay for the first retry.
        max_delay: Maximum delay cap.
        multiplier: Factor to multiply delay by each attempt.
        jitter: Maximum random delay to add.
    """

    def __init__(
        self,
        max_attempts: int,
        initial_delay: float,
        max_delay: float,
        multiplier: float = 2.0,
        jitter: float = 0.0,
    ):
        """Initialize exponential backoff policy.

        Args:
            max_attempts: Maximum total attempts (must be >= 1).
            initial_delay: Initial delay in seconds (must be > 0).
            max_delay: Maximum delay cap in seconds (must be >= initial_delay).
            multiplier: Delay multiplier per attempt (must be >= 1).
            jitter: Maximum random jitter to add (must be >= 0).

        Raises:
            ValueError: If parameters are invalid.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if initial_delay <= 0:
            raise ValueError(f"initial_delay must be > 0, got {initial_delay}")
        if max_delay < initial_delay:
            raise ValueError(f"max_delay ({max_delay}) must be >= initial_delay ({initial_delay})")
        if multiplier < 1:
            raise ValueError(f"multiplier must be >= 1, got {multiplier}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")

        self._max_attempts = max_attempts
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._multiplier = multiplier
        self._jitter = jitter

        logger.debug(
            "ExponentialBackoff created: max_attempts=%d, initial=%.3fs, "
            "max=%.3fs, multiplier=%.1f, jitter=%.3fs",
            max_attempts,
            initial_delay,
            max_delay,
            multiplier,
            jitter,
        )

    @property
    def max_attempts(self) -> int:
        """Maximum number of total attempts."""
        return self._max_attempts

    @property
    def initial_delay(self) -> float:
        """Initial delay for first retry."""
        return self._initial_delay

    @property
    def max_delay(self) -> float:
        """Maximum delay cap."""
        return self._max_delay

    @property
    def multiplier(self) -> float:
        """Delay multiplier per attempt."""
        return self._multiplier

    @property
    def jitter(self) -> float:
        """Maximum random jitter."""
        return self._jitter

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Check if another retry should be attempted.

        Args:
            attempt: Current attempt number (1-based).
            error: The error that occurred (ignored for this policy).

        Returns:
            True if attempt < max_attempts.
        """
        return attempt < self._max_attempts

    def get_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter.

        Args:
            attempt: Current attempt number (1-based).

        Returns:
            Delay in seconds for this attempt.
        """
        # Exponential: initial * multiplier^(attempt-1)
        # attempt=1 (first try): no delay needed (but if called, use initial)
        # attempt=2 (first retry): initial_delay
        # attempt=3 (second retry): initial_delay * multiplier
        exponent = max(0, attempt - 2)  # 0 for first retry
        delay = self._initial_delay * (self._multiplier**exponent)

        # Cap at max_delay
        delay = min(delay, self._max_delay)

        # Add jitter
        if self._jitter > 0:
            delay += random.uniform(0, self._jitter)

        return delay


class DecorrelatedJitter:
    """AWS-style decorrelated jitter backoff.

    A more sophisticated jitter algorithm that decorrelates retry times
    across multiple clients. This helps prevent synchronized retries
    (thundering herd) more effectively than simple jitter.

    The delay formula is:
        delay = random(base_delay, min(max_delay, previous_delay * 3))

    Reference:
        https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

    Attributes:
        max_attempts: Maximum number of total attempts.
        base_delay: Minimum delay between attempts.
        max_delay: Maximum delay cap.
    """

    def __init__(self, max_attempts: int, base_delay: float, max_delay: float):
        """Initialize decorrelated jitter policy.

        Args:
            max_attempts: Maximum total attempts (must be >= 1).
            base_delay: Minimum delay in seconds (must be > 0).
            max_delay: Maximum delay cap in seconds (must be >= base_delay).

        Raises:
            ValueError: If parameters are invalid.
        """
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        if base_delay <= 0:
            raise ValueError(f"base_delay must be > 0, got {base_delay}")
        if max_delay < base_delay:
            raise ValueError(f"max_delay ({max_delay}) must be >= base_delay ({base_delay})")

        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._previous_delay = base_delay

        logger.debug(
            "DecorrelatedJitter created: max_attempts=%d, base=%.3fs, max=%.3fs",
            max_attempts,
            base_delay,
            max_delay,
        )

    @property
    def max_attempts(self) -> int:
        """Maximum number of total attempts."""
        return self._max_attempts

    @property
    def base_delay(self) -> float:
        """Minimum delay between attempts."""
        return self._base_delay

    @property
    def max_delay(self) -> float:
        """Maximum delay cap."""
        return self._max_delay

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """Check if another retry should be attempted.

        Args:
            attempt: Current attempt number (1-based).
            error: The error that occurred (ignored for this policy).

        Returns:
            True if attempt < max_attempts.
        """
        return attempt < self._max_attempts

    def get_delay(self, attempt: int) -> float:
        """Calculate decorrelated jitter delay.

        Uses the formula: random(base, min(max, prev * 3))

        Args:
            attempt: Current attempt number (1-based).

        Returns:
            Delay in seconds for this attempt.
        """
        # Calculate upper bound: min(max_delay, previous * 3)
        upper = min(self._max_delay, self._previous_delay * 3)

        # Random between base and upper
        delay = random.uniform(self._base_delay, upper)

        # Store for next calculation
        self._previous_delay = delay

        return delay

    def reset(self) -> None:
        """Reset the previous delay to base_delay.

        Call this when starting a new request sequence.
        """
        self._previous_delay = self._base_delay
