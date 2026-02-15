"""Concurrency control strategies for servers.

Provides pluggable concurrency models that control how many requests
a server can process simultaneously. Models range from simple fixed
limits to dynamic scaling and weighted capacity pools.
"""

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ConcurrencyModel(Protocol):
    """Protocol for concurrency control strategies.

    Implementations manage a pool of processing slots that requests
    must acquire before processing and release when complete.
    """

    def acquire(self, weight: int = 1) -> bool:
        """Attempt to acquire processing capacity.

        Args:
            weight: Amount of capacity to acquire (default 1).

        Returns:
            True if capacity was acquired, False if unavailable.
        """
        ...

    def release(self, weight: int = 1) -> None:
        """Release previously acquired processing capacity.

        Args:
            weight: Amount of capacity to release (default 1).
        """
        ...

    def has_capacity(self, weight: int = 1) -> bool:
        """Check if capacity is available without acquiring.

        Args:
            weight: Amount of capacity needed (default 1).

        Returns:
            True if the requested capacity is available.
        """
        ...

    @property
    def available(self) -> int:
        """Number of available slots/capacity units."""
        ...

    @property
    def active(self) -> int:
        """Number of currently used slots/capacity units."""
        ...

    @property
    def limit(self) -> int:
        """Total capacity limit."""
        ...


class FixedConcurrency:
    """Fixed number of concurrent slots.

    The simplest concurrency model with a static limit that cannot
    change during simulation. Each request consumes exactly one slot.

    Attributes:
        max_concurrent: Maximum number of simultaneous requests.
    """

    def __init__(self, max_concurrent: int):
        """Initialize with a fixed concurrency limit.

        Args:
            max_concurrent: Maximum simultaneous requests.

        Raises:
            ValueError: If max_concurrent < 1.
        """
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")

        self._max_concurrent = max_concurrent
        self._active = 0

        logger.debug("FixedConcurrency created: max=%d", max_concurrent)

    def acquire(self, weight: int = 1) -> bool:
        """Acquire a processing slot.

        Args:
            weight: Ignored for FixedConcurrency (always 1).

        Returns:
            True if a slot was acquired, False if at capacity.
        """
        if self._active >= self._max_concurrent:
            return False
        self._active += 1
        return True

    def release(self, weight: int = 1) -> None:
        """Release a processing slot.

        Args:
            weight: Ignored for FixedConcurrency (always 1).
        """
        self._active = max(0, self._active - 1)

    def has_capacity(self, weight: int = 1) -> bool:
        """Check if a slot is available.

        Args:
            weight: Ignored for FixedConcurrency.

        Returns:
            True if active < max_concurrent.
        """
        return self._active < self._max_concurrent

    @property
    def available(self) -> int:
        """Number of available slots."""
        return self._max_concurrent - self._active

    @property
    def active(self) -> int:
        """Number of currently used slots."""
        return self._active

    @property
    def limit(self) -> int:
        """Total slot limit."""
        return self._max_concurrent


class DynamicConcurrency:
    """Adjustable concurrency limit for autoscaling simulation.

    Allows the concurrency limit to be changed at runtime within
    configured bounds. Useful for simulating autoscaling behavior
    where capacity adjusts based on load.

    Attributes:
        current_limit: Current concurrency limit.
        min_limit: Minimum allowed limit.
        max_limit: Maximum allowed limit.
    """

    def __init__(self, initial: int, min_limit: int = 1, max_limit: int | None = None):
        """Initialize with adjustable concurrency bounds.

        Args:
            initial: Starting concurrency limit.
            min_limit: Minimum concurrency (default 1).
            max_limit: Maximum concurrency (default None = unlimited).

        Raises:
            ValueError: If initial < min_limit or initial > max_limit.
        """
        if min_limit < 1:
            raise ValueError(f"min_limit must be >= 1, got {min_limit}")
        if max_limit is not None and max_limit < min_limit:
            raise ValueError(f"max_limit ({max_limit}) must be >= min_limit ({min_limit})")
        if initial < min_limit:
            raise ValueError(f"initial ({initial}) must be >= min_limit ({min_limit})")
        if max_limit is not None and initial > max_limit:
            raise ValueError(f"initial ({initial}) must be <= max_limit ({max_limit})")

        self._current_limit = initial
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._active = 0

        logger.debug(
            "DynamicConcurrency created: initial=%d, min=%d, max=%s",
            initial,
            min_limit,
            max_limit,
        )

    @property
    def current_limit(self) -> int:
        """Current concurrency limit."""
        return self._current_limit

    @property
    def min_limit(self) -> int:
        """Minimum allowed limit."""
        return self._min_limit

    @property
    def max_limit(self) -> int | None:
        """Maximum allowed limit (None = unlimited)."""
        return self._max_limit

    def set_limit(self, new_limit: int) -> None:
        """Adjust the concurrency limit.

        The new limit is clamped to [min_limit, max_limit].
        If active requests exceed the new limit, they continue
        processing but no new requests are admitted.

        Args:
            new_limit: Desired new concurrency limit.
        """
        clamped = max(self._min_limit, new_limit)
        if self._max_limit is not None:
            clamped = min(self._max_limit, clamped)

        old_limit = self._current_limit
        self._current_limit = clamped

        logger.debug(
            "DynamicConcurrency limit changed: %d -> %d (requested %d)",
            old_limit,
            clamped,
            new_limit,
        )

    def scale_up(self, amount: int = 1) -> None:
        """Increase the concurrency limit.

        Args:
            amount: How much to increase (default 1).
        """
        self.set_limit(self._current_limit + amount)

    def scale_down(self, amount: int = 1) -> None:
        """Decrease the concurrency limit.

        Args:
            amount: How much to decrease (default 1).
        """
        self.set_limit(self._current_limit - amount)

    def acquire(self, weight: int = 1) -> bool:
        """Acquire a processing slot.

        Args:
            weight: Ignored for DynamicConcurrency (always 1).

        Returns:
            True if a slot was acquired, False if at capacity.
        """
        if self._active >= self._current_limit:
            return False
        self._active += 1
        return True

    def release(self, weight: int = 1) -> None:
        """Release a processing slot.

        Args:
            weight: Ignored for DynamicConcurrency (always 1).
        """
        self._active = max(0, self._active - 1)

    def has_capacity(self, weight: int = 1) -> bool:
        """Check if a slot is available.

        Args:
            weight: Ignored for DynamicConcurrency.

        Returns:
            True if active < current_limit.
        """
        return self._active < self._current_limit

    @property
    def available(self) -> int:
        """Number of available slots."""
        return max(0, self._current_limit - self._active)

    @property
    def active(self) -> int:
        """Number of currently used slots."""
        return self._active

    @property
    def limit(self) -> int:
        """Current total slot limit."""
        return self._current_limit


class WeightedConcurrency:
    """Requests consume variable weight from a capacity pool.

    Instead of fixed slots, this model provides a capacity pool
    where different requests can consume different amounts. Useful
    for modeling resources like memory, CPU cores, or database
    connections where operations have varying resource requirements.

    Example:
        pool = WeightedConcurrency(total_capacity=100)
        pool.acquire(weight=10)  # Heavy query
        pool.acquire(weight=1)   # Light query
        pool.available  # Returns 89

    Attributes:
        total_capacity: Total capacity units available.
    """

    def __init__(self, total_capacity: int):
        """Initialize with a total capacity pool.

        Args:
            total_capacity: Total capacity units available.

        Raises:
            ValueError: If total_capacity < 1.
        """
        if total_capacity < 1:
            raise ValueError(f"total_capacity must be >= 1, got {total_capacity}")

        self._total_capacity = total_capacity
        self._used_capacity = 0

        logger.debug("WeightedConcurrency created: capacity=%d", total_capacity)

    @property
    def total_capacity(self) -> int:
        """Total capacity units available."""
        return self._total_capacity

    def acquire(self, weight: int = 1) -> bool:
        """Acquire capacity from the pool.

        Args:
            weight: Amount of capacity to acquire.

        Returns:
            True if capacity was acquired, False if insufficient.
        """
        if weight < 1:
            raise ValueError(f"weight must be >= 1, got {weight}")

        if self._used_capacity + weight > self._total_capacity:
            return False

        self._used_capacity += weight
        logger.debug(
            "WeightedConcurrency acquired: weight=%d, used=%d/%d",
            weight,
            self._used_capacity,
            self._total_capacity,
        )
        return True

    def release(self, weight: int = 1) -> None:
        """Release capacity back to the pool.

        Args:
            weight: Amount of capacity to release.
        """
        if weight < 1:
            raise ValueError(f"weight must be >= 1, got {weight}")

        self._used_capacity = max(0, self._used_capacity - weight)
        logger.debug(
            "WeightedConcurrency released: weight=%d, used=%d/%d",
            weight,
            self._used_capacity,
            self._total_capacity,
        )

    def has_capacity(self, weight: int = 1) -> bool:
        """Check if capacity is available.

        Args:
            weight: Amount of capacity needed.

        Returns:
            True if the requested capacity is available.
        """
        return self._used_capacity + weight <= self._total_capacity

    @property
    def available(self) -> int:
        """Available capacity units."""
        return self._total_capacity - self._used_capacity

    @property
    def active(self) -> int:
        """Currently used capacity units."""
        return self._used_capacity

    @property
    def limit(self) -> int:
        """Total capacity limit."""
        return self._total_capacity

    @property
    def utilization(self) -> float:
        """Current utilization as fraction of total capacity."""
        if self._total_capacity == 0:
            return 0.0
        return self._used_capacity / self._total_capacity
