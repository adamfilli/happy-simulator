"""Semaphore implementation.

Provides a counting semaphore for limiting concurrent access to resources.
Unlike a mutex, a semaphore allows multiple holders up to a specified count.

Example:
    from happysimulator.components.sync import Semaphore

    # Allow up to 5 concurrent database connections
    db_pool = Semaphore(name="db_connections", initial_count=5)

    def handle_event(self, event):
        yield from db_pool.acquire()
        try:
            yield 0.05  # Use connection
        finally:
            return db_pool.release()
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Generator, Callable, Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


@dataclass
class SemaphoreStats:
    """Statistics tracked by Semaphore."""

    acquisitions: int = 0  # Successful permit acquisitions
    releases: int = 0  # Permit releases
    contentions: int = 0  # Times a waiter had to queue
    total_wait_time_ns: int = 0  # Total time spent waiting
    peak_waiters: int = 0  # Maximum concurrent waiters


@dataclass
class _Waiter:
    """A waiting acquirer."""

    count: int  # Number of permits requested
    callback: Callable[[], Any]
    enqueue_time_ns: int


class Semaphore(Entity):
    """Counting semaphore for resource limiting.

    Allows up to N concurrent holders, where N is the initial count.
    Acquirers can request multiple permits at once.

    Attributes:
        name: Entity name for identification.
        available: Number of permits currently available.
        capacity: Maximum number of permits (initial count).
    """

    def __init__(self, name: str, initial_count: int):
        """Initialize the semaphore.

        Args:
            name: Name for this semaphore entity.
            initial_count: Initial number of available permits.

        Raises:
            ValueError: If initial_count < 1.
        """
        if initial_count < 1:
            raise ValueError(f"initial_count must be >= 1, got {initial_count}")

        super().__init__(name)
        self._count = initial_count
        self._capacity = initial_count
        self._waiters: deque[_Waiter] = deque()

        # Statistics
        self.stats = SemaphoreStats()

    @property
    def available(self) -> int:
        """Number of permits currently available."""
        return self._count

    @property
    def capacity(self) -> int:
        """Maximum number of permits."""
        return self._capacity

    @property
    def waiters(self) -> int:
        """Number of entities waiting to acquire."""
        return len(self._waiters)

    def try_acquire(self, count: int = 1) -> bool:
        """Try to acquire permits without blocking.

        Args:
            count: Number of permits to acquire.

        Returns:
            True if permits were acquired, False if not enough available.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        if self._count >= count:
            self._count -= count
            self.stats.acquisitions += count
            return True

        return False

    def acquire(self, count: int = 1) -> Generator[float, None, None]:
        """Acquire permits, blocking if necessary.

        This is a generator that yields control while waiting for permits.
        Use with 'yield from' in an event handler.

        Args:
            count: Number of permits to acquire.

        Yields:
            0.0 when permits are acquired.

        Raises:
            ValueError: If count < 1 or count > capacity.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        if count > self._capacity:
            raise ValueError(
                f"cannot acquire {count} permits from semaphore with capacity {self._capacity}"
            )

        if self.try_acquire(count):
            yield 0.0
            return

        # Must wait
        self.stats.contentions += 1
        enqueue_time = self._clock.now.nanoseconds if self._clock else 0

        acquired = [False]

        def on_wake():
            acquired[0] = True

        waiter = _Waiter(count=count, callback=on_wake, enqueue_time_ns=enqueue_time)
        self._waiters.append(waiter)

        # Track peak waiters
        if len(self._waiters) > self.stats.peak_waiters:
            self.stats.peak_waiters = len(self._waiters)

        while not acquired[0]:
            yield 0.0

        self.stats.acquisitions += count

        if self._clock:
            wait_time = self._clock.now.nanoseconds - enqueue_time
            self.stats.total_wait_time_ns += wait_time

    def release(self, count: int = 1) -> list[Event]:
        """Release permits and wake waiting acquirers.

        Args:
            count: Number of permits to release.

        Returns:
            Empty list (waking is handled internally).

        Raises:
            ValueError: If count < 1 or would exceed capacity.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")

        # Check we won't exceed capacity
        future_count = self._count + count
        if future_count > self._capacity:
            raise ValueError(
                f"releasing {count} would exceed capacity "
                f"({self._count} + {count} > {self._capacity})"
            )

        self._count += count
        self.stats.releases += count

        # Wake waiters that can now be satisfied
        self._wake_waiters()

        return []

    def _wake_waiters(self) -> None:
        """Wake waiters whose requests can now be satisfied."""
        while self._waiters:
            waiter = self._waiters[0]

            if self._count >= waiter.count:
                # Can satisfy this waiter
                self._waiters.popleft()
                self._count -= waiter.count
                waiter.callback()
            else:
                # Not enough permits for next waiter
                break

    def handle_event(self, event: Event) -> None:
        """Semaphore doesn't directly handle events."""
        pass
