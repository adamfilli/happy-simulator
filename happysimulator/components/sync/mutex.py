"""Mutex (Mutual Exclusion Lock) implementation.

Provides a mutual exclusion lock that allows only one holder at a time.
Other acquirers are queued and wake in FIFO order when the lock is released.

Example:
    from happysimulator.components.sync import Mutex

    mutex = Mutex(name="resource_lock")

    def handle_event(self, event):
        yield from mutex.acquire()
        try:
            yield 0.01  # Critical section work
        finally:
            return mutex.release()
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Generator, Callable, Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


@dataclass
class MutexStats:
    """Statistics tracked by Mutex."""

    acquisitions: int = 0  # Successful lock acquisitions
    releases: int = 0  # Lock releases
    contentions: int = 0  # Times a waiter had to queue
    total_wait_time_ns: int = 0  # Total time spent waiting (nanoseconds)


@dataclass
class _Waiter:
    """A waiting acquirer."""

    callback: Callable[[], Any]
    enqueue_time_ns: int


class Mutex(Entity):
    """Mutual exclusion lock with queued waiters.

    Only one entity can hold the lock at a time. Other acquirers block
    until the lock is released, then wake in FIFO order.

    The acquire() method is a generator that yields until the lock is
    obtained. The release() method returns events to wake the next waiter.

    Attributes:
        name: Entity name for identification.
        is_locked: Whether the lock is currently held.
        waiters: Number of entities waiting to acquire.
    """

    def __init__(self, name: str):
        """Initialize the mutex.

        Args:
            name: Name for this mutex entity.
        """
        super().__init__(name)
        self._locked = False
        self._waiters: deque[_Waiter] = deque()
        self._owner: str | None = None

        # Statistics
        self.stats = MutexStats()

    @property
    def is_locked(self) -> bool:
        """Whether the lock is currently held."""
        return self._locked

    @property
    def waiters(self) -> int:
        """Number of entities waiting to acquire."""
        return len(self._waiters)

    @property
    def owner(self) -> str | None:
        """Current lock owner (if set via acquire with owner parameter)."""
        return self._owner

    def try_acquire(self, owner: str | None = None) -> bool:
        """Try to acquire the lock without blocking.

        Args:
            owner: Optional owner identifier for debugging.

        Returns:
            True if lock was acquired, False if already held.
        """
        if self._locked:
            return False

        self._locked = True
        self._owner = owner
        self.stats.acquisitions += 1
        return True

    def acquire(self, owner: str | None = None) -> Generator[float, None, None]:
        """Acquire the lock, blocking if necessary.

        This is a generator that yields control while waiting for the lock.
        Use with 'yield from' in an event handler.

        Args:
            owner: Optional owner identifier for debugging.

        Yields:
            0.0 when lock is acquired (no additional delay).

        Example:
            def handle_event(self, event):
                yield from self.mutex.acquire()
                # ... critical section ...
                return self.mutex.release()
        """
        if self.try_acquire(owner):
            # Lock acquired immediately
            yield 0.0
            return

        # Must wait - record contention
        self.stats.contentions += 1
        enqueue_time = self._clock.now.nanoseconds if self._clock else 0

        # Create a flag that will be set when we get the lock
        acquired = [False]

        def on_wake():
            acquired[0] = True

        waiter = _Waiter(callback=on_wake, enqueue_time_ns=enqueue_time)
        self._waiters.append(waiter)

        # Yield control until woken
        while not acquired[0]:
            yield 0.0

        # Now we have the lock
        self._owner = owner
        self.stats.acquisitions += 1

        if self._clock:
            wait_time = self._clock.now.nanoseconds - enqueue_time
            self.stats.total_wait_time_ns += wait_time

    def release(self) -> list[Event]:
        """Release the lock and wake the next waiter.

        Returns:
            List of events to wake the next waiter, or empty if no waiters.

        Raises:
            RuntimeError: If the lock is not currently held.
        """
        if not self._locked:
            raise RuntimeError(f"Mutex {self.name} released when not locked")

        self.stats.releases += 1
        self._owner = None

        if self._waiters:
            # Wake the next waiter
            waiter = self._waiters.popleft()
            waiter.callback()
            # Lock transfers directly to next waiter
            self._locked = True
            return []
        else:
            # No waiters, unlock
            self._locked = False
            return []

    def handle_event(self, event: Event) -> None:
        """Mutex doesn't directly handle events."""
        pass
