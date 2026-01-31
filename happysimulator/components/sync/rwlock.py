"""Read-Write Lock implementation.

Provides a lock that allows concurrent reads or exclusive writes.
Multiple readers can hold the lock simultaneously, but writers get
exclusive access.

Example:
    from happysimulator.components.sync import RWLock

    cache_lock = RWLock(name="cache")

    # Reader
    def read_cache(self, event):
        yield from cache_lock.acquire_read()
        try:
            yield 0.001  # Read from cache
        finally:
            return cache_lock.release_read()

    # Writer
    def update_cache(self, event):
        yield from cache_lock.acquire_write()
        try:
            yield 0.01  # Update cache
        finally:
            return cache_lock.release_write()
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Callable, Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class _WaiterType(Enum):
    """Type of waiter in the queue."""

    READER = "reader"
    WRITER = "writer"


@dataclass
class RWLockStats:
    """Statistics tracked by RWLock."""

    read_acquisitions: int = 0
    write_acquisitions: int = 0
    read_releases: int = 0
    write_releases: int = 0
    read_contentions: int = 0  # Readers that had to wait
    write_contentions: int = 0  # Writers that had to wait
    total_read_wait_ns: int = 0
    total_write_wait_ns: int = 0
    peak_readers: int = 0  # Maximum concurrent readers


@dataclass
class _Waiter:
    """A waiting acquirer."""

    waiter_type: _WaiterType
    callback: Callable[[], Any]
    enqueue_time_ns: int


class RWLock(Entity):
    """Read-write lock allowing concurrent reads or exclusive write.

    Multiple readers can hold the lock simultaneously. Writers get
    exclusive access - no readers or other writers allowed.

    Write requests are prioritized to prevent writer starvation:
    once a writer is waiting, new readers must wait behind it.

    Attributes:
        name: Entity name for identification.
        active_readers: Number of readers currently holding the lock.
        is_write_locked: Whether a writer holds the lock.
        max_readers: Maximum concurrent readers (None = unlimited).
    """

    def __init__(self, name: str, max_readers: int | None = None):
        """Initialize the read-write lock.

        Args:
            name: Name for this lock entity.
            max_readers: Maximum concurrent readers (None = unlimited).

        Raises:
            ValueError: If max_readers < 1.
        """
        if max_readers is not None and max_readers < 1:
            raise ValueError(f"max_readers must be >= 1 or None, got {max_readers}")

        super().__init__(name)
        self._max_readers = max_readers
        self._active_readers = 0
        self._write_locked = False
        self._waiters: deque[_Waiter] = deque()

        # Statistics
        self.stats = RWLockStats()

    @property
    def active_readers(self) -> int:
        """Number of readers currently holding the lock."""
        return self._active_readers

    @property
    def is_write_locked(self) -> bool:
        """Whether a writer holds the lock."""
        return self._write_locked

    @property
    def max_readers(self) -> int | None:
        """Maximum concurrent readers."""
        return self._max_readers

    @property
    def waiters(self) -> int:
        """Total number of waiting readers and writers."""
        return len(self._waiters)

    def _has_waiting_writer(self) -> bool:
        """Check if there's a writer in the wait queue."""
        return any(w.waiter_type == _WaiterType.WRITER for w in self._waiters)

    def try_acquire_read(self) -> bool:
        """Try to acquire a read lock without blocking.

        Returns:
            True if read lock was acquired, False otherwise.
        """
        # Can't acquire if write-locked or writer waiting (to prevent starvation)
        if self._write_locked:
            return False
        if self._has_waiting_writer():
            return False
        if self._max_readers and self._active_readers >= self._max_readers:
            return False

        self._active_readers += 1
        self.stats.read_acquisitions += 1

        if self._active_readers > self.stats.peak_readers:
            self.stats.peak_readers = self._active_readers

        return True

    def try_acquire_write(self) -> bool:
        """Try to acquire a write lock without blocking.

        Returns:
            True if write lock was acquired, False otherwise.
        """
        if self._write_locked or self._active_readers > 0:
            return False

        self._write_locked = True
        self.stats.write_acquisitions += 1
        return True

    def acquire_read(self) -> Generator[float, None, None]:
        """Acquire a read lock, blocking if necessary.

        Blocks if a writer holds the lock or a writer is waiting.

        Yields:
            0.0 when read lock is acquired.
        """
        if self.try_acquire_read():
            yield 0.0
            return

        # Must wait
        self.stats.read_contentions += 1
        enqueue_time = self._clock.now.nanoseconds if self._clock else 0

        acquired = [False]

        def on_wake():
            acquired[0] = True

        waiter = _Waiter(
            waiter_type=_WaiterType.READER,
            callback=on_wake,
            enqueue_time_ns=enqueue_time,
        )
        self._waiters.append(waiter)

        while not acquired[0]:
            yield 0.0

        self.stats.read_acquisitions += 1

        if self._clock:
            wait_time = self._clock.now.nanoseconds - enqueue_time
            self.stats.total_read_wait_ns += wait_time

    def acquire_write(self) -> Generator[float, None, None]:
        """Acquire a write lock, blocking if necessary.

        Blocks if any readers or another writer holds the lock.

        Yields:
            0.0 when write lock is acquired.
        """
        if self.try_acquire_write():
            yield 0.0
            return

        # Must wait
        self.stats.write_contentions += 1
        enqueue_time = self._clock.now.nanoseconds if self._clock else 0

        acquired = [False]

        def on_wake():
            acquired[0] = True

        waiter = _Waiter(
            waiter_type=_WaiterType.WRITER,
            callback=on_wake,
            enqueue_time_ns=enqueue_time,
        )
        self._waiters.append(waiter)

        while not acquired[0]:
            yield 0.0

        self.stats.write_acquisitions += 1

        if self._clock:
            wait_time = self._clock.now.nanoseconds - enqueue_time
            self.stats.total_write_wait_ns += wait_time

    def release_read(self) -> list[Event]:
        """Release a read lock.

        Returns:
            Empty list (waking is handled internally).

        Raises:
            RuntimeError: If no read lock is held.
        """
        if self._active_readers < 1:
            raise RuntimeError(f"RWLock {self.name}: release_read when no readers")

        self._active_readers -= 1
        self.stats.read_releases += 1

        self._wake_waiters()
        return []

    def release_write(self) -> list[Event]:
        """Release a write lock.

        Returns:
            Empty list (waking is handled internally).

        Raises:
            RuntimeError: If no write lock is held.
        """
        if not self._write_locked:
            raise RuntimeError(f"RWLock {self.name}: release_write when not write-locked")

        self._write_locked = False
        self.stats.write_releases += 1

        self._wake_waiters()
        return []

    def _wake_waiters(self) -> None:
        """Wake waiters that can now proceed."""
        if not self._waiters:
            return

        # If write-locked, can't wake anyone
        if self._write_locked:
            return

        # Check what's at the front
        front = self._waiters[0]

        if front.waiter_type == _WaiterType.WRITER:
            # Writer at front - can only wake if no active readers
            if self._active_readers == 0:
                self._waiters.popleft()
                self._write_locked = True
                front.callback()
        else:
            # Readers at front - wake all readers until we hit a writer
            while self._waiters:
                waiter = self._waiters[0]

                if waiter.waiter_type == _WaiterType.WRITER:
                    # Stop at writer
                    break

                # Check max readers limit
                if self._max_readers and self._active_readers >= self._max_readers:
                    break

                self._waiters.popleft()
                self._active_readers += 1
                waiter.callback()

                if self._active_readers > self.stats.peak_readers:
                    self.stats.peak_readers = self._active_readers

    def handle_event(self, event: Event) -> None:
        """RWLock doesn't directly handle events."""
        pass
