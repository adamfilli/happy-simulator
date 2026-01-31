"""Condition Variable implementation.

Provides a condition variable for complex synchronization patterns.
Threads can wait on a condition and be notified when the condition
might have changed.

Example:
    from happysimulator.components.sync import Mutex, Condition

    mutex = Mutex(name="queue_lock")
    not_empty = Condition(name="not_empty", lock=mutex)
    queue = []

    # Consumer
    def consume(self, event):
        yield from mutex.acquire()
        while not queue:
            yield from not_empty.wait()
        item = queue.pop(0)
        return mutex.release()

    # Producer
    def produce(self, event, item):
        yield from mutex.acquire()
        queue.append(item)
        events = not_empty.notify()
        return mutex.release() + events
"""

from collections import deque
from dataclasses import dataclass
from typing import Generator, Callable, Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.components.sync.mutex import Mutex


@dataclass
class ConditionStats:
    """Statistics tracked by Condition."""

    waits: int = 0  # Total wait() calls
    notifies: int = 0  # Total notify() calls
    notify_alls: int = 0  # Total notify_all() calls
    wakeups: int = 0  # Total waiters woken
    total_wait_time_ns: int = 0


@dataclass
class _Waiter:
    """A waiting thread."""

    callback: Callable[[], Any]
    enqueue_time_ns: int


class Condition(Entity):
    """Condition variable for complex synchronization.

    A condition variable allows threads to wait for a condition to become
    true. It must be used with an associated mutex that protects the
    shared state being checked.

    The wait() method atomically releases the mutex and waits. When woken,
    it reacquires the mutex before returning.

    Attributes:
        name: Entity name for identification.
        lock: Associated mutex.
        waiters: Number of threads waiting on this condition.
    """

    def __init__(self, name: str, lock: Mutex):
        """Initialize the condition variable.

        Args:
            name: Name for this condition entity.
            lock: Associated mutex that protects the condition.
        """
        super().__init__(name)
        self._lock = lock
        self._waiters: deque[_Waiter] = deque()

        # Statistics
        self.stats = ConditionStats()

    @property
    def lock(self) -> Mutex:
        """The associated mutex."""
        return self._lock

    @property
    def waiters(self) -> int:
        """Number of threads waiting on this condition."""
        return len(self._waiters)

    def wait(self) -> Generator[float, None, None]:
        """Wait for the condition to be signaled.

        Atomically releases the associated mutex, waits for a signal,
        then reacquires the mutex before returning.

        The caller should hold the mutex when calling wait(), and should
        check the actual condition in a loop (spurious wakeups are possible).

        Yields:
            0.0 while waiting and during mutex reacquisition.

        Raises:
            RuntimeError: If the mutex is not held.

        Example:
            yield from mutex.acquire()
            while not condition_is_true():
                yield from condition.wait()
            # condition is now true, mutex is held
        """
        if not self._lock.is_locked:
            raise RuntimeError(
                f"Condition {self.name}: wait() called without holding mutex"
            )

        self.stats.waits += 1
        enqueue_time = self._clock.now.nanoseconds if self._clock else 0

        # Set up wakeup callback
        woken = [False]

        def on_wake():
            woken[0] = True

        waiter = _Waiter(callback=on_wake, enqueue_time_ns=enqueue_time)
        self._waiters.append(waiter)

        # Release the mutex (this may wake other waiters on the mutex)
        self._lock.release()

        # Wait for signal
        while not woken[0]:
            yield 0.0

        # Reacquire the mutex
        yield from self._lock.acquire()

        if self._clock:
            wait_time = self._clock.now.nanoseconds - enqueue_time
            self.stats.total_wait_time_ns += wait_time

    def wait_for(
        self,
        predicate: Callable[[], bool],
        timeout: float | None = None,
    ) -> Generator[float, None, bool]:
        """Wait for a predicate to become true.

        A convenience method that handles the wait loop automatically.

        Args:
            predicate: Callable that returns True when condition is met.
            timeout: Maximum time to wait in seconds (None = forever).

        Yields:
            0.0 while waiting.

        Returns:
            True if predicate became true, False if timed out.
        """
        if not self._lock.is_locked:
            raise RuntimeError(
                f"Condition {self.name}: wait_for() called without holding mutex"
            )

        start_time = self._clock.now.nanoseconds if self._clock else 0

        while not predicate():
            if timeout is not None and self._clock:
                elapsed_ns = self._clock.now.nanoseconds - start_time
                elapsed_s = elapsed_ns / 1_000_000_000
                if elapsed_s >= timeout:
                    return False

            yield from self.wait()

        return True

    def notify(self, n: int = 1) -> list[Event]:
        """Wake up to n waiting threads.

        The woken threads will not run immediately; they will compete
        to reacquire the mutex when the current holder releases it.

        Args:
            n: Maximum number of threads to wake.

        Returns:
            Empty list (waking is handled internally).
        """
        self.stats.notifies += 1

        woken = 0
        while self._waiters and woken < n:
            waiter = self._waiters.popleft()
            waiter.callback()
            woken += 1

        self.stats.wakeups += woken
        return []

    def notify_all(self) -> list[Event]:
        """Wake all waiting threads.

        Returns:
            Empty list (waking is handled internally).
        """
        self.stats.notify_alls += 1

        woken = 0
        while self._waiters:
            waiter = self._waiters.popleft()
            waiter.callback()
            woken += 1

        self.stats.wakeups += woken
        return []

    def handle_event(self, event: Event) -> None:
        """Condition doesn't directly handle events."""
        pass
