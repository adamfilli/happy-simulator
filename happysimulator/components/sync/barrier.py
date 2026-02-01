"""Barrier synchronization primitive.

Provides a synchronization point where multiple parties must wait until
all have arrived before any can proceed.

Example:
    from happysimulator.components.sync import Barrier

    barrier = Barrier(name="phase_sync", parties=3)

    # In each of 3 workers:
    def handle_event(self, event):
        # Do phase 1 work
        yield 0.01
        # Wait for all workers to finish phase 1
        yield from barrier.wait()
        # All workers now proceed to phase 2
        yield 0.02
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Generator, Callable, Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


@dataclass
class BarrierStats:
    """Statistics tracked by Barrier."""

    wait_calls: int = 0  # Total wait() calls
    barrier_breaks: int = 0  # Times barrier was broken (all parties arrived)
    resets: int = 0  # Manual resets
    total_wait_time_ns: int = 0  # Total time spent waiting (nanoseconds)


@dataclass
class _BarrierWaiter:
    """A party waiting at the barrier."""

    callback: Callable[[], Any]
    enqueue_time_ns: int


class Barrier(Entity):
    """Synchronization point for multiple parties.

    A barrier blocks all arriving parties until the specified number of
    parties have called wait(). Once all parties arrive, they are all
    released simultaneously and the barrier resets for the next round.

    This is useful for phased computations where all workers must complete
    one phase before any can start the next.

    Attributes:
        name: Entity name for identification.
        parties: Number of parties required to break the barrier.
        waiting: Number of parties currently waiting.
        broken: Whether the barrier has been broken (error state).
        generation: Current barrier generation (increments each break).
    """

    def __init__(self, name: str, parties: int):
        """Initialize the barrier.

        Args:
            name: Name for this barrier entity.
            parties: Number of parties required to break the barrier.

        Raises:
            ValueError: If parties < 1.
        """
        if parties < 1:
            raise ValueError(f"parties must be >= 1, got {parties}")

        super().__init__(name)
        self._parties = parties
        self._waiters: deque[_BarrierWaiter] = deque()
        self._generation = 0
        self._broken = False

        # Statistics
        self.stats = BarrierStats()

    @property
    def parties(self) -> int:
        """Number of parties required to break the barrier."""
        return self._parties

    @property
    def waiting(self) -> int:
        """Number of parties currently waiting at the barrier."""
        return len(self._waiters)

    @property
    def broken(self) -> bool:
        """Whether the barrier is in a broken state."""
        return self._broken

    @property
    def generation(self) -> int:
        """Current barrier generation (increments each time barrier breaks)."""
        return self._generation

    def wait(self) -> Generator[float, None, int]:
        """Wait at the barrier until all parties arrive.

        This is a generator that yields control while waiting for other
        parties. Use with 'yield from' in an event handler.

        Returns:
            The arrival index (0 to parties-1). The last party to arrive
            gets index 0, making it easy to designate a "leader".

        Raises:
            RuntimeError: If the barrier is broken.

        Yields:
            0.0 while waiting for other parties.

        Example:
            def handle_event(self, event):
                arrival_index = yield from barrier.wait()
                if arrival_index == 0:
                    # I'm the last to arrive, do leader tasks
                    pass
        """
        if self._broken:
            raise RuntimeError(f"Barrier {self.name} is broken")

        self.stats.wait_calls += 1
        enqueue_time = self._clock.now.nanoseconds if self._clock else 0
        my_generation = self._generation

        # Check if we're the last party (breaks the barrier)
        if len(self._waiters) + 1 >= self._parties:
            # We're the last one - break the barrier!
            self._break_barrier(enqueue_time)
            # Last to arrive gets index 0 (leader)
            return 0

        # Not the last - must wait
        released = [False]

        def on_release():
            released[0] = True

        waiter = _BarrierWaiter(callback=on_release, enqueue_time_ns=enqueue_time)
        self._waiters.append(waiter)
        arrival_index = self._parties - len(self._waiters)

        # Yield control until released
        while not released[0]:
            # Check for broken barrier
            if self._broken:
                raise RuntimeError(f"Barrier {self.name} is broken")
            # Check for generation change (we were released)
            if self._generation != my_generation:
                released[0] = True
                break
            yield 0.0

        # Record wait time
        if self._clock:
            wait_time = self._clock.now.nanoseconds - enqueue_time
            self.stats.total_wait_time_ns += wait_time

        return arrival_index

    def _break_barrier(self, trigger_time_ns: int) -> None:
        """Break the barrier and release all waiting parties."""
        self.stats.barrier_breaks += 1

        # Wake all waiters
        while self._waiters:
            waiter = self._waiters.popleft()
            # Record wait time for each waiter
            if self._clock:
                wait_time = self._clock.now.nanoseconds - waiter.enqueue_time_ns
                self.stats.total_wait_time_ns += wait_time
            waiter.callback()

        # Advance to next generation
        self._generation += 1

    def reset(self) -> None:
        """Reset the barrier to its initial state.

        Any parties currently waiting will be released with a RuntimeError
        on their next check (barrier broken).

        This is useful for recovery scenarios or reusing the barrier
        after an error condition.
        """
        self.stats.resets += 1
        self._broken = True

        # Wake all waiters (they'll see broken state)
        while self._waiters:
            waiter = self._waiters.popleft()
            waiter.callback()

        # Reset to clean state
        self._broken = False
        self._generation += 1

    def abort(self) -> None:
        """Permanently break the barrier.

        All current and future wait() calls will raise RuntimeError.
        Use reset() to recover from this state.
        """
        self._broken = True

        # Wake all waiters
        while self._waiters:
            waiter = self._waiters.popleft()
            waiter.callback()

    def handle_event(self, event: Event) -> None:
        """Barrier doesn't directly handle events."""
        pass
