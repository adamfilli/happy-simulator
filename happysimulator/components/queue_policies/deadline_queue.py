"""Deadline Queue implementation.

Provides deadline-based priority queuing that automatically drops expired items.

Example:
    from happysimulator.components.queue_policies import DeadlineQueue
    from happysimulator.core.temporal import Instant

    def get_deadline(event):
        # Deadline is 100ms after event time
        return event.time + Duration.from_seconds(0.100)

    queue = DeadlineQueue(
        get_deadline=get_deadline,
        clock_func=lambda: simulation.clock.now,
    )
"""

import heapq
from dataclasses import dataclass, field
from typing import TypeVar, Optional, Callable, Generic

from happysimulator.components.queue_policy import QueuePolicy
from happysimulator.core.temporal import Instant

T = TypeVar("T")


@dataclass(frozen=True)
class DeadlineQueueStats:
    """Statistics tracked by DeadlineQueue."""

    enqueued: int = 0
    dequeued: int = 0
    expired: int = 0
    capacity_rejected: int = 0


@dataclass(order=True)
class _DeadlineEntry(Generic[T]):
    """Entry wrapper for deadline-based heap ordering."""

    deadline_ns: int  # Nanoseconds for comparison
    insert_order: int  # Tie-breaker for same deadline
    item: T = field(compare=False)
    deadline: Instant = field(compare=False)


class DeadlineQueue(QueuePolicy[T]):
    """Priority queue by deadline that drops expired items.

    Items are dequeued in deadline order (earliest first).
    Expired items (deadline < current time) are automatically dropped.

    This is useful for request queues where stale requests should be
    discarded rather than processed late.

    Attributes:
        get_deadline: Function to extract deadline from items.
        capacity: Maximum queue capacity.
    """

    def __init__(
        self,
        get_deadline: Callable[[T], Instant],
        capacity: int | None = None,
        clock_func: Callable[[], Instant] | None = None,
    ):
        """Initialize the deadline queue.

        Args:
            get_deadline: Function that extracts deadline from an item.
            capacity: Maximum queue capacity (None = unlimited).
            clock_func: Function returning current time as Instant.
                       Required for expiration checking.

        Raises:
            ValueError: If parameters are invalid.
        """
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")

        self._get_deadline = get_deadline
        self._capacity = float("inf") if capacity is None else capacity
        self._clock_func = clock_func

        # Min-heap for deadline ordering
        self._heap: list[_DeadlineEntry[T]] = []
        self._insert_counter = 0

        # Statistics
        self._enqueued = 0
        self._dequeued = 0
        self._expired = 0
        self._capacity_rejected = 0

    @property
    def stats(self) -> DeadlineQueueStats:
        """Return a frozen snapshot of queue statistics."""
        return DeadlineQueueStats(
            enqueued=self._enqueued,
            dequeued=self._dequeued,
            expired=self._expired,
            capacity_rejected=self._capacity_rejected,
        )

    @property
    def capacity(self) -> float:
        """Maximum queue capacity."""
        return self._capacity

    def set_clock(self, clock_func: Callable[[], Instant]) -> None:
        """Set the clock function for expiration checking."""
        self._clock_func = clock_func

    def _now(self) -> Instant | None:
        """Get current time, or None if no clock."""
        if self._clock_func is None:
            return None
        return self._clock_func()

    def push(self, item: T) -> bool:
        """Add item to queue with its deadline.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if capacity exceeded.
        """
        if len(self._heap) >= self._capacity:
            self._capacity_rejected += 1
            return False

        deadline = self._get_deadline(item)
        entry = _DeadlineEntry(
            deadline_ns=deadline.nanoseconds,
            insert_order=self._insert_counter,
            item=item,
            deadline=deadline,
        )
        self._insert_counter += 1

        heapq.heappush(self._heap, entry)
        self._enqueued += 1
        return True

    def pop(self) -> Optional[T]:
        """Remove and return the item with earliest deadline.

        Automatically drops expired items until finding a valid one.

        Returns:
            The next non-expired item, or None if all expired/empty.
        """
        now = self._now()

        while self._heap:
            entry = heapq.heappop(self._heap)

            # Check if expired
            if now is not None and entry.deadline < now:
                self._expired += 1
                continue

            self._dequeued += 1
            return entry.item

        return None

    def peek(self) -> Optional[T]:
        """Return the next non-expired item without removing it.

        Note: Does not remove expired items (use pop for that).
        """
        now = self._now()

        for entry in self._heap:
            if now is None or entry.deadline >= now:
                return entry.item

        return None

    def purge_expired(self) -> int:
        """Remove all expired items from the queue.

        Returns:
            Number of items removed.
        """
        now = self._now()
        if now is None:
            return 0

        removed = 0
        new_heap = []

        for entry in self._heap:
            if entry.deadline >= now:
                new_heap.append(entry)
            else:
                removed += 1
                self._expired += 1

        if removed > 0:
            heapq.heapify(new_heap)
            self._heap = new_heap

        return removed

    def is_empty(self) -> bool:
        """Return True if no items in queue.

        Note: May include expired items. Use pop() to skip expired.
        """
        return len(self._heap) == 0

    def __len__(self) -> int:
        """Return number of items in queue (including potentially expired)."""
        return len(self._heap)

    def count_expired(self) -> int:
        """Count currently expired items without removing them."""
        now = self._now()
        if now is None:
            return 0

        return sum(1 for entry in self._heap if entry.deadline < now)

    def count_valid(self) -> int:
        """Count non-expired items."""
        return len(self._heap) - self.count_expired()
