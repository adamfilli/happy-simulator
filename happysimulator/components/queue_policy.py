"""Pluggable queue ordering policies for bounded buffers.

Defines the QueuePolicy interface and three implementations:
- FIFOQueue: First-In-First-Out (standard queue behavior)
- LIFOQueue: Last-In-First-Out (stack behavior)
- PriorityQueue: Ordered by priority value (lowest first)

All policies support optional capacity limits. Items exceeding capacity
are rejected at push time. Subclass QueuePolicy to implement custom
ordering strategies.
"""

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T")


class QueuePolicy[T](ABC):
    """
    Abstract queue policy defining how items are stored and retrieved.

    Implementations control ordering semantics (FIFO, LIFO, priority, etc.)
    and optionally enforce capacity limits.
    """

    @property
    @abstractmethod
    def capacity(self) -> float:
        """Return the maximum capacity of this queue."""

    @abstractmethod
    def push(self, item: T) -> bool:
        """
        Add item to queue.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if rejected (e.g., capacity exceeded).
        """

    @abstractmethod
    def pop(self) -> T | None:
        """
        Remove and return the next item.

        Returns:
            The next item according to the policy, or None if empty.
        """

    @abstractmethod
    def peek(self) -> T | None:
        """
        Return the next item without removing it.

        Returns:
            The next item according to the policy, or None if empty.
        """

    @abstractmethod
    def is_empty(self) -> bool:
        """Return True if no items in queue."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of items in queue."""


class FIFOQueue(QueuePolicy[T]):
    """
    First-In-First-Out queue with optional capacity limit.

    Items are processed in arrival order. This is the default policy
    for most server-like entities.

    Args:
        capacity: Maximum number of items. Defaults to unlimited.
    """

    def __init__(self, capacity: float = float("inf")):
        self._capacity = capacity
        self._queue: deque[T] = deque()

    @property
    def capacity(self) -> float:
        return self._capacity

    def push(self, item: T) -> bool:
        if len(self._queue) >= self.capacity:
            return False
        self._queue.append(item)
        return True

    def pop(self) -> T | None:
        if not self._queue:
            return None
        return self._queue.popleft()

    def peek(self) -> T | None:
        if not self._queue:
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


class LIFOQueue(QueuePolicy[T]):
    """
    Last-In-First-Out (stack) queue with optional capacity limit.

    Items are processed in reverse arrival order. Useful for modeling
    systems where recent items have priority (e.g., cache eviction).

    Args:
        capacity: Maximum number of items. Defaults to unlimited.
    """

    def __init__(self, capacity: float = float("inf")):
        self._capacity = capacity
        self._queue: deque[T] = deque()

    @property
    def capacity(self) -> float:
        return self._capacity

    def push(self, item: T) -> bool:
        if len(self._queue) >= self.capacity:
            return False
        self._queue.append(item)
        return True

    def pop(self) -> T | None:
        if not self._queue:
            return None
        return self._queue.pop()  # LIFO: pop from right

    def peek(self) -> T | None:
        if not self._queue:
            return None
        return self._queue[-1]

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)


# --- Priority Queue Support ---


@runtime_checkable
class Prioritized(Protocol):
    """
    Protocol for items that know their own priority.

    Items implementing this protocol can be used directly with PriorityQueue
    without providing a key function.

    Lower priority values are dequeued first (like Unix nice values).

    Example:
        @dataclass
        class Request:
            user_id: str
            is_premium: bool

            @property
            def priority(self) -> float:
                return 0.0 if self.is_premium else 1.0
    """

    @property
    def priority(self) -> float:
        """Return priority value. Lower values = higher priority."""
        ...


@dataclass(order=True)
class _PriorityEntry[T]:
    """
    Internal wrapper for heap entries.

    Maintains insertion order for stable sorting when priorities are equal.
    The item itself is excluded from comparison to avoid requiring items
    to be comparable.
    """

    priority: float
    insert_order: int
    item: T = field(compare=False)


class PriorityQueue(QueuePolicy[T]):
    """
    Priority queue with optional capacity limit.

    Items are dequeued in priority order (lowest priority value first).
    Items with equal priority are dequeued in FIFO order (stable).

    Priority can be determined in two ways:
    1. Provide a `key` function that extracts priority from items
    2. Use items that implement the `Prioritized` protocol (have a `priority` property)

    If neither is provided, items must be comparable (have `__lt__`).

    Args:
        capacity: Maximum number of items. Defaults to unlimited.
        key: Optional function to extract priority from items.
             Lower values = higher priority (dequeued first).

    Examples:
        # Using a key function (most flexible)
        queue = PriorityQueue(key=lambda req: req.deadline)

        # Using Prioritized protocol
        @dataclass
        class Task:
            name: str
            @property
            def priority(self) -> float:
                return self._compute_priority()

        queue = PriorityQueue()  # Will use task.priority automatically

        # Using comparable items directly
        queue = PriorityQueue()
        queue.push(3)  # ints are comparable
        queue.push(1)
        queue.pop()  # returns 1
    """

    def __init__(self, capacity: float = float("inf"), key: Callable[[T], float] | None = None):
        self._capacity = capacity
        self._key = key
        self._heap: list[_PriorityEntry[T]] = []
        self._insert_counter = 0

    @property
    def capacity(self) -> float:
        return self._capacity

    def _get_priority(self, item: T) -> float:
        """Extract priority from item using configured strategy."""
        if self._key is not None:
            return self._key(item)
        if isinstance(item, Prioritized):
            return item.priority
        # Fallback: assume item is directly comparable (will fail if not)
        return float(item)  # type: ignore[arg-type]

    def push(self, item: T) -> bool:
        if len(self._heap) >= self.capacity:
            return False

        priority = self._get_priority(item)
        entry = _PriorityEntry(priority, self._insert_counter, item)
        self._insert_counter += 1
        heapq.heappush(self._heap, entry)
        return True

    def pop(self) -> T | None:
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        return entry.item

    def peek(self) -> T | None:
        if not self._heap:
            return None
        return self._heap[0].item

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def __len__(self) -> int:
        return len(self._heap)
