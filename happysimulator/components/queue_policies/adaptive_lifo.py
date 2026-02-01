"""Adaptive LIFO queue implementation.

Provides a queue that switches between FIFO and LIFO based on congestion.
Under normal load, FIFO provides fairness. Under congestion, LIFO provides
better latency for recent requests.

Example:
    from happysimulator.components.queue_policies import AdaptiveLIFO

    queue = AdaptiveLIFO(
        congestion_threshold=10,  # Switch to LIFO above 10 items
        capacity=100,
    )
"""

from collections import deque
from dataclasses import dataclass
from typing import TypeVar, Optional

from happysimulator.components.queue_policy import QueuePolicy

T = TypeVar("T")


@dataclass
class AdaptiveLIFOStats:
    """Statistics tracked by AdaptiveLIFO."""

    enqueued: int = 0
    dequeued_fifo: int = 0  # Dequeued in FIFO mode
    dequeued_lifo: int = 0  # Dequeued in LIFO mode
    capacity_rejected: int = 0
    mode_switches: int = 0  # Times switched between FIFO and LIFO


class AdaptiveLIFO(QueuePolicy[T]):
    """Adaptive queue that switches between FIFO and LIFO based on congestion.

    When queue depth is below the congestion threshold, items are dequeued
    in FIFO order (fair to all requests). When congestion exceeds the
    threshold, switches to LIFO to prioritize recent requests.

    The rationale is that under congestion, older requests have likely
    already timed out from the client's perspective, so processing newer
    requests provides better overall user experience.

    Attributes:
        congestion_threshold: Queue depth to switch to LIFO mode.
        capacity: Maximum queue capacity.
    """

    def __init__(
        self,
        congestion_threshold: int,
        capacity: int | None = None,
    ):
        """Initialize the adaptive queue.

        Args:
            congestion_threshold: Queue depth at which to switch to LIFO.
            capacity: Maximum queue capacity (None = unlimited).

        Raises:
            ValueError: If parameters are invalid.
        """
        if congestion_threshold < 1:
            raise ValueError(f"congestion_threshold must be >= 1, got {congestion_threshold}")
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")

        self._congestion_threshold = congestion_threshold
        self._capacity = float("inf") if capacity is None else capacity

        # Queue storage
        self._queue: deque[T] = deque()

        # Track current mode for statistics
        self._was_congested = False

        # Statistics
        self.stats = AdaptiveLIFOStats()

    @property
    def congestion_threshold(self) -> int:
        """Queue depth at which to switch to LIFO."""
        return self._congestion_threshold

    @property
    def capacity(self) -> float:
        """Maximum queue capacity."""
        return self._capacity

    @property
    def is_congested(self) -> bool:
        """Whether queue is currently in congested (LIFO) mode."""
        return len(self._queue) >= self._congestion_threshold

    @property
    def mode(self) -> str:
        """Current queue mode: 'FIFO' or 'LIFO'."""
        return "LIFO" if self.is_congested else "FIFO"

    def push(self, item: T) -> bool:
        """Add item to queue.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if capacity exceeded.
        """
        if len(self._queue) >= self._capacity:
            self.stats.capacity_rejected += 1
            return False

        self._queue.append(item)
        self.stats.enqueued += 1
        return True

    def pop(self) -> Optional[T]:
        """Remove and return the next item.

        Uses FIFO when below congestion threshold, LIFO when above.

        Returns:
            The next item, or None if empty.
        """
        if not self._queue:
            return None

        is_congested = self.is_congested

        # Track mode switches
        if is_congested != self._was_congested:
            self.stats.mode_switches += 1
            self._was_congested = is_congested

        if is_congested:
            # LIFO: pop from right (most recent)
            item = self._queue.pop()
            self.stats.dequeued_lifo += 1
        else:
            # FIFO: pop from left (oldest)
            item = self._queue.popleft()
            self.stats.dequeued_fifo += 1

        return item

    def peek(self) -> Optional[T]:
        """Return the next item without removing it.

        Returns item based on current mode (FIFO or LIFO).
        """
        if not self._queue:
            return None

        if self.is_congested:
            # LIFO: peek at right
            return self._queue[-1]
        else:
            # FIFO: peek at left
            return self._queue[0]

    def is_empty(self) -> bool:
        """Return True if no items in queue."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Return number of items in queue."""
        return len(self._queue)
