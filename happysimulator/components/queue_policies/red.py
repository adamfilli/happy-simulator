"""RED (Random Early Detection) queue management algorithm.

RED is an active queue management algorithm that probabilistically drops
packets before the queue is full, signaling congestion to senders early.

Example:
    from happysimulator.components.queue_policies import REDQueue

    queue = REDQueue(
        min_threshold=10,   # Start dropping at 10 items
        max_threshold=30,   # 100% drop at 30 items
        max_probability=0.1,
    )
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import TypeVar

from happysimulator.components.queue_policy import QueuePolicy

T = TypeVar("T")


@dataclass(frozen=True)
class REDStats:
    """Statistics tracked by REDQueue."""

    enqueued: int = 0
    dequeued: int = 0
    dropped_probabilistic: int = 0  # Dropped by RED algorithm
    dropped_forced: int = 0  # Dropped because queue full
    capacity_rejected: int = 0


class REDQueue(QueuePolicy[T]):
    """Random Early Detection queue management.

    RED maintains an exponentially-weighted moving average of queue size
    and drops packets probabilistically when average queue exceeds thresholds.

    Drop probability:
    - avg_queue < min_threshold: 0% (no drops)
    - min_threshold <= avg_queue < max_threshold: linear 0% to max_probability
    - avg_queue >= max_threshold: 100% (drop all)

    The averaging smooths out burst traffic, preventing unnecessary drops
    during temporary spikes.

    Attributes:
        min_threshold: Queue length to start probabilistic drops.
        max_threshold: Queue length for 100% drop rate.
        max_probability: Maximum drop probability (at max_threshold).
        capacity: Hard queue capacity limit.
    """

    def __init__(
        self,
        min_threshold: int,
        max_threshold: int,
        max_probability: float = 0.1,
        capacity: int | None = None,
        weight: float = 0.002,
    ):
        """Initialize the RED queue.

        Args:
            min_threshold: Minimum average queue length to start dropping.
            max_threshold: Average queue length for 100% drop rate.
            max_probability: Maximum drop probability (default 0.1 = 10%).
            capacity: Hard capacity limit. Defaults to 2 * max_threshold.
            weight: Weight for exponential moving average (default 0.002).

        Raises:
            ValueError: If parameters are invalid.
        """
        if min_threshold < 0:
            raise ValueError(f"min_threshold must be >= 0, got {min_threshold}")
        if max_threshold <= min_threshold:
            raise ValueError(
                f"max_threshold must be > min_threshold, got {max_threshold} <= {min_threshold}"
            )
        if not 0 < max_probability <= 1:
            raise ValueError(f"max_probability must be in (0, 1], got {max_probability}")
        if not 0 < weight < 1:
            raise ValueError(f"weight must be in (0, 1), got {weight}")

        self._min_threshold = min_threshold
        self._max_threshold = max_threshold
        self._max_probability = max_probability
        self._weight = weight

        # Default capacity is 2x max threshold
        if capacity is None:
            capacity = max_threshold * 2
        if capacity < max_threshold:
            raise ValueError(f"capacity must be >= max_threshold, got {capacity} < {max_threshold}")
        self._capacity = capacity

        # Queue storage
        self._queue: deque[T] = deque()

        # RED state
        self._avg_queue: float = 0.0  # Exponential moving average
        self._count_since_last_drop = 0  # Packets since last drop

        # Statistics
        self._enqueued = 0
        self._dequeued = 0
        self._dropped_probabilistic = 0
        self._dropped_forced = 0
        self._capacity_rejected = 0

    @property
    def stats(self) -> REDStats:
        """Return a frozen snapshot of queue statistics."""
        return REDStats(
            enqueued=self._enqueued,
            dequeued=self._dequeued,
            dropped_probabilistic=self._dropped_probabilistic,
            dropped_forced=self._dropped_forced,
            capacity_rejected=self._capacity_rejected,
        )

    @property
    def min_threshold(self) -> int:
        """Minimum queue length to start dropping."""
        return self._min_threshold

    @property
    def max_threshold(self) -> int:
        """Queue length for 100% drop rate."""
        return self._max_threshold

    @property
    def max_probability(self) -> float:
        """Maximum drop probability."""
        return self._max_probability

    @property
    def capacity(self) -> float:
        """Hard queue capacity limit."""
        return self._capacity

    @property
    def avg_queue_length(self) -> float:
        """Current exponential moving average of queue length."""
        return self._avg_queue

    def push(self, item: T) -> bool:
        """Add item to queue, applying RED algorithm.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if dropped.
        """
        # Update average queue length
        self._update_avg()

        # Check hard capacity
        if len(self._queue) >= self._capacity:
            self._capacity_rejected += 1
            return False

        # Apply RED algorithm
        drop_probability = self._calculate_drop_probability()

        if drop_probability > 0:
            # Adjust probability based on time since last drop
            # This prevents synchronized drops
            denominator = 1 - self._count_since_last_drop * drop_probability
            if denominator <= 0:
                # If denominator <= 0, probability should be 1.0
                adjusted_prob = 1.0
            else:
                adjusted_prob = drop_probability / denominator
                adjusted_prob = min(adjusted_prob, 1.0)

            if random.random() < adjusted_prob:
                # Drop this packet
                self._count_since_last_drop = 0
                if self._avg_queue >= self._max_threshold:
                    self._dropped_forced += 1
                else:
                    self._dropped_probabilistic += 1
                return False

        # Accept packet
        self._queue.append(item)
        self._count_since_last_drop += 1
        self._enqueued += 1
        return True

    def _update_avg(self) -> None:
        """Update exponential moving average of queue length."""
        current_len = len(self._queue)
        self._avg_queue = (1 - self._weight) * self._avg_queue + self._weight * current_len

    def _calculate_drop_probability(self) -> float:
        """Calculate drop probability based on average queue length."""
        if self._avg_queue < self._min_threshold:
            return 0.0

        if self._avg_queue >= self._max_threshold:
            return 1.0

        # Linear interpolation between min and max thresholds
        range_size = self._max_threshold - self._min_threshold
        position = (self._avg_queue - self._min_threshold) / range_size
        return position * self._max_probability

    def pop(self) -> T | None:
        """Remove and return the next item.

        Returns:
            The next item, or None if empty.
        """
        if not self._queue:
            return None

        item = self._queue.popleft()
        self._dequeued += 1
        return item

    def peek(self) -> T | None:
        """Return the next item without removing it."""
        if not self._queue:
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        """Return True if no items in queue."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Return number of items in queue."""
        return len(self._queue)
