"""CoDel (Controlled Delay) queue management algorithm.

CoDel is an active queue management algorithm that controls queue delay
by dropping packets when the minimum queue delay over an interval exceeds
a target threshold.

Unlike RED, CoDel doesn't require tuning based on queue size or bandwidth.
It adapts automatically to changing network conditions.

Example:
    from happysimulator.components.queue_policies import CoDelQueue

    queue = CoDelQueue(
        target_delay=0.005,  # 5ms target
        interval=0.100,      # 100ms measurement interval
    )
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import TypeVar, Optional, Generic

from happysimulator.components.queue_policy import QueuePolicy
from happysimulator.core.temporal import Instant, Duration

T = TypeVar("T")


@dataclass(frozen=True)
class CoDelStats:
    """Statistics tracked by CoDelQueue."""

    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    capacity_rejected: int = 0
    drop_intervals: int = 0  # Number of times we entered dropping state


@dataclass
class _QueuedItem(Generic[T]):
    """Item wrapper that tracks enqueue time."""

    item: T
    enqueue_time: Instant


class CoDelQueue(QueuePolicy[T]):
    """CoDel (Controlled Delay) active queue management.

    CoDel monitors the minimum queue delay over an interval. When this
    minimum delay exceeds the target, it begins dropping packets. The
    drop rate increases if congestion persists.

    Key properties:
    - Controls delay, not queue length
    - Self-tuning (no configuration based on bandwidth/queue size)
    - Fair to bursty traffic
    - Distinguishes between good queue (temporary burst) and bad queue
      (persistent congestion)

    Attributes:
        target_delay: Target minimum delay in seconds.
        interval: Measurement interval in seconds.
        capacity: Maximum queue size.
    """

    def __init__(
        self,
        target_delay: float = 0.005,
        interval: float = 0.100,
        capacity: int | None = None,
        clock_func: callable = None,
    ):
        """Initialize the CoDel queue.

        Args:
            target_delay: Target delay threshold in seconds (default 5ms).
            interval: Measurement interval in seconds (default 100ms).
            capacity: Maximum queue capacity (None = unlimited).
            clock_func: Function returning current time as Instant.
                       Required for time-based operations.

        Raises:
            ValueError: If parameters are invalid.
        """
        if target_delay <= 0:
            raise ValueError(f"target_delay must be > 0, got {target_delay}")
        if interval <= 0:
            raise ValueError(f"interval must be > 0, got {interval}")
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")

        self._target_delay = target_delay
        self._interval = interval
        self._capacity = float("inf") if capacity is None else capacity
        self._clock_func = clock_func

        # Queue storage
        self._queue: deque[_QueuedItem[T]] = deque()

        # CoDel state
        self._first_above_time: Instant | None = None  # When delay first exceeded target
        self._drop_next: Instant | None = None  # Next scheduled drop time
        self._count = 0  # Drops since entering drop state
        self._dropping = False  # Currently in dropping state
        self._last_count = 0  # Count from last dropping interval

        # Statistics
        self._enqueued = 0
        self._dequeued = 0
        self._dropped = 0
        self._capacity_rejected = 0
        self._drop_intervals = 0

    @property
    def stats(self) -> CoDelStats:
        """Return a frozen snapshot of queue statistics."""
        return CoDelStats(
            enqueued=self._enqueued,
            dequeued=self._dequeued,
            dropped=self._dropped,
            capacity_rejected=self._capacity_rejected,
            drop_intervals=self._drop_intervals,
        )

    @property
    def target_delay(self) -> float:
        """Target delay threshold in seconds."""
        return self._target_delay

    @property
    def interval(self) -> float:
        """Measurement interval in seconds."""
        return self._interval

    @property
    def capacity(self) -> float:
        """Maximum queue capacity."""
        return self._capacity

    @property
    def dropping(self) -> bool:
        """Whether queue is currently in dropping state."""
        return self._dropping

    def set_clock(self, clock_func: callable) -> None:
        """Set the clock function for time-based operations."""
        self._clock_func = clock_func

    def _now(self) -> Instant:
        """Get current time."""
        if self._clock_func is None:
            raise RuntimeError("CoDelQueue requires a clock function to be set")
        return self._clock_func()

    def push(self, item: T) -> bool:
        """Add item to queue with timestamp.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if capacity exceeded.
        """
        if len(self._queue) >= self._capacity:
            self._capacity_rejected += 1
            return False

        queued = _QueuedItem(item=item, enqueue_time=self._now())
        self._queue.append(queued)
        self._enqueued += 1
        return True

    def pop(self) -> Optional[T]:
        """Remove and return the next item, applying CoDel algorithm.

        May drop items if delay exceeds target for too long.

        Returns:
            The next item, or None if empty.
        """
        if not self._queue:
            return None

        now = self._now()

        # Get item and calculate sojourn time
        queued = self._queue.popleft()
        sojourn_time = (now - queued.enqueue_time).to_seconds()

        # Apply CoDel algorithm
        self._codel_dequeue(now, sojourn_time)

        self._dequeued += 1
        return queued.item

    def _codel_dequeue(self, now: Instant, sojourn_time: float) -> None:
        """Apply CoDel algorithm on dequeue.

        Updates state based on current sojourn time.
        """
        ok_to_drop = self._should_mark_or_drop(now, sojourn_time)

        if self._dropping:
            if not ok_to_drop:
                # Delay is acceptable, exit dropping state
                self._dropping = False
            elif self._drop_next is not None and now >= self._drop_next:
                # Time to drop
                while self._drop_next is not None and now >= self._drop_next and self._queue:
                    self._drop()
                    self._count += 1
                    self._drop_next = self._control_law(now)
        elif ok_to_drop:
            # Enter dropping state
            self._drop()
            self._dropping = True
            self._drop_intervals += 1

            # If we were recently dropping, start faster
            delta = self._count - self._last_count
            self._count = 1 if delta < 1 or (now - self._drop_next).to_seconds() < self._interval else delta
            self._last_count = self._count
            self._drop_next = self._control_law(now)

    def _should_mark_or_drop(self, now: Instant, sojourn_time: float) -> bool:
        """Determine if current packet should be marked/dropped."""
        if sojourn_time < self._target_delay or len(self._queue) == 0:
            # Good state: delay is acceptable or queue nearly empty
            self._first_above_time = None
            return False

        if self._first_above_time is None:
            # Just crossed threshold, start timer
            self._first_above_time = now + Duration.from_seconds(self._interval)
            return False

        # Check if we've been above threshold for entire interval
        return now >= self._first_above_time

    def _control_law(self, now: Instant) -> Instant:
        """Calculate next drop time using inverse sqrt law."""
        # CoDel increases drop rate as sqrt(count)
        drop_interval = self._interval / math.sqrt(self._count)
        return now + Duration.from_seconds(drop_interval)

    def _drop(self) -> None:
        """Drop the head-of-line packet."""
        if self._queue:
            self._queue.popleft()
            self._dropped += 1

    def peek(self) -> Optional[T]:
        """Return the next item without removing it."""
        if not self._queue:
            return None
        return self._queue[0].item

    def is_empty(self) -> bool:
        """Return True if no items in queue."""
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Return number of items in queue."""
        return len(self._queue)
