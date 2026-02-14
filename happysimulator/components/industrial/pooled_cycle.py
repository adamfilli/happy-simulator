"""Pool of identical units with automatic fixed-duration release.

PooledCycleResource models a pool of N identical units (washing machines,
ride seats, rental cars) where each use is a fixed-duration cycle. Units
are automatically returned to the pool after the cycle completes.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PooledCycleStats:
    """Snapshot of pooled cycle resource statistics."""

    pool_size: int
    available: int
    active: int
    queued: int
    completed: int
    rejected: int
    utilization: float


class PooledCycleResource(Entity):
    """Entity modeling a pool of identical units with fixed cycle times.

    Each unit is discrete: when an event arrives, if a unit is available
    it begins a fixed-duration cycle. After ``cycle_time`` seconds the
    unit is released back to the pool and the event is forwarded downstream.
    If no unit is available, the event is queued (or rejected if at capacity).

    Args:
        name: Identifier for logging.
        pool_size: Number of units in the pool.
        cycle_time: Duration of each use cycle in seconds.
        downstream: Entity to forward completed items to (optional).
        queue_capacity: Maximum queue size (0 = unlimited).
    """

    def __init__(
        self,
        name: str,
        pool_size: int,
        cycle_time: float,
        downstream: Entity | None = None,
        queue_capacity: int = 0,
    ):
        if pool_size <= 0:
            raise ValueError(f"pool_size must be > 0, got {pool_size}")
        if cycle_time < 0:
            raise ValueError(f"cycle_time must be >= 0, got {cycle_time}")
        super().__init__(name)
        self.pool_size = pool_size
        self.cycle_time = cycle_time
        self.downstream = downstream
        self._queue_capacity = queue_capacity

        self._available = pool_size
        self._active = 0
        self._queue: deque[Event] = deque()
        self._completed = 0
        self._rejected = 0

    @property
    def available(self) -> int:
        return self._available

    @property
    def active(self) -> int:
        return self._active

    @property
    def queued(self) -> int:
        return len(self._queue)

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def rejected(self) -> int:
        return self._rejected

    @property
    def utilization(self) -> float:
        if self.pool_size == 0:
            return 0.0
        return self._active / self.pool_size

    @property
    def stats(self) -> PooledCycleStats:
        return PooledCycleStats(
            pool_size=self.pool_size,
            available=self._available,
            active=self._active,
            queued=len(self._queue),
            completed=self._completed,
            rejected=self._rejected,
            utilization=self.utilization,
        )

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event]:
        if self._available > 0:
            return self._start_cycle(event)

        # No unit available — try to queue
        if self._queue_capacity > 0 and len(self._queue) >= self._queue_capacity:
            self._rejected += 1
            logger.debug(
                "[%s] Rejected (queue full: %d/%d)",
                self.name, len(self._queue), self._queue_capacity,
            )
            return []

        self._queue.append(event)
        logger.debug(
            "[%s] Queued (no units available, queue depth=%d)",
            self.name, len(self._queue),
        )
        return []

    def _start_cycle(self, event: Event) -> Generator[float, None, list[Event]]:
        self._available -= 1
        self._active += 1

        try:
            yield self.cycle_time
        finally:
            self._active -= 1
            self._available += 1

        self._completed += 1

        results: list[Event] = []
        if self.downstream is not None:
            results.append(
                Event(
                    time=self.now,
                    event_type=event.event_type,
                    target=self.downstream,
                    context=event.context,
                )
            )

        # Try to dequeue next waiting item
        if self._queue and self._available > 0:
            next_event = self._queue.popleft()
            # Schedule dequeued item for immediate processing
            results.append(
                Event(
                    time=self.now,
                    event_type=next_event.event_type,
                    target=self,
                    context=next_event.context,
                )
            )

        return results
