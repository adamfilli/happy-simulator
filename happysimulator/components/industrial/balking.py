"""Queue policy decorator that rejects arrivals based on queue depth.

BalkingQueue wraps any inner QueuePolicy. On ``push()``, it checks the
current depth against a threshold and probabilistically rejects items,
simulating customers who see a long line and leave.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, TypeVar

from happysimulator.components.queue_policy import QueuePolicy

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BalkingQueue(QueuePolicy[T]):
    """Queue policy that probabilistically rejects items when depth exceeds a threshold.

    When the inner queue's depth is at or above ``balk_threshold``, new items
    are rejected with probability ``balk_probability``. Items below the
    threshold are always accepted (subject to inner policy capacity).

    Args:
        inner: The underlying queue policy to delegate to.
        balk_threshold: Queue depth at which balking begins.
        balk_probability: Probability of balking when at/above threshold (0.0-1.0).
    """

    def __init__(
        self,
        inner: QueuePolicy[T],
        balk_threshold: int = 5,
        balk_probability: float = 1.0,
    ):
        if not (0.0 <= balk_probability <= 1.0):
            raise ValueError(f"balk_probability must be in [0.0, 1.0], got {balk_probability}")
        self._inner = inner
        self.balk_threshold = balk_threshold
        self.balk_probability = balk_probability
        self.balked: int = 0

    @property
    def capacity(self) -> float:
        return self._inner.capacity

    @property
    def inner(self) -> QueuePolicy[T]:
        return self._inner

    def push(self, item: T) -> bool:
        if len(self._inner) >= self.balk_threshold:
            if random.random() < self.balk_probability:
                self.balked += 1
                logger.debug(
                    "BalkingQueue: item balked (depth=%d >= threshold=%d)",
                    len(self._inner),
                    self.balk_threshold,
                )
                return False
        return self._inner.push(item)

    def pop(self) -> Optional[T]:
        return self._inner.pop()

    def peek(self) -> Optional[T]:
        return self._inner.peek()

    def is_empty(self) -> bool:
        return self._inner.is_empty()

    def __len__(self) -> int:
        return len(self._inner)
