"""Write policy implementations for cached stores.

Defines how writes propagate between cache and backing store.

Example:
    from happysimulator.components.datastore import WriteThrough, WriteBack

    # Immediate consistency
    policy = WriteThrough()

    # Better performance, eventual consistency
    policy = WriteBack(flush_interval=1.0, max_dirty=100)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class WritePolicy(ABC):
    """Abstract base class for write policies.

    Write policies determine how writes to a cached store propagate
    to the backing store.
    """

    @abstractmethod
    def should_write_through(self) -> bool:
        """Whether to write to backing store immediately.

        Returns:
            True if writes should go to backing store immediately.
        """

    @abstractmethod
    def on_write(self, key: str, value: Any) -> None:
        """Called when a key is written.

        Args:
            key: The written key.
            value: The written value.
        """

    @abstractmethod
    def should_flush(self) -> bool:
        """Whether pending writes should be flushed.

        Returns:
            True if flush should be triggered.
        """

    @abstractmethod
    def get_keys_to_flush(self) -> list[str]:
        """Get keys that need to be flushed.

        Returns:
            List of keys to write to backing store.
        """

    @abstractmethod
    def on_flush(self, keys: list[str]) -> None:
        """Called after keys have been flushed.

        Args:
            keys: Keys that were flushed.
        """


@dataclass
class WriteThrough(WritePolicy):
    """Write-through policy.

    Writes go to both cache and backing store synchronously.
    Provides strong consistency but higher latency.
    """

    def should_write_through(self) -> bool:
        """Always write through."""
        return True

    def on_write(self, key: str, value: Any) -> None:
        """No tracking needed for write-through."""

    def should_flush(self) -> bool:
        """Never needs flush - writes are immediate."""
        return False

    def get_keys_to_flush(self) -> list[str]:
        """No pending flushes."""
        return []

    def on_flush(self, keys: list[str]) -> None:
        """No action needed."""


class WriteBack(WritePolicy):
    """Write-back (write-behind) policy.

    Writes go to cache only, flushed to backing store later.
    Provides better performance but eventual consistency.

    Flush is triggered by:
    - Number of dirty entries exceeding max_dirty
    - Time since last flush exceeding flush_interval

    Attributes:
        flush_interval: Maximum time between flushes in seconds.
        max_dirty: Maximum dirty entries before forced flush.
    """

    def __init__(
        self,
        flush_interval: float = 1.0,
        max_dirty: int = 100,
    ):
        """Initialize write-back policy.

        Args:
            flush_interval: Maximum seconds between flushes.
            max_dirty: Maximum dirty entries before flush.

        Raises:
            ValueError: If parameters are invalid.
        """
        if flush_interval <= 0:
            raise ValueError(f"flush_interval must be > 0, got {flush_interval}")
        if max_dirty < 1:
            raise ValueError(f"max_dirty must be >= 1, got {max_dirty}")

        self._flush_interval = flush_interval
        self._max_dirty = max_dirty
        self._dirty_keys: set[str] = set()
        self._last_flush_time: float = 0.0

    @property
    def flush_interval(self) -> float:
        """Maximum time between flushes."""
        return self._flush_interval

    @property
    def max_dirty(self) -> int:
        """Maximum dirty entries before flush."""
        return self._max_dirty

    @property
    def dirty_count(self) -> int:
        """Number of dirty entries."""
        return len(self._dirty_keys)

    def should_write_through(self) -> bool:
        """Don't write through - buffer in cache."""
        return False

    def on_write(self, key: str, value: Any) -> None:
        """Track dirty key."""
        self._dirty_keys.add(key)

    def should_flush(self) -> bool:
        """Check if flush is needed based on dirty count."""
        return len(self._dirty_keys) >= self._max_dirty

    def get_keys_to_flush(self) -> list[str]:
        """Get all dirty keys."""
        return list(self._dirty_keys)

    def on_flush(self, keys: list[str]) -> None:
        """Remove flushed keys from dirty set."""
        for key in keys:
            self._dirty_keys.discard(key)


class WriteAround(WritePolicy):
    """Write-around policy.

    Writes go directly to backing store, bypassing cache.
    Good for write-heavy workloads where written data isn't
    immediately re-read.

    On write, the key is invalidated from cache if present.
    """

    def __init__(self):
        """Initialize write-around policy."""
        self._invalidated_keys: list[str] = []

    def should_write_through(self) -> bool:
        """Write directly to backing store."""
        return True

    def on_write(self, key: str, value: Any) -> None:
        """Track key for cache invalidation."""
        self._invalidated_keys.append(key)

    def should_flush(self) -> bool:
        """No buffered writes."""
        return False

    def get_keys_to_flush(self) -> list[str]:
        """No pending flushes."""
        return []

    def on_flush(self, keys: list[str]) -> None:
        """No action needed."""

    def get_keys_to_invalidate(self) -> list[str]:
        """Get keys to remove from cache.

        Returns:
            Keys that should be invalidated from cache.
        """
        keys = self._invalidated_keys
        self._invalidated_keys = []
        return keys
