"""Key-Value Store implementation.

Provides an in-memory key-value store with configurable latency for
simulating database or cache behavior.

Example:
    from happysimulator.components.datastore import KVStore

    store = KVStore(
        name="database",
        read_latency=0.001,   # 1ms reads
        write_latency=0.005,  # 5ms writes
    )

    def handle_event(self, event):
        # Read
        value = yield from store.get("user:123")

        # Write
        yield from store.put("user:123", {"name": "Alice"})
"""

from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


@dataclass
class KVStoreStats:
    """Statistics tracked by KVStore."""

    reads: int = 0
    writes: int = 0
    deletes: int = 0
    hits: int = 0  # Key existed on read
    misses: int = 0  # Key not found on read
    evictions: int = 0  # Evicted due to capacity


class KVStore(Entity):
    """In-memory key-value store with latency simulation.

    Stores key-value pairs with configurable read and write latencies.
    Optionally limits capacity and evicts oldest entries when full.

    Attributes:
        name: Entity name for identification.
        read_latency: Time in seconds for read operations.
        write_latency: Time in seconds for write operations.
        capacity: Maximum number of keys (None = unlimited).
    """

    def __init__(
        self,
        name: str,
        read_latency: float = 0.001,
        write_latency: float = 0.005,
        delete_latency: float | None = None,
        capacity: int | None = None,
    ):
        """Initialize the key-value store.

        Args:
            name: Name for this store entity.
            read_latency: Latency for get operations in seconds.
            write_latency: Latency for put operations in seconds.
            delete_latency: Latency for delete operations (defaults to write_latency).
            capacity: Maximum number of keys (None = unlimited).

        Raises:
            ValueError: If parameters are invalid.
        """
        if read_latency < 0:
            raise ValueError(f"read_latency must be >= 0, got {read_latency}")
        if write_latency < 0:
            raise ValueError(f"write_latency must be >= 0, got {write_latency}")
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")

        super().__init__(name)
        self._read_latency = read_latency
        self._write_latency = write_latency
        self._delete_latency = delete_latency if delete_latency is not None else write_latency
        self._capacity = capacity

        # Storage
        self._data: dict[str, Any] = {}
        self._insertion_order: list[str] = []  # For FIFO eviction

        # Statistics
        self.stats = KVStoreStats()

    @property
    def read_latency(self) -> float:
        """Latency for read operations in seconds."""
        return self._read_latency

    @property
    def write_latency(self) -> float:
        """Latency for write operations in seconds."""
        return self._write_latency

    @property
    def capacity(self) -> int | None:
        """Maximum number of keys."""
        return self._capacity

    @property
    def size(self) -> int:
        """Current number of stored keys."""
        return len(self._data)

    def get(self, key: str) -> Generator[float, None, Optional[Any]]:
        """Get a value by key.

        Args:
            key: The key to look up.

        Yields:
            Read latency delay.

        Returns:
            The value if found, None otherwise.
        """
        yield self._read_latency

        self.stats.reads += 1

        if key in self._data:
            self.stats.hits += 1
            return self._data[key]
        else:
            self.stats.misses += 1
            return None

    def get_sync(self, key: str) -> Optional[Any]:
        """Get a value synchronously (no latency, for internal use).

        Args:
            key: The key to look up.

        Returns:
            The value if found, None otherwise.
        """
        return self._data.get(key)

    def put(self, key: str, value: Any) -> Generator[float, None, None]:
        """Store a value.

        Args:
            key: The key to store under.
            value: The value to store.

        Yields:
            Write latency delay.
        """
        yield self._write_latency

        self.stats.writes += 1

        # Check capacity and evict if needed
        if self._capacity is not None and key not in self._data:
            while len(self._data) >= self._capacity:
                self._evict_oldest()

        # Store value
        if key not in self._data:
            self._insertion_order.append(key)
        self._data[key] = value

    def put_sync(self, key: str, value: Any) -> None:
        """Store a value synchronously (no latency, for internal use).

        Args:
            key: The key to store under.
            value: The value to store.
        """
        if self._capacity is not None and key not in self._data:
            while len(self._data) >= self._capacity:
                self._evict_oldest()

        if key not in self._data:
            self._insertion_order.append(key)
        self._data[key] = value

    def delete(self, key: str) -> Generator[float, None, bool]:
        """Delete a key.

        Args:
            key: The key to delete.

        Yields:
            Delete latency delay.

        Returns:
            True if key existed, False otherwise.
        """
        yield self._delete_latency

        self.stats.deletes += 1

        if key in self._data:
            del self._data[key]
            self._insertion_order.remove(key)
            return True
        return False

    def delete_sync(self, key: str) -> bool:
        """Delete a key synchronously (no latency, for internal use).

        Args:
            key: The key to delete.

        Returns:
            True if key existed, False otherwise.
        """
        if key in self._data:
            del self._data[key]
            self._insertion_order.remove(key)
            return True
        return False

    def contains(self, key: str) -> bool:
        """Check if a key exists (no latency).

        Args:
            key: The key to check.

        Returns:
            True if key exists.
        """
        return key in self._data

    def keys(self) -> list[str]:
        """Get all keys (no latency).

        Returns:
            List of all keys.
        """
        return list(self._data.keys())

    def clear(self) -> None:
        """Clear all data (no latency)."""
        self._data.clear()
        self._insertion_order.clear()

    def _evict_oldest(self) -> None:
        """Evict the oldest entry (FIFO)."""
        if self._insertion_order:
            oldest_key = self._insertion_order.pop(0)
            del self._data[oldest_key]
            self.stats.evictions += 1

    def handle_event(self, event: Event) -> None:
        """KVStore can handle events for get/put operations."""
        pass
