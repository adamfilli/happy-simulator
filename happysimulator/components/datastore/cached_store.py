"""Cached Store implementation.

Provides a caching layer in front of a backing store with configurable
eviction policies and write strategies.

Example:
    from happysimulator.components.datastore import (
        KVStore, CachedStore, LRUEviction
    )

    backing = KVStore(name="db", read_latency=0.010)
    cache = CachedStore(
        name="cached_db",
        backing_store=backing,
        cache_capacity=1000,
        eviction_policy=LRUEviction(),
    )

    def handle_event(self, event):
        # Fast if cached, slow if miss
        value = yield from cache.get("user:123")
"""

from dataclasses import dataclass
from typing import Any, Generator, Optional

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.components.datastore.eviction_policies import CacheEvictionPolicy
from happysimulator.components.datastore.kv_store import KVStore


@dataclass
class CachedStoreStats:
    """Statistics tracked by CachedStore."""

    reads: int = 0
    writes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    writebacks: int = 0  # For write-back policy


class CachedStore(Entity):
    """Cache layer in front of a backing store.

    Provides fast access to frequently-used data by caching it in memory.
    Supports various eviction policies and write strategies.

    Attributes:
        name: Entity name for identification.
        cache_capacity: Maximum number of cached entries.
        hit_rate: Ratio of cache hits to total reads.
        miss_rate: Ratio of cache misses to total reads.
    """

    def __init__(
        self,
        name: str,
        backing_store: KVStore,
        cache_capacity: int,
        eviction_policy: CacheEvictionPolicy,
        cache_read_latency: float = 0.0001,
        write_through: bool = True,
    ):
        """Initialize the cached store.

        Args:
            name: Name for this cache entity.
            backing_store: The underlying storage to cache.
            cache_capacity: Maximum number of entries to cache.
            eviction_policy: Policy for selecting entries to evict.
            cache_read_latency: Latency for cache hits in seconds.
            write_through: If True, writes go to both cache and backing store.
                          If False, writes only go to cache (write-back).

        Raises:
            ValueError: If parameters are invalid.
        """
        if cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1, got {cache_capacity}")
        if cache_read_latency < 0:
            raise ValueError(f"cache_read_latency must be >= 0, got {cache_read_latency}")

        super().__init__(name)
        self._backing_store = backing_store
        self._cache_capacity = cache_capacity
        self._eviction_policy = eviction_policy
        self._cache_read_latency = cache_read_latency
        self._write_through = write_through

        # Cache storage
        self._cache: dict[str, Any] = {}
        self._dirty_keys: set[str] = set()  # For write-back

        # Statistics
        self.stats = CachedStoreStats()

    @property
    def backing_store(self) -> KVStore:
        """The underlying backing store."""
        return self._backing_store

    @property
    def cache_capacity(self) -> int:
        """Maximum number of cached entries."""
        return self._cache_capacity

    @property
    def cache_size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Ratio of cache hits to total reads."""
        if self.stats.reads == 0:
            return 0.0
        return self.stats.hits / self.stats.reads

    @property
    def miss_rate(self) -> float:
        """Ratio of cache misses to total reads."""
        if self.stats.reads == 0:
            return 0.0
        return self.stats.misses / self.stats.reads

    def get(self, key: str) -> Generator[float, None, Optional[Any]]:
        """Get a value, checking cache first.

        Args:
            key: The key to look up.

        Yields:
            Cache or backing store latency.

        Returns:
            The value if found, None otherwise.
        """
        self.stats.reads += 1

        # Check cache first
        if key in self._cache:
            self.stats.hits += 1
            self._eviction_policy.on_access(key)
            yield self._cache_read_latency
            return self._cache[key]

        # Cache miss - fetch from backing store
        self.stats.misses += 1
        value = yield from self._backing_store.get(key)

        if value is not None:
            # Cache the value
            self._cache_put(key, value)

        return value

    def put(self, key: str, value: Any) -> Generator[float, None, None]:
        """Store a value.

        With write-through, writes to both cache and backing store.
        With write-back, writes only to cache (must flush later).

        Args:
            key: The key to store under.
            value: The value to store.

        Yields:
            Write latency.
        """
        self.stats.writes += 1

        # Update cache
        self._cache_put(key, value)

        if self._write_through:
            # Write to backing store
            yield from self._backing_store.put(key, value)
        else:
            # Mark as dirty for later writeback
            self._dirty_keys.add(key)
            yield self._cache_read_latency  # Just cache write latency

    def delete(self, key: str) -> Generator[float, None, bool]:
        """Delete a key from cache and backing store.

        Args:
            key: The key to delete.

        Yields:
            Delete latency.

        Returns:
            True if key existed in either cache or backing store.
        """
        existed_in_cache = key in self._cache
        if existed_in_cache:
            self._cache_remove(key)

        existed_in_store = yield from self._backing_store.delete(key)
        return existed_in_cache or existed_in_store

    def invalidate(self, key: str) -> None:
        """Remove a key from cache only (not backing store).

        Args:
            key: The key to invalidate.
        """
        if key in self._cache:
            self._cache_remove(key)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._dirty_keys.clear()
        self._eviction_policy.clear()

    def flush(self) -> Generator[float, None, int]:
        """Write all dirty entries to backing store.

        Only relevant for write-back mode.

        Yields:
            Write latencies.

        Returns:
            Number of entries flushed.
        """
        flushed = 0
        for key in list(self._dirty_keys):
            if key in self._cache:
                yield from self._backing_store.put(key, self._cache[key])
                self._dirty_keys.discard(key)
                self.stats.writebacks += 1
                flushed += 1
        return flushed

    def _cache_put(self, key: str, value: Any) -> None:
        """Add or update a cache entry, evicting if necessary."""
        if key not in self._cache:
            # Evict if at capacity
            while len(self._cache) >= self._cache_capacity:
                evict_key = self._eviction_policy.evict()
                if evict_key is None:
                    break
                self._cache.pop(evict_key, None)
                self._dirty_keys.discard(evict_key)
                self.stats.evictions += 1

            self._eviction_policy.on_insert(key)
        else:
            self._eviction_policy.on_access(key)

        self._cache[key] = value

    def _cache_remove(self, key: str) -> None:
        """Remove an entry from cache."""
        self._cache.pop(key, None)
        self._dirty_keys.discard(key)
        self._eviction_policy.on_remove(key)

    def contains_cached(self, key: str) -> bool:
        """Check if a key is in the cache.

        Args:
            key: The key to check.

        Returns:
            True if key is cached.
        """
        return key in self._cache

    def get_cached_keys(self) -> list[str]:
        """Get all cached keys.

        Returns:
            List of cached keys.
        """
        return list(self._cache.keys())

    def get_dirty_keys(self) -> list[str]:
        """Get keys pending writeback.

        Returns:
            List of dirty keys.
        """
        return list(self._dirty_keys)

    def handle_event(self, event: Event) -> None:
        """CachedStore can handle events for cache operations."""
        pass
