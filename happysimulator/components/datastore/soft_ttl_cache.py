"""Soft TTL Cache implementation with stale-while-revalidate pattern.

Provides a cache layer that distinguishes between fresh, stale, and expired data:

| Zone      | Condition               | Behavior                                    |
|-----------|-------------------------|---------------------------------------------|
| Fresh     | age < soft_ttl          | Serve from cache immediately                |
| Stale     | soft_ttl <= age < hard_ttl | Serve stale + trigger background refresh |
| Expired   | age >= hard_ttl         | Block until fresh data fetched              |

This pattern enables low-latency reads while maintaining data freshness,
commonly used in CDNs, API gateways, and distributed caches.

Example:
    from happysimulator.components.datastore import KVStore, SoftTTLCache

    backing = KVStore(name="db", read_latency=0.010)
    cache = SoftTTLCache(
        name="api_cache",
        backing_store=backing,
        soft_ttl=30.0,   # Fresh for 30 seconds
        hard_ttl=300.0,  # Valid for 5 minutes
    )

    def handle_event(self, event):
        # Fast if fresh/stale, blocking only if expired
        value = yield from cache.get("user:123")
"""

from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.instant import Duration, Instant
from happysimulator.components.datastore.kv_store import KVStore


@dataclass
class CacheEntry:
    """A cached value with its timestamp.

    Attributes:
        value: The cached data.
        cached_at: Simulation time when the entry was cached.
    """

    value: Any
    cached_at: Instant

    def is_fresh(self, now: Instant, soft_ttl: Duration) -> bool:
        """Check if entry is within soft TTL (fresh zone).

        Args:
            now: Current simulation time.
            soft_ttl: The soft TTL duration.

        Returns:
            True if age < soft_ttl.
        """
        age = now - self.cached_at
        return age < soft_ttl

    def is_valid(self, now: Instant, hard_ttl: Duration) -> bool:
        """Check if entry is within hard TTL (still usable).

        Args:
            now: Current simulation time.
            hard_ttl: The hard TTL duration.

        Returns:
            True if age < hard_ttl.
        """
        age = now - self.cached_at
        return age < hard_ttl


@dataclass
class SoftTTLCacheStats:
    """Statistics tracked by SoftTTLCache.

    Attributes:
        reads: Total read operations.
        fresh_hits: Reads served from cache within soft TTL.
        stale_hits: Reads served from cache that triggered background refresh.
        hard_misses: Reads requiring blocking fetch (expired or uncached).
        background_refreshes: Background refresh operations started.
        refresh_successes: Background refreshes that completed successfully.
        coalesced_requests: Requests that joined an in-flight refresh.
        evictions: Entries removed due to capacity limits.
    """

    reads: int = 0
    fresh_hits: int = 0
    stale_hits: int = 0
    hard_misses: int = 0
    background_refreshes: int = 0
    refresh_successes: int = 0
    coalesced_requests: int = 0
    evictions: int = 0

    @property
    def fresh_hit_rate(self) -> float:
        """Ratio of fresh hits to total reads."""
        if self.reads == 0:
            return 0.0
        return self.fresh_hits / self.reads

    @property
    def stale_hit_rate(self) -> float:
        """Ratio of stale hits to total reads."""
        if self.reads == 0:
            return 0.0
        return self.stale_hits / self.reads

    @property
    def total_hit_rate(self) -> float:
        """Ratio of all cache hits (fresh + stale) to total reads."""
        if self.reads == 0:
            return 0.0
        return (self.fresh_hits + self.stale_hits) / self.reads

    @property
    def miss_rate(self) -> float:
        """Ratio of hard misses to total reads."""
        if self.reads == 0:
            return 0.0
        return self.hard_misses / self.reads


class SoftTTLCache(Entity):
    """Cache with stale-while-revalidate semantics.

    Implements the soft TTL / hard TTL pattern where entries transition
    through three zones:

    1. **Fresh** (age < soft_ttl): Serve immediately from cache
    2. **Stale** (soft_ttl <= age < hard_ttl): Serve stale + background refresh
    3. **Expired** (age >= hard_ttl): Block until fresh data is fetched

    Features:
    - Request coalescing: Multiple requests during refresh share one fetch
    - LRU eviction when at capacity
    - Statistics for monitoring cache effectiveness

    Attributes:
        name: Entity name for identification.
        stats: SoftTTLCacheStats with operational metrics.
    """

    def __init__(
        self,
        name: str,
        backing_store: KVStore,
        soft_ttl: float | Duration,
        hard_ttl: float | Duration,
        cache_capacity: int | None = None,
        cache_read_latency: float = 0.0001,
    ):
        """Initialize the soft TTL cache.

        Args:
            name: Name for this cache entity.
            backing_store: The underlying storage to cache.
            soft_ttl: Time in seconds (or Duration) before entries become stale.
            hard_ttl: Time in seconds (or Duration) before entries expire.
            cache_capacity: Maximum entries (None = unlimited).
            cache_read_latency: Latency for cache reads in seconds.

        Raises:
            ValueError: If parameters are invalid (e.g., soft_ttl > hard_ttl).
        """
        super().__init__(name)

        # Convert float to Duration if needed
        self._soft_ttl = (
            soft_ttl if isinstance(soft_ttl, Duration)
            else Duration.from_seconds(soft_ttl)
        )
        self._hard_ttl = (
            hard_ttl if isinstance(hard_ttl, Duration)
            else Duration.from_seconds(hard_ttl)
        )

        # Validation
        if self._soft_ttl.nanoseconds < 0:
            raise ValueError(f"soft_ttl must be >= 0, got {soft_ttl}")
        if self._hard_ttl.nanoseconds < 0:
            raise ValueError(f"hard_ttl must be >= 0, got {hard_ttl}")
        if self._soft_ttl > self._hard_ttl:
            raise ValueError(
                f"soft_ttl ({soft_ttl}) must be <= hard_ttl ({hard_ttl})"
            )
        if cache_capacity is not None and cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1 or None, got {cache_capacity}")
        if cache_read_latency < 0:
            raise ValueError(f"cache_read_latency must be >= 0, got {cache_read_latency}")

        self._backing_store = backing_store
        self._cache_capacity = cache_capacity
        self._cache_read_latency = cache_read_latency

        # State
        self._cache: dict[str, CacheEntry] = {}
        self._refreshing_keys: set[str] = set()
        self._access_order: list[str] = []  # For LRU eviction

        # Statistics
        self.stats = SoftTTLCacheStats()

    @property
    def backing_store(self) -> KVStore:
        """The underlying backing store."""
        return self._backing_store

    @property
    def soft_ttl(self) -> Duration:
        """Duration before entries become stale."""
        return self._soft_ttl

    @property
    def hard_ttl(self) -> Duration:
        """Duration before entries expire completely."""
        return self._hard_ttl

    @property
    def cache_capacity(self) -> int | None:
        """Maximum number of cached entries (None = unlimited)."""
        return self._cache_capacity

    @property
    def cache_size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    def get(self, key: str) -> Generator[float, None, Optional[Any]]:
        """Get a value with soft TTL semantics.

        Behavior depends on cache state:
        - Fresh hit: Return immediately from cache
        - Stale hit: Return from cache AND trigger background refresh
        - Hard miss: Block until fresh data is fetched

        Args:
            key: The key to look up.

        Yields:
            Latency delays (cache read or backing store fetch).

        Returns:
            The value if found, None otherwise.
        """
        self.stats.reads += 1
        now = self.now

        if key in self._cache:
            entry = self._cache[key]
            self._touch_for_lru(key)

            # Fresh hit - serve immediately
            if entry.is_fresh(now, self._soft_ttl):
                self.stats.fresh_hits += 1
                yield self._cache_read_latency
                return entry.value

            # Stale hit - serve AND trigger background refresh
            if entry.is_valid(now, self._hard_ttl):
                self.stats.stale_hits += 1
                side_effects = self._maybe_start_refresh(key)
                if side_effects:
                    yield self._cache_read_latency, side_effects
                else:
                    yield self._cache_read_latency
                return entry.value

        # Hard miss - need to fetch from backing store
        self.stats.hard_misses += 1

        # Request coalescing: if refresh is in progress, wait for it
        if key in self._refreshing_keys:
            self.stats.coalesced_requests += 1
            # Wait for backing store latency (simulating waiting for the refresh)
            yield self._backing_store.read_latency
            # Check if the refresh completed
            if key in self._cache:
                return self._cache[key].value
            return None

        # Fetch from backing store (blocking)
        value = yield from self._backing_store.get(key)
        if value is not None:
            self._store(key, value)
        return value

    def put(self, key: str, value: Any) -> Generator[float, None, None]:
        """Store a value in cache and backing store.

        Updates the cache timestamp and writes through to backing store.

        Args:
            key: The key to store under.
            value: The value to store.

        Yields:
            Write latency.
        """
        # Write to backing store first
        yield from self._backing_store.put(key, value)
        # Then update cache
        self._store(key, value)

    def invalidate(self, key: str) -> None:
        """Remove a key from cache only (not backing store).

        Args:
            key: The key to invalidate.
        """
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._access_order.clear()
        self._refreshing_keys.clear()

    def contains_cached(self, key: str) -> bool:
        """Check if a key is in the cache (regardless of freshness).

        Args:
            key: The key to check.

        Returns:
            True if key is cached.
        """
        return key in self._cache

    def is_refreshing(self, key: str) -> bool:
        """Check if a background refresh is in progress for a key.

        Args:
            key: The key to check.

        Returns:
            True if refresh is in progress.
        """
        return key in self._refreshing_keys

    def get_cached_keys(self) -> list[str]:
        """Get all cached keys.

        Returns:
            List of cached keys.
        """
        return list(self._cache.keys())

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        """Handle internal cache events (background refresh).

        Args:
            event: The event to process.

        Yields:
            Processing delays.

        Returns:
            None (background refresh doesn't produce follow-up events).
        """
        if event.event_type == "_sttl_refresh":
            key = event.context["metadata"]["key"]
            try:
                value = yield from self._backing_store.get(key)
                if value is not None:
                    self._store(key, value)
                    self.stats.refresh_successes += 1
            finally:
                self._refreshing_keys.discard(key)
        return None

    def _maybe_start_refresh(self, key: str) -> list[Event] | None:
        """Start a background refresh if not already in progress.

        Args:
            key: The key to refresh.

        Returns:
            List containing the refresh event, or None if already refreshing.
        """
        if key in self._refreshing_keys:
            return None

        self._refreshing_keys.add(key)
        self.stats.background_refreshes += 1

        return [Event(
            time=self.now,
            event_type="_sttl_refresh",
            target=self,
            context={"metadata": {"key": key}},
        )]

    def _store(self, key: str, value: Any) -> None:
        """Store a value in the cache with current timestamp.

        Handles LRU tracking and eviction if at capacity.

        Args:
            key: The key to store under.
            value: The value to store.
        """
        # Handle capacity - evict if needed
        if self._cache_capacity is not None and key not in self._cache:
            while len(self._cache) >= self._cache_capacity:
                self._evict_lru()

        # Update LRU tracking
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        # Store with current timestamp
        self._cache[key] = CacheEntry(value=value, cached_at=self.now)

    def _touch_for_lru(self, key: str) -> None:
        """Update LRU tracking for a key access.

        Args:
            key: The accessed key.
        """
        if key in self._access_order:
            self._access_order.remove(key)
            self._access_order.append(key)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)
            self.stats.evictions += 1
