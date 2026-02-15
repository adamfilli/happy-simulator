"""Multi-tier cache implementation.

Provides L1/L2/etc hierarchical caching where faster caches are checked
first, with promotion of frequently accessed items to higher tiers.

Example:
    from happysimulator.components.datastore import (
        KVStore, CachedStore, LRUEviction, MultiTierCache
    )

    backing = KVStore(name="db", read_latency=0.010)

    l1 = CachedStore(
        name="l1_cache",
        backing_store=backing,
        cache_capacity=100,
        cache_read_latency=0.0001,  # 0.1ms
        eviction_policy=LRUEviction(),
    )

    l2 = CachedStore(
        name="l2_cache",
        backing_store=backing,
        cache_capacity=1000,
        cache_read_latency=0.001,  # 1ms
        eviction_policy=LRUEviction(),
    )

    multi_tier = MultiTierCache(
        name="tiered_cache",
        tiers=[l1, l2],
        backing_store=backing,
    )
"""

from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class PromotionPolicy(Enum):
    """Policy for promoting items to higher cache tiers."""

    ALWAYS = "always"  # Always promote on access
    ON_SECOND_ACCESS = "on_second_access"  # Promote only after second access
    NEVER = "never"  # Never promote (each tier independent)


@dataclass(frozen=True)
class MultiTierCacheStats:
    """Statistics tracked by MultiTierCache."""

    reads: int = 0
    writes: int = 0
    tier_hits: dict[int, int] = field(default_factory=dict)  # Hits per tier (0=L1, 1=L2, etc.)
    backing_store_hits: int = 0
    misses: int = 0  # Not found anywhere
    promotions: int = 0


class MultiTierCache(Entity):
    """Hierarchical multi-tier cache.

    Checks caches from fastest (L1) to slowest (Ln), then backing store.
    Supports promotion of items to faster tiers on access.

    Tiers should be ordered from fastest to slowest, with L1 typically
    being smallest/fastest and Ln being largest/slowest.

    Attributes:
        name: Entity name for identification.
        num_tiers: Number of cache tiers.
        hit_rate: Overall cache hit rate across all tiers.
    """

    def __init__(
        self,
        name: str,
        tiers: list[Entity],
        backing_store: Entity,
        promotion_policy: PromotionPolicy | str = PromotionPolicy.ALWAYS,
    ):
        """Initialize the multi-tier cache.

        Args:
            name: Name for this cache entity.
            tiers: List of cache tiers from fastest (L1) to slowest.
            backing_store: The underlying storage behind all caches.
            promotion_policy: Policy for promoting items to higher tiers.

        Raises:
            ValueError: If no tiers provided.
        """
        if not tiers:
            raise ValueError("At least one cache tier is required")

        if isinstance(promotion_policy, str):
            promotion_policy = PromotionPolicy(promotion_policy)

        super().__init__(name)
        self._tiers = tiers
        self._backing_store = backing_store
        self._promotion_policy = promotion_policy

        # Track access counts for ON_SECOND_ACCESS policy
        self._access_counts: dict[str, int] = {}

        # Statistics
        self._reads = 0
        self._writes = 0
        self._tier_hits: dict[int, int] = dict.fromkeys(range(len(tiers)), 0)
        self._backing_store_hits = 0
        self._misses = 0
        self._promotions = 0

    @property
    def stats(self) -> MultiTierCacheStats:
        """Frozen snapshot of multi-tier cache statistics."""
        return MultiTierCacheStats(
            reads=self._reads,
            writes=self._writes,
            tier_hits=dict(self._tier_hits),
            backing_store_hits=self._backing_store_hits,
            misses=self._misses,
            promotions=self._promotions,
        )

    @property
    def num_tiers(self) -> int:
        """Number of cache tiers."""
        return len(self._tiers)

    @property
    def tiers(self) -> list[Entity]:
        """The cache tiers (L1 to Ln)."""
        return self._tiers

    @property
    def backing_store(self) -> Entity:
        """The underlying backing store."""
        return self._backing_store

    @property
    def promotion_policy(self) -> PromotionPolicy:
        """Policy for promoting items between tiers."""
        return self._promotion_policy

    @property
    def hit_rate(self) -> float:
        """Overall cache hit rate across all tiers."""
        if self._reads == 0:
            return 0.0
        total_hits = sum(self._tier_hits.values())
        return total_hits / self._reads

    def get(self, key: str) -> Generator[float, None, Any | None]:
        """Get a value, checking tiers from fastest to slowest.

        Args:
            key: The key to look up.

        Yields:
            Cache or backing store latency.

        Returns:
            The value if found, None otherwise.
        """
        self._reads += 1
        self._access_counts[key] = self._access_counts.get(key, 0) + 1

        # Check each tier in order
        for tier_idx, tier in enumerate(self._tiers):
            # Check if tier has the key cached
            if hasattr(tier, "contains_cached") and tier.contains_cached(key):
                value = yield from tier.get(key)
                if value is not None:
                    self._tier_hits[tier_idx] = self._tier_hits.get(tier_idx, 0) + 1

                    # Promote to higher tier if applicable
                    if tier_idx > 0:
                        self._maybe_promote(key, value, tier_idx)

                    return value

        # Not in any cache - fetch from backing store
        value = yield from self._backing_store.get(key)

        if value is not None:
            self._backing_store_hits += 1
            # Cache in appropriate tier(s)
            self._cache_value(key, value)
        else:
            self._misses += 1

        return value

    def put(self, key: str, value: Any) -> Generator[float]:
        """Store a value in cache and backing store.

        Writes to backing store and invalidates/updates all cache tiers.

        Args:
            key: The key to store under.
            value: The value to store.

        Yields:
            Write latency.
        """
        self._writes += 1

        # Write to backing store
        yield from self._backing_store.put(key, value)

        # Update L1 cache (highest priority)
        if self._tiers:
            # Invalidate from all tiers first
            for tier in self._tiers:
                if hasattr(tier, "invalidate"):
                    tier.invalidate(key)

            # Write to L1
            yield from self._tiers[0].put(key, value)

    def delete(self, key: str) -> Generator[float, None, bool]:
        """Delete a key from all tiers and backing store.

        Args:
            key: The key to delete.

        Yields:
            Delete latency.

        Returns:
            True if key existed anywhere.
        """
        existed = False

        # Remove from all tiers
        for tier in self._tiers:
            if hasattr(tier, "invalidate"):
                tier.invalidate(key)
                existed = True

        # Remove from backing store
        store_existed = yield from self._backing_store.delete(key)

        # Clean up access tracking
        self._access_counts.pop(key, None)

        return existed or store_existed

    def invalidate(self, key: str) -> None:
        """Remove a key from all cache tiers (not backing store).

        Args:
            key: The key to invalidate.
        """
        for tier in self._tiers:
            if hasattr(tier, "invalidate"):
                tier.invalidate(key)

    def invalidate_all(self) -> None:
        """Clear all cache tiers."""
        for tier in self._tiers:
            if hasattr(tier, "invalidate_all"):
                tier.invalidate_all()
        self._access_counts.clear()

    def _should_promote(self, key: str) -> bool:
        """Check if a key should be promoted to a higher tier."""
        if self._promotion_policy == PromotionPolicy.NEVER:
            return False
        if self._promotion_policy == PromotionPolicy.ALWAYS:
            return True
        if self._promotion_policy == PromotionPolicy.ON_SECOND_ACCESS:
            return self._access_counts.get(key, 0) >= 2
        return False

    def _maybe_promote(self, key: str, value: Any, from_tier: int) -> None:
        """Promote a value to a higher tier if policy allows."""
        if from_tier <= 0:
            return  # Already at L1

        if not self._should_promote(key):
            return

        # Promote to L1 (synchronously, no yield)
        target_tier = self._tiers[0]
        if hasattr(target_tier, "_cache_put"):
            target_tier._cache_put(key, value)
            self._promotions += 1

    def _cache_value(self, key: str, value: Any) -> None:
        """Cache a value in the appropriate tier(s)."""
        # For new values, cache in L1
        if self._tiers:
            target_tier = self._tiers[0]
            if hasattr(target_tier, "_cache_put"):
                target_tier._cache_put(key, value)

    def get_tier_stats(self) -> dict[int, dict]:
        """Get statistics for each tier.

        Returns:
            Dictionary mapping tier index to tier stats.
        """
        result = {}
        for i, tier in enumerate(self._tiers):
            if hasattr(tier, "stats"):
                result[i] = {
                    "hits": self._tier_hits.get(i, 0),
                    "cache_size": tier.cache_size if hasattr(tier, "cache_size") else 0,
                    "capacity": tier.cache_capacity if hasattr(tier, "cache_capacity") else 0,
                }
        return result

    def handle_event(self, event: Event) -> None:
        """MultiTierCache can handle events for cache operations."""
