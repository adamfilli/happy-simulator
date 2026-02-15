"""Tests for MultiTierCache."""

import pytest

from happysimulator.components.datastore import (
    CachedStore,
    KVStore,
    LRUEviction,
    MultiTierCache,
    PromotionPolicy,
)


def create_tiered_setup():
    """Create a standard tiered cache setup for testing."""
    backing = KVStore(name="backing", read_latency=0.010)

    l1 = CachedStore(
        name="l1",
        backing_store=backing,
        cache_capacity=10,
        cache_read_latency=0.0001,
        eviction_policy=LRUEviction(),
    )

    l2 = CachedStore(
        name="l2",
        backing_store=backing,
        cache_capacity=100,
        cache_read_latency=0.001,
        eviction_policy=LRUEviction(),
    )

    return backing, l1, l2


class TestMultiTierCacheCreation:
    """Tests for MultiTierCache creation."""

    def test_creates_with_tiers(self):
        """MultiTierCache is created with tiers."""
        backing, l1, l2 = create_tiered_setup()

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        assert cache.name == "multi"
        assert cache.num_tiers == 2
        assert cache.tiers == [l1, l2]

    def test_rejects_no_tiers(self):
        """Rejects empty tier list."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError):
            MultiTierCache(
                name="multi",
                tiers=[],
                backing_store=backing,
            )

    def test_default_promotion_policy(self):
        """Default promotion policy is ALWAYS."""
        backing, l1, l2 = create_tiered_setup()

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        assert cache.promotion_policy == PromotionPolicy.ALWAYS

    def test_accepts_string_promotion_policy(self):
        """Accepts string for promotion policy."""
        backing, l1, l2 = create_tiered_setup()

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
            promotion_policy="never",
        )

        assert cache.promotion_policy == PromotionPolicy.NEVER


class TestMultiTierCacheAccess:
    """Tests for cache access patterns."""

    def test_fetches_from_backing_store(self):
        """Fetches missing keys from backing store."""
        backing, l1, l2 = create_tiered_setup()
        backing.put_sync("key1", "value1")

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        gen = cache.get("key1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == "value1"
        assert cache.stats.backing_store_hits == 1

    def test_caches_in_l1(self):
        """Caches fetched values in L1."""
        backing, l1, l2 = create_tiered_setup()
        backing.put_sync("key1", "value1")

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        # First access - miss, fetch from backing
        gen = cache.get("key1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Should now be in L1
        assert l1.contains_cached("key1")

    def test_tracks_tier_hits(self):
        """Tracks hits per tier."""
        backing, l1, l2 = create_tiered_setup()
        backing.put_sync("key1", "value1")

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        # First access - backing store hit
        gen = cache.get("key1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Second access - L1 hit
        gen = cache.get("key1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert cache.stats.tier_hits[0] == 1  # L1 hit
        assert cache.stats.backing_store_hits == 1


class TestMultiTierCacheWrite:
    """Tests for cache write operations."""

    def test_write_goes_to_backing(self):
        """Writes go to backing store."""
        backing, l1, l2 = create_tiered_setup()

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        gen = cache.put("key1", "value1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert backing.get_sync("key1") == "value1"

    def test_write_updates_l1(self):
        """Writes update L1 cache."""
        backing, l1, l2 = create_tiered_setup()

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        gen = cache.put("key1", "value1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert l1.contains_cached("key1")


class TestMultiTierCacheStatistics:
    """Tests for MultiTierCache statistics."""

    def test_hit_rate_calculation(self):
        """Hit rate is calculated correctly."""
        backing, l1, l2 = create_tiered_setup()
        backing.put_sync("key1", "value1")

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        # Miss then hit
        for _ in range(2):
            gen = cache.get("key1")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert cache.hit_rate == 0.5  # 1 hit out of 2 reads
