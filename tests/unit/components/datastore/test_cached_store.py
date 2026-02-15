"""Tests for CachedStore."""

import pytest

from happysimulator.components.datastore import (
    CachedStore,
    FIFOEviction,
    KVStore,
    LRUEviction,
)


class TestCachedStoreCreation:
    """Tests for CachedStore creation."""

    def test_creates_with_basic_params(self):
        """CachedStore creates with basic parameters."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        assert cache.name == "cache"
        assert cache.backing_store is backing
        assert cache.cache_capacity == 100
        assert cache.cache_size == 0

    def test_rejects_invalid_capacity(self):
        """CachedStore rejects capacity < 1."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError):
            CachedStore(
                name="cache",
                backing_store=backing,
                cache_capacity=0,
                eviction_policy=LRUEviction(),
            )

    def test_rejects_negative_latency(self):
        """CachedStore rejects negative cache latency."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError):
            CachedStore(
                name="cache",
                backing_store=backing,
                cache_capacity=100,
                eviction_policy=LRUEviction(),
                cache_read_latency=-1,
            )


class TestCachedStoreGet:
    """Tests for CachedStore.get()."""

    def test_cache_hit(self):
        """Cache hit returns cached value with low latency."""
        backing = KVStore(name="backing", read_latency=0.010)
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
            cache_read_latency=0.001,
        )

        # Put directly in backing store and fetch to cache
        backing.put_sync("key1", "value1")
        list(cache.get("key1"))  # Miss, fetches from backing

        # Second get should be cache hit
        gen = cache.get("key1")
        latency = next(gen)

        assert latency == 0.001  # Cache latency, not backing

    def test_cache_miss_fetches_from_backing(self):
        """Cache miss fetches from backing store."""
        backing = KVStore(name="backing", read_latency=0.010)
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        backing.put_sync("key1", "value1")

        gen = cache.get("key1")
        latency = next(gen)

        # Should have backing store latency on miss
        assert latency == 0.010

    def test_cache_miss_caches_value(self):
        """Cache miss stores value in cache."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        backing.put_sync("key1", "value1")
        list(cache.get("key1"))

        assert cache.contains_cached("key1") is True

    def test_missing_key_returns_none(self):
        """Missing key returns None."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        gen = cache.get("missing")
        next(gen)
        try:
            next(gen)
        except StopIteration as e:
            value = e.value

        assert value is None


class TestCachedStorePut:
    """Tests for CachedStore.put()."""

    def test_write_through_writes_both(self):
        """Write-through writes to cache and backing store."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
            write_through=True,
        )

        list(cache.put("key1", "value1"))

        assert cache.contains_cached("key1") is True
        assert backing.get_sync("key1") == "value1"

    def test_write_back_caches_only(self):
        """Write-back writes only to cache."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
            write_through=False,
        )

        list(cache.put("key1", "value1"))

        assert cache.contains_cached("key1") is True
        assert backing.get_sync("key1") is None  # Not in backing yet

    def test_write_back_marks_dirty(self):
        """Write-back marks keys as dirty."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
            write_through=False,
        )

        list(cache.put("key1", "value1"))

        assert "key1" in cache.get_dirty_keys()


class TestCachedStoreEviction:
    """Tests for cache eviction."""

    def test_evicts_on_overflow(self):
        """Cache evicts when capacity exceeded."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=2,
            eviction_policy=FIFOEviction(),
        )

        list(cache.put("key1", "value1"))
        list(cache.put("key2", "value2"))
        list(cache.put("key3", "value3"))  # Should evict key1

        assert cache.cache_size == 2
        assert cache.contains_cached("key1") is False
        assert cache.contains_cached("key2") is True
        assert cache.contains_cached("key3") is True

    def test_tracks_evictions(self):
        """Statistics track evictions."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=1,
            eviction_policy=FIFOEviction(),
        )

        list(cache.put("key1", "value1"))
        list(cache.put("key2", "value2"))

        assert cache.stats.evictions == 1


class TestCachedStoreFlush:
    """Tests for write-back flush."""

    def test_flush_writes_dirty_keys(self):
        """flush() writes dirty keys to backing store."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
            write_through=False,
        )

        list(cache.put("key1", "value1"))
        list(cache.put("key2", "value2"))

        gen = cache.flush()
        while True:
            try:
                next(gen)
            except StopIteration as e:
                flushed = e.value
                break

        assert flushed == 2
        assert backing.get_sync("key1") == "value1"
        assert backing.get_sync("key2") == "value2"

    def test_flush_clears_dirty_set(self):
        """flush() clears dirty key tracking."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
            write_through=False,
        )

        list(cache.put("key1", "value1"))
        list(cache.flush())

        assert len(cache.get_dirty_keys()) == 0


class TestCachedStoreInvalidate:
    """Tests for cache invalidation."""

    def test_invalidate_removes_from_cache(self):
        """invalidate() removes key from cache only."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        list(cache.put("key1", "value1"))
        cache.invalidate("key1")

        assert cache.contains_cached("key1") is False
        assert backing.get_sync("key1") == "value1"

    def test_invalidate_all_clears_cache(self):
        """invalidate_all() clears entire cache."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        list(cache.put("key1", "value1"))
        list(cache.put("key2", "value2"))
        cache.invalidate_all()

        assert cache.cache_size == 0


class TestCachedStoreStatistics:
    """Tests for CachedStore statistics."""

    def test_hit_rate(self):
        """hit_rate calculates correctly."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        list(cache.put("key1", "value1"))
        list(cache.get("key1"))  # Hit
        list(cache.get("key1"))  # Hit
        list(cache.get("missing"))  # Miss

        assert cache.hit_rate == 2 / 3

    def test_miss_rate(self):
        """miss_rate calculates correctly."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        list(cache.put("key1", "value1"))
        list(cache.get("key1"))  # Hit
        list(cache.get("missing"))  # Miss

        assert cache.miss_rate == 0.5

    def test_zero_reads_rates(self):
        """Hit/miss rates are 0 with no reads."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        assert cache.hit_rate == 0.0
        assert cache.miss_rate == 0.0
