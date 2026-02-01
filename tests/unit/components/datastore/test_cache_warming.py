"""Tests for CacheWarmer."""

import pytest

from happysimulator.components.datastore import CacheWarmer, KVStore, CachedStore, LRUEviction


class TestCacheWarmerCreation:
    """Tests for CacheWarmer creation."""

    def test_creates_with_list(self):
        """CacheWarmer is created with a list of keys."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        warmer = CacheWarmer(
            name="warmer",
            cache=cache,
            keys_to_warm=["key1", "key2", "key3"],
        )

        assert warmer.name == "warmer"
        assert warmer.warmup_rate == 100.0
        assert warmer.is_started is False
        assert warmer.is_complete is False

    def test_creates_with_callable(self):
        """CacheWarmer is created with a callable for keys."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        def get_keys():
            return ["dynamic1", "dynamic2"]

        warmer = CacheWarmer(
            name="warmer",
            cache=cache,
            keys_to_warm=get_keys,
        )

        keys = warmer.get_keys_to_warm()
        assert keys == ["dynamic1", "dynamic2"]

    def test_rejects_invalid_rate(self):
        """Rejects warmup_rate <= 0."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        with pytest.raises(ValueError):
            CacheWarmer(
                name="warmer",
                cache=cache,
                keys_to_warm=[],
                warmup_rate=0,
            )


class TestCacheWarmerProgress:
    """Tests for warming progress tracking."""

    def test_progress_starts_at_zero(self):
        """Progress is 0 before starting."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        warmer = CacheWarmer(
            name="warmer",
            cache=cache,
            keys_to_warm=["key1", "key2"],
        )

        assert warmer.progress == 0.0

    def test_start_warming_returns_event(self):
        """start_warming returns an event."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        warmer = CacheWarmer(
            name="warmer",
            cache=cache,
            keys_to_warm=["key1", "key2"],
        )

        event = warmer.start_warming()

        assert event is not None
        assert event.target is warmer
        assert warmer.is_started is True


class TestCacheWarmerStatistics:
    """Tests for CacheWarmer statistics."""

    def test_tracks_keys_to_warm(self):
        """Statistics track total keys to warm."""
        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=100,
            eviction_policy=LRUEviction(),
        )

        warmer = CacheWarmer(
            name="warmer",
            cache=cache,
            keys_to_warm=["k1", "k2", "k3"],
        )

        warmer.start_warming()

        assert warmer.stats.keys_to_warm == 3
        assert warmer.stats.keys_warmed == 0
