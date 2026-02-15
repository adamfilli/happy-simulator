"""Tests for SoftTTLCache."""

import pytest

from happysimulator.components.datastore import (
    CacheEntry,
    KVStore,
    SoftTTLCache,
    SoftTTLCacheStats,
)
from happysimulator.core.clock import Clock
from happysimulator.core.temporal import Duration, Instant


class MockClock(Clock):
    """Clock with manually controllable time for testing."""

    def __init__(self, start_time: Instant = Instant.Epoch):
        self._now = start_time

    @property
    def now(self) -> Instant:
        return self._now

    def advance(self, seconds: float) -> None:
        """Advance the clock by the given number of seconds."""
        self._now = self._now + seconds

    def set_time(self, time: Instant) -> None:
        """Set the clock to an absolute time."""
        self._now = time


def exhaust_generator(gen):
    """Run a generator to completion and return its final value."""
    result = None
    try:
        while True:
            next(gen)
    except StopIteration as e:
        result = e.value
    return result


def get_first_yield(gen):
    """Get the first yielded value from a generator."""
    return next(gen)


def get_yield_with_side_effects(gen):
    """Get first yield, handling both plain yields and (delay, events) tuples."""
    yielded = next(gen)
    if isinstance(yielded, tuple):
        return yielded[0], yielded[1]
    return yielded, None


class TestCacheEntryCreation:
    """Tests for CacheEntry dataclass."""

    def test_creates_with_value_and_timestamp(self):
        """CacheEntry stores value and timestamp."""
        now = Instant.from_seconds(10.0)
        entry = CacheEntry(value="test_value", cached_at=now)

        assert entry.value == "test_value"
        assert entry.cached_at == now

    def test_is_fresh_within_soft_ttl(self):
        """Entry is fresh when age < soft_ttl."""
        cached_at = Instant.from_seconds(10.0)
        entry = CacheEntry(value="test", cached_at=cached_at)

        now = Instant.from_seconds(15.0)  # 5 seconds later
        soft_ttl = Duration.from_seconds(10.0)

        assert entry.is_fresh(now, soft_ttl) is True

    def test_not_fresh_beyond_soft_ttl(self):
        """Entry is not fresh when age >= soft_ttl."""
        cached_at = Instant.from_seconds(10.0)
        entry = CacheEntry(value="test", cached_at=cached_at)

        now = Instant.from_seconds(25.0)  # 15 seconds later
        soft_ttl = Duration.from_seconds(10.0)

        assert entry.is_fresh(now, soft_ttl) is False

    def test_is_valid_within_hard_ttl(self):
        """Entry is valid when age < hard_ttl."""
        cached_at = Instant.from_seconds(10.0)
        entry = CacheEntry(value="test", cached_at=cached_at)

        now = Instant.from_seconds(50.0)  # 40 seconds later
        hard_ttl = Duration.from_seconds(60.0)

        assert entry.is_valid(now, hard_ttl) is True

    def test_not_valid_beyond_hard_ttl(self):
        """Entry is not valid when age >= hard_ttl."""
        cached_at = Instant.from_seconds(10.0)
        entry = CacheEntry(value="test", cached_at=cached_at)

        now = Instant.from_seconds(80.0)  # 70 seconds later
        hard_ttl = Duration.from_seconds(60.0)

        assert entry.is_valid(now, hard_ttl) is False


class TestSoftTTLCacheCreation:
    """Tests for SoftTTLCache creation."""

    def test_creates_with_basic_params(self):
        """SoftTTLCache creates with basic parameters."""
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )

        assert cache.name == "cache"
        assert cache.backing_store is backing
        assert cache.soft_ttl == Duration.from_seconds(30.0)
        assert cache.hard_ttl == Duration.from_seconds(300.0)
        assert cache.cache_size == 0

    def test_creates_with_duration_params(self):
        """SoftTTLCache accepts Duration objects."""
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=Duration.from_seconds(30.0),
            hard_ttl=Duration.from_seconds(300.0),
        )

        assert cache.soft_ttl == Duration.from_seconds(30.0)
        assert cache.hard_ttl == Duration.from_seconds(300.0)

    def test_rejects_soft_ttl_greater_than_hard_ttl(self):
        """SoftTTLCache rejects soft_ttl > hard_ttl."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError, match=r"soft_ttl.*must be <= hard_ttl"):
            SoftTTLCache(
                name="cache",
                backing_store=backing,
                soft_ttl=100.0,
                hard_ttl=50.0,
            )

    def test_rejects_negative_soft_ttl(self):
        """SoftTTLCache rejects negative soft_ttl."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError, match="soft_ttl must be >= 0"):
            SoftTTLCache(
                name="cache",
                backing_store=backing,
                soft_ttl=-1.0,
                hard_ttl=100.0,
            )

    def test_rejects_negative_hard_ttl(self):
        """SoftTTLCache rejects negative hard_ttl."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError, match="hard_ttl must be >= 0"):
            SoftTTLCache(
                name="cache",
                backing_store=backing,
                soft_ttl=10.0,
                hard_ttl=-1.0,
            )

    def test_rejects_invalid_capacity(self):
        """SoftTTLCache rejects capacity < 1."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError, match="cache_capacity must be >= 1"):
            SoftTTLCache(
                name="cache",
                backing_store=backing,
                soft_ttl=30.0,
                hard_ttl=300.0,
                cache_capacity=0,
            )

    def test_rejects_negative_cache_read_latency(self):
        """SoftTTLCache rejects negative cache_read_latency."""
        backing = KVStore(name="backing")

        with pytest.raises(ValueError, match="cache_read_latency must be >= 0"):
            SoftTTLCache(
                name="cache",
                backing_store=backing,
                soft_ttl=30.0,
                hard_ttl=300.0,
                cache_read_latency=-0.001,
            )


class TestSoftTTLCacheGet:
    """Tests for SoftTTLCache.get()."""

    def test_fresh_hit(self):
        """Entry within soft_ttl served immediately, no refresh."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing", read_latency=0.010)
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_read_latency=0.001,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Put value in backing store and fetch to cache
        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))

        # Advance time but stay within soft_ttl
        clock.advance(10.0)

        # Second get should be fresh hit
        gen = cache.get("key1")
        latency, side_effects = get_yield_with_side_effects(gen)

        assert latency == 0.001  # Cache latency
        assert side_effects is None  # No background refresh
        assert cache.stats.fresh_hits == 1

    def test_stale_hit_triggers_refresh(self):
        """Entry between soft/hard TTL triggers background refresh."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing", read_latency=0.010)
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_read_latency=0.001,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Put value in backing store and fetch to cache
        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))

        # Advance past soft_ttl but within hard_ttl
        clock.advance(60.0)

        # Get should return stale value and trigger refresh
        gen = cache.get("key1")
        latency, side_effects = get_yield_with_side_effects(gen)

        assert latency == 0.001  # Cache latency (not blocking)
        assert side_effects is not None
        assert len(side_effects) == 1
        assert side_effects[0].event_type == "_sttl_refresh"
        assert cache.stats.stale_hits == 1
        assert cache.stats.background_refreshes == 1

    def test_hard_miss_blocks_on_fetch(self):
        """Entry past hard_ttl requires blocking fetch."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing", read_latency=0.010)
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_read_latency=0.001,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Put value in backing store and fetch to cache
        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))
        initial_hard_misses = cache.stats.hard_misses  # Initial fetch was also a hard miss

        # Advance past hard_ttl
        clock.advance(400.0)

        # Get should do blocking fetch
        gen = cache.get("key1")
        latency = get_first_yield(gen)

        assert latency == 0.010  # Backing store latency
        assert cache.stats.hard_misses == initial_hard_misses + 1  # One more hard miss

    def test_uncached_key_fetches_from_backing(self):
        """First access to a key fetches from backing store."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing", read_latency=0.010)
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_read_latency=0.001,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        backing.put_sync("key1", "value1")

        gen = cache.get("key1")
        latency = get_first_yield(gen)

        assert latency == 0.010  # Backing store latency
        assert cache.stats.hard_misses == 1

    def test_uncached_key_populates_cache(self):
        """First access caches the value."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))

        assert cache.contains_cached("key1") is True
        assert cache.cache_size == 1

    def test_missing_key_returns_none(self):
        """Missing key returns None."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        value = exhaust_generator(cache.get("missing"))

        assert value is None

    def test_request_coalescing(self):
        """Multiple requests during refresh share single fetch."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing", read_latency=0.010)
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_read_latency=0.001,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Put value and fetch to cache
        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))

        # Advance past soft_ttl to trigger refresh
        clock.advance(60.0)

        # First request triggers refresh
        gen1 = cache.get("key1")
        get_yield_with_side_effects(gen1)  # Triggers refresh

        # Key is now refreshing
        assert cache.is_refreshing("key1") is True

        # Advance past hard_ttl so next request would be a hard miss
        clock.advance(300.0)

        # Second request should coalesce (join in-flight refresh)
        gen2 = cache.get("key1")
        latency = get_first_yield(gen2)

        assert latency == 0.010  # Waiting for backing store
        assert cache.stats.coalesced_requests == 1

    def test_stale_hit_does_not_duplicate_refresh(self):
        """Multiple stale hits don't start multiple refreshes."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))

        # Advance past soft_ttl
        clock.advance(60.0)

        # Multiple stale hits
        gen1 = cache.get("key1")
        _, effects1 = get_yield_with_side_effects(gen1)

        gen2 = cache.get("key1")
        _, effects2 = get_yield_with_side_effects(gen2)

        assert effects1 is not None  # First triggers refresh
        assert effects2 is None  # Second does not
        assert cache.stats.background_refreshes == 1


class TestSoftTTLCachePut:
    """Tests for SoftTTLCache.put()."""

    def test_put_updates_cache(self):
        """put() stores value in cache."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))

        assert cache.contains_cached("key1") is True

    def test_put_writes_through_to_backing(self):
        """put() writes to backing store."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))

        assert backing.get_sync("key1") == "value1"

    def test_put_resets_ttl(self):
        """put() resets the cache timestamp."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Initial put
        exhaust_generator(cache.put("key1", "value1"))

        # Advance past soft_ttl
        clock.advance(60.0)

        # Update value
        exhaust_generator(cache.put("key1", "value2"))

        # Should now be fresh again
        gen = cache.get("key1")
        _latency, side_effects = get_yield_with_side_effects(gen)

        assert side_effects is None  # No refresh needed
        assert cache.stats.fresh_hits == 1


class TestSoftTTLCacheInvalidate:
    """Tests for cache invalidation."""

    def test_invalidate_removes_from_cache(self):
        """invalidate() removes key from cache only."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))
        cache.invalidate("key1")

        assert cache.contains_cached("key1") is False
        assert backing.get_sync("key1") == "value1"

    def test_invalidate_all_clears_cache(self):
        """invalidate_all() clears entire cache."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))
        exhaust_generator(cache.put("key2", "value2"))
        cache.invalidate_all()

        assert cache.cache_size == 0


class TestSoftTTLCacheEviction:
    """Tests for cache eviction."""

    def test_evicts_lru_on_overflow(self):
        """Cache evicts least recently used when at capacity."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_capacity=2,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))
        exhaust_generator(cache.put("key2", "value2"))
        exhaust_generator(cache.put("key3", "value3"))  # Should evict key1

        assert cache.cache_size == 2
        assert cache.contains_cached("key1") is False
        assert cache.contains_cached("key2") is True
        assert cache.contains_cached("key3") is True
        assert cache.stats.evictions == 1

    def test_access_updates_lru_order(self):
        """Accessing a key moves it to most-recently-used."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
            cache_capacity=2,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))
        exhaust_generator(cache.put("key2", "value2"))

        # Access key1 to make it most recently used
        exhaust_generator(cache.get("key1"))

        # Add key3 - should evict key2 (now LRU)
        exhaust_generator(cache.put("key3", "value3"))

        assert cache.contains_cached("key1") is True
        assert cache.contains_cached("key2") is False
        assert cache.contains_cached("key3") is True


class TestSoftTTLCacheBackgroundRefresh:
    """Tests for background refresh handling."""

    def test_handle_event_completes_refresh(self):
        """handle_event processes _sttl_refresh events."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Put stale value in cache
        backing.put_sync("key1", "old_value")
        exhaust_generator(cache.get("key1"))

        # Update backing store
        backing.put_sync("key1", "new_value")

        # Advance past soft_ttl
        clock.advance(60.0)

        # Trigger stale hit to get refresh event
        gen = cache.get("key1")
        _, side_effects = get_yield_with_side_effects(gen)
        refresh_event = side_effects[0]

        # Simulate running the refresh
        exhaust_generator(cache.handle_event(refresh_event))

        assert cache.stats.refresh_successes == 1
        assert cache.is_refreshing("key1") is False

    def test_refresh_updates_cache(self):
        """Background refresh updates cached value."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        # Initial cache population
        backing.put_sync("key1", "old_value")
        exhaust_generator(cache.get("key1"))

        # Update backing store
        backing.put_sync("key1", "new_value")

        # Advance past soft_ttl
        clock.advance(60.0)

        # Trigger refresh
        gen = cache.get("key1")
        _, side_effects = get_yield_with_side_effects(gen)
        refresh_event = side_effects[0]

        # Run refresh
        exhaust_generator(cache.handle_event(refresh_event))

        # Cache should have new value
        # Reset stats to check fresh hit
        cache._fresh_hits = 0
        value = exhaust_generator(cache.get("key1"))

        # The value might still be stale depending on clock position
        # but it should be the new value
        assert value == "new_value"


class TestSoftTTLCacheStatistics:
    """Tests for SoftTTLCacheStats."""

    def test_fresh_hit_rate(self):
        """fresh_hit_rate calculates correctly."""
        stats = SoftTTLCacheStats(reads=10, fresh_hits=7)
        assert stats.fresh_hit_rate == 0.7

    def test_stale_hit_rate(self):
        """stale_hit_rate calculates correctly."""
        stats = SoftTTLCacheStats(reads=10, stale_hits=2)
        assert stats.stale_hit_rate == 0.2

    def test_total_hit_rate(self):
        """total_hit_rate calculates correctly."""
        stats = SoftTTLCacheStats(reads=10, fresh_hits=5, stale_hits=3)
        assert stats.total_hit_rate == 0.8

    def test_miss_rate(self):
        """miss_rate calculates correctly."""
        stats = SoftTTLCacheStats(reads=10, hard_misses=4)
        assert stats.miss_rate == 0.4

    def test_zero_reads_rates(self):
        """All rates are 0 with no reads."""
        stats = SoftTTLCacheStats()
        assert stats.fresh_hit_rate == 0.0
        assert stats.stale_hit_rate == 0.0
        assert stats.total_hit_rate == 0.0
        assert stats.miss_rate == 0.0

    def test_statistics_accuracy(self):
        """All counters increment correctly during operations."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=10.0,
            hard_ttl=60.0,
            cache_capacity=2,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        backing.put_sync("key1", "value1")
        backing.put_sync("key2", "value2")
        backing.put_sync("key3", "value3")

        # Cold miss
        exhaust_generator(cache.get("key1"))
        assert cache.stats.hard_misses == 1
        assert cache.stats.reads == 1

        # Fresh hit
        clock.advance(5.0)
        exhaust_generator(cache.get("key1"))
        assert cache.stats.fresh_hits == 1
        assert cache.stats.reads == 2

        # Stale hit
        clock.advance(10.0)
        exhaust_generator(cache.get("key1"))
        assert cache.stats.stale_hits == 1
        assert cache.stats.background_refreshes == 1
        assert cache.stats.reads == 3

        # Another cold miss
        exhaust_generator(cache.get("key2"))
        assert cache.stats.hard_misses == 2
        assert cache.stats.reads == 4

        # Trigger eviction
        exhaust_generator(cache.put("key3", "value3"))
        assert cache.stats.evictions == 1


class TestSoftTTLCacheHelperMethods:
    """Tests for helper methods."""

    def test_contains_cached(self):
        """contains_cached returns correct status."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        assert cache.contains_cached("key1") is False

        exhaust_generator(cache.put("key1", "value1"))
        assert cache.contains_cached("key1") is True

    def test_is_refreshing(self):
        """is_refreshing returns correct status."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        backing.put_sync("key1", "value1")
        exhaust_generator(cache.get("key1"))

        assert cache.is_refreshing("key1") is False

        clock.advance(60.0)
        gen = cache.get("key1")
        get_yield_with_side_effects(gen)

        assert cache.is_refreshing("key1") is True

    def test_get_cached_keys(self):
        """get_cached_keys returns all cached keys."""
        clock = MockClock(Instant.from_seconds(0.0))
        backing = KVStore(name="backing")
        cache = SoftTTLCache(
            name="cache",
            backing_store=backing,
            soft_ttl=30.0,
            hard_ttl=300.0,
        )
        cache.set_clock(clock)
        backing.set_clock(clock)

        exhaust_generator(cache.put("key1", "value1"))
        exhaust_generator(cache.put("key2", "value2"))

        keys = cache.get_cached_keys()
        assert set(keys) == {"key1", "key2"}
