"""Unit tests for PageCache."""

import pytest

from happysimulator.components.infrastructure.page_cache import (
    PageCache,
    PageCacheStats,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestPageCacheCreation:
    def test_defaults(self):
        cache = PageCache("cache")
        assert cache.name == "cache"
        assert cache.pages_cached == 0
        assert cache.dirty_pages == 0

    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity_pages must be >= 1"):
            PageCache("bad", capacity_pages=0)

    def test_stats_initial(self):
        cache = PageCache("cache")
        stats = cache.stats
        assert isinstance(stats, PageCacheStats)
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.hit_rate == 0.0


class TestPageCacheBehavior:
    def _make_cache(self, **kwargs) -> tuple[PageCache, Simulation]:
        cache = PageCache("test_cache", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[cache],
        )
        return cache, sim

    def _exhaust(self, gen):
        """Run a generator to completion, collecting yielded values."""
        values = []
        try:
            while True:
                values.append(next(gen))
        except StopIteration as e:
            return values, e.value

    def test_read_miss_then_hit(self):
        cache, sim = self._make_cache()

        # First read: miss
        values, _ = self._exhaust(cache.read_page(1))
        assert len(values) > 0  # disk read latency yielded
        assert cache.stats.misses == 1

        # Second read: hit
        values, _ = self._exhaust(cache.read_page(1))
        assert len(values) == 0  # no latency for hit
        assert cache.stats.hits == 1

    def test_write_marks_dirty(self):
        cache, sim = self._make_cache()

        self._exhaust(cache.write_page(1))
        assert cache.dirty_pages == 1
        assert cache.pages_cached == 1

    def test_write_hit_marks_dirty(self):
        cache, sim = self._make_cache()

        # Read page first (clean)
        self._exhaust(cache.read_page(1))
        assert cache.dirty_pages == 0

        # Write same page (dirty)
        self._exhaust(cache.write_page(1))
        assert cache.dirty_pages == 1

    def test_eviction_on_full_cache(self):
        cache, sim = self._make_cache(capacity_pages=3)

        for i in range(3):
            self._exhaust(cache.read_page(i))
        assert cache.pages_cached == 3

        # Reading page 3 should evict page 0
        self._exhaust(cache.read_page(3))
        assert cache.pages_cached == 3
        assert cache.stats.evictions == 1

    def test_lru_eviction_order(self):
        cache, sim = self._make_cache(capacity_pages=3)

        # Load pages 0, 1, 2
        for i in range(3):
            self._exhaust(cache.read_page(i))

        # Access page 0 to make it recent
        self._exhaust(cache.read_page(0))

        # Add page 3 — should evict page 1 (least recently used)
        self._exhaust(cache.read_page(3))

        # Page 0 should still be cached (hit)
        values, _ = self._exhaust(cache.read_page(0))
        assert len(values) == 0  # hit

        # Page 1 should be evicted (miss)
        values, _ = self._exhaust(cache.read_page(1))
        assert len(values) > 0  # miss

    def test_dirty_eviction_causes_writeback(self):
        cache, sim = self._make_cache(capacity_pages=2)

        # Write page 0 (dirty)
        self._exhaust(cache.write_page(0))
        # Read page 1 (clean)
        self._exhaust(cache.read_page(1))

        # Add page 2 — evicts page 0 (dirty) which causes writeback
        values, _ = self._exhaust(cache.read_page(2))
        assert cache.stats.dirty_writebacks == 1

    def test_readahead(self):
        cache, sim = self._make_cache(capacity_pages=10, readahead_pages=2)

        self._exhaust(cache.read_page(5))
        # Pages 5, 6, 7 should be cached
        assert cache.pages_cached == 3
        assert cache.stats.readaheads == 2

        # Pages 6 and 7 should be hits
        values_6, _ = self._exhaust(cache.read_page(6))
        values_7, _ = self._exhaust(cache.read_page(7))
        assert len(values_6) == 0
        assert len(values_7) == 0

    def test_flush(self):
        cache, sim = self._make_cache()

        self._exhaust(cache.write_page(1))
        self._exhaust(cache.write_page(2))
        assert cache.dirty_pages == 2

        values, flushed = self._exhaust(cache.flush())
        assert flushed == 2
        assert cache.dirty_pages == 0
        assert cache.stats.dirty_writebacks == 2

    def test_flush_no_dirty(self):
        cache, sim = self._make_cache()

        self._exhaust(cache.read_page(1))
        values, flushed = self._exhaust(cache.flush())
        assert flushed == 0

    def test_hit_rate(self):
        cache, sim = self._make_cache()

        self._exhaust(cache.read_page(1))  # miss
        self._exhaust(cache.read_page(1))  # hit
        self._exhaust(cache.read_page(1))  # hit

        assert cache.stats.hit_rate == pytest.approx(2.0 / 3.0)

    def test_repr(self):
        cache, sim = self._make_cache()
        assert "test_cache" in repr(cache)

    def test_handle_event_is_noop(self):
        cache, sim = self._make_cache()
        from happysimulator.core.event import Event
        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=cache,
        )
        result = cache.handle_event(event)
        assert result is None
