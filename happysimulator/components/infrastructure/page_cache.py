"""OS page cache model with LRU eviction, read-ahead, and writeback.

Sits between storage engines and DiskIO, caching recently accessed pages
in memory. Read-ahead prefetches adjacent pages on sequential access.
Dirty pages from writes are flushed to disk periodically or on eviction.

Key behaviors:
- LRU eviction when cache exceeds capacity.
- Read-ahead prefetch of adjacent pages on cache miss.
- Write-back: writes go to cache as dirty pages, flushed asynchronously.
- Dirty page eviction forces synchronous writeback before eviction.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class _CachedPage:
    """Internal: a page held in the cache."""

    page_id: int
    dirty: bool = False


@dataclass(frozen=True)
class PageCacheStats:
    """Frozen snapshot of page cache statistics.

    Attributes:
        hits: Cache hit count.
        misses: Cache miss count.
        evictions: Pages evicted from cache.
        dirty_writebacks: Dirty pages written back to disk.
        readaheads: Pages prefetched via read-ahead.
        pages_cached: Current number of cached pages.
        dirty_pages: Current number of dirty pages.
    """

    hits: int
    misses: int
    evictions: int
    dirty_writebacks: int
    readaheads: int
    pages_cached: int
    dirty_pages: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# PageCache entity
# ---------------------------------------------------------------------------


class PageCache(Entity):
    """OS page cache with LRU eviction, read-ahead, and writeback.

    Provides ``read_page()`` and ``write_page()`` generator methods that
    check the cache first, falling back to disk I/O on miss.

    Args:
        name: Entity name.
        capacity_pages: Maximum number of pages in cache.
        page_size_bytes: Size of each page in bytes (default 4096).
        readahead_pages: Number of pages to prefetch on miss (default 0).
        disk_read_latency_s: Latency for reading a page from disk.
        disk_write_latency_s: Latency for writing a page to disk.

    Example::

        cache = PageCache("os_cache", capacity_pages=1000, readahead_pages=4)
        sim = Simulation(entities=[cache, ...], ...)

        # In another entity's handle_event:
        yield from cache.read_page(page_id=42)
        yield from cache.write_page(page_id=42)
    """

    def __init__(
        self,
        name: str,
        *,
        capacity_pages: int = 1000,
        page_size_bytes: int = 4096,
        readahead_pages: int = 0,
        disk_read_latency_s: float = 0.0001,
        disk_write_latency_s: float = 0.0002,
    ) -> None:
        if capacity_pages < 1:
            raise ValueError(f"capacity_pages must be >= 1, got {capacity_pages}")

        super().__init__(name)
        self._capacity = capacity_pages
        self._page_size = page_size_bytes
        self._readahead = readahead_pages
        self._disk_read_latency_s = disk_read_latency_s
        self._disk_write_latency_s = disk_write_latency_s

        # LRU cache: OrderedDict with most-recently-used at end
        self._pages: OrderedDict[int, _CachedPage] = OrderedDict()

        # Stats
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._dirty_writebacks: int = 0
        self._readaheads: int = 0

    @property
    def pages_cached(self) -> int:
        """Number of pages currently in cache."""
        return len(self._pages)

    @property
    def dirty_pages(self) -> int:
        """Number of dirty pages in cache."""
        return sum(1 for p in self._pages.values() if p.dirty)

    @property
    def stats(self) -> PageCacheStats:
        """Frozen snapshot of page cache statistics."""
        return PageCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            dirty_writebacks=self._dirty_writebacks,
            readaheads=self._readaheads,
            pages_cached=self.pages_cached,
            dirty_pages=self.dirty_pages,
        )

    def _touch(self, page_id: int) -> None:
        """Move page to most-recently-used position."""
        self._pages.move_to_end(page_id)

    def _evict_one(self) -> Generator[float, None, None]:
        """Evict the least-recently-used page, flushing if dirty."""
        if not self._pages:
            return

        oldest_id, oldest = next(iter(self._pages.items()))
        if oldest.dirty:
            yield self._disk_write_latency_s
            self._dirty_writebacks += 1

        del self._pages[oldest_id]
        self._evictions += 1

    def _ensure_space(self) -> Generator[float, None, None]:
        """Evict pages until there is room for at least one new page."""
        while len(self._pages) >= self._capacity:
            yield from self._evict_one()

    def _load_page(self, page_id: int) -> Generator[float, None, None]:
        """Load a page from disk into cache."""
        yield from self._ensure_space()
        yield self._disk_read_latency_s
        self._pages[page_id] = _CachedPage(page_id=page_id)

    def read_page(self, page_id: int) -> Generator[float, None, None]:
        """Read a page, serving from cache if present.

        On cache miss, loads from disk and optionally prefetches
        adjacent pages via read-ahead.
        """
        if page_id in self._pages:
            self._hits += 1
            self._touch(page_id)
            return

        self._misses += 1
        yield from self._load_page(page_id)

        # Read-ahead: prefetch adjacent pages
        for i in range(1, self._readahead + 1):
            ahead_id = page_id + i
            if ahead_id not in self._pages and len(self._pages) < self._capacity:
                yield from self._ensure_space()
                yield self._disk_read_latency_s
                self._pages[ahead_id] = _CachedPage(page_id=ahead_id)
                self._readaheads += 1

    def write_page(self, page_id: int) -> Generator[float, None, None]:
        """Write a page to cache, marking it dirty.

        If the page is already cached, it is updated in place.
        Otherwise, space is made and a new dirty page is inserted.
        """
        if page_id in self._pages:
            self._hits += 1
            self._pages[page_id].dirty = True
            self._touch(page_id)
            return

        self._misses += 1
        yield from self._ensure_space()
        self._pages[page_id] = _CachedPage(page_id=page_id, dirty=True)

    def flush(self) -> Generator[float, None, int]:
        """Flush all dirty pages to disk.

        Returns the number of pages flushed.
        """
        flushed = 0
        for page in self._pages.values():
            if page.dirty:
                yield self._disk_write_latency_s
                page.dirty = False
                self._dirty_writebacks += 1
                flushed += 1
        return flushed

    def handle_event(self, event: Event) -> None:
        """PageCache does not process events directly."""
        pass

    def __repr__(self) -> str:
        return (
            f"PageCache('{self.name}', cached={self.pages_cached}/{self._capacity}, "
            f"dirty={self.dirty_pages})"
        )
