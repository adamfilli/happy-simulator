"""Immutable sorted-string table for LSM-tree storage engines.

An SSTable stores a sorted sequence of key-value pairs with a sparse index
and bloom filter for efficient lookups. SSTables are immutable once created
and form the on-disk component of LSM-tree storage.

Key properties:
- Sorted by key for efficient range scans and merging
- Sparse index for O(log n) point lookups
- Bloom filter to skip tables that definitely don't contain a key
- Page-level I/O cost tracking for simulation

This is a pure data structure, NOT an Entity.
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from happysimulator.sketching.bloom_filter import BloomFilter


@dataclass(frozen=True)
class SSTableStats:
    """Frozen snapshot of SSTable statistics.

    Attributes:
        key_count: Number of key-value pairs stored.
        size_bytes: Estimated size in bytes.
        index_entries: Number of sparse index entries.
        bloom_filter_fp_rate: Current bloom filter false positive rate.
        bloom_filter_size_bits: Size of bloom filter bit array.
    """

    key_count: int = 0
    size_bytes: int = 0
    index_entries: int = 0
    bloom_filter_fp_rate: float = 0.0
    bloom_filter_size_bits: int = 0


class SSTable:
    """Immutable sorted-string table with bloom filter and sparse index.

    Stores a sorted collection of key-value pairs. Once created, data cannot
    be modified. Provides efficient point lookups via bloom filter + sparse
    index, and range scans via sorted iteration.

    Args:
        data: List of (key, value) tuples. Will be sorted by key.
        index_interval: Build one index entry per this many keys.
            Smaller values use more memory but enable faster lookups.
        bloom_fp_rate: Target false positive rate for the bloom filter.
        level: LSM level this SSTable belongs to (0 = most recent).
        sequence: Sequence number for ordering within a level.

    Example::

        sstable = SSTable([("a", 1), ("b", 2), ("c", 3)])
        assert sstable.get("b") == 2
        assert sstable.contains("a")
        assert not sstable.contains("z")
    """

    def __init__(
        self,
        data: list[tuple[str, Any]],
        *,
        index_interval: int = 16,
        bloom_fp_rate: float = 0.01,
        level: int = 0,
        sequence: int = 0,
    ) -> None:
        if index_interval < 1:
            raise ValueError(f"index_interval must be >= 1, got {index_interval}")
        if not 0 < bloom_fp_rate < 1:
            raise ValueError(f"bloom_fp_rate must be in (0, 1), got {bloom_fp_rate}")

        # Sort data by key
        self._data = sorted(data, key=lambda kv: kv[0])
        self._keys = [kv[0] for kv in self._data]
        self._values = [kv[1] for kv in self._data]
        self._level = level
        self._sequence = sequence
        self._index_interval = index_interval

        # Build sparse index: maps index_key -> position in _data
        self._index_keys: list[str] = []
        self._index_positions: list[int] = []
        for i in range(0, len(self._data), index_interval):
            self._index_keys.append(self._keys[i])
            self._index_positions.append(i)

        # Build bloom filter
        n = max(len(self._data), 1)
        self._bloom: BloomFilter = BloomFilter.from_expected_items(n=n, fp_rate=bloom_fp_rate)
        for key, _ in self._data:
            self._bloom.add(key)

        # Estimate size: ~64 bytes per key-value pair (rough estimate)
        self._size_bytes = len(self._data) * 64

    @property
    def key_count(self) -> int:
        """Number of key-value pairs in this SSTable."""
        return len(self._data)

    @property
    def size_bytes(self) -> int:
        """Estimated size in bytes."""
        return self._size_bytes

    @property
    def level(self) -> int:
        """LSM level this SSTable belongs to."""
        return self._level

    @property
    def sequence(self) -> int:
        """Sequence number for ordering within a level."""
        return self._sequence

    @property
    def min_key(self) -> str | None:
        """Smallest key in this SSTable, or None if empty."""
        return self._keys[0] if self._keys else None

    @property
    def max_key(self) -> str | None:
        """Largest key in this SSTable, or None if empty."""
        return self._keys[-1] if self._keys else None

    @property
    def bloom_filter(self) -> BloomFilter:
        """The bloom filter for this SSTable."""
        return self._bloom

    @property
    def stats(self) -> SSTableStats:
        """Frozen snapshot of SSTable statistics."""
        return SSTableStats(
            key_count=len(self._data),
            size_bytes=self._size_bytes,
            index_entries=len(self._index_keys),
            bloom_filter_fp_rate=self._bloom.false_positive_rate,
            bloom_filter_size_bits=self._bloom.size_bits,
        )

    def contains(self, key: str) -> bool:
        """Check if this SSTable might contain the key (bloom filter).

        Returns False only if the key is definitely not present.
        May return True for keys not actually in the table (false positive).
        """
        return self._bloom.contains(key)

    def get(self, key: str) -> Any | None:
        """Look up a key using sparse index + binary search.

        Returns the value if found, None otherwise.
        """
        if not self._bloom.contains(key):
            return None

        # Use sparse index to narrow search range
        start, end = self._index_range_for(key)

        # Binary search within the range
        idx = bisect.bisect_left(self._keys, key, start, end)
        if idx < end and self._keys[idx] == key:
            return self._values[idx]
        return None

    def scan(
        self, start_key: str | None = None, end_key: str | None = None
    ) -> list[tuple[str, Any]]:
        """Return all key-value pairs in [start_key, end_key) range.

        Args:
            start_key: Inclusive lower bound. None means from the beginning.
            end_key: Exclusive upper bound. None means to the end.

        Returns:
            Sorted list of (key, value) tuples in the range.
        """
        if start_key is None:
            lo = 0
        else:
            lo = bisect.bisect_left(self._keys, start_key)

        if end_key is None:
            hi = len(self._keys)
        else:
            hi = bisect.bisect_left(self._keys, end_key)

        return list(self._data[lo:hi])

    def page_reads_for_get(self, key: str) -> int:
        """Estimate page reads needed for a point lookup.

        Returns 0 if the bloom filter says the key is absent.
        Otherwise returns 1 (index page) + 1 (data page) = 2.
        """
        if not self._bloom.contains(key):
            return 0
        if len(self._data) == 0:
            return 0
        # One index page read + one data page read
        return 2

    def page_reads_for_scan(self, start_key: str | None = None, end_key: str | None = None) -> int:
        """Estimate page reads for a range scan.

        Returns the number of data pages that would be read,
        based on index_interval as page size.
        """
        if len(self._data) == 0:
            return 0

        if start_key is None:
            lo = 0
        else:
            lo = bisect.bisect_left(self._keys, start_key)

        if end_key is None:
            hi = len(self._keys)
        else:
            hi = bisect.bisect_left(self._keys, end_key)

        n_keys = hi - lo
        if n_keys <= 0:
            return 0
        # One page per index_interval keys, plus one index page
        return 1 + (n_keys + self._index_interval - 1) // self._index_interval

    def overlaps(self, other: SSTable) -> bool:
        """Check if this SSTable's key range overlaps with another's."""
        if not self._keys or not other._keys:
            return False
        return self._keys[0] <= other._keys[-1] and other._keys[0] <= self._keys[-1]

    def _index_range_for(self, key: str) -> tuple[int, int]:
        """Find the data range to search for a key using the sparse index."""
        if not self._index_keys:
            return 0, len(self._keys)

        idx = bisect.bisect_right(self._index_keys, key) - 1
        if idx < 0:
            start = 0
        else:
            start = self._index_positions[idx]

        if idx + 1 < len(self._index_positions):
            end = self._index_positions[idx + 1]
        else:
            end = len(self._keys)

        return start, end

    def __repr__(self) -> str:
        key_range = ""
        if self._keys:
            key_range = f", keys=[{self._keys[0]!r}..{self._keys[-1]!r}]"
        return (
            f"SSTable(level={self._level}, seq={self._sequence}, "
            f"count={len(self._data)}{key_range})"
        )

    def __len__(self) -> int:
        return len(self._data)
