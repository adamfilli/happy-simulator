"""In-memory sorted write buffer for LSM-tree storage engines.

A Memtable accumulates writes in a sorted in-memory structure. When it
reaches a size threshold, it is flushed to an SSTable on disk. Reads
check the memtable first (most recent data) before falling through to
SSTables.

Uses a Python dict internally (sorted on flush), modeling the performance
characteristics of a skip list or red-black tree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.components.storage.sstable import SSTable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemtableStats:
    """Frozen snapshot of Memtable statistics.

    Attributes:
        writes: Total put operations.
        reads: Total get operations.
        hits: Get operations that found the key.
        misses: Get operations that did not find the key.
        flushes: Number of times the memtable was flushed to SSTable.
        current_size: Current number of entries.
        total_bytes_written: Estimated total bytes written.
    """

    writes: int = 0
    reads: int = 0
    hits: int = 0
    misses: int = 0
    flushes: int = 0
    current_size: int = 0
    total_bytes_written: int = 0


class Memtable(Entity):
    """In-memory sorted write buffer that flushes to SSTables.

    Writes are accumulated in memory until the size threshold is reached.
    On flush, the contents are frozen into an immutable SSTable and the
    memtable is cleared.

    Args:
        name: Entity name.
        size_threshold: Max number of entries before the memtable is full.
        write_latency: Seconds per put operation.
        read_latency: Seconds per get operation.
        rwlock: Optional RWLock for concurrent read/write modeling.

    Example::

        memtable = Memtable("mem", size_threshold=1000)
        is_full = yield from memtable.put("key1", "value1")
        if is_full:
            sstable = memtable.flush()
    """

    def __init__(
        self,
        name: str,
        *,
        size_threshold: int = 1000,
        write_latency: float = 0.00001,
        read_latency: float = 0.000005,
        rwlock: Any | None = None,
    ) -> None:
        super().__init__(name)
        self._size_threshold = size_threshold
        self._write_latency = write_latency
        self._read_latency = read_latency
        self._rwlock = rwlock

        self._data: dict[str, Any] = {}
        self._sequence: int = 0  # Tracks flush sequence for SSTable numbering

        # Stats
        self._total_writes: int = 0
        self._total_reads: int = 0
        self._total_hits: int = 0
        self._total_misses: int = 0
        self._total_flushes: int = 0
        self._total_bytes_written: int = 0

    @property
    def is_full(self) -> bool:
        """Whether the memtable has reached its size threshold."""
        return len(self._data) >= self._size_threshold

    @property
    def size(self) -> int:
        """Current number of entries in the memtable."""
        return len(self._data)

    @property
    def stats(self) -> MemtableStats:
        """Frozen snapshot of memtable statistics."""
        return MemtableStats(
            writes=self._total_writes,
            reads=self._total_reads,
            hits=self._total_hits,
            misses=self._total_misses,
            flushes=self._total_flushes,
            current_size=len(self._data),
            total_bytes_written=self._total_bytes_written,
        )

    def put(self, key: str, value: Any) -> Generator[float, None, bool]:
        """Write a key-value pair, yielding write latency.

        Returns True if the memtable is now full and should be flushed.
        """
        self._data[key] = value
        self._total_writes += 1
        self._total_bytes_written += 64  # estimate
        yield self._write_latency
        return self.is_full

    def put_sync(self, key: str, value: Any) -> bool:
        """Write without yielding latency. For testing or fast paths.

        Returns True if the memtable is now full.
        """
        self._data[key] = value
        self._total_writes += 1
        self._total_bytes_written += 64
        return self.is_full

    def get(self, key: str) -> Generator[float, None, Any | None]:
        """Look up a key, yielding read latency.

        Returns the value if found, None otherwise.
        """
        self._total_reads += 1
        yield self._read_latency
        value = self._data.get(key)
        if value is not None:
            self._total_hits += 1
        else:
            self._total_misses += 1
        return value

    def get_sync(self, key: str) -> Any | None:
        """Look up without yielding latency."""
        self._total_reads += 1
        value = self._data.get(key)
        if value is not None:
            self._total_hits += 1
        else:
            self._total_misses += 1
        return value

    def contains(self, key: str) -> bool:
        """Check if a key is in the memtable (no I/O cost)."""
        return key in self._data

    def flush(self) -> SSTable:
        """Freeze contents into an SSTable and clear the memtable.

        Returns the new SSTable containing all current entries.
        """
        data = [(k, v) for k, v in self._data.items()]
        sstable = SSTable(data, level=0, sequence=self._sequence)
        self._sequence += 1
        self._total_flushes += 1
        self._data.clear()
        logger.debug(
            "[%s] Flushed %d entries to SSTable(seq=%d)",
            self.name, sstable.key_count, sstable.sequence,
        )
        return sstable

    def handle_event(self, event: Event) -> None:
        """Memtable does not process events directly."""
        pass

    def __repr__(self) -> str:
        return (
            f"Memtable('{self.name}', size={len(self._data)}/{self._size_threshold}, "
            f"flushes={self._total_flushes})"
        )
