"""Log-Structured Merge Tree storage engine.

Composes WAL + Memtable + SSTable levels into a full LSM storage engine.
Supports pluggable compaction strategies (size-tiered, leveled, FIFO) and
tracks read/write/space amplification metrics.

Write path: WAL append -> Memtable put -> (flush to L0 if full) -> (compact if needed)
Read path: Memtable -> L0 SSTables -> L1 -> ... (bloom filter skips)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.components.storage.sstable import SSTable
from happysimulator.components.storage.wal import WriteAheadLog
from happysimulator.components.storage.memtable import Memtable

logger = logging.getLogger(__name__)

# Tombstone sentinel — distinct from None so we can delete keys
_TOMBSTONE = object()


# ---------------------------------------------------------------------------
# Compaction strategies
# ---------------------------------------------------------------------------


class CompactionStrategy(ABC):
    """Strategy for deciding when and what to compact."""

    @abstractmethod
    def should_compact(self, levels: list[list[SSTable]]) -> bool:
        """Return True if compaction should be triggered."""
        ...

    @abstractmethod
    def select_compaction(self, levels: list[list[SSTable]]) -> tuple[int, list[SSTable]]:
        """Select which level and SSTables to compact.

        Returns:
            (source_level, sstables_to_compact) tuple.
        """
        ...


class SizeTieredCompaction(CompactionStrategy):
    """Compact when a level accumulates enough similarly-sized SSTables.

    Triggers when any level has >= min_sstables. Merges all SSTables
    in the most populated level into the next level.

    Args:
        min_sstables: Minimum SSTables in a level to trigger compaction.
    """

    def __init__(self, min_sstables: int = 4) -> None:
        self.min_sstables = min_sstables

    def should_compact(self, levels: list[list[SSTable]]) -> bool:
        return any(len(level) >= self.min_sstables for level in levels)

    def select_compaction(self, levels: list[list[SSTable]]) -> tuple[int, list[SSTable]]:
        # Pick the most populated level
        best_level = 0
        best_count = 0
        for i, level in enumerate(levels):
            if len(level) > best_count:
                best_count = len(level)
                best_level = i
        return best_level, list(levels[best_level])


class LeveledCompaction(CompactionStrategy):
    """Level-based compaction with size ratio between levels.

    L0 compacts when it has too many SSTables. Other levels compact
    when total keys exceed level_size_limit = base_size * ratio^level.

    Args:
        level_0_max: Max SSTables in L0 before compaction.
        size_ratio: Size multiplier between adjacent levels.
        base_size_keys: Target key count for L1.
    """

    def __init__(
        self,
        level_0_max: int = 4,
        size_ratio: int = 10,
        base_size_keys: int = 1000,
    ) -> None:
        self.level_0_max = level_0_max
        self.size_ratio = size_ratio
        self.base_size_keys = base_size_keys

    def should_compact(self, levels: list[list[SSTable]]) -> bool:
        if not levels:
            return False
        # L0: too many SSTables
        if len(levels[0]) >= self.level_0_max:
            return True
        # Other levels: too many keys
        for i in range(1, len(levels)):
            limit = self.base_size_keys * (self.size_ratio ** i)
            total_keys = sum(s.key_count for s in levels[i])
            if total_keys > limit:
                return True
        return False

    def select_compaction(self, levels: list[list[SSTable]]) -> tuple[int, list[SSTable]]:
        # L0 first
        if levels and len(levels[0]) >= self.level_0_max:
            return 0, list(levels[0])
        # Then check other levels
        for i in range(1, len(levels)):
            limit = self.base_size_keys * (self.size_ratio ** i)
            total_keys = sum(s.key_count for s in levels[i])
            if total_keys > limit:
                return i, list(levels[i])
        # Fallback: compact L0
        return (0, list(levels[0])) if levels and levels[0] else (0, [])


class FIFOCompaction(CompactionStrategy):
    """Drop oldest SSTables when total count exceeds threshold.

    Simple TTL-like compaction for time-series data where old data
    can be discarded.

    Args:
        max_total_sstables: Maximum total SSTables across all levels.
    """

    def __init__(self, max_total_sstables: int = 100) -> None:
        self.max_total_sstables = max_total_sstables

    def should_compact(self, levels: list[list[SSTable]]) -> bool:
        total = sum(len(level) for level in levels)
        return total > self.max_total_sstables

    def select_compaction(self, levels: list[list[SSTable]]) -> tuple[int, list[SSTable]]:
        # Find the highest (oldest) non-empty level
        for i in range(len(levels) - 1, -1, -1):
            if levels[i]:
                return i, list(levels[i])
        return 0, []


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LSMTreeStats:
    """Frozen snapshot of LSM tree statistics.

    Attributes:
        writes: Total put/delete operations.
        reads: Total get operations.
        read_hits: Gets that found a value.
        read_misses: Gets that returned None.
        wal_writes: Total WAL append operations.
        memtable_flushes: Number of memtable flushes.
        compactions: Number of compaction operations.
        total_sstables: Current SSTable count across all levels.
        levels: Number of occupied levels.
        read_amplification: Average SSTables checked per read.
        write_amplification: Total SSTable bytes / user write bytes.
        space_amplification: Total stored bytes / logical data bytes.
        bloom_filter_saves: Reads avoided by bloom filter.
    """

    writes: int = 0
    reads: int = 0
    read_hits: int = 0
    read_misses: int = 0
    wal_writes: int = 0
    memtable_flushes: int = 0
    compactions: int = 0
    total_sstables: int = 0
    levels: int = 0
    read_amplification: float = 0.0
    write_amplification: float = 0.0
    space_amplification: float = 0.0
    bloom_filter_saves: int = 0


# ---------------------------------------------------------------------------
# LSMTree entity
# ---------------------------------------------------------------------------


class LSMTree(Entity):
    """Log-Structured Merge Tree storage engine.

    Composes WAL, Memtable, and SSTable levels. Write path appends to WAL
    then memtable, flushing to L0 SSTables when full. Reads check memtable
    first, then each level using bloom filters to skip irrelevant SSTables.

    Args:
        name: Entity name.
        memtable_size: Max entries in the memtable before flush.
        compaction_strategy: Strategy for compaction triggers and selection.
        wal: Optional WriteAheadLog for durability. Created internally if None.
        disk: Optional Resource for disk I/O contention.
        sstable_read_latency: Seconds per SSTable page read.
        sstable_write_latency: Seconds per SSTable page write.
        max_levels: Maximum number of levels.

    Example::

        lsm = LSMTree("db", memtable_size=1000,
                       compaction_strategy=LeveledCompaction())
        sim = Simulation(entities=[lsm], ...)
    """

    def __init__(
        self,
        name: str,
        *,
        memtable_size: int = 1000,
        compaction_strategy: CompactionStrategy | None = None,
        wal: WriteAheadLog | None = None,
        disk: Any | None = None,
        sstable_read_latency: float = 0.001,
        sstable_write_latency: float = 0.002,
        max_levels: int = 7,
    ) -> None:
        super().__init__(name)
        self._compaction_strategy = compaction_strategy or SizeTieredCompaction()
        self._wal = wal
        self._disk = disk
        self._sstable_read_latency = sstable_read_latency
        self._sstable_write_latency = sstable_write_latency
        self._max_levels = max_levels

        # Active memtable
        self._memtable = Memtable(f"{name}_memtable", size_threshold=memtable_size)

        # Immutable memtables awaiting flush (for reads during flush)
        self._immutable_memtables: list[Memtable] = []

        # SSTable levels: levels[0] is L0 (most recent)
        self._levels: list[list[SSTable]] = [[] for _ in range(max_levels)]

        # Logical data tracking for amplification metrics
        self._logical_data: dict[str, Any] = {}  # current logical state
        self._user_bytes_written: int = 0
        self._sstable_bytes_written: int = 0

        # Stats
        self._total_writes: int = 0
        self._total_reads: int = 0
        self._total_read_hits: int = 0
        self._total_read_misses: int = 0
        self._total_wal_writes: int = 0
        self._total_memtable_flushes: int = 0
        self._total_compactions: int = 0
        self._total_sstables_checked: int = 0
        self._total_bloom_saves: int = 0

    def set_clock(self, clock) -> None:
        """Inject clock into this entity and internal components."""
        super().set_clock(clock)
        self._memtable.set_clock(clock)
        if self._wal is not None:
            self._wal.set_clock(clock)

    @property
    def stats(self) -> LSMTreeStats:
        """Frozen snapshot of LSM tree statistics."""
        total_sst = sum(len(level) for level in self._levels)
        occupied_levels = sum(1 for level in self._levels if level)
        logical_bytes = len(self._logical_data) * 64

        read_amp = (
            self._total_sstables_checked / self._total_reads
            if self._total_reads > 0 else 0.0
        )
        write_amp = (
            self._sstable_bytes_written / self._user_bytes_written
            if self._user_bytes_written > 0 else 1.0
        )
        total_stored = sum(s.size_bytes for level in self._levels for s in level)
        space_amp = (
            total_stored / logical_bytes
            if logical_bytes > 0 else 1.0
        )

        return LSMTreeStats(
            writes=self._total_writes,
            reads=self._total_reads,
            read_hits=self._total_read_hits,
            read_misses=self._total_read_misses,
            wal_writes=self._total_wal_writes,
            memtable_flushes=self._total_memtable_flushes,
            compactions=self._total_compactions,
            total_sstables=total_sst,
            levels=occupied_levels,
            read_amplification=read_amp,
            write_amplification=write_amp,
            space_amplification=space_amp,
            bloom_filter_saves=self._total_bloom_saves,
        )

    @property
    def level_summary(self) -> list[dict]:
        """Summary of each level: count, total keys, total bytes."""
        result = []
        for i, level in enumerate(self._levels):
            if level:
                result.append({
                    "level": i,
                    "sstables": len(level),
                    "total_keys": sum(s.key_count for s in level),
                    "total_bytes": sum(s.size_bytes for s in level),
                })
        return result

    def put(self, key: str, value: Any) -> Generator[float, None, None]:
        """Write a key-value pair through WAL -> memtable -> (flush).

        Yields I/O latency for WAL and potential flush operations.
        """
        self._total_writes += 1
        self._user_bytes_written += 64
        self._logical_data[key] = value

        # WAL append
        if self._wal is not None:
            yield from self._wal.append(key, value)
            self._total_wal_writes += 1

        # Memtable put
        is_full = yield from self._memtable.put(key, value)

        # Flush if full
        if is_full:
            yield from self._flush_memtable()

    def put_sync(self, key: str, value: Any) -> None:
        """Write without yielding latency."""
        self._total_writes += 1
        self._user_bytes_written += 64
        self._logical_data[key] = value

        if self._wal is not None:
            self._wal.append_sync(key, value)
            self._total_wal_writes += 1

        is_full = self._memtable.put_sync(key, value)
        if is_full:
            self._flush_memtable_sync()

    def get(self, key: str) -> Generator[float, None, Any | None]:
        """Read a key, checking memtable then SSTable levels.

        Uses bloom filters to skip SSTables that don't contain the key.
        Yields I/O latency for SSTable reads.
        """
        self._total_reads += 1

        # Check active memtable first (no I/O)
        value = self._memtable.get_sync(key)
        if value is not None:
            self._total_read_hits += 1
            if value is _TOMBSTONE:
                return None
            return value

        # Check immutable memtables
        for imm in reversed(self._immutable_memtables):
            value = imm.get_sync(key)
            if value is not None:
                self._total_read_hits += 1
                if value is _TOMBSTONE:
                    return None
                return value

        # Check each level, L0 first (most recent)
        for level in self._levels:
            # L0: check all SSTables (may have overlapping key ranges)
            for sstable in reversed(level):
                self._total_sstables_checked += 1

                if not sstable.contains(key):
                    self._total_bloom_saves += 1
                    continue

                # Bloom says maybe — do the actual read
                page_reads = sstable.page_reads_for_get(key)
                if page_reads > 0:
                    yield page_reads * self._sstable_read_latency

                result = sstable.get(key)
                if result is not None:
                    self._total_read_hits += 1
                    if result is _TOMBSTONE:
                        return None
                    return result

        self._total_read_misses += 1
        return None

    def get_sync(self, key: str) -> Any | None:
        """Read without yielding latency."""
        self._total_reads += 1

        value = self._memtable.get_sync(key)
        if value is not None:
            self._total_read_hits += 1
            return None if value is _TOMBSTONE else value

        for imm in reversed(self._immutable_memtables):
            value = imm.get_sync(key)
            if value is not None:
                self._total_read_hits += 1
                return None if value is _TOMBSTONE else value

        for level in self._levels:
            for sstable in reversed(level):
                self._total_sstables_checked += 1
                if not sstable.contains(key):
                    self._total_bloom_saves += 1
                    continue
                result = sstable.get(key)
                if result is not None:
                    self._total_read_hits += 1
                    return None if result is _TOMBSTONE else result

        self._total_read_misses += 1
        return None

    def delete(self, key: str) -> Generator[float, None, None]:
        """Delete a key by writing a tombstone marker."""
        self._total_writes += 1
        self._user_bytes_written += 64
        self._logical_data.pop(key, None)

        if self._wal is not None:
            yield from self._wal.append(key, _TOMBSTONE)
            self._total_wal_writes += 1

        is_full = yield from self._memtable.put(key, _TOMBSTONE)
        if is_full:
            yield from self._flush_memtable()

    def scan(self, start_key: str, end_key: str) -> Generator[float, None, list[tuple[str, Any]]]:
        """Range scan across memtable and all SSTable levels.

        Returns merged, deduplicated results with most recent values.
        Yields I/O latency for SSTable reads.
        """
        # Collect from memtable
        merged: dict[str, Any] = {}
        for k, v in self._memtable._data.items():
            if start_key <= k < end_key:
                merged[k] = v

        # Collect from immutable memtables (newer first)
        for imm in reversed(self._immutable_memtables):
            for k, v in imm._data.items():
                if start_key <= k < end_key and k not in merged:
                    merged[k] = v

        # Collect from SSTables (newer levels first)
        for level in self._levels:
            for sstable in reversed(level):
                page_reads = sstable.page_reads_for_scan(start_key, end_key)
                if page_reads > 0:
                    yield page_reads * self._sstable_read_latency

                for k, v in sstable.scan(start_key, end_key):
                    if k not in merged:
                        merged[k] = v

        # Filter out tombstones and sort
        result = [(k, v) for k, v in sorted(merged.items()) if v is not _TOMBSTONE]
        return result

    def _flush_memtable(self) -> Generator[float, None, None]:
        """Flush the active memtable to an L0 SSTable."""
        if self._memtable.size == 0:
            return

        # Move active memtable to immutable list
        old_memtable = self._memtable
        self._immutable_memtables.append(old_memtable)

        # Create new active memtable
        self._memtable = Memtable(
            f"{self.name}_memtable",
            size_threshold=old_memtable._size_threshold,
        )
        self._memtable.set_clock(self._clock)

        # Flush to SSTable
        sstable = old_memtable.flush()
        self._sstable_bytes_written += sstable.size_bytes

        # Write latency for creating SSTable on disk
        pages = max(1, sstable.key_count // 16)
        yield pages * self._sstable_write_latency

        # Add to L0
        self._levels[0].append(sstable)
        self._total_memtable_flushes += 1

        # Remove from immutable list
        self._immutable_memtables.remove(old_memtable)

        # Truncate WAL
        if self._wal is not None:
            self._wal.truncate(self._wal._next_sequence - 1)

        logger.debug(
            "[%s] Flushed memtable to L0 SSTable(%d keys), L0 now has %d SSTables",
            self.name, sstable.key_count, len(self._levels[0]),
        )

        # Check if compaction needed
        if self._compaction_strategy.should_compact(self._levels):
            yield from self._compact()

    def _flush_memtable_sync(self) -> None:
        """Flush without yielding latency."""
        if self._memtable.size == 0:
            return

        sstable = self._memtable.flush()
        self._sstable_bytes_written += sstable.size_bytes
        self._levels[0].append(sstable)
        self._total_memtable_flushes += 1

        # Reset memtable (flush() already clears it)

        if self._wal is not None:
            self._wal.truncate(self._wal._next_sequence - 1)

        if self._compaction_strategy.should_compact(self._levels):
            self._compact_sync()

    def _compact(self) -> Generator[float, None, None]:
        """Run a compaction cycle."""
        source_level, sstables = self._compaction_strategy.select_compaction(self._levels)
        if not sstables:
            return

        target_level = min(source_level + 1, self._max_levels - 1)

        # Merge all selected SSTables
        merged_data: dict[str, Any] = {}
        # Process from oldest to newest so newer values win
        for sst in sstables:
            for k, v in sst.scan():
                merged_data[k] = v

        # Also include overlapping SSTables from target level
        overlapping = []
        if target_level != source_level:
            for sst in self._levels[target_level]:
                if any(sst.overlaps(s) for s in sstables):
                    overlapping.append(sst)
                    for k, v in sst.scan():
                        if k not in merged_data:
                            merged_data[k] = v

        # Filter out tombstones in the deepest level
        if target_level == self._max_levels - 1:
            merged_data = {k: v for k, v in merged_data.items() if v is not _TOMBSTONE}

        # Create new SSTable
        data_list = sorted(merged_data.items())
        if data_list:
            new_sst = SSTable(data_list, level=target_level, sequence=self._total_compactions)
            self._sstable_bytes_written += new_sst.size_bytes

            # Write latency
            pages = max(1, new_sst.key_count // 16)
            yield pages * self._sstable_write_latency

            # Remove old SSTables and add new one
            for sst in sstables:
                if sst in self._levels[source_level]:
                    self._levels[source_level].remove(sst)
            for sst in overlapping:
                if sst in self._levels[target_level]:
                    self._levels[target_level].remove(sst)
            self._levels[target_level].append(new_sst)

        self._total_compactions += 1
        logger.debug(
            "[%s] Compacted L%d -> L%d (%d SSTables merged)",
            self.name, source_level, target_level, len(sstables) + len(overlapping),
        )

    def _compact_sync(self) -> None:
        """Run compaction without yielding latency."""
        source_level, sstables = self._compaction_strategy.select_compaction(self._levels)
        if not sstables:
            return

        target_level = min(source_level + 1, self._max_levels - 1)

        merged_data: dict[str, Any] = {}
        for sst in sstables:
            for k, v in sst.scan():
                merged_data[k] = v

        overlapping = []
        if target_level != source_level:
            for sst in self._levels[target_level]:
                if any(sst.overlaps(s) for s in sstables):
                    overlapping.append(sst)
                    for k, v in sst.scan():
                        if k not in merged_data:
                            merged_data[k] = v

        if target_level == self._max_levels - 1:
            merged_data = {k: v for k, v in merged_data.items() if v is not _TOMBSTONE}

        data_list = sorted(merged_data.items())
        if data_list:
            new_sst = SSTable(data_list, level=target_level, sequence=self._total_compactions)
            self._sstable_bytes_written += new_sst.size_bytes

            for sst in sstables:
                if sst in self._levels[source_level]:
                    self._levels[source_level].remove(sst)
            for sst in overlapping:
                if sst in self._levels[target_level]:
                    self._levels[target_level].remove(sst)
            self._levels[target_level].append(new_sst)

        self._total_compactions += 1

    def crash(self) -> dict:
        """Simulate power loss: lose memtable and unsynced WAL entries.

        SSTables survive (already flushed to disk). The memtable and any
        immutable memtables awaiting flush are lost (volatile memory).
        WAL entries beyond the last sync point are also lost.

        Returns:
            Summary with counts of lost entries.
        """
        memtable_lost = self._memtable.size
        immutable_lost = sum(m.size for m in self._immutable_memtables)

        # Clear volatile state
        self._memtable = Memtable(
            f"{self.name}_memtable",
            size_threshold=self._memtable._size_threshold,
        )
        if self._clock is not None:
            self._memtable.set_clock(self._clock)
        self._immutable_memtables.clear()

        # Crash WAL — discard unsynced entries
        wal_lost = 0
        if self._wal is not None:
            wal_lost = self._wal.crash()

        return {
            "memtable_entries_lost": memtable_lost,
            "immutable_memtable_entries_lost": immutable_lost,
            "wal_entries_lost": wal_lost,
        }

    def recover_from_crash(self) -> dict:
        """Recover durable state after a crash.

        Replays surviving WAL entries into a fresh memtable. SSTables
        are already intact on disk and need no recovery.

        Returns:
            Summary with counts of recovered data.
        """
        wal_recovered = 0
        if self._wal is not None:
            entries = self._wal.recover()
            for entry in entries:
                self._memtable.put_sync(entry.key, entry.value)
            wal_recovered = len(entries)

        sstable_keys = sum(
            s.key_count for level in self._levels for s in level
        )

        return {
            "wal_entries_replayed": wal_recovered,
            "sstable_keys": sstable_keys,
            "total_keys_recovered": self._memtable.size + sstable_keys,
        }

    def handle_event(self, event: Event) -> Generator[float, None, None] | None:
        """Handle CompactionTrigger events."""
        if event.event_type == "CompactionTrigger":
            if self._compaction_strategy.should_compact(self._levels):
                return self._compact()
        return None

    def __repr__(self) -> str:
        total_sst = sum(len(level) for level in self._levels)
        return (
            f"LSMTree('{self.name}', memtable={self._memtable.size}, "
            f"sstables={total_sst}, compactions={self._total_compactions})"
        )
