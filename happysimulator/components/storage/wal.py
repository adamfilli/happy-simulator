"""Write-Ahead Log for crash recovery and durability.

Append-only durability log with pluggable sync policies. Every write is
recorded before being applied, enabling recovery after crashes. The sync
policy controls the tradeoff between durability and throughput.

Sync policies:
- SyncEveryWrite: Maximum durability, lowest throughput
- SyncPeriodic: Sync at regular time intervals
- SyncOnBatch: Sync after N accumulated writes
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sync policies
# ---------------------------------------------------------------------------


class SyncPolicy(ABC):
    """Strategy for when to fsync the WAL to disk."""

    @abstractmethod
    def should_sync(self, writes_since_sync: int, time_since_sync_s: float) -> bool:
        """Return True if the WAL should be synced now."""
        ...


class SyncEveryWrite(SyncPolicy):
    """Sync after every write — maximum durability."""

    def should_sync(self, writes_since_sync: int, time_since_sync_s: float) -> bool:
        return True


class SyncPeriodic(SyncPolicy):
    """Sync when a time interval has elapsed since the last sync.

    Args:
        interval_s: Seconds between syncs.
    """

    def __init__(self, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError(f"interval_s must be > 0, got {interval_s}")
        self.interval_s = interval_s

    def should_sync(self, writes_since_sync: int, time_since_sync_s: float) -> bool:
        return time_since_sync_s >= self.interval_s


class SyncOnBatch(SyncPolicy):
    """Sync after a fixed number of writes accumulate.

    Args:
        batch_size: Number of writes before syncing.
    """

    def __init__(self, batch_size: int) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.batch_size = batch_size

    def should_sync(self, writes_since_sync: int, time_since_sync_s: float) -> bool:
        return writes_since_sync >= self.batch_size


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WALEntry:
    """A single entry in the write-ahead log.

    Attributes:
        sequence_number: Monotonically increasing ID for this entry.
        key: The key being written.
        value: The value being written.
        timestamp_s: Simulation time when the entry was written.
    """

    sequence_number: int
    key: str
    value: Any
    timestamp_s: float


@dataclass(frozen=True)
class WALStats:
    """Frozen snapshot of WAL statistics.

    Attributes:
        writes: Total append operations.
        bytes_written: Estimated bytes written.
        syncs: Number of fsync operations performed.
        total_sync_latency_s: Cumulative sync latency.
        entries_recovered: Number of entries returned by last recover().
    """

    writes: int = 0
    bytes_written: int = 0
    syncs: int = 0
    total_sync_latency_s: float = 0.0
    entries_recovered: int = 0


# ---------------------------------------------------------------------------
# WriteAheadLog entity
# ---------------------------------------------------------------------------


class WriteAheadLog(Entity):
    """Append-only durability log with pluggable sync policies.

    Records key-value writes for crash recovery. The sync policy controls
    when data is flushed to persistent storage. Optional disk Resource
    models I/O contention.

    Args:
        name: Entity name.
        sync_policy: When to fsync. Defaults to SyncEveryWrite.
        disk: Optional Resource for disk I/O contention modeling.
        write_latency: Seconds per append operation.
        sync_latency: Seconds per fsync operation.

    Example::

        wal = WriteAheadLog("wal", sync_policy=SyncOnBatch(batch_size=10))
        sim = Simulation(entities=[wal], ...)

        # In another entity's handle_event:
        seq = yield from wal.append("key1", "value1")
    """

    def __init__(
        self,
        name: str,
        *,
        sync_policy: SyncPolicy | None = None,
        disk: Any | None = None,
        write_latency: float = 0.0001,
        sync_latency: float = 0.001,
    ) -> None:
        super().__init__(name)
        self._sync_policy = sync_policy or SyncEveryWrite()
        self._disk = disk
        self._write_latency = write_latency
        self._sync_latency = sync_latency

        self._entries: list[WALEntry] = []
        self._next_sequence: int = 1
        self._writes_since_sync: int = 0
        self._last_sync_time_s: float = 0.0
        self._synced_up_to_sequence: int = 0

        # Stats
        self._total_writes: int = 0
        self._total_bytes: int = 0
        self._total_syncs: int = 0
        self._total_sync_latency_s: float = 0.0
        self._entries_recovered: int = 0

    @property
    def synced_up_to(self) -> int:
        """Sequence number of the last entry synced to durable storage."""
        return self._synced_up_to_sequence

    @property
    def size(self) -> int:
        """Number of entries currently in the log."""
        return len(self._entries)

    @property
    def stats(self) -> WALStats:
        """Frozen snapshot of WAL statistics."""
        return WALStats(
            writes=self._total_writes,
            bytes_written=self._total_bytes,
            syncs=self._total_syncs,
            total_sync_latency_s=self._total_sync_latency_s,
            entries_recovered=self._entries_recovered,
        )

    def append(self, key: str, value: Any) -> Generator[float, None, int]:
        """Append a key-value pair to the log, yielding I/O latency.

        Returns the sequence number assigned to this entry.
        """
        seq = self._next_sequence
        self._next_sequence += 1

        now_s = self.now.to_seconds()
        entry = WALEntry(
            sequence_number=seq,
            key=key,
            value=value,
            timestamp_s=now_s,
        )
        self._entries.append(entry)

        # Estimate 64 bytes per entry
        self._total_bytes += 64
        self._total_writes += 1
        self._writes_since_sync += 1

        # Write latency
        yield self._write_latency

        # Check sync policy
        time_since_sync = self.now.to_seconds() - self._last_sync_time_s
        if self._sync_policy.should_sync(self._writes_since_sync, time_since_sync):
            yield self._sync_latency
            self._synced_up_to_sequence = seq
            self._total_syncs += 1
            self._total_sync_latency_s += self._sync_latency
            self._writes_since_sync = 0
            self._last_sync_time_s = self.now.to_seconds()

        return seq

    def append_sync(self, key: str, value: Any) -> int:
        """Append without yielding I/O latency. For testing or fast paths.

        Returns the sequence number assigned to this entry.
        """
        seq = self._next_sequence
        self._next_sequence += 1

        now_s = self.now.to_seconds() if self._clock is not None else 0.0
        entry = WALEntry(
            sequence_number=seq,
            key=key,
            value=value,
            timestamp_s=now_s,
        )
        self._entries.append(entry)
        self._total_bytes += 64
        self._total_writes += 1
        self._writes_since_sync += 1

        return seq

    def recover(self) -> list[WALEntry]:
        """Return all entries in the log for crash recovery.

        Returns entries in sequence order.
        """
        result = sorted(self._entries, key=lambda e: e.sequence_number)
        self._entries_recovered = len(result)
        return result

    def truncate(self, up_to_sequence: int) -> None:
        """Remove entries with sequence_number <= up_to_sequence.

        Called after a checkpoint to reclaim space.
        """
        self._entries = [e for e in self._entries if e.sequence_number > up_to_sequence]

    def crash(self) -> int:
        """Simulate power loss: discard entries not yet synced to disk.

        Entries with sequence_number > synced_up_to are lost — they were
        in volatile memory only. Returns the number of entries lost.
        """
        before = len(self._entries)
        self._entries = [
            e for e in self._entries
            if e.sequence_number <= self._synced_up_to_sequence
        ]
        lost = before - len(self._entries)
        self._writes_since_sync = 0
        return lost

    def handle_event(self, event: Event) -> None:
        """WAL does not process events directly."""
        pass

    def __repr__(self) -> str:
        return f"WriteAheadLog('{self.name}', entries={len(self._entries)}, writes={self._total_writes})"
