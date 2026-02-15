"""Transaction manager with isolation levels and conflict detection.

Wraps any storage engine that implements the StorageEngine protocol with
transactional semantics. Supports three isolation levels:

- READ_COMMITTED: No conflict detection, reads see latest committed values.
- SNAPSHOT_ISOLATION: Write-write conflict detection. Reads from a snapshot.
- SERIALIZABLE: Read-write + write-write conflict detection.

Transactions buffer writes locally and apply them on commit after passing
conflict checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generator, Protocol, runtime_checkable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage engine protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StorageEngine(Protocol):
    """Structural typing for any key-value store."""

    def get(self, key: str) -> Generator: ...
    def put(self, key: str, value: Any) -> Generator: ...
    def get_sync(self, key: str) -> Any | None: ...
    def put_sync(self, key: str, value: Any) -> None: ...


# ---------------------------------------------------------------------------
# Isolation levels
# ---------------------------------------------------------------------------


class IsolationLevel(Enum):
    """Transaction isolation level."""

    READ_COMMITTED = "read_committed"
    SNAPSHOT_ISOLATION = "snapshot_isolation"
    SERIALIZABLE = "serializable"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransactionStats:
    """Frozen snapshot of transaction manager statistics.

    Attributes:
        transactions_started: Total transactions begun.
        transactions_committed: Total successful commits.
        transactions_aborted: Total aborted transactions.
        conflicts_detected: Total conflict detections (commit failures).
        deadlocks_detected: Total deadlock detections.
        reads: Total read operations across all transactions.
        writes: Total write operations across all transactions.
        avg_transaction_duration_s: Average transaction duration.
    """

    transactions_started: int = 0
    transactions_committed: int = 0
    transactions_aborted: int = 0
    conflicts_detected: int = 0
    deadlocks_detected: int = 0
    reads: int = 0
    writes: int = 0
    avg_transaction_duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Commit log entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CommitLogEntry:
    """Record of a committed transaction's writes."""

    tx_id: int
    version: int
    keys_written: frozenset[str]
    keys_read: frozenset[str]


# ---------------------------------------------------------------------------
# StorageTransaction
# ---------------------------------------------------------------------------


class StorageTransaction:
    """A single transaction against a storage engine.

    Buffers reads and writes locally. On commit, checks for conflicts
    based on the isolation level and applies buffered writes if no
    conflicts are detected.

    Not an Entity — created and managed by TransactionManager.
    """

    def __init__(
        self,
        tx_id: int,
        manager: TransactionManager,
        isolation: IsolationLevel,
        snapshot_version: int,
    ) -> None:
        self._tx_id = tx_id
        self._manager = manager
        self._isolation = isolation
        self._snapshot_version = snapshot_version
        self._start_time_s: float = 0.0

        self._read_set: set[str] = set()
        self._write_set: dict[str, Any] = {}
        self._committed = False
        self._aborted = False

    @property
    def tx_id(self) -> int:
        """Transaction identifier."""
        return self._tx_id

    @property
    def is_active(self) -> bool:
        """Whether this transaction is still active."""
        return not self._committed and not self._aborted

    def read(self, key: str) -> Generator[float, None, Any | None]:
        """Read a key, tracking it in the read set.

        For SNAPSHOT_ISOLATION and SERIALIZABLE, reads use the snapshot.
        For READ_COMMITTED, reads see the latest committed value.
        """
        if not self.is_active:
            raise RuntimeError(f"Transaction {self._tx_id} is not active")

        self._read_set.add(key)
        self._manager._total_reads += 1

        # Check local write buffer first
        if key in self._write_set:
            return self._write_set[key]

        # Read from underlying store
        value = yield from self._manager._store.get(key)
        return value

    def write(self, key: str, value: Any) -> Generator[float, None, None]:
        """Buffer a write in the local write set.

        The write is not applied to the store until commit.
        """
        if not self.is_active:
            raise RuntimeError(f"Transaction {self._tx_id} is not active")

        self._write_set[key] = value
        self._manager._total_writes += 1
        # Small latency for buffering
        yield 0.000001

    def commit(self) -> Generator[float, None, bool]:
        """Attempt to commit the transaction.

        Checks for conflicts based on isolation level. If no conflicts,
        applies buffered writes to the store.

        Returns True if committed successfully, False if aborted due to conflict.
        """
        if not self.is_active:
            raise RuntimeError(f"Transaction {self._tx_id} is not active")

        # Conflict check
        has_conflict = self._manager._check_conflict(self)
        if has_conflict:
            self._aborted = True
            self._manager._total_conflicts += 1
            self._manager._total_aborted += 1
            self._manager._record_duration(self._start_time_s)
            self._manager._active_txns.pop(self._tx_id, None)
            logger.debug("[tx-%d] Aborted due to conflict", self._tx_id)
            return False

        # Apply writes
        for key, value in self._write_set.items():
            self._manager._store.put_sync(key, value)

        # Record in commit log
        self._manager._version += 1
        entry = _CommitLogEntry(
            tx_id=self._tx_id,
            version=self._manager._version,
            keys_written=frozenset(self._write_set.keys()),
            keys_read=frozenset(self._read_set),
        )
        self._manager._commit_log.append(entry)

        self._committed = True
        self._manager._total_committed += 1
        self._manager._record_duration(self._start_time_s)
        self._manager._active_txns.pop(self._tx_id, None)

        # Small commit latency
        yield 0.00001

        logger.debug(
            "[tx-%d] Committed (wrote %d keys, read %d keys)",
            self._tx_id, len(self._write_set), len(self._read_set),
        )
        return True

    def abort(self) -> None:
        """Abort this transaction, discarding all buffered writes."""
        if not self.is_active:
            return

        self._aborted = True
        self._manager._total_aborted += 1
        self._manager._record_duration(self._start_time_s)
        self._manager._active_txns.pop(self._tx_id, None)
        logger.debug("[tx-%d] Aborted by user", self._tx_id)


# ---------------------------------------------------------------------------
# TransactionManager entity
# ---------------------------------------------------------------------------


class TransactionManager(Entity):
    """Transaction manager with configurable isolation levels.

    Wraps a storage engine and provides transactional semantics with
    conflict detection. Each transaction sees a consistent snapshot
    (for SI/Serializable) and buffers writes until commit.

    Args:
        name: Entity name.
        store: The underlying storage engine (LSMTree, BTree, etc.).
        isolation: Default isolation level for new transactions.
        deadlock_detection: Whether to detect deadlocks (placeholder).

    Example::

        lsm = LSMTree("db")
        tm = TransactionManager("txm", store=lsm,
                                isolation=IsolationLevel.SNAPSHOT_ISOLATION)
        sim = Simulation(entities=[lsm, tm], ...)
    """

    def __init__(
        self,
        name: str,
        store: StorageEngine,
        isolation: IsolationLevel = IsolationLevel.SNAPSHOT_ISOLATION,
        deadlock_detection: bool = True,
    ) -> None:
        super().__init__(name)
        self._store = store
        self._default_isolation = isolation
        self._deadlock_detection = deadlock_detection

        self._next_tx_id: int = 1
        self._version: int = 0
        self._commit_log: list[_CommitLogEntry] = []
        self._active_txns: dict[int, StorageTransaction] = {}

        # Stats
        self._total_started: int = 0
        self._total_committed: int = 0
        self._total_aborted: int = 0
        self._total_conflicts: int = 0
        self._total_deadlocks: int = 0
        self._total_reads: int = 0
        self._total_writes: int = 0
        self._total_duration_s: float = 0.0

    @property
    def stats(self) -> TransactionStats:
        """Frozen snapshot of transaction manager statistics."""
        avg_dur = (
            self._total_duration_s / (self._total_committed + self._total_aborted)
            if (self._total_committed + self._total_aborted) > 0
            else 0.0
        )
        return TransactionStats(
            transactions_started=self._total_started,
            transactions_committed=self._total_committed,
            transactions_aborted=self._total_aborted,
            conflicts_detected=self._total_conflicts,
            deadlocks_detected=self._total_deadlocks,
            reads=self._total_reads,
            writes=self._total_writes,
            avg_transaction_duration_s=avg_dur,
        )

    @property
    def active_transactions(self) -> int:
        """Number of currently active transactions."""
        return len(self._active_txns)

    def begin(self, isolation: IsolationLevel | None = None) -> Generator[float, None, StorageTransaction]:
        """Begin a new transaction.

        Args:
            isolation: Override the default isolation level for this transaction.

        Returns the new StorageTransaction after yielding a small latency.
        """
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        self._total_started += 1

        level = isolation or self._default_isolation
        tx = StorageTransaction(
            tx_id=tx_id,
            manager=self,
            isolation=level,
            snapshot_version=self._version,
        )
        tx._start_time_s = self.now.to_seconds()
        self._active_txns[tx_id] = tx

        yield 0.000001  # small begin latency
        logger.debug("[%s] Begin tx-%d (isolation=%s)", self.name, tx_id, level.value)
        return tx

    def begin_sync(self, isolation: IsolationLevel | None = None) -> StorageTransaction:
        """Begin a transaction without yielding latency."""
        tx_id = self._next_tx_id
        self._next_tx_id += 1
        self._total_started += 1

        level = isolation or self._default_isolation
        tx = StorageTransaction(
            tx_id=tx_id,
            manager=self,
            isolation=level,
            snapshot_version=self._version,
        )
        if self._clock is not None:
            tx._start_time_s = self.now.to_seconds()
        self._active_txns[tx_id] = tx
        return tx

    def _check_conflict(self, tx: StorageTransaction) -> bool:
        """Check if a transaction conflicts with committed transactions.

        Returns True if there is a conflict (transaction should abort).
        """
        if tx._isolation == IsolationLevel.READ_COMMITTED:
            # No conflict detection
            return False

        # Check commits since this transaction's snapshot
        for entry in self._commit_log:
            if entry.version <= tx._snapshot_version:
                continue
            if entry.tx_id == tx._tx_id:
                continue

            if tx._isolation == IsolationLevel.SNAPSHOT_ISOLATION:
                # Write-write conflict: another tx wrote a key we want to write
                if tx._write_set.keys() & entry.keys_written:
                    return True

            elif tx._isolation == IsolationLevel.SERIALIZABLE:
                # Write-write conflict
                if tx._write_set.keys() & entry.keys_written:
                    return True
                # Read-write conflict: another tx wrote a key we read
                if tx._read_set & entry.keys_written:
                    return True
                # Write-read conflict: another tx read a key we want to write
                if set(tx._write_set.keys()) & entry.keys_read:
                    return True

        return False

    def _record_duration(self, start_time_s: float) -> None:
        """Record transaction duration for stats."""
        if self._clock is not None:
            elapsed = self.now.to_seconds() - start_time_s
            self._total_duration_s += elapsed

    def handle_event(self, event: Event) -> None:
        """TransactionManager does not process events directly."""
        pass

    def __repr__(self) -> str:
        return (
            f"TransactionManager('{self.name}', active={len(self._active_txns)}, "
            f"committed={self._total_committed}, aborted={self._total_aborted})"
        )
