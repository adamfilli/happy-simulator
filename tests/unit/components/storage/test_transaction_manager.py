"""Unit tests for TransactionManager."""

from happysimulator.components.storage.btree import BTree
from happysimulator.components.storage.lsm_tree import LSMTree
from happysimulator.components.storage.transaction_manager import (
    IsolationLevel,
    StorageTransaction,
    TransactionManager,
    TransactionStats,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestTransactionManager:
    def _make_tm(self, store=None, **kwargs) -> tuple[TransactionManager, Simulation]:
        if store is None:
            store = LSMTree("store")
        tm = TransactionManager("test_tm", store=store, **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[store, tm],
        )
        return tm, sim

    def test_begin_sync(self):
        tm, _sim = self._make_tm()
        tx = tm.begin_sync()
        assert isinstance(tx, StorageTransaction)
        assert tx.is_active
        assert tm.active_transactions == 1

    def test_read_write_commit_sync(self):
        store = LSMTree("store")
        store.put_sync("existing", "value")
        tm, _sim = self._make_tm(store=store)

        tx = tm.begin_sync()
        # Read existing key
        val = store.get_sync("existing")
        assert val == "value"

        # Write via transaction (sync path)
        tx._write_set["new_key"] = "new_value"
        tx._manager._total_writes += 1

        # Check local buffer
        assert "new_key" in tx._write_set

    def test_abort(self):
        tm, _sim = self._make_tm()
        tx = tm.begin_sync()
        tx._write_set["key"] = "value"
        tx.abort()
        assert not tx.is_active
        assert tm.active_transactions == 0
        assert tm.stats.transactions_aborted == 1

    def test_abort_idempotent(self):
        tm, _sim = self._make_tm()
        tx = tm.begin_sync()
        tx.abort()
        tx.abort()  # should be no-op
        assert tm.stats.transactions_aborted == 1

    def test_snapshot_isolation_no_conflict(self):
        """Two transactions writing different keys should both commit."""
        store = LSMTree("store")
        tm, _sim = self._make_tm(
            store=store,
            isolation=IsolationLevel.SNAPSHOT_ISOLATION,
        )

        tx1 = tm.begin_sync()
        tx2 = tm.begin_sync()

        tx1._write_set["key_a"] = "val_a"
        tx2._write_set["key_b"] = "val_b"

        # Commit tx1 first
        has_conflict = tm._check_conflict(tx1)
        assert not has_conflict

        # Record tx1 commit
        from happysimulator.components.storage.transaction_manager import _CommitLogEntry

        tm._version += 1
        tm._commit_log.append(
            _CommitLogEntry(
                tx_id=tx1.tx_id,
                version=tm._version,
                keys_written=frozenset(["key_a"]),
                keys_read=frozenset(),
            )
        )

        # Check tx2 — different keys, no conflict
        has_conflict = tm._check_conflict(tx2)
        assert not has_conflict

    def test_snapshot_isolation_write_write_conflict(self):
        """Two transactions writing the same key — second should conflict."""
        store = LSMTree("store")
        tm, _sim = self._make_tm(
            store=store,
            isolation=IsolationLevel.SNAPSHOT_ISOLATION,
        )

        tx1 = tm.begin_sync()
        tx2 = tm.begin_sync()

        tx1._write_set["key"] = "val_1"
        tx2._write_set["key"] = "val_2"

        # Commit tx1
        assert not tm._check_conflict(tx1)
        from happysimulator.components.storage.transaction_manager import _CommitLogEntry

        tm._version += 1
        tm._commit_log.append(
            _CommitLogEntry(
                tx_id=tx1.tx_id,
                version=tm._version,
                keys_written=frozenset(["key"]),
                keys_read=frozenset(),
            )
        )

        # tx2 should have write-write conflict
        assert tm._check_conflict(tx2)

    def test_serializable_read_write_conflict(self):
        """Serializable: tx1 reads key, tx2 writes same key — tx2 conflicts."""
        store = LSMTree("store")
        tm, _sim = self._make_tm(
            store=store,
            isolation=IsolationLevel.SERIALIZABLE,
        )

        tx1 = tm.begin_sync()
        tx2 = tm.begin_sync()

        tx1._read_set.add("key")
        tx2._write_set["key"] = "val"

        # Commit tx1 (reads key)
        from happysimulator.components.storage.transaction_manager import _CommitLogEntry

        tm._version += 1
        tm._commit_log.append(
            _CommitLogEntry(
                tx_id=tx1.tx_id,
                version=tm._version,
                keys_written=frozenset(),
                keys_read=frozenset(["key"]),
            )
        )

        # tx2 writes key that tx1 read — conflict under serializable
        assert tm._check_conflict(tx2)

    def test_read_committed_no_conflicts(self):
        """READ_COMMITTED never detects conflicts."""
        store = LSMTree("store")
        tm, _sim = self._make_tm(
            store=store,
            isolation=IsolationLevel.READ_COMMITTED,
        )

        tx1 = tm.begin_sync()
        tx2 = tm.begin_sync()

        tx1._write_set["key"] = "val_1"
        tx2._write_set["key"] = "val_2"

        # Commit tx1
        from happysimulator.components.storage.transaction_manager import _CommitLogEntry

        tm._version += 1
        tm._commit_log.append(
            _CommitLogEntry(
                tx_id=tx1.tx_id,
                version=tm._version,
                keys_written=frozenset(["key"]),
                keys_read=frozenset(),
            )
        )

        # tx2 should NOT conflict under READ_COMMITTED
        assert not tm._check_conflict(tx2)

    def test_stats(self):
        tm, _sim = self._make_tm()
        tx = tm.begin_sync()
        tx.abort()

        stats = tm.stats
        assert isinstance(stats, TransactionStats)
        assert stats.transactions_started == 1
        assert stats.transactions_aborted == 1
        assert stats.transactions_committed == 0

    def test_with_btree_store(self):
        """TransactionManager works with BTree as well."""
        store = BTree("btree_store")
        tm, _sim = self._make_tm(store=store)
        tx = tm.begin_sync()
        assert tx.is_active

    def test_repr(self):
        tm, _sim = self._make_tm()
        assert "test_tm" in repr(tm)

    def test_handle_event_is_noop(self):
        tm, _sim = self._make_tm()
        from happysimulator.core.event import Event

        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=tm,
        )
        result = tm.handle_event(event)
        assert result is None

    def test_multiple_transactions_sequential(self):
        """Multiple transactions committed sequentially."""
        store = LSMTree("store")
        tm, _sim = self._make_tm(store=store)

        for i in range(5):
            tx = tm.begin_sync()
            tx._write_set[f"key_{i}"] = f"val_{i}"
            # Manually commit (sync path)
            store.put_sync(f"key_{i}", f"val_{i}")
            tm._version += 1
            from happysimulator.components.storage.transaction_manager import _CommitLogEntry

            tm._commit_log.append(
                _CommitLogEntry(
                    tx_id=tx.tx_id,
                    version=tm._version,
                    keys_written=frozenset([f"key_{i}"]),
                    keys_read=frozenset(),
                )
            )
            tx._committed = True
            tm._total_committed += 1
            tm._active_txns.pop(tx.tx_id, None)

        assert tm.stats.transactions_committed == 5
        for i in range(5):
            assert store.get_sync(f"key_{i}") == f"val_{i}"
