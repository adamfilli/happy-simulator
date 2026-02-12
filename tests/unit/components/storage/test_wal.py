"""Unit tests for WriteAheadLog."""

import pytest

from happysimulator.components.storage.wal import (
    WriteAheadLog,
    WALEntry,
    WALStats,
    SyncEveryWrite,
    SyncPeriodic,
    SyncOnBatch,
)
from happysimulator.components.storage.lsm_tree import LSMTree
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestSyncPolicies:
    def test_sync_every_write(self):
        policy = SyncEveryWrite()
        assert policy.should_sync(1, 0.0)
        assert policy.should_sync(0, 0.0)

    def test_sync_periodic(self):
        policy = SyncPeriodic(interval_s=1.0)
        assert not policy.should_sync(5, 0.5)
        assert policy.should_sync(5, 1.0)
        assert policy.should_sync(5, 2.0)

    def test_sync_periodic_invalid(self):
        with pytest.raises(ValueError):
            SyncPeriodic(interval_s=0.0)

    def test_sync_on_batch(self):
        policy = SyncOnBatch(batch_size=10)
        assert not policy.should_sync(5, 0.0)
        assert policy.should_sync(10, 0.0)
        assert policy.should_sync(15, 0.0)

    def test_sync_on_batch_invalid(self):
        with pytest.raises(ValueError):
            SyncOnBatch(batch_size=0)


class TestWriteAheadLog:
    def _make_wal(self, **kwargs) -> tuple[WriteAheadLog, Simulation]:
        wal = WriteAheadLog("test_wal", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[wal],
        )
        return wal, sim

    def test_append_sync(self):
        wal, sim = self._make_wal()
        seq1 = wal.append_sync("key1", "value1")
        seq2 = wal.append_sync("key2", "value2")
        assert seq1 == 1
        assert seq2 == 2
        assert wal.size == 2

    def test_recover(self):
        wal, sim = self._make_wal()
        wal.append_sync("key1", "value1")
        wal.append_sync("key2", "value2")
        wal.append_sync("key3", "value3")

        entries = wal.recover()
        assert len(entries) == 3
        assert entries[0].key == "key1"
        assert entries[1].key == "key2"
        assert entries[2].key == "key3"
        assert all(isinstance(e, WALEntry) for e in entries)

    def test_truncate(self):
        wal, sim = self._make_wal()
        wal.append_sync("key1", "value1")
        wal.append_sync("key2", "value2")
        wal.append_sync("key3", "value3")

        wal.truncate(up_to_sequence=2)
        entries = wal.recover()
        assert len(entries) == 1
        assert entries[0].key == "key3"

    def test_truncate_all(self):
        wal, sim = self._make_wal()
        wal.append_sync("key1", "value1")
        wal.truncate(up_to_sequence=1)
        assert wal.size == 0

    def test_stats(self):
        wal, sim = self._make_wal()
        wal.append_sync("key1", "value1")
        wal.append_sync("key2", "value2")
        stats = wal.stats
        assert isinstance(stats, WALStats)
        assert stats.writes == 2
        assert stats.bytes_written > 0

    def test_recover_updates_stats(self):
        wal, sim = self._make_wal()
        wal.append_sync("k", "v")
        wal.recover()
        assert wal.stats.entries_recovered == 1

    def test_repr(self):
        wal, sim = self._make_wal()
        assert "test_wal" in repr(wal)

    def test_handle_event_is_noop(self):
        wal, sim = self._make_wal()
        from happysimulator.core.event import Event
        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=wal,
        )
        result = wal.handle_event(event)
        assert result is None

    def test_synced_up_to_starts_at_zero(self):
        wal, sim = self._make_wal()
        assert wal.synced_up_to == 0

    def test_crash_no_entries(self):
        wal, sim = self._make_wal()
        lost = wal.crash()
        assert lost == 0
        assert wal.size == 0

    def test_crash_discards_unsynced_entries(self):
        wal, sim = self._make_wal(sync_policy=SyncOnBatch(batch_size=100))
        # append_sync doesn't trigger sync — all entries are unsynced
        for i in range(10):
            wal.append_sync(f"key_{i}", f"val_{i}")
        assert wal.size == 10
        assert wal.synced_up_to == 0

        lost = wal.crash()
        assert lost == 10
        assert wal.size == 0

    def test_crash_preserves_synced_entries(self):
        wal, sim = self._make_wal(sync_policy=SyncEveryWrite())
        # SyncEveryWrite syncs on every append(), but append_sync
        # bypasses the sync path. Use append_sync to manually
        # set up the scenario.
        wal.append_sync("key_1", "val_1")  # seq 1
        wal.append_sync("key_2", "val_2")  # seq 2
        wal.append_sync("key_3", "val_3")  # seq 3

        # Simulate that the first 2 were synced
        wal._synced_up_to_sequence = 2

        lost = wal.crash()
        assert lost == 1
        assert wal.size == 2
        entries = wal.recover()
        assert [e.key for e in entries] == ["key_1", "key_2"]


class TestLSMTreeCrashRecovery:
    """Tests for crash() and recover_from_crash() on LSMTree."""

    def _make_lsm(self, sync_policy, **kwargs):
        wal = WriteAheadLog("wal", sync_policy=sync_policy)
        lsm = LSMTree("lsm", wal=wal, **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[lsm, wal],
        )
        return lsm, wal, sim

    def test_sstables_survive_crash(self):
        """Data flushed to SSTables is durable across crashes."""
        from happysimulator.components.storage.lsm_tree import LSMTree
        lsm, wal, sim = self._make_lsm(
            SyncEveryWrite(), memtable_size=5,
        )
        # Write enough to trigger a flush (5 entries → SSTable)
        for i in range(6):
            lsm.put_sync(f"key_{i}", f"val_{i}")
        assert lsm._total_memtable_flushes >= 1

        lsm.crash()
        lsm.recover_from_crash()

        # First 5 keys were flushed to SSTable — they survive
        for i in range(5):
            assert lsm.get_sync(f"key_{i}") == f"val_{i}"

    def test_sync_every_write_no_data_loss(self):
        """With SyncEveryWrite, WAL entries survive crash."""
        from happysimulator.components.storage.lsm_tree import LSMTree
        lsm, wal, sim = self._make_lsm(
            SyncEveryWrite(), memtable_size=1000,
        )
        # Manually set synced_up_to to simulate sync-every-write behavior
        # (append_sync bypasses the sync path, so we set it manually)
        for i in range(10):
            lsm.put_sync(f"key_{i}", f"val_{i}")
        wal._synced_up_to_sequence = wal._next_sequence - 1

        crash_info = lsm.crash()
        assert crash_info["memtable_entries_lost"] == 10

        recovery_info = lsm.recover_from_crash()
        assert recovery_info["wal_entries_replayed"] == 10

        # All keys recovered from WAL replay
        for i in range(10):
            assert lsm.get_sync(f"key_{i}") == f"val_{i}"

    def test_async_sync_loses_unsynced_writes(self):
        """With SyncOnBatch, unsynced WAL entries are lost on crash."""
        from happysimulator.components.storage.lsm_tree import LSMTree
        lsm, wal, sim = self._make_lsm(
            SyncOnBatch(batch_size=50), memtable_size=1000,
        )
        for i in range(20):
            lsm.put_sync(f"key_{i}", f"val_{i}")
        # No sync happened (batch_size=50, only 20 writes)
        assert wal.synced_up_to == 0

        crash_info = lsm.crash()
        assert crash_info["wal_entries_lost"] == 20

        recovery_info = lsm.recover_from_crash()
        assert recovery_info["wal_entries_replayed"] == 0
        assert recovery_info["total_keys_recovered"] == 0

        # All data lost
        for i in range(20):
            assert lsm.get_sync(f"key_{i}") is None

    def test_partial_batch_loss(self):
        """Only the tail of writes after the last sync is lost."""
        from happysimulator.components.storage.lsm_tree import LSMTree
        lsm, wal, sim = self._make_lsm(
            SyncOnBatch(batch_size=10), memtable_size=1000,
        )
        for i in range(25):
            lsm.put_sync(f"key_{i:03d}", f"val_{i}")
        # Simulate: first 20 writes synced (2 batches of 10), last 5 unsynced
        wal._synced_up_to_sequence = 20

        crash_info = lsm.crash()
        assert crash_info["wal_entries_lost"] == 5

        recovery_info = lsm.recover_from_crash()
        assert recovery_info["wal_entries_replayed"] == 20

        # First 20 keys recovered, last 5 lost
        for i in range(20):
            assert lsm.get_sync(f"key_{i:03d}") == f"val_{i}"
        for i in range(20, 25):
            assert lsm.get_sync(f"key_{i:03d}") is None
