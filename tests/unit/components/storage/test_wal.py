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
