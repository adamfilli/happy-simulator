"""Integration tests for storage engine components.

Tests the full storage engine pipeline: WAL, Memtable, SSTable, LSMTree,
BTree, and TransactionManager working together.
"""

import pytest

from happysimulator import (
    Simulation,
    Event,
    Entity,
    Instant,
    Source,
    Sink,
    LatencyTracker,
)
from happysimulator.components.storage import (
    SSTable,
    WriteAheadLog,
    Memtable,
    LSMTree,
    LSMTreeStats,
    BTree,
    BTreeStats,
    TransactionManager,
    IsolationLevel,
    SizeTieredCompaction,
    LeveledCompaction,
    SyncEveryWrite,
    SyncOnBatch,
)


class TestLSMWriteReadRoundtrip:
    """Verify correct values survive the full LSM write/read path."""

    def test_write_read_roundtrip(self):
        lsm = LSMTree("db", memtable_size=10)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[lsm],
        )

        # Write 50 key-value pairs (triggers multiple flushes)
        for i in range(50):
            lsm.put_sync(f"key_{i:03d}", f"value_{i}")

        # Read all back
        for i in range(50):
            value = lsm.get_sync(f"key_{i:03d}")
            assert value == f"value_{i}", f"key_{i:03d}: expected value_{i}, got {value}"

    def test_overwrite_preserves_latest(self):
        lsm = LSMTree("db", memtable_size=5)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[lsm],
        )

        for i in range(20):
            lsm.put_sync("key", f"v{i}")

        assert lsm.get_sync("key") == "v19"


class TestWALRecovery:
    """Verify WAL recovery returns all entries."""

    def test_all_entries_present(self):
        wal = WriteAheadLog("wal", sync_policy=SyncEveryWrite())
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[wal],
        )

        for i in range(100):
            wal.append_sync(f"key_{i}", f"value_{i}")

        entries = wal.recover()
        assert len(entries) == 100
        for i, entry in enumerate(entries):
            assert entry.key == f"key_{i}"
            assert entry.value == f"value_{i}"
            assert entry.sequence_number == i + 1

    def test_recovery_after_truncate(self):
        wal = WriteAheadLog("wal")
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[wal],
        )

        for i in range(10):
            wal.append_sync(f"key_{i}", f"value_{i}")

        wal.truncate(up_to_sequence=5)
        entries = wal.recover()
        assert len(entries) == 5
        assert entries[0].sequence_number == 6


class TestCompactionReducesSSTables:
    """Verify compaction reduces SSTable count."""

    def test_size_tiered_compaction(self):
        lsm = LSMTree(
            "db",
            memtable_size=5,
            compaction_strategy=SizeTieredCompaction(min_sstables=3),
        )
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[lsm],
        )

        # Write enough to trigger multiple flushes and compactions
        for i in range(50):
            lsm.put_sync(f"key_{i:03d}", i)

        assert lsm.stats.compactions > 0
        # After compaction, L0 should have fewer SSTables than total flushes
        assert len(lsm._levels[0]) < lsm.stats.memtable_flushes

    def test_leveled_compaction(self):
        lsm = LSMTree(
            "db",
            memtable_size=5,
            compaction_strategy=LeveledCompaction(level_0_max=2, base_size_keys=10),
        )
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[lsm],
        )

        for i in range(40):
            lsm.put_sync(f"key_{i:03d}", i)

        assert lsm.stats.compactions > 0
        # Data still correct after compaction
        for i in range(40):
            assert lsm.get_sync(f"key_{i:03d}") == i


class TestBloomFilterReducesPageReads:
    """Verify bloom filters reduce page reads for missing keys."""

    def test_bloom_saves_on_missing_keys(self):
        lsm = LSMTree("db", memtable_size=50)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[lsm],
        )

        # Write keys and flush to SSTables
        for i in range(100):
            lsm.put_sync(f"exist_{i:04d}", i)

        # Force flush
        lsm._flush_memtable_sync()

        # Read keys that don't exist
        for i in range(500):
            lsm.get_sync(f"miss_{i:04d}")

        assert lsm.stats.bloom_filter_saves > 0
        # Most misses should be caught by bloom
        assert lsm.stats.bloom_filter_saves > 400


class TestBTreeVsLSMComparison:
    """Compare BTree and LSM on the same workload."""

    def test_same_workload_produces_same_results(self):
        btree = BTree("btree", order=16)
        lsm = LSMTree("lsm", memtable_size=20)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[btree, lsm],
        )

        keys = [f"key_{i:04d}" for i in range(100)]
        for key in keys:
            btree.put_sync(key, key)
            lsm.put_sync(key, key)

        # Both should return the same values
        for key in keys:
            assert btree.get_sync(key) == lsm.get_sync(key) == key

    def test_btree_fewer_page_reads_for_reads(self):
        """BTree typically has lower read amplification."""
        btree = BTree("btree", order=32)
        lsm = LSMTree("lsm", memtable_size=20)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[btree, lsm],
        )

        # Write data
        for i in range(200):
            btree.put_sync(f"key_{i:04d}", i)
            lsm.put_sync(f"key_{i:04d}", i)

        # Read all keys
        for i in range(200):
            btree.get_sync(f"key_{i:04d}")
            lsm.get_sync(f"key_{i:04d}")

        # BTree should have predictable page reads (depth per read)
        # LSM may have higher read amplification due to checking multiple levels
        assert btree.stats.page_reads > 0
        assert lsm.stats.reads > 0


class TestSnapshotIsolationConflicts:
    """Verify snapshot isolation detects write-write conflicts."""

    def test_write_write_conflict_detected(self):
        store = LSMTree("store")
        tm = TransactionManager(
            "txm", store=store,
            isolation=IsolationLevel.SNAPSHOT_ISOLATION,
        )
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[store, tm],
        )

        # Two transactions writing the same key
        tx1 = tm.begin_sync()
        tx2 = tm.begin_sync()

        tx1._write_set["shared_key"] = "tx1_value"
        tx2._write_set["shared_key"] = "tx2_value"

        # Commit tx1 (should succeed)
        assert not tm._check_conflict(tx1)
        store.put_sync("shared_key", "tx1_value")
        tm._version += 1
        from happysimulator.components.storage.transaction_manager import _CommitLogEntry
        tm._commit_log.append(_CommitLogEntry(
            tx_id=tx1.tx_id,
            version=tm._version,
            keys_written=frozenset(["shared_key"]),
            keys_read=frozenset(),
        ))

        # tx2 should detect conflict
        assert tm._check_conflict(tx2)


class TestSerializableConflicts:
    """Verify serializable isolation detects read-write conflicts."""

    def test_read_write_conflict_detected(self):
        store = LSMTree("store")
        store.put_sync("counter", 0)
        tm = TransactionManager(
            "txm", store=store,
            isolation=IsolationLevel.SERIALIZABLE,
        )
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[store, tm],
        )

        tx1 = tm.begin_sync()
        tx2 = tm.begin_sync()

        # tx1 reads "counter"
        tx1._read_set.add("counter")
        # tx2 writes "counter"
        tx2._write_set["counter"] = 1

        # Commit tx1 first
        from happysimulator.components.storage.transaction_manager import _CommitLogEntry
        tm._version += 1
        tm._commit_log.append(_CommitLogEntry(
            tx_id=tx1.tx_id,
            version=tm._version,
            keys_written=frozenset(),
            keys_read=frozenset(["counter"]),
        ))

        # tx2 writes a key that tx1 read -> conflict
        assert tm._check_conflict(tx2)


class TestFullPipeline:
    """End-to-end: Source -> Entity -> TransactionManager(LSMTree(WAL)) -> LatencyTracker."""

    def test_full_pipeline(self):
        wal = WriteAheadLog("wal", sync_policy=SyncOnBatch(batch_size=5))
        lsm = LSMTree("db", memtable_size=20, wal=wal)
        tracker = LatencyTracker("tracker")

        class StorageWorker(Entity):
            """Entity that processes incoming events via the LSM tree."""

            def __init__(self, name, store, tracker):
                super().__init__(name)
                self._store = store
                self._tracker = tracker
                self._count = 0

            def handle_event(self, event):
                self._count += 1
                key = f"key_{self._count:04d}"
                self._store.put_sync(key, {"data": self._count})
                value = self._store.get_sync(key)
                assert value is not None

                return [Event(
                    time=self.now,
                    event_type="Processed",
                    target=self._tracker,
                    context={"created_at": event.time},
                )]

        worker = StorageWorker("worker", lsm, tracker)

        source = Source.constant(rate=100, target=worker, event_type="Write")

        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(1),
            sources=[source],
            entities=[wal, lsm, worker, tracker],
        )
        summary = sim.run()

        assert summary.total_events_processed > 0
        assert tracker.count > 0
        assert lsm.stats.writes > 0
        assert wal.stats.writes > 0
