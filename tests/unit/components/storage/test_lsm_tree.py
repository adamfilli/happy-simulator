"""Unit tests for LSMTree."""

from happysimulator.components.storage.lsm_tree import (
    FIFOCompaction,
    LeveledCompaction,
    LSMTree,
    LSMTreeStats,
    SizeTieredCompaction,
)
from happysimulator.components.storage.sstable import SSTable
from happysimulator.components.storage.wal import WriteAheadLog
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestCompactionStrategies:
    def test_size_tiered_should_compact(self):
        strategy = SizeTieredCompaction(min_sstables=4)
        levels = [
            [SSTable([(f"k{j}", j)]) for j in range(4)],
            [],
        ]
        assert strategy.should_compact(levels)

    def test_size_tiered_should_not_compact(self):
        strategy = SizeTieredCompaction(min_sstables=4)
        levels = [
            [SSTable([(f"k{j}", j)]) for j in range(3)],
            [],
        ]
        assert not strategy.should_compact(levels)

    def test_size_tiered_select(self):
        strategy = SizeTieredCompaction(min_sstables=2)
        sst1 = SSTable([("a", 1)])
        sst2 = SSTable([("b", 2)])
        levels = [[sst1, sst2], []]
        level, sstables = strategy.select_compaction(levels)
        assert level == 0
        assert len(sstables) == 2

    def test_leveled_l0_trigger(self):
        strategy = LeveledCompaction(level_0_max=4)
        levels = [
            [SSTable([(f"k{j}", j)]) for j in range(4)],
            [],
        ]
        assert strategy.should_compact(levels)

    def test_leveled_level_size_trigger(self):
        strategy = LeveledCompaction(level_0_max=10, base_size_keys=10, size_ratio=10)
        # L1 limit = base_size_keys * size_ratio^1 = 100; put 150 keys
        data = [(f"key_{i:04d}", i) for i in range(150)]
        levels = [[], [SSTable(data)]]
        assert strategy.should_compact(levels)

    def test_fifo_trigger(self):
        strategy = FIFOCompaction(max_total_sstables=3)
        levels = [
            [SSTable([("a", 1)]) for _ in range(4)],
        ]
        assert strategy.should_compact(levels)

    def test_fifo_no_trigger(self):
        strategy = FIFOCompaction(max_total_sstables=10)
        levels = [[SSTable([("a", 1)])]]
        assert not strategy.should_compact(levels)


class TestLSMTree:
    def _make_lsm(self, **kwargs) -> tuple[LSMTree, Simulation]:
        lsm = LSMTree("test_lsm", **kwargs)
        entities = [lsm]
        if lsm._wal is not None:
            entities.append(lsm._wal)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=entities,
        )
        return lsm, sim

    def test_put_sync_and_get_sync(self):
        lsm, _sim = self._make_lsm()
        lsm.put_sync("key1", "value1")
        assert lsm.get_sync("key1") == "value1"

    def test_get_sync_missing(self):
        lsm, _sim = self._make_lsm()
        assert lsm.get_sync("missing") is None

    def test_multiple_puts_and_gets(self):
        lsm, _sim = self._make_lsm()
        for i in range(50):
            lsm.put_sync(f"key_{i:03d}", i)
        for i in range(50):
            assert lsm.get_sync(f"key_{i:03d}") == i

    def test_overwrite_key(self):
        lsm, _sim = self._make_lsm()
        lsm.put_sync("key", "old")
        lsm.put_sync("key", "new")
        assert lsm.get_sync("key") == "new"

    def test_flush_on_full_memtable(self):
        lsm, _sim = self._make_lsm(memtable_size=5)
        for i in range(6):
            lsm.put_sync(f"key_{i}", i)
        # After 5 puts, memtable should have flushed
        assert lsm._total_memtable_flushes >= 1
        assert len(lsm._levels[0]) >= 1

    def test_data_survives_flush(self):
        lsm, _sim = self._make_lsm(memtable_size=3)
        lsm.put_sync("a", 1)
        lsm.put_sync("b", 2)
        lsm.put_sync("c", 3)  # triggers flush
        lsm.put_sync("d", 4)

        assert lsm.get_sync("a") == 1
        assert lsm.get_sync("d") == 4

    def test_compaction_triggered(self):
        lsm, _sim = self._make_lsm(
            memtable_size=5,
            compaction_strategy=SizeTieredCompaction(min_sstables=2),
        )
        # Fill and flush twice to trigger compaction
        for i in range(15):
            lsm.put_sync(f"key_{i:03d}", i)

        # Should have had at least one compaction
        assert lsm._total_compactions >= 1

    def test_data_survives_compaction(self):
        lsm, _sim = self._make_lsm(
            memtable_size=5,
            compaction_strategy=SizeTieredCompaction(min_sstables=2),
        )
        keys = []
        for i in range(20):
            key = f"key_{i:03d}"
            lsm.put_sync(key, i)
            keys.append((key, i))

        for key, value in keys:
            assert lsm.get_sync(key) == value

    def test_delete(self):
        lsm, _sim = self._make_lsm()
        lsm.put_sync("key", "value")
        assert lsm.get_sync("key") == "value"
        # Delete via put_sync with tombstone (using internal API for sync delete)
        from happysimulator.components.storage.lsm_tree import _TOMBSTONE

        lsm._memtable.put_sync("key", _TOMBSTONE)
        lsm._total_writes += 1
        assert lsm.get_sync("key") is None

    def test_with_wal(self):
        wal = WriteAheadLog("test_wal")
        lsm, _sim = self._make_lsm(wal=wal)
        lsm.put_sync("key", "value")
        assert lsm.get_sync("key") == "value"
        assert wal.size >= 0  # WAL may have been truncated
        assert lsm.stats.wal_writes >= 1

    def test_stats(self):
        lsm, _sim = self._make_lsm()
        lsm.put_sync("a", 1)
        lsm.get_sync("a")
        lsm.get_sync("missing")

        stats = lsm.stats
        assert isinstance(stats, LSMTreeStats)
        assert stats.writes == 1
        assert stats.reads == 2
        assert stats.read_hits >= 1
        assert stats.read_misses >= 1

    def test_level_summary(self):
        lsm, _sim = self._make_lsm(memtable_size=3)
        for i in range(10):
            lsm.put_sync(f"key_{i}", i)

        summary = lsm.level_summary
        assert isinstance(summary, list)
        # Should have at least L0
        if lsm._total_memtable_flushes > 0:
            assert len(summary) > 0

    def test_bloom_filter_saves(self):
        lsm, _sim = self._make_lsm(memtable_size=5)
        # Put data and flush to SSTables
        for i in range(10):
            lsm.put_sync(f"key_{i}", i)

        # Read keys that don't exist — bloom filters should save reads
        for i in range(100, 200):
            lsm.get_sync(f"key_{i}")

        assert lsm.stats.bloom_filter_saves > 0

    def test_repr(self):
        lsm, _sim = self._make_lsm()
        assert "test_lsm" in repr(lsm)

    def test_leveled_compaction_integration(self):
        lsm, _sim = self._make_lsm(
            memtable_size=5,
            compaction_strategy=LeveledCompaction(level_0_max=2, base_size_keys=10),
        )
        for i in range(30):
            lsm.put_sync(f"key_{i:03d}", i)

        # Data should still be correct
        for i in range(30):
            assert lsm.get_sync(f"key_{i:03d}") == i

    def test_fifo_compaction_integration(self):
        lsm, _sim = self._make_lsm(
            memtable_size=3,
            compaction_strategy=FIFOCompaction(max_total_sstables=3),
        )
        for i in range(20):
            lsm.put_sync(f"key_{i:03d}", i)

        # FIFO drops oldest — some data may be lost
        total_sst = sum(len(level) for level in lsm._levels)
        assert total_sst <= 5  # Should be kept under control
