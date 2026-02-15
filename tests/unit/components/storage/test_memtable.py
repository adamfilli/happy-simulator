"""Unit tests for Memtable."""

from happysimulator.components.storage.memtable import Memtable, MemtableStats
from happysimulator.components.storage.sstable import SSTable
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestMemtable:
    def _make_memtable(self, **kwargs) -> tuple[Memtable, Simulation]:
        mem = Memtable("test_mem", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[mem],
        )
        return mem, sim

    def test_put_sync_and_get_sync(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("key1", "value1")
        assert mem.get_sync("key1") == "value1"

    def test_get_sync_missing(self):
        mem, _sim = self._make_memtable()
        assert mem.get_sync("missing") is None

    def test_put_sync_returns_is_full(self):
        mem, _sim = self._make_memtable(size_threshold=3)
        assert not mem.put_sync("a", 1)
        assert not mem.put_sync("b", 2)
        assert mem.put_sync("c", 3)  # Now full

    def test_is_full(self):
        mem, _sim = self._make_memtable(size_threshold=2)
        assert not mem.is_full
        mem.put_sync("a", 1)
        assert not mem.is_full
        mem.put_sync("b", 2)
        assert mem.is_full

    def test_size(self):
        mem, _sim = self._make_memtable()
        assert mem.size == 0
        mem.put_sync("a", 1)
        assert mem.size == 1
        mem.put_sync("b", 2)
        assert mem.size == 2

    def test_contains(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("a", 1)
        assert mem.contains("a")
        assert not mem.contains("b")

    def test_overwrite_key(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("a", 1)
        mem.put_sync("a", 2)
        assert mem.get_sync("a") == 2
        assert mem.size == 1  # Still one entry

    def test_flush_produces_sstable(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("c", 3)
        mem.put_sync("a", 1)
        mem.put_sync("b", 2)

        sst = mem.flush()
        assert isinstance(sst, SSTable)
        assert sst.key_count == 3
        assert sst.get("a") == 1
        assert sst.get("b") == 2
        assert sst.get("c") == 3
        assert sst.level == 0

    def test_flush_clears_memtable(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("a", 1)
        mem.flush()
        assert mem.size == 0
        assert mem.get_sync("a") is None

    def test_multiple_flushes_increment_sequence(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("a", 1)
        sst1 = mem.flush()
        mem.put_sync("b", 2)
        sst2 = mem.flush()
        assert sst1.sequence == 0
        assert sst2.sequence == 1

    def test_flush_empty_memtable(self):
        mem, _sim = self._make_memtable()
        sst = mem.flush()
        assert sst.key_count == 0

    def test_stats(self):
        mem, _sim = self._make_memtable()
        mem.put_sync("a", 1)
        mem.get_sync("a")
        mem.get_sync("missing")
        mem.flush()

        stats = mem.stats
        assert isinstance(stats, MemtableStats)
        assert stats.writes == 1
        assert stats.reads == 2
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.flushes == 1
        assert stats.current_size == 0

    def test_repr(self):
        mem, _sim = self._make_memtable(size_threshold=100)
        assert "test_mem" in repr(mem)
        assert "100" in repr(mem)

    def test_handle_event_is_noop(self):
        mem, _sim = self._make_memtable()
        from happysimulator.core.event import Event

        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=mem,
        )
        result = mem.handle_event(event)
        assert result is None
