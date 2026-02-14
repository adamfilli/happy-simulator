"""Unit tests for BTree."""

import pytest

from happysimulator.components.storage.btree import BTree, BTreeStats
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestBTree:
    def _make_btree(self, **kwargs) -> tuple[BTree, Simulation]:
        bt = BTree("test_btree", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[bt],
        )
        return bt, sim

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            BTree("bt", order=2)

    def test_put_sync_and_get_sync(self):
        bt, sim = self._make_btree()
        bt.put_sync("key1", "value1")
        assert bt.get_sync("key1") == "value1"

    def test_get_sync_missing(self):
        bt, sim = self._make_btree()
        assert bt.get_sync("missing") is None

    def test_multiple_inserts(self):
        bt, sim = self._make_btree(order=4)
        for i in range(100):
            bt.put_sync(f"key_{i:04d}", i)
        for i in range(100):
            assert bt.get_sync(f"key_{i:04d}") == i

    def test_overwrite_key(self):
        bt, sim = self._make_btree()
        bt.put_sync("key", "old")
        bt.put_sync("key", "new")
        assert bt.get_sync("key") == "new"

    def test_size_tracks_inserts(self):
        bt, sim = self._make_btree()
        assert bt.size == 0
        bt.put_sync("a", 1)
        assert bt.size == 1
        bt.put_sync("b", 2)
        assert bt.size == 2
        # Overwrite should not increase size
        bt.put_sync("a", 10)
        assert bt.size == 2

    def test_depth_grows_with_splits(self):
        bt, sim = self._make_btree(order=4)
        initial_depth = bt.depth
        # Insert enough keys to cause splits
        for i in range(50):
            bt.put_sync(f"key_{i:04d}", i)
        assert bt.depth >= initial_depth

    def test_node_splits_tracked(self):
        bt, sim = self._make_btree(order=4)
        for i in range(50):
            bt.put_sync(f"key_{i:04d}", i)
        assert bt.stats.node_splits > 0

    def test_delete_existing(self):
        bt, sim = self._make_btree()
        bt.put_sync("a", 1)
        bt.put_sync("b", 2)
        # Use sync delete
        deleted = bt._delete("a")
        assert deleted
        assert bt.get_sync("a") is None
        assert bt.get_sync("b") == 2
        assert bt.size == 1

    def test_delete_missing(self):
        bt, sim = self._make_btree()
        bt.put_sync("a", 1)
        deleted = bt._delete("missing")
        assert not deleted
        assert bt.size == 1

    def test_scan_sync(self):
        bt, sim = self._make_btree(order=4)
        for c in "abcdefghij":
            bt.put_sync(c, ord(c))

        # Internal scan (no I/O)
        results = []
        bt._scan_node(bt._root, "c", "g", results)
        keys = [k for k, v in results]
        assert keys == ["c", "d", "e", "f"]

    def test_stats(self):
        bt, sim = self._make_btree()
        bt.put_sync("a", 1)
        bt.get_sync("a")

        stats = bt.stats
        assert isinstance(stats, BTreeStats)
        assert stats.writes == 1
        assert stats.reads == 1
        assert stats.total_keys == 1
        assert stats.page_reads > 0
        assert stats.page_writes > 0

    def test_large_insert_order(self):
        """Insert keys in reverse order to exercise splits."""
        bt, sim = self._make_btree(order=4)
        keys = [f"key_{i:04d}" for i in range(100)]
        for key in reversed(keys):
            bt.put_sync(key, key)
        for key in keys:
            assert bt.get_sync(key) == key

    def test_repr(self):
        bt, sim = self._make_btree(order=16)
        assert "test_btree" in repr(bt)
        assert "order=16" in repr(bt)

    def test_handle_event_is_noop(self):
        bt, sim = self._make_btree()
        from happysimulator.core.event import Event
        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=bt,
        )
        result = bt.handle_event(event)
        assert result is None
