"""Tests for KVStore."""

import pytest

from happysimulator.components.datastore import KVStore


class TestKVStoreCreation:
    """Tests for KVStore creation."""

    def test_creates_with_defaults(self):
        """KVStore creates with default latencies."""
        store = KVStore(name="test")

        assert store.name == "test"
        assert store.read_latency == 0.001
        assert store.write_latency == 0.005
        assert store.capacity is None
        assert store.size == 0

    def test_creates_with_custom_latencies(self):
        """KVStore creates with custom latencies."""
        store = KVStore(
            name="test",
            read_latency=0.010,
            write_latency=0.050,
        )

        assert store.read_latency == 0.010
        assert store.write_latency == 0.050

    def test_creates_with_capacity(self):
        """KVStore creates with capacity limit."""
        store = KVStore(name="test", capacity=100)

        assert store.capacity == 100

    def test_rejects_negative_latency(self):
        """KVStore rejects negative latencies."""
        with pytest.raises(ValueError):
            KVStore(name="test", read_latency=-1)

        with pytest.raises(ValueError):
            KVStore(name="test", write_latency=-1)

    def test_rejects_invalid_capacity(self):
        """KVStore rejects capacity < 1."""
        with pytest.raises(ValueError):
            KVStore(name="test", capacity=0)


class TestKVStoreSync:
    """Tests for synchronous KVStore operations."""

    def test_put_sync_and_get_sync(self):
        """put_sync and get_sync work correctly."""
        store = KVStore(name="test")

        store.put_sync("key1", "value1")

        assert store.get_sync("key1") == "value1"
        assert store.size == 1

    def test_get_sync_missing_returns_none(self):
        """get_sync returns None for missing key."""
        store = KVStore(name="test")

        assert store.get_sync("missing") is None

    def test_delete_sync(self):
        """delete_sync removes key."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")

        result = store.delete_sync("key1")

        assert result is True
        assert store.get_sync("key1") is None

    def test_delete_sync_missing_returns_false(self):
        """delete_sync returns False for missing key."""
        store = KVStore(name="test")

        result = store.delete_sync("missing")

        assert result is False

    def test_contains(self):
        """contains checks key existence."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")

        assert store.contains("key1") is True
        assert store.contains("missing") is False

    def test_keys(self):
        """keys returns all stored keys."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")
        store.put_sync("key2", "value2")

        keys = store.keys()

        assert set(keys) == {"key1", "key2"}

    def test_clear(self):
        """clear removes all data."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")
        store.put_sync("key2", "value2")

        store.clear()

        assert store.size == 0
        assert store.get_sync("key1") is None


class TestKVStoreGenerators:
    """Tests for generator-based KVStore operations."""

    def test_get_yields_latency(self):
        """get() yields read latency."""
        store = KVStore(name="test", read_latency=0.010)
        store.put_sync("key1", "value1")

        gen = store.get("key1")
        latency = next(gen)

        assert latency == 0.010

    def test_get_returns_value(self):
        """get() returns value after iteration."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")

        gen = store.get("key1")
        next(gen)
        try:
            next(gen)
        except StopIteration as e:
            value = e.value

        assert value == "value1"

    def test_put_yields_latency(self):
        """put() yields write latency."""
        store = KVStore(name="test", write_latency=0.020)

        gen = store.put("key1", "value1")
        latency = next(gen)

        assert latency == 0.020

    def test_delete_yields_latency(self):
        """delete() yields delete latency."""
        store = KVStore(name="test", write_latency=0.015)
        store.put_sync("key1", "value1")

        gen = store.delete("key1")
        latency = next(gen)

        assert latency == 0.015


class TestKVStoreCapacity:
    """Tests for capacity-limited KVStore."""

    def test_evicts_oldest_on_overflow(self):
        """Evicts oldest entry when capacity exceeded."""
        store = KVStore(name="test", capacity=2)

        store.put_sync("key1", "value1")
        store.put_sync("key2", "value2")
        store.put_sync("key3", "value3")  # Should evict key1

        assert store.size == 2
        assert store.get_sync("key1") is None
        assert store.get_sync("key2") == "value2"
        assert store.get_sync("key3") == "value3"

    def test_tracks_evictions(self):
        """Statistics track evictions."""
        store = KVStore(name="test", capacity=1)

        store.put_sync("key1", "value1")
        store.put_sync("key2", "value2")

        assert store.stats.evictions == 1

    def test_update_doesnt_evict(self):
        """Updating existing key doesn't evict."""
        store = KVStore(name="test", capacity=2)

        store.put_sync("key1", "value1")
        store.put_sync("key2", "value2")
        store.put_sync("key1", "updated")  # Update, not insert

        assert store.size == 2
        assert store.get_sync("key1") == "updated"
        assert store.stats.evictions == 0


class TestKVStoreStatistics:
    """Tests for KVStore statistics."""

    def test_tracks_reads(self):
        """Statistics track read operations."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")

        list(store.get("key1"))
        list(store.get("key1"))

        assert store.stats.reads == 2

    def test_tracks_hits_and_misses(self):
        """Statistics track cache hits and misses."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")

        list(store.get("key1"))  # Hit
        list(store.get("missing"))  # Miss

        assert store.stats.hits == 1
        assert store.stats.misses == 1

    def test_tracks_writes(self):
        """Statistics track write operations."""
        store = KVStore(name="test")

        list(store.put("key1", "value1"))
        list(store.put("key2", "value2"))

        assert store.stats.writes == 2

    def test_tracks_deletes(self):
        """Statistics track delete operations."""
        store = KVStore(name="test")
        store.put_sync("key1", "value1")

        list(store.delete("key1"))
        list(store.delete("missing"))

        assert store.stats.deletes == 2
