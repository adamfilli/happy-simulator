"""Tests for ReplicatedStore."""

import pytest

from happysimulator.components.datastore import (
    ConsistencyLevel,
    KVStore,
    ReplicatedStore,
)


class TestReplicatedStoreCreation:
    """Tests for ReplicatedStore creation."""

    def test_creates_with_replicas(self):
        """ReplicatedStore is created with replicas."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]

        store = ReplicatedStore(
            name="distributed",
            replicas=replicas,
        )

        assert store.name == "distributed"
        assert store.num_replicas == 3
        assert store.quorum_size == 2

    def test_rejects_no_replicas(self):
        """Rejects empty replica list."""
        with pytest.raises(ValueError):
            ReplicatedStore(name="distributed", replicas=[])

    def test_default_consistency(self):
        """Default consistency is QUORUM."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]

        store = ReplicatedStore(name="distributed", replicas=replicas)

        assert store.read_consistency == ConsistencyLevel.QUORUM
        assert store.write_consistency == ConsistencyLevel.QUORUM

    def test_quorum_calculation(self):
        """Quorum size is calculated correctly."""
        for n, expected_quorum in [(1, 1), (2, 2), (3, 2), (4, 3), (5, 3)]:
            replicas = [KVStore(name=f"node{i}") for i in range(n)]
            store = ReplicatedStore(name="test", replicas=replicas)
            assert store.quorum_size == expected_quorum


class TestReplicatedStoreRead:
    """Tests for read operations."""

    def test_reads_from_replicas(self):
        """Reads from replicas with consistency."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]
        # Write to all replicas
        for r in replicas:
            r.put_sync("key1", "value1")

        store = ReplicatedStore(name="distributed", replicas=replicas)

        gen = store.get("key1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == "value1"
        assert store.stats.read_successes == 1

    def test_read_with_one_consistency(self):
        """Reads with ONE consistency need only one replica."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]
        # Only first replica has the value
        replicas[0].put_sync("key1", "value1")

        store = ReplicatedStore(
            name="distributed",
            replicas=replicas,
            read_consistency=ConsistencyLevel.ONE,
        )

        gen = store.get("key1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == "value1"

    def test_read_miss_returns_none(self):
        """Reading missing key returns None."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]

        store = ReplicatedStore(name="distributed", replicas=replicas)

        gen = store.get("missing")
        result = "not_none"
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result is None


class TestReplicatedStoreWrite:
    """Tests for write operations."""

    def test_writes_to_all_replicas(self):
        """Writes are sent to all replicas."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]

        store = ReplicatedStore(name="distributed", replicas=replicas)

        gen = store.put("key1", "value1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        for r in replicas:
            assert r.get_sync("key1") == "value1"

    def test_write_success_with_quorum(self):
        """Write succeeds with quorum acknowledgment."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]

        store = ReplicatedStore(
            name="distributed",
            replicas=replicas,
            write_consistency=ConsistencyLevel.QUORUM,
        )

        gen = store.put("key1", "value1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result is True
        assert store.stats.write_successes == 1


class TestReplicatedStoreStatistics:
    """Tests for ReplicatedStore statistics."""

    def test_tracks_read_latencies(self):
        """Statistics track read latencies."""
        replicas = [KVStore(name=f"node{i}", read_latency=0.005) for i in range(3)]
        for r in replicas:
            r.put_sync("key1", "value1")

        store = ReplicatedStore(name="distributed", replicas=replicas)

        gen = store.get("key1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert len(store.stats.read_latencies) == 1
        assert store.stats.read_latencies[0] > 0

    def test_replica_status(self):
        """Can get replica status."""
        replicas = [KVStore(name=f"node{i}") for i in range(3)]
        replicas[0].put_sync("key1", "value1")

        store = ReplicatedStore(name="distributed", replicas=replicas)

        status = store.get_replica_status()

        assert len(status) == 3
        assert status[0]["name"] == "node0"
        assert status[0]["size"] == 1
