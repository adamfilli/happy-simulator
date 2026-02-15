"""Tests for ShardedStore."""

import pytest

from happysimulator.components.datastore import (
    ConsistentHashSharding,
    HashSharding,
    KVStore,
    RangeSharding,
    ShardedStore,
)


class TestShardingStrategies:
    """Tests for sharding strategies."""

    def test_hash_sharding_deterministic(self):
        """HashSharding is deterministic."""
        strategy = HashSharding()

        shard1 = strategy.get_shard("key1", 4)
        shard2 = strategy.get_shard("key1", 4)

        assert shard1 == shard2

    def test_hash_sharding_distribution(self):
        """HashSharding distributes keys across shards."""
        strategy = HashSharding()
        shards = [strategy.get_shard(f"key{i}", 4) for i in range(100)]

        # All shards should have some keys
        for shard in range(4):
            assert shard in shards

    def test_range_sharding(self):
        """RangeSharding assigns based on key range."""
        strategy = RangeSharding()

        # Keys starting with 'a' and 'z' should go to different shards
        shard_a = strategy.get_shard("apple", 4)
        shard_z = strategy.get_shard("zebra", 4)

        # They should be in different shards (a is early, z is late)
        assert shard_a != shard_z or shard_a == 0  # Could be in first shard

    def test_range_sharding_with_boundaries(self):
        """RangeSharding with explicit boundaries."""
        strategy = RangeSharding(boundaries=["m"])

        shard_a = strategy.get_shard("apple", 2)
        shard_z = strategy.get_shard("zebra", 2)

        assert shard_a == 0  # Before 'm'
        assert shard_z == 1  # After 'm'

    def test_consistent_hash_deterministic(self):
        """ConsistentHashSharding is deterministic."""
        strategy = ConsistentHashSharding(seed=42)

        shard1 = strategy.get_shard("key1", 4)
        shard2 = strategy.get_shard("key1", 4)

        assert shard1 == shard2

    def test_consistent_hash_same_seed(self):
        """Same seed produces same results."""
        strategy1 = ConsistentHashSharding(seed=42)
        strategy2 = ConsistentHashSharding(seed=42)

        for i in range(10):
            key = f"key{i}"
            assert strategy1.get_shard(key, 4) == strategy2.get_shard(key, 4)


class TestShardedStoreCreation:
    """Tests for ShardedStore creation."""

    def test_creates_with_shards(self):
        """ShardedStore is created with shards."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]

        store = ShardedStore(name="sharded", shards=shards)

        assert store.name == "sharded"
        assert store.num_shards == 4

    def test_rejects_no_shards(self):
        """Rejects empty shard list."""
        with pytest.raises(ValueError):
            ShardedStore(name="sharded", shards=[])

    def test_default_strategy_is_hash(self):
        """Default sharding strategy is HashSharding."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]

        store = ShardedStore(name="sharded", shards=shards)

        assert isinstance(store.sharding_strategy, HashSharding)


class TestShardedStoreOperations:
    """Tests for sharded store operations."""

    def test_writes_to_correct_shard(self):
        """Writes go to the correct shard."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        gen = store.put("key1", "value1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Find which shard has the key
        shard_idx = store.get_shard_for_key("key1")
        assert shards[shard_idx].get_sync("key1") == "value1"

        # Other shards shouldn't have it
        for i, shard in enumerate(shards):
            if i != shard_idx:
                assert shard.get_sync("key1") is None

    def test_reads_from_correct_shard(self):
        """Reads come from the correct shard."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        # Write directly to a shard
        shard_idx = store.get_shard_for_key("key1")
        shards[shard_idx].put_sync("key1", "value1")

        gen = store.get("key1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == "value1"

    def test_delete_from_correct_shard(self):
        """Deletes target the correct shard."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        # Write first
        gen = store.put("key1", "value1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Delete
        gen = store.delete("key1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result is True

        # Verify deleted
        shard_idx = store.get_shard_for_key("key1")
        assert shards[shard_idx].get_sync("key1") is None


class TestShardedStoreStatistics:
    """Tests for ShardedStore statistics."""

    def test_tracks_shard_reads(self):
        """Statistics track reads per shard."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        for i in range(10):
            gen = store.get(f"key{i}")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert store.stats.reads == 10
        assert sum(store.stats.shard_reads.values()) == 10

    def test_tracks_shard_writes(self):
        """Statistics track writes per shard."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        for i in range(10):
            gen = store.put(f"key{i}", f"value{i}")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert store.stats.writes == 10
        assert sum(store.stats.shard_writes.values()) == 10

    def test_get_shard_sizes(self):
        """Can get size of each shard."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        # Write some keys
        for i in range(10):
            gen = store.put(f"key{i}", f"value{i}")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        sizes = store.get_shard_sizes()
        assert sum(sizes.values()) == 10


class TestShardedStoreScatterGather:
    """Tests for scatter-gather operations."""

    def test_scatter_gather_multiple_keys(self):
        """Scatter-gather fetches from multiple shards."""
        shards = [KVStore(name=f"shard{i}") for i in range(4)]
        store = ShardedStore(name="sharded", shards=shards)

        # Write some keys
        for i in range(5):
            gen = store.put(f"key{i}", f"value{i}")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # Scatter-gather
        gen = store.scatter_gather(["key0", "key1", "key2", "missing"])
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result["key0"] == "value0"
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"
        assert "missing" not in result
