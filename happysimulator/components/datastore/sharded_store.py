"""Sharded store implementation.

Provides horizontal partitioning of data across multiple shards.
Supports various sharding strategies including hash, range, and consistent hashing.

Example:
    from happysimulator.components.datastore import (
        KVStore, ShardedStore, HashSharding
    )

    # Create shard nodes
    shards = [
        KVStore(name=f"shard{i}", read_latency=0.005)
        for i in range(4)
    ]

    # Create sharded store
    store = ShardedStore(
        name="sharded_db",
        shards=shards,
        sharding_strategy=HashSharding(),
    )
"""

import bisect
import hashlib
from abc import abstractmethod
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any, Protocol

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class ShardingStrategy(Protocol):
    """Protocol for sharding strategies."""

    @abstractmethod
    def get_shard(self, key: str, num_shards: int) -> int:
        """Determine which shard a key belongs to.

        Args:
            key: The key to shard.
            num_shards: Total number of shards.

        Returns:
            Shard index (0 to num_shards-1).
        """
        ...


class HashSharding:
    """Simple hash-based sharding.

    Uses MD5 hash of the key modulo number of shards.
    Fast and uniform, but reshards all keys when shard count changes.
    """

    def get_shard(self, key: str, num_shards: int) -> int:
        """Get shard index using hash modulo."""
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return hash_value % num_shards


class RangeSharding:
    """Range-based sharding.

    Assigns key ranges to shards based on alphabetical ordering.
    Good for range queries, but can lead to hot spots.
    """

    def __init__(self, boundaries: list[str] | None = None):
        """Initialize range sharding.

        Args:
            boundaries: Optional list of key boundaries for shards.
                       If None, keys are distributed alphabetically.
        """
        self._boundaries = boundaries

    def get_shard(self, key: str, num_shards: int) -> int:
        """Get shard index using range boundaries."""
        if self._boundaries:
            # Use provided boundaries
            for i, boundary in enumerate(self._boundaries):
                if key < boundary:
                    return i
            return len(self._boundaries)
        # Simple alphabetical distribution
        # Assumes keys are strings that can be compared
        if not key:
            return 0
        # Use first character to determine shard
        first_char = ord(key[0].lower())
        # Map a-z (97-122) to shard indices
        if first_char < 97:
            return 0
        if first_char > 122:
            return num_shards - 1
        return (first_char - 97) * num_shards // 26


class ConsistentHashSharding:
    """Consistent hash sharding.

    Uses consistent hashing with virtual nodes to minimize key
    redistribution when shards are added or removed.
    """

    def __init__(self, virtual_nodes: int = 100, seed: int | None = None):
        """Initialize consistent hash sharding.

        Args:
            virtual_nodes: Number of virtual nodes per shard.
            seed: Optional random seed for reproducibility.
        """
        self._virtual_nodes = virtual_nodes
        self._seed = seed
        self._ring: list[tuple[int, int]] = []  # (hash, shard_index)
        self._initialized_for: int = 0

    def _build_ring(self, num_shards: int) -> None:
        """Build the hash ring for given number of shards."""
        if self._initialized_for == num_shards:
            return

        self._ring = []
        for shard_idx in range(num_shards):
            for vnode in range(self._virtual_nodes):
                # Create unique identifier for this virtual node
                vnode_key = f"shard{shard_idx}:vnode{vnode}"
                if self._seed is not None:
                    vnode_key = f"{self._seed}:{vnode_key}"
                hash_value = int(hashlib.md5(vnode_key.encode()).hexdigest(), 16)
                self._ring.append((hash_value, shard_idx))

        # Sort by hash value
        self._ring.sort(key=lambda x: x[0])
        self._initialized_for = num_shards

    def get_shard(self, key: str, num_shards: int) -> int:
        """Get shard index using consistent hashing."""
        self._build_ring(num_shards)

        if not self._ring:
            return 0

        key_hash = int(hashlib.md5(key.encode()).hexdigest(), 16)

        # Binary search for the first node with hash >= key_hash
        hashes = [h for h, _ in self._ring]
        idx = bisect.bisect_left(hashes, key_hash)

        # Wrap around if we're past the end
        if idx >= len(self._ring):
            idx = 0

        return self._ring[idx][1]


@dataclass(frozen=True)
class ShardedStoreStats:
    """Statistics tracked by ShardedStore."""

    reads: int = 0
    writes: int = 0
    deletes: int = 0
    shard_reads: dict[int, int] = field(default_factory=dict)
    shard_writes: dict[int, int] = field(default_factory=dict)

    def get_shard_distribution(self) -> dict[int, float]:
        """Get read distribution across shards as percentages."""
        total = sum(self.shard_reads.values())
        if total == 0:
            return {}
        return {k: v / total for k, v in self.shard_reads.items()}


class ShardedStore(Entity):
    """Horizontally partitioned key-value store.

    Distributes data across multiple shards using a configurable
    sharding strategy. Each key maps to exactly one shard.

    Attributes:
        name: Entity name for identification.
        num_shards: Number of shard nodes.
        sharding_strategy: Strategy used for key distribution.
    """

    def __init__(
        self,
        name: str,
        shards: list[Entity],
        sharding_strategy: ShardingStrategy | None = None,
    ):
        """Initialize the sharded store.

        Args:
            name: Name for this store entity.
            shards: List of shard nodes (KVStore or similar).
            sharding_strategy: Strategy for distributing keys.
                              Defaults to HashSharding.

        Raises:
            ValueError: If no shards provided.
        """
        if not shards:
            raise ValueError("At least one shard is required")

        super().__init__(name)
        self._shards = shards
        self._sharding_strategy = sharding_strategy or HashSharding()

        # Statistics
        self._reads = 0
        self._writes = 0
        self._deletes = 0
        self._shard_reads: dict[int, int] = dict.fromkeys(range(len(shards)), 0)
        self._shard_writes: dict[int, int] = dict.fromkeys(range(len(shards)), 0)

    @property
    def stats(self) -> ShardedStoreStats:
        """Frozen snapshot of sharded store statistics."""
        return ShardedStoreStats(
            reads=self._reads,
            writes=self._writes,
            deletes=self._deletes,
            shard_reads=dict(self._shard_reads),
            shard_writes=dict(self._shard_writes),
        )

    @property
    def num_shards(self) -> int:
        """Number of shard nodes."""
        return len(self._shards)

    @property
    def shards(self) -> list[Entity]:
        """The shard nodes."""
        return self._shards

    @property
    def sharding_strategy(self) -> ShardingStrategy:
        """Strategy used for key distribution."""
        return self._sharding_strategy

    def _get_shard_for_key(self, key: str) -> tuple[int, Entity]:
        """Get the shard index and node for a key."""
        shard_idx = self._sharding_strategy.get_shard(key, len(self._shards))
        return shard_idx, self._shards[shard_idx]

    def get(self, key: str) -> Generator[float, None, Any | None]:
        """Get a value from the appropriate shard.

        Args:
            key: The key to look up.

        Yields:
            Read latency delay.

        Returns:
            The value if found, None otherwise.
        """
        self._reads += 1
        shard_idx, shard = self._get_shard_for_key(key)
        self._shard_reads[shard_idx] = self._shard_reads.get(shard_idx, 0) + 1

        value = yield from shard.get(key)
        return value

    def put(self, key: str, value: Any) -> Generator[float]:
        """Store a value in the appropriate shard.

        Args:
            key: The key to store under.
            value: The value to store.

        Yields:
            Write latency delay.
        """
        self._writes += 1
        shard_idx, shard = self._get_shard_for_key(key)
        self._shard_writes[shard_idx] = self._shard_writes.get(shard_idx, 0) + 1

        yield from shard.put(key, value)

    def delete(self, key: str) -> Generator[float, None, bool]:
        """Delete a key from the appropriate shard.

        Args:
            key: The key to delete.

        Yields:
            Delete latency delay.

        Returns:
            True if key existed.
        """
        self._deletes += 1
        _shard_idx, shard = self._get_shard_for_key(key)

        result = yield from shard.delete(key)
        return result

    def get_shard_for_key(self, key: str) -> int:
        """Get the shard index for a key without accessing the shard.

        Args:
            key: The key to check.

        Returns:
            Shard index (0 to num_shards-1).
        """
        return self._sharding_strategy.get_shard(key, len(self._shards))

    def get_shard_sizes(self) -> dict[int, int]:
        """Get the size of each shard.

        Returns:
            Dictionary mapping shard index to key count.
        """
        sizes = {}
        for i, shard in enumerate(self._shards):
            if hasattr(shard, "size"):
                sizes[i] = shard.size
            else:
                sizes[i] = 0
        return sizes

    def get_all_keys(self) -> list[str]:
        """Get all keys across all shards.

        Returns:
            List of all keys.
        """
        all_keys = []
        for shard in self._shards:
            if hasattr(shard, "keys"):
                all_keys.extend(shard.keys())
        return all_keys

    def scatter_gather(self, keys: list[str]) -> Generator[float, None, dict[str, Any]]:
        """Get multiple keys, optimizing for parallel shard access.

        Groups keys by shard and fetches from each shard.

        Args:
            keys: List of keys to fetch.

        Yields:
            Latency delays.

        Returns:
            Dictionary mapping keys to values (missing keys omitted).
        """
        # Group keys by shard
        shard_keys: dict[int, list[str]] = {}
        for key in keys:
            shard_idx = self.get_shard_for_key(key)
            if shard_idx not in shard_keys:
                shard_keys[shard_idx] = []
            shard_keys[shard_idx].append(key)

        # Fetch from each shard
        results = {}
        for shard_idx, keys_for_shard in shard_keys.items():
            shard = self._shards[shard_idx]
            for key in keys_for_shard:
                value = yield from shard.get(key)
                if value is not None:
                    results[key] = value

        return results

    def handle_event(self, event: Event) -> None:
        """ShardedStore can handle events for store operations."""
