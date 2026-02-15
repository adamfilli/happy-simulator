"""Data store components for simulating storage systems.

This module provides key-value stores, caching layers, and related
storage infrastructure for simulating realistic data access patterns.

Example:
    from happysimulator.components.datastore import KVStore, CachedStore, LRUEviction

    # Simple key-value store
    store = KVStore(name="db")

    # Cached store with LRU eviction
    cache = CachedStore(
        name="cached_db",
        backing_store=store,
        cache_capacity=1000,
        eviction_policy=LRUEviction(),
    )
"""

from happysimulator.components.datastore.cache_warming import CacheWarmer, CacheWarmerStats
from happysimulator.components.datastore.cached_store import CachedStore, CachedStoreStats
from happysimulator.components.datastore.database import (
    Database,
    DatabaseStats,
    Transaction,
    TransactionState,
)
from happysimulator.components.datastore.eviction_policies import (
    CacheEvictionPolicy,
    ClockEviction,
    FIFOEviction,
    LFUEviction,
    LRUEviction,
    RandomEviction,
    SampledLRUEviction,
    SLRUEviction,
    TTLEviction,
    TwoQueueEviction,
)
from happysimulator.components.datastore.kv_store import KVStore, KVStoreStats
from happysimulator.components.datastore.multi_tier_cache import (
    MultiTierCache,
    MultiTierCacheStats,
    PromotionPolicy,
)
from happysimulator.components.datastore.replicated_store import (
    ConsistencyLevel,
    ReplicatedStore,
    ReplicatedStoreStats,
)
from happysimulator.components.datastore.sharded_store import (
    ConsistentHashSharding,
    HashSharding,
    RangeSharding,
    ShardedStore,
    ShardedStoreStats,
    ShardingStrategy,
)
from happysimulator.components.datastore.soft_ttl_cache import (
    CacheEntry,
    SoftTTLCache,
    SoftTTLCacheStats,
)
from happysimulator.components.datastore.write_policies import (
    WriteAround,
    WriteBack,
    WritePolicy,
    WriteThrough,
)

__all__ = [
    "CacheEntry",
    # Eviction Policies
    "CacheEvictionPolicy",
    # Cache Warming
    "CacheWarmer",
    "CacheWarmerStats",
    # Cached Store
    "CachedStore",
    "CachedStoreStats",
    "ClockEviction",
    "ConsistencyLevel",
    "ConsistentHashSharding",
    # Database
    "Database",
    "DatabaseStats",
    "FIFOEviction",
    "HashSharding",
    # Key-Value Store
    "KVStore",
    "KVStoreStats",
    "LFUEviction",
    "LRUEviction",
    # Multi-Tier Cache
    "MultiTierCache",
    "MultiTierCacheStats",
    "PromotionPolicy",
    "RandomEviction",
    "RangeSharding",
    # Replicated Store
    "ReplicatedStore",
    "ReplicatedStoreStats",
    "SLRUEviction",
    "SampledLRUEviction",
    # Sharded Store
    "ShardedStore",
    "ShardedStoreStats",
    "ShardingStrategy",
    # Soft TTL Cache
    "SoftTTLCache",
    "SoftTTLCacheStats",
    "TTLEviction",
    "Transaction",
    "TransactionState",
    "TwoQueueEviction",
    "WriteAround",
    "WriteBack",
    # Write Policies
    "WritePolicy",
    "WriteThrough",
]
