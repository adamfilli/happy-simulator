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

from happysimulator.components.datastore.kv_store import KVStore, KVStoreStats
from happysimulator.components.datastore.eviction_policies import (
    CacheEvictionPolicy,
    LRUEviction,
    LFUEviction,
    TTLEviction,
    FIFOEviction,
    RandomEviction,
    SLRUEviction,
    SampledLRUEviction,
    ClockEviction,
    TwoQueueEviction,
)
from happysimulator.components.datastore.cached_store import CachedStore, CachedStoreStats
from happysimulator.components.datastore.write_policies import (
    WritePolicy,
    WriteThrough,
    WriteBack,
    WriteAround,
)
from happysimulator.components.datastore.cache_warming import CacheWarmer, CacheWarmerStats
from happysimulator.components.datastore.multi_tier_cache import (
    MultiTierCache,
    MultiTierCacheStats,
    PromotionPolicy,
)
from happysimulator.components.datastore.replicated_store import (
    ReplicatedStore,
    ReplicatedStoreStats,
    ConsistencyLevel,
)
from happysimulator.components.datastore.sharded_store import (
    ShardedStore,
    ShardedStoreStats,
    ShardingStrategy,
    HashSharding,
    RangeSharding,
    ConsistentHashSharding,
)
from happysimulator.components.datastore.database import (
    Database,
    DatabaseStats,
    Transaction,
    TransactionState,
)
from happysimulator.components.datastore.soft_ttl_cache import (
    SoftTTLCache,
    SoftTTLCacheStats,
    CacheEntry,
)

__all__ = [
    # Key-Value Store
    "KVStore",
    "KVStoreStats",
    # Eviction Policies
    "CacheEvictionPolicy",
    "LRUEviction",
    "LFUEviction",
    "TTLEviction",
    "FIFOEviction",
    "RandomEviction",
    "SLRUEviction",
    "SampledLRUEviction",
    "ClockEviction",
    "TwoQueueEviction",
    # Cached Store
    "CachedStore",
    "CachedStoreStats",
    # Write Policies
    "WritePolicy",
    "WriteThrough",
    "WriteBack",
    "WriteAround",
    # Cache Warming
    "CacheWarmer",
    "CacheWarmerStats",
    # Multi-Tier Cache
    "MultiTierCache",
    "MultiTierCacheStats",
    "PromotionPolicy",
    # Replicated Store
    "ReplicatedStore",
    "ReplicatedStoreStats",
    "ConsistencyLevel",
    # Sharded Store
    "ShardedStore",
    "ShardedStoreStats",
    "ShardingStrategy",
    "HashSharding",
    "RangeSharding",
    "ConsistentHashSharding",
    # Database
    "Database",
    "DatabaseStats",
    "Transaction",
    "TransactionState",
    # Soft TTL Cache
    "SoftTTLCache",
    "SoftTTLCacheStats",
    "CacheEntry",
]
