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
]
