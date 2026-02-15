"""Cache eviction policy implementations.

Provides various cache eviction strategies including LRU, LFU, TTL, FIFO,
and random eviction.

Example:
    from happysimulator.components.datastore import LRUEviction, CachedStore

    cache = CachedStore(
        name="cache",
        backing_store=store,
        cache_capacity=100,
        eviction_policy=LRUEviction(),
    )
"""

import random
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable


class CacheEvictionPolicy(ABC):
    """Abstract base class for cache eviction policies.

    Eviction policies track key access patterns and decide which key
    to evict when the cache is full.
    """

    @abstractmethod
    def on_access(self, key: str) -> None:
        """Called when a key is accessed (read or write).

        Args:
            key: The accessed key.
        """

    @abstractmethod
    def on_insert(self, key: str) -> None:
        """Called when a new key is inserted.

        Args:
            key: The inserted key.
        """

    @abstractmethod
    def on_remove(self, key: str) -> None:
        """Called when a key is explicitly removed.

        Args:
            key: The removed key.
        """

    @abstractmethod
    def evict(self) -> str | None:
        """Select a key to evict.

        Returns:
            The key to evict, or None if no keys to evict.
        """

    @abstractmethod
    def clear(self) -> None:
        """Clear all tracking state."""


class LRUEviction(CacheEvictionPolicy):
    """Least Recently Used eviction policy.

    Evicts the key that hasn't been accessed for the longest time.
    Good for workloads with temporal locality.
    """

    def __init__(self):
        """Initialize LRU eviction policy."""
        self._order: OrderedDict[str, None] = OrderedDict()

    def on_access(self, key: str) -> None:
        """Move key to end (most recently used)."""
        if key in self._order:
            self._order.move_to_end(key)

    def on_insert(self, key: str) -> None:
        """Add key as most recently used."""
        self._order[key] = None

    def on_remove(self, key: str) -> None:
        """Remove key from tracking."""
        self._order.pop(key, None)

    def evict(self) -> str | None:
        """Return least recently used key."""
        if not self._order:
            return None
        # First key is least recently used
        key = next(iter(self._order))
        del self._order[key]
        return key

    def clear(self) -> None:
        """Clear tracking state."""
        self._order.clear()


class LFUEviction(CacheEvictionPolicy):
    """Least Frequently Used eviction policy.

    Evicts the key with the lowest access count.
    Good for workloads where popular items should stay cached.
    """

    def __init__(self):
        """Initialize LFU eviction policy."""
        self._counts: dict[str, int] = {}
        self._min_count = 0

    def on_access(self, key: str) -> None:
        """Increment access count for key."""
        if key in self._counts:
            self._counts[key] += 1

    def on_insert(self, key: str) -> None:
        """Initialize count for new key."""
        self._counts[key] = 1
        self._min_count = 1

    def on_remove(self, key: str) -> None:
        """Remove key from tracking."""
        self._counts.pop(key, None)

    def evict(self) -> str | None:
        """Return least frequently used key."""
        if not self._counts:
            return None

        # Find minimum count
        min_count = min(self._counts.values())

        # Find first key with minimum count
        for key, count in self._counts.items():
            if count == min_count:
                del self._counts[key]
                return key

        return None

    def clear(self) -> None:
        """Clear tracking state."""
        self._counts.clear()
        self._min_count = 0


class TTLEviction(CacheEvictionPolicy):
    """Time-To-Live based eviction policy.

    Evicts keys that have exceeded their TTL. If no expired keys,
    evicts the oldest key.

    Attributes:
        ttl: Time-to-live in seconds.
    """

    def __init__(self, ttl: float, clock_func: Callable[[], float] | None = None):
        """Initialize TTL eviction policy.

        Args:
            ttl: Time-to-live in seconds.
            clock_func: Function returning current time. Defaults to time.time().

        Raises:
            ValueError: If ttl <= 0.
        """
        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")

        self._ttl = ttl
        self._clock_func = clock_func or time.time
        self._insert_times: dict[str, float] = {}

    @property
    def ttl(self) -> float:
        """Time-to-live in seconds."""
        return self._ttl

    def on_access(self, key: str) -> None:
        """TTL doesn't update on access (only insertion time matters)."""

    def on_insert(self, key: str) -> None:
        """Record insertion time."""
        self._insert_times[key] = self._clock_func()

    def on_remove(self, key: str) -> None:
        """Remove key from tracking."""
        self._insert_times.pop(key, None)

    def evict(self) -> str | None:
        """Return an expired key, or oldest key if none expired."""
        if not self._insert_times:
            return None

        now = self._clock_func()

        # Find expired keys
        for key, insert_time in list(self._insert_times.items()):
            if now - insert_time >= self._ttl:
                del self._insert_times[key]
                return key

        # No expired keys - evict oldest
        oldest_key = min(self._insert_times, key=lambda k: self._insert_times[k])
        del self._insert_times[oldest_key]
        return oldest_key

    def is_expired(self, key: str) -> bool:
        """Check if a key has expired.

        Args:
            key: The key to check.

        Returns:
            True if expired or not tracked.
        """
        if key not in self._insert_times:
            return True
        return self._clock_func() - self._insert_times[key] >= self._ttl

    def get_expired_keys(self) -> list[str]:
        """Get all expired keys.

        Returns:
            List of expired keys.
        """
        now = self._clock_func()
        return [
            key for key, insert_time in self._insert_times.items() if now - insert_time >= self._ttl
        ]

    def clear(self) -> None:
        """Clear tracking state."""
        self._insert_times.clear()


class FIFOEviction(CacheEvictionPolicy):
    """First-In-First-Out eviction policy.

    Evicts the key that was inserted earliest.
    Simple and predictable, but doesn't consider access patterns.
    """

    def __init__(self):
        """Initialize FIFO eviction policy."""
        self._order: list[str] = []

    def on_access(self, key: str) -> None:
        """FIFO doesn't update on access."""

    def on_insert(self, key: str) -> None:
        """Add key to end of queue."""
        if key not in self._order:
            self._order.append(key)

    def on_remove(self, key: str) -> None:
        """Remove key from queue."""
        if key in self._order:
            self._order.remove(key)

    def evict(self) -> str | None:
        """Return first inserted key."""
        if not self._order:
            return None
        return self._order.pop(0)

    def clear(self) -> None:
        """Clear tracking state."""
        self._order.clear()


class RandomEviction(CacheEvictionPolicy):
    """Random eviction policy.

    Evicts a random key. Useful as a baseline for comparison.
    """

    def __init__(self, seed: int | None = None):
        """Initialize random eviction policy.

        Args:
            seed: Random seed for reproducibility.
        """
        self._keys: set[str] = set()
        self._rng = random.Random(seed)

    def on_access(self, key: str) -> None:
        """Random doesn't track access."""

    def on_insert(self, key: str) -> None:
        """Track key."""
        self._keys.add(key)

    def on_remove(self, key: str) -> None:
        """Remove key from tracking."""
        self._keys.discard(key)

    def evict(self) -> str | None:
        """Return a random key."""
        if not self._keys:
            return None
        key = self._rng.choice(list(self._keys))
        self._keys.discard(key)
        return key

    def clear(self) -> None:
        """Clear tracking state."""
        self._keys.clear()


class SLRUEviction(CacheEvictionPolicy):
    """Segmented LRU (SLRU) eviction policy.

    Divides the cache into two segments:
    - Probationary segment: New items enter here
    - Protected segment: Items that are accessed again move here

    Eviction order: probationary first (LRU), then protected (LRU).
    This provides scan resistance - a single scan won't evict hot items.

    Attributes:
        protected_ratio: Fraction of capacity for protected segment (0.0-1.0).
    """

    def __init__(self, protected_ratio: float = 0.8):
        """Initialize SLRU eviction policy.

        Args:
            protected_ratio: Fraction of total capacity for protected segment.
                            Default 0.8 means 80% protected, 20% probationary.

        Raises:
            ValueError: If protected_ratio not in (0, 1).
        """
        if not 0 < protected_ratio < 1:
            raise ValueError(f"protected_ratio must be in (0, 1), got {protected_ratio}")

        self._protected_ratio = protected_ratio
        self._probationary: OrderedDict[str, None] = OrderedDict()
        self._protected: OrderedDict[str, None] = OrderedDict()

    @property
    def protected_ratio(self) -> float:
        """Fraction of capacity for protected segment."""
        return self._protected_ratio

    @property
    def probationary_size(self) -> int:
        """Current size of probationary segment."""
        return len(self._probationary)

    @property
    def protected_size(self) -> int:
        """Current size of protected segment."""
        return len(self._protected)

    def on_access(self, key: str) -> None:
        """Promote key from probationary to protected on re-access."""
        if key in self._probationary:
            # Promote to protected
            del self._probationary[key]
            self._protected[key] = None
            self._protected.move_to_end(key)
        elif key in self._protected:
            # Already protected, move to end (most recently used)
            self._protected.move_to_end(key)

    def on_insert(self, key: str) -> None:
        """Insert new key into probationary segment."""
        # New items always go to probationary
        self._probationary[key] = None

    def on_remove(self, key: str) -> None:
        """Remove key from whichever segment it's in."""
        self._probationary.pop(key, None)
        self._protected.pop(key, None)

    def evict(self) -> str | None:
        """Evict from probationary first, then protected."""
        # Try probationary first
        if self._probationary:
            key = next(iter(self._probationary))
            del self._probationary[key]
            return key

        # Fall back to protected
        if self._protected:
            key = next(iter(self._protected))
            del self._protected[key]
            return key

        return None

    def clear(self) -> None:
        """Clear both segments."""
        self._probationary.clear()
        self._protected.clear()


class SampledLRUEviction(CacheEvictionPolicy):
    """Sampled LRU (Probabilistic LRU) eviction policy.

    Instead of tracking exact LRU order for all keys, this policy:
    1. Samples N random keys from the cache
    2. Evicts the least recently used among the sample

    This provides O(1) access updates (just update timestamp) and O(N) eviction,
    where N is the sample size. For large caches, this is much cheaper than
    true LRU's O(log n) or O(n) operations.

    Used by Redis for its approximated LRU implementation.

    Attributes:
        sample_size: Number of keys to sample during eviction.
    """

    def __init__(self, sample_size: int = 5, seed: int | None = None):
        """Initialize Sampled LRU eviction policy.

        Args:
            sample_size: Number of keys to sample for eviction decision.
                        Higher = more accurate but slower. Redis uses 5.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If sample_size < 1.
        """
        if sample_size < 1:
            raise ValueError(f"sample_size must be >= 1, got {sample_size}")

        self._sample_size = sample_size
        self._rng = random.Random(seed)
        # Track access time for each key (logical clock)
        self._access_times: dict[str, int] = {}
        self._clock = 0

    @property
    def sample_size(self) -> int:
        """Number of keys sampled during eviction."""
        return self._sample_size

    def on_access(self, key: str) -> None:
        """Update access time for key."""
        if key in self._access_times:
            self._clock += 1
            self._access_times[key] = self._clock

    def on_insert(self, key: str) -> None:
        """Record insertion time as access time."""
        self._clock += 1
        self._access_times[key] = self._clock

    def on_remove(self, key: str) -> None:
        """Remove key from tracking."""
        self._access_times.pop(key, None)

    def evict(self) -> str | None:
        """Sample keys and evict the least recently accessed."""
        if not self._access_times:
            return None

        keys = list(self._access_times.keys())

        # Sample up to sample_size keys
        sample_count = min(self._sample_size, len(keys))
        sample = self._rng.sample(keys, sample_count)

        # Find the least recently used in the sample
        lru_key = min(sample, key=lambda k: self._access_times[k])

        del self._access_times[lru_key]
        return lru_key

    def clear(self) -> None:
        """Clear tracking state."""
        self._access_times.clear()
        self._clock = 0


class ClockEviction(CacheEvictionPolicy):
    """Clock (Second-Chance) eviction policy.

    Approximates LRU using a circular buffer with reference bits.
    Each key has a reference bit that is set on access.

    On eviction:
    - Scan keys in circular order
    - If reference bit is set, clear it and move on (second chance)
    - If reference bit is clear, evict that key

    This provides O(1) access updates and amortized O(1) eviction,
    making it very efficient for large caches.

    Also known as: Second-Chance algorithm, Not Recently Used (NRU).
    """

    def __init__(self):
        """Initialize Clock eviction policy."""
        # Circular buffer of keys
        self._keys: list[str] = []
        # Reference bits (True = recently accessed)
        self._ref_bits: dict[str, bool] = {}
        # Current position in circular buffer (clock hand)
        self._hand = 0

    @property
    def size(self) -> int:
        """Number of tracked keys."""
        return len(self._keys)

    def on_access(self, key: str) -> None:
        """Set reference bit for accessed key."""
        if key in self._ref_bits:
            self._ref_bits[key] = True

    def on_insert(self, key: str) -> None:
        """Insert key with reference bit set."""
        if key not in self._ref_bits:
            self._keys.append(key)
            self._ref_bits[key] = True

    def on_remove(self, key: str) -> None:
        """Remove key from tracking."""
        if key in self._ref_bits:
            self._keys.remove(key)
            del self._ref_bits[key]
            # Adjust hand if needed
            if self._hand >= len(self._keys) and self._keys:
                self._hand = 0

    def evict(self) -> str | None:
        """Evict using clock algorithm."""
        if not self._keys:
            return None

        # Scan until we find a key with ref bit = False
        # (or we've given everyone a second chance)
        scanned = 0
        while scanned < len(self._keys) * 2:  # Max 2 full rotations
            key = self._keys[self._hand]

            if self._ref_bits[key]:
                # Give second chance - clear ref bit
                self._ref_bits[key] = False
            else:
                # Evict this key
                self._keys.pop(self._hand)
                del self._ref_bits[key]

                # Adjust hand
                if self._hand >= len(self._keys) and self._keys:
                    self._hand = 0

                return key

            # Advance hand (circular)
            self._hand = (self._hand + 1) % len(self._keys)
            scanned += 1

        # If all keys had ref bits set, evict current position
        if self._keys:
            key = self._keys[self._hand]
            self._keys.pop(self._hand)
            del self._ref_bits[key]
            if self._hand >= len(self._keys) and self._keys:
                self._hand = 0
            return key

        return None

    def clear(self) -> None:
        """Clear tracking state."""
        self._keys.clear()
        self._ref_bits.clear()
        self._hand = 0


class TwoQueueEviction(CacheEvictionPolicy):
    """2Q (Two Queue) eviction policy.

    Similar to SLRU but uses FIFO for the first queue (A1).
    Provides excellent scan resistance.

    Structure:
    - A1in: FIFO queue for first-time accessed items
    - A1out: Ghost queue tracking recently evicted from A1in (keys only)
    - Am: LRU queue for items accessed more than once

    On first access: item goes to A1in
    On access while in A1in: nothing (FIFO, no promotion yet)
    On access while in Am: move to MRU position in Am
    On access of item in A1out (ghost): add to Am (it's a valuable item)
    On eviction: remove from A1in first, then Am

    Attributes:
        kin_ratio: Fraction of cache for A1in queue (default 0.25).
    """

    def __init__(self, kin_ratio: float = 0.25):
        """Initialize 2Q eviction policy.

        Args:
            kin_ratio: Fraction of total cache for A1in queue.

        Raises:
            ValueError: If kin_ratio not in (0, 1).
        """
        if not 0 < kin_ratio < 1:
            raise ValueError(f"kin_ratio must be in (0, 1), got {kin_ratio}")

        self._kin_ratio = kin_ratio
        self._a1in: list[str] = []  # FIFO queue for new items
        self._a1out: list[str] = []  # Ghost queue (evicted from A1in)
        self._am: OrderedDict[str, None] = OrderedDict()  # LRU for hot items
        self._a1out_max = 50  # Max size of ghost queue

    @property
    def kin_ratio(self) -> float:
        """Fraction of cache for A1in queue."""
        return self._kin_ratio

    def on_access(self, key: str) -> None:
        """Handle access - promote if in ghost queue or Am."""
        if key in self._am:
            # Already in Am, move to MRU position
            self._am.move_to_end(key)
        # Note: items in A1in stay there (FIFO, no promotion on access)

    def on_insert(self, key: str) -> None:
        """Insert new key, checking ghost queue first."""
        if key in self._a1out:
            # Was recently evicted, must be valuable - add to Am
            self._a1out.remove(key)
            self._am[key] = None
        else:
            # First time seeing this key - add to A1in
            self._a1in.append(key)

    def on_remove(self, key: str) -> None:
        """Remove key from all queues."""
        if key in self._a1in:
            self._a1in.remove(key)
        if key in self._a1out:
            self._a1out.remove(key)
        self._am.pop(key, None)

    def evict(self) -> str | None:
        """Evict from A1in first, then Am."""
        if self._a1in:
            key = self._a1in.pop(0)
            # Add to ghost queue
            self._a1out.append(key)
            # Trim ghost queue
            while len(self._a1out) > self._a1out_max:
                self._a1out.pop(0)
            return key

        if self._am:
            key = next(iter(self._am))
            del self._am[key]
            return key

        return None

    def clear(self) -> None:
        """Clear all queues."""
        self._a1in.clear()
        self._a1out.clear()
        self._am.clear()
