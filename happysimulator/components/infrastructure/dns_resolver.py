"""DNS resolver model with caching and hierarchical lookup.

Simulates DNS name resolution with a multi-level cache and hierarchical
lookup through root, TLD, and authoritative nameservers. Each level
adds latency, and cached results expire after their TTL.

Key behaviors:
- Cache-first: returns cached result if TTL has not expired.
- Hierarchical lookup: root -> TLD -> authoritative, each with latency.
- Configurable TTL per record.
- Cache eviction when capacity is exceeded.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DNSRecord:
    """A DNS record mapping a hostname to an IP address.

    Attributes:
        hostname: The queried hostname.
        ip_address: The resolved IP address.
        ttl_s: Time-to-live in seconds.
    """

    hostname: str
    ip_address: str
    ttl_s: float = 300.0


@dataclass
class _CacheEntry:
    """Internal: a cached DNS record with expiry time."""

    record: DNSRecord
    expires_at_s: float


@dataclass(frozen=True)
class DNSStats:
    """Frozen snapshot of DNS resolver statistics.

    Attributes:
        lookups: Total lookup requests.
        cache_hits: Lookups served from cache.
        cache_misses: Lookups requiring hierarchical resolution.
        cache_expirations: Cache entries that expired (TTL).
        cache_evictions: Cache entries evicted due to capacity.
        cache_size: Current number of entries in cache.
        total_resolution_latency_s: Cumulative resolution latency.
    """

    lookups: int
    cache_hits: int
    cache_misses: int
    cache_expirations: int
    cache_evictions: int
    cache_size: int
    total_resolution_latency_s: float

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.lookups if self.lookups > 0 else 0.0

    @property
    def avg_resolution_latency_s(self) -> float:
        return self.total_resolution_latency_s / self.lookups if self.lookups > 0 else 0.0


# ---------------------------------------------------------------------------
# DNSResolver entity
# ---------------------------------------------------------------------------


class DNSResolver(Entity):
    """DNS resolver with caching, TTL, and hierarchical lookup latency.

    Provides ``resolve()`` to look up a hostname. Results are cached
    according to the record's TTL. Cache misses trigger a hierarchical
    lookup through root, TLD, and authoritative nameservers.

    Args:
        name: Entity name.
        cache_capacity: Maximum number of cached records (default 1000).
        root_latency_s: Latency for root nameserver query (default 20ms).
        tld_latency_s: Latency for TLD nameserver query (default 15ms).
        auth_latency_s: Latency for authoritative nameserver query (default 10ms).
        records: Pre-configured DNS records (hostname -> DNSRecord).

    Example::

        dns = DNSResolver("dns", records={
            "api.example.com": DNSRecord("api.example.com", "10.0.0.1", ttl_s=60),
        })
        sim = Simulation(entities=[dns, ...], ...)

        # In another entity's handle_event:
        ip = yield from dns.resolve("api.example.com")
    """

    def __init__(
        self,
        name: str,
        *,
        cache_capacity: int = 1000,
        root_latency_s: float = 0.02,
        tld_latency_s: float = 0.015,
        auth_latency_s: float = 0.01,
        records: dict[str, DNSRecord] | None = None,
    ) -> None:
        if cache_capacity < 1:
            raise ValueError(f"cache_capacity must be >= 1, got {cache_capacity}")

        super().__init__(name)
        self._cache_capacity = cache_capacity
        self._root_latency_s = root_latency_s
        self._tld_latency_s = tld_latency_s
        self._auth_latency_s = auth_latency_s

        # Authoritative records (source of truth)
        self._records: dict[str, DNSRecord] = dict(records) if records else {}

        # LRU cache
        self._cache: OrderedDict[str, _CacheEntry] = OrderedDict()

        # Stats
        self._lookups: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._cache_expirations: int = 0
        self._cache_evictions: int = 0
        self._total_latency_s: float = 0.0

    @property
    def cache_size(self) -> int:
        """Number of entries currently in the DNS cache."""
        return len(self._cache)

    @property
    def stats(self) -> DNSStats:
        """Frozen snapshot of DNS resolver statistics."""
        return DNSStats(
            lookups=self._lookups,
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            cache_expirations=self._cache_expirations,
            cache_evictions=self._cache_evictions,
            cache_size=len(self._cache),
            total_resolution_latency_s=self._total_latency_s,
        )

    def add_record(self, record: DNSRecord) -> None:
        """Add or update an authoritative DNS record.

        Args:
            record: The DNS record to add.
        """
        self._records[record.hostname] = record

    def _evict_expired(self) -> None:
        """Remove expired entries from cache."""
        now_s = self.now.to_seconds()
        expired = [h for h, e in self._cache.items() if e.expires_at_s <= now_s]
        for h in expired:
            del self._cache[h]
            self._cache_expirations += 1

    def _evict_lru(self) -> None:
        """Evict least-recently-used entries until under capacity."""
        while len(self._cache) >= self._cache_capacity:
            self._cache.popitem(last=False)
            self._cache_evictions += 1

    def resolve(self, hostname: str) -> Generator[float, None, str | None]:
        """Resolve a hostname to an IP address.

        Checks cache first, then performs hierarchical DNS lookup
        (root -> TLD -> authoritative). Returns the IP address
        or None if the hostname is not found.

        Args:
            hostname: The hostname to resolve.
        """
        self._lookups += 1
        now_s = self.now.to_seconds()

        # Check cache
        if hostname in self._cache:
            entry = self._cache[hostname]
            if entry.expires_at_s > now_s:
                self._cache_hits += 1
                self._cache.move_to_end(hostname)
                return entry.record.ip_address
            # Expired
            del self._cache[hostname]
            self._cache_expirations += 1

        # Cache miss — hierarchical lookup
        self._cache_misses += 1
        total_latency = 0.0

        # Root nameserver
        yield self._root_latency_s
        total_latency += self._root_latency_s

        # TLD nameserver
        yield self._tld_latency_s
        total_latency += self._tld_latency_s

        # Authoritative nameserver
        yield self._auth_latency_s
        total_latency += self._auth_latency_s

        self._total_latency_s += total_latency

        # Look up record
        record = self._records.get(hostname)
        if record is None:
            return None

        # Cache the result
        self._evict_lru()
        self._cache[hostname] = _CacheEntry(
            record=record,
            expires_at_s=now_s + total_latency + record.ttl_s,
        )

        return record.ip_address

    def handle_event(self, event: Event) -> None:
        """DNSResolver does not process events directly."""
        pass

    def __repr__(self) -> str:
        return (
            f"DNSResolver('{self.name}', cache={len(self._cache)}/{self._cache_capacity}, "
            f"records={len(self._records)})"
        )
