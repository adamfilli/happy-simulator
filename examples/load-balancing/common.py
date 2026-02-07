"""Shared components for load balancing examples.

This module provides:
- CachingServer: Server entity with local TTL-based cache backed by shared datastore
- Helper functions for metrics collection and visualization
- Custom event provider for customer-based requests
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generator

from happysimulator import (
    Entity,
    Event,
    EventProvider,
    Instant,
    QueuedResource,
    FIFOQueue,
)
from happysimulator.components.datastore.kv_store import KVStore
from happysimulator.components.datastore.cached_store import CachedStore
from happysimulator.components.datastore.eviction_policies import TTLEviction
from happysimulator.components.load_balancer.strategies import ConsistentHash
from happysimulator.distributions.value_distribution import ValueDistribution


# =============================================================================
# Custom Key Extraction for ConsistentHash
# =============================================================================


def customer_id_key_extractor(event: Event) -> str | None:
    """Extract customer_id from event for consistent hashing.

    This function is passed to ConsistentHash(get_key=...) to enable
    routing based on customer_id rather than the default keys.
    """
    ctx = event.context

    # Check root context
    if "customer_id" in ctx:
        return str(ctx["customer_id"])

    # Check metadata
    metadata = ctx.get("metadata", {})
    if "customer_id" in metadata:
        return str(metadata["customer_id"])

    return None


def create_customer_consistent_hash(virtual_nodes: int = 100) -> ConsistentHash:
    """Create a ConsistentHash strategy configured for customer_id routing.

    Args:
        virtual_nodes: Number of virtual nodes per backend.

    Returns:
        ConsistentHash configured with customer_id key extraction.
    """
    return ConsistentHash(
        virtual_nodes=virtual_nodes,
        get_key=customer_id_key_extractor,
    )


def create_customer_ip_hash():
    """Create an IPHash strategy configured for customer_id routing.

    IPHash uses modulo hashing (hash % N), which causes catastrophic
    cache invalidation when the number of backends changes.

    Returns:
        IPHash configured with customer_id key extraction.
    """
    from happysimulator.components.load_balancer.strategies import IPHash
    return IPHash(get_key=customer_id_key_extractor)


# =============================================================================
# CachingServer Entity
# =============================================================================


@dataclass
class CachingServerStats:
    """Statistics tracked by CachingServer."""

    requests_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


class CachingServer(QueuedResource):
    """Server with local TTL-based cache backed by a shared datastore.

    Each server maintains its own cache. With consistent hashing, the same
    customer always routes to the same server, maximizing cache hits.

    Probed metrics:
        - hit_rate: Cache hit ratio (0.0-1.0)
        - miss_rate: Cache miss ratio (0.0-1.0)
        - cache_size: Current number of cached entries
        - requests_processed: Total requests handled

    Args:
        name: Entity name for identification.
        server_id: Unique server identifier.
        datastore: Shared backing KVStore.
        cache_capacity: Maximum cache entries.
        cache_ttl_s: Time-to-live for cache entries in seconds.
        cache_read_latency_s: Latency for cache reads.
        datastore_read_latency_s: Latency for datastore reads (on miss).
        processing_latency_s: Additional processing time per request.
    """

    def __init__(
        self,
        name: str,
        server_id: int,
        datastore: KVStore,
        cache_capacity: int = 100,
        cache_ttl_s: float = 30.0,
        cache_read_latency_s: float = 0.0001,
        datastore_read_latency_s: float = 0.005,
        processing_latency_s: float = 0.001,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.server_id = server_id
        self._datastore = datastore
        self._cache_capacity = cache_capacity
        self._cache_ttl_s = cache_ttl_s
        self._cache_read_latency_s = cache_read_latency_s
        self._datastore_read_latency_s = datastore_read_latency_s
        self._processing_latency_s = processing_latency_s

        # Will be initialized in _ensure_cache_initialized
        self._cache: CachedStore | None = None
        self._eviction_policy: TTLEviction | None = None

        # Statistics
        self.stats = CachingServerStats()

    def _ensure_cache_initialized(self) -> CachedStore:
        """Lazily initialize cache with simulation clock."""
        if self._cache is None:
            # Create TTL eviction policy with simulation clock
            self._eviction_policy = TTLEviction(
                ttl=self._cache_ttl_s,
                clock_func=lambda: self.now.to_seconds(),
            )
            self._cache = CachedStore(
                name=f"{self.name}_cache",
                backing_store=self._datastore,
                cache_capacity=self._cache_capacity,
                eviction_policy=self._eviction_policy,
                cache_read_latency=self._cache_read_latency_s,
                write_through=True,
            )
        return self._cache

    @property
    def hit_rate(self) -> float:
        """Cache hit ratio (0.0-1.0)."""
        total = self.stats.cache_hits + self.stats.cache_misses
        if total == 0:
            return 0.0
        return self.stats.cache_hits / total

    @property
    def miss_rate(self) -> float:
        """Cache miss ratio (0.0-1.0)."""
        total = self.stats.cache_hits + self.stats.cache_misses
        if total == 0:
            return 0.0
        return self.stats.cache_misses / total

    @property
    def cache_size(self) -> int:
        """Current number of cached entries."""
        if self._cache is None:
            return 0
        return self._cache.cache_size

    @property
    def requests_processed(self) -> int:
        """Total requests handled."""
        return self.stats.requests_processed

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a request, checking cache first.

        Extracts customer_id from event context and uses it as cache key.
        On cache miss, reads from datastore and populates cache.
        """
        cache = self._ensure_cache_initialized()

        # Extract customer_id from context (supports multiple locations)
        customer_id = self._extract_customer_id(event)
        cache_key = f"customer:{customer_id}"

        # Check cache - must verify key exists AND is not expired
        was_in_cache = (
            cache_key in cache._cache
            and not self._eviction_policy.is_expired(cache_key)
        )

        # Simulate cache lookup time
        yield self._cache_read_latency_s

        if was_in_cache:
            self.stats.cache_hits += 1
            # Access updates eviction policy
            cache._eviction_policy.on_access(cache_key)
        else:
            self.stats.cache_misses += 1
            # Simulate datastore read on cache miss
            yield self._datastore_read_latency_s

            # Populate cache (simulating the data)
            self._populate_cache(cache, cache_key, {"customer_id": customer_id})

        # Additional processing time
        yield self._processing_latency_s

        self.stats.requests_processed += 1

        return []

    def _extract_customer_id(self, event: Event) -> Any:
        """Extract customer_id from event context.

        Supports multiple locations for flexibility:
        1. context["customer_id"]
        2. context["metadata"]["customer_id"]
        3. context["metadata"]["client_id"]
        """
        ctx = event.context

        # Direct field
        if "customer_id" in ctx:
            return ctx["customer_id"]

        # Nested in metadata
        metadata = ctx.get("metadata", {})
        if "customer_id" in metadata:
            return metadata["customer_id"]
        if "client_id" in metadata:
            return metadata["client_id"]

        # Fallback
        return "unknown"

    def _populate_cache(self, cache: CachedStore, key: str, value: Any) -> None:
        """Populate cache entry, handling eviction if needed."""
        # Check if eviction needed
        if cache.cache_size >= cache.cache_capacity:
            evicted_key = cache._eviction_policy.evict()
            if evicted_key and evicted_key in cache._cache:
                del cache._cache[evicted_key]
                cache.stats.evictions += 1

        # Insert into cache
        cache._cache[key] = value
        cache._eviction_policy.on_insert(key)

    def invalidate_cache(self) -> None:
        """Invalidate all cached entries."""
        if self._cache is not None:
            self._cache._cache.clear()
            if self._eviction_policy is not None:
                self._eviction_policy.clear()


# =============================================================================
# Custom Event Provider
# =============================================================================


class CustomerRequestProvider(EventProvider):
    """Event provider that creates customer requests with proper metadata structure.

    Generates events with customer_id in both root context and metadata,
    ensuring compatibility with ConsistentHash key extraction.
    """

    def __init__(
        self,
        target: Entity,
        customer_distribution: ValueDistribution,
        event_type: str = "Request",
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._customer_dist = customer_distribution
        self._event_type = event_type
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        customer_id = self._customer_dist.sample()

        # Place customer_id in metadata for ConsistentHash compatibility
        context = {
            "created_at": time,
            "request_id": self.generated,
            "customer_id": customer_id,
            "metadata": {
                "customer_id": customer_id,
            },
        }

        return [
            Event(
                time=time,
                event_type=self._event_type,
                target=self._target,
                context=context,
            )
        ]


# =============================================================================
# Metrics Collection
# =============================================================================


@dataclass
class AggregateMetrics:
    """Aggregated metrics across all servers."""

    total_requests: int = 0
    total_hits: int = 0
    total_misses: int = 0
    aggregate_hit_rate: float = 0.0
    per_server_hit_rates: dict[str, float] = field(default_factory=dict)
    per_server_request_counts: dict[str, int] = field(default_factory=dict)


def collect_aggregate_metrics(servers: list[CachingServer]) -> AggregateMetrics:
    """Collect aggregated statistics across all servers.

    Args:
        servers: List of CachingServer instances.

    Returns:
        AggregateMetrics with totals and per-server breakdowns.
    """
    metrics = AggregateMetrics()

    for server in servers:
        metrics.total_requests += server.stats.requests_processed
        metrics.total_hits += server.stats.cache_hits
        metrics.total_misses += server.stats.cache_misses
        metrics.per_server_hit_rates[server.name] = server.hit_rate
        metrics.per_server_request_counts[server.name] = server.stats.requests_processed

    if metrics.total_requests > 0:
        metrics.aggregate_hit_rate = metrics.total_hits / (
            metrics.total_hits + metrics.total_misses
        )

    return metrics


def compute_key_distribution(
    strategy: Any,
    backends: list[Entity],
    keys: list[Any],
) -> dict[str, int]:
    """Analyze how keys would be distributed across backends.

    Args:
        strategy: Load balancing strategy with select() method.
        backends: List of backend entities.
        keys: List of keys to distribute.

    Returns:
        Dict mapping backend name to count of assigned keys.
    """
    from happysimulator import Event, Instant

    distribution: dict[str, int] = {b.name: 0 for b in backends}

    for key in keys:
        # Create mock event with customer_id
        mock_event = Event(
            time=Instant.Epoch,
            event_type="Mock",
            target=backends[0],
            context={"metadata": {"customer_id": key}},
        )
        selected = strategy.select(backends, mock_event)
        if selected:
            distribution[selected.name] += 1

    return distribution


# =============================================================================
# Visualization Helpers
# =============================================================================


def plot_hit_rate_comparison(
    time_series: list[tuple[float, float, float]],
    labels: tuple[str, str],
    title: str,
    output_path: str,
) -> None:
    """Plot hit rate comparison between two strategies.

    Args:
        time_series: List of (time, strategy1_rate, strategy2_rate) tuples.
        labels: Tuple of (strategy1_name, strategy2_name).
        title: Plot title.
        output_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    times = [t for t, _, _ in time_series]
    rates1 = [r1 for _, r1, _ in time_series]
    rates2 = [r2 for _, _, r2 in time_series]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, rates1, 'b-', linewidth=2, label=labels[0])
    ax.plot(times, rates2, 'r--', linewidth=2, label=labels[1])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_key_distribution(
    distribution: dict[str, int],
    title: str,
    output_path: str,
) -> None:
    """Plot bar chart of key distribution across servers.

    Args:
        distribution: Dict mapping server name to key count.
        title: Plot title.
        output_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    servers = list(distribution.keys())
    counts = list(distribution.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(servers, counts, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha='center',
            va='bottom',
            fontsize=9,
        )

    ax.set_xlabel("Server")
    ax.set_ylabel("Number of Keys")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')

    # Add coefficient of variation annotation
    mean = sum(counts) / len(counts)
    std = (sum((c - mean) ** 2 for c in counts) / len(counts)) ** 0.5
    cov = std / mean if mean > 0 else 0
    ax.annotate(
        f"CoV: {cov:.3f}",
        xy=(0.95, 0.95),
        xycoords='axes fraction',
        ha='right',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_fleet_change_impact(
    time_series_consistent: list[tuple[float, float]],
    time_series_modulo: list[tuple[float, float]],
    change_time: float,
    title: str,
    output_path: str,
) -> None:
    """Plot hit rate over time showing fleet change impact.

    Args:
        time_series_consistent: (time, hit_rate) for consistent hashing.
        time_series_modulo: (time, hit_rate) for modulo hashing.
        change_time: Time when fleet change occurred.
        title: Plot title.
        output_path: Path to save the figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    times_c = [t for t, _ in time_series_consistent]
    rates_c = [r for _, r in time_series_consistent]
    times_m = [t for t, _ in time_series_modulo]
    rates_m = [r for _, r in time_series_modulo]

    ax.plot(times_c, rates_c, 'b-', linewidth=2, label='Consistent Hashing')
    ax.plot(times_m, rates_m, 'r--', linewidth=2, label='Modulo Hashing (IPHash)')

    # Mark fleet change
    ax.axvline(x=change_time, color='green', linestyle=':', linewidth=2, label='Fleet Change')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
