"""Cold start simulation demonstrating cache behavior during warmup and reset.

This example shows:
1. Cache warmup from empty state - hit rate gradually improves
2. Mid-simulation cache reset - triggers cold start recovery
3. Visualization of hit rate, datastore load spikes, and latency impact

## Architecture Diagram

The datastore is a SEPARATE entity from the server, representing a remote database
(like DynamoDB, Redis, or PostgreSQL). The server has a local in-memory cache and
communicates with the remote datastore over the network on cache misses.

```
Customer Traffic -> [ingress delay] -> CachedServer (local cache)
                                            |
                                            | [cache miss]
                                            v
                                    [db network delay]
                                            |
                                            v
                                    Remote Datastore (KVStore)
                                            |
                                            v
                                    [hydrate local cache]
                                            |
                                            v
                                       Response
```

## Timeline

```
t=0          t=warmup     t=reset      t=end
|--------------|------------|------------|
  Cache fills    Steady      Cold start
  (warmup)       state       recovery
```

During the warmup phase, the cache gradually fills with frequently-accessed
customer data. Once warm, most requests hit the cache. When we reset the cache,
it must warm up again, causing a spike in datastore load and increased latency.

The Zipf distribution ensures some customers are accessed much more frequently
than others (the "hot" keys), which makes caching effective once warm.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    DistributedFieldProvider,
    Entity,
    Event,
    FIFOQueue,
    Instant,
    PoissonArrivalTimeProvider,
    Probe,
    QueuedResource,
    Simulation,
    Source,
    UniformDistribution,
    ZipfDistribution,
)
from happysimulator.components.datastore import CachedStore, KVStore, LRUEviction


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ColdStartConfig:
    """Configuration for cold start simulation.

    Attributes:
        arrival_rate: Requests per second.
        num_customers: Number of unique customer IDs.
        distribution_type: "zipf" or "uniform" for customer ID selection.
        zipf_s: Zipf exponent (higher = more skewed, only used if distribution_type="zipf").
        cache_capacity: Maximum number of entries in cache.
        cache_read_latency_s: Latency for cache hits in seconds.
        ingress_latency_s: Network delay from customer to server.
        db_network_latency_s: Network round-trip delay from server to datastore.
        datastore_read_latency_s: Processing latency at the datastore.
        cold_start_time_s: When to reset the cache (None = no reset).
        duration_s: Total simulation duration.
        probe_interval_s: Metric sampling interval.
        seed: Random seed for reproducibility.
        use_poisson: Use Poisson arrivals (True) or constant rate (False).
    """

    arrival_rate: float = 1000.0
    num_customers: int = 200
    distribution_type: str = "zipf"
    zipf_s: float = 1.5
    cache_capacity: int = 150
    cache_read_latency_s: float = 0.0001  # 100 microseconds (local cache)
    ingress_latency_s: float = 0.010  # 10ms network delay (customer -> server)
    db_network_latency_s: float = 0.010  # 10ms network RTT (server <-> datastore)
    datastore_read_latency_s: float = 0.001  # 1ms processing at datastore
    cold_start_time_s: float | None = 90.0
    duration_s: float = 180.0
    probe_interval_s: float = 0.1
    seed: int | None = 42
    use_poisson: bool = True


# =============================================================================
# Entities
# =============================================================================


class CachedServer(QueuedResource):
    """Application server with a local cache backed by a remote datastore.

    The server has a local in-memory cache (CachedStore) that wraps an external
    datastore (KVStore). On cache hits, responses are fast. On cache misses,
    requests go to the remote datastore over the network.

    Properties exposed for probing:
    - hit_rate: Windowed cache hit rate since last reset (0.0 to 1.0)
    - miss_rate: Windowed cache miss rate since last reset (0.0 to 1.0)
    - cache_size: Current number of cached entries
    - datastore_reads: Total reads from backing store
    """

    def __init__(
        self,
        name: str,
        *,
        datastore: KVStore,
        cache_capacity: int,
        cache_read_latency_s: float,
        ingress_latency_s: float,
        downstream: Entity | None = None,
    ):
        """Initialize the cached server.

        Args:
            name: Server name.
            datastore: External KVStore representing the remote database.
            cache_capacity: Maximum entries in local cache.
            cache_read_latency_s: Latency for local cache hits.
            ingress_latency_s: Network delay from customer to this server.
            downstream: Entity to forward responses to.
        """
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self._ingress_latency_s = ingress_latency_s
        self._datastore = datastore

        # Create local cache layer wrapping the external datastore
        self._cache = CachedStore(
            name=f"{name}_cache",
            backing_store=datastore,
            cache_capacity=cache_capacity,
            eviction_policy=LRUEviction(),
            cache_read_latency=cache_read_latency_s,
            write_through=True,
        )

        # Track windowed hit rate (reset when cache is cleared)
        self._last_cache_hits = 0
        self._last_cache_misses = 0

        # Track in-flight requests (currently being processed)
        self._in_flight = 0

    @property
    def hit_rate(self) -> float:
        """Windowed cache hit rate since last reset (0.0 to 1.0)."""
        current_hits = self._cache.stats.hits - self._last_cache_hits
        current_misses = self._cache.stats.misses - self._last_cache_misses
        total = current_hits + current_misses
        if total == 0:
            return 0.0
        return current_hits / total

    @property
    def miss_rate(self) -> float:
        """Windowed cache miss rate since last reset (0.0 to 1.0)."""
        current_hits = self._cache.stats.hits - self._last_cache_hits
        current_misses = self._cache.stats.misses - self._last_cache_misses
        total = current_hits + current_misses
        if total == 0:
            return 0.0
        return current_misses / total

    @property
    def cache_size(self) -> int:
        """Current number of cached entries."""
        return self._cache.cache_size

    @property
    def datastore_reads(self) -> int:
        """Total reads from the remote datastore."""
        return self._datastore.stats.reads

    @property
    def in_flight(self) -> int:
        """Number of requests currently being processed."""
        return self._in_flight

    def reset_cache(self) -> None:
        """Clear the local cache, triggering cold start behavior."""
        self._cache.invalidate_all()
        # Reset windowed stats tracking
        self._last_cache_hits = self._cache.stats.hits
        self._last_cache_misses = self._cache.stats.misses

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a customer request.

        Yields:
            Delays in seconds for network and processing latencies.
        """
        self._in_flight += 1
        try:
            # Simulate network ingress delay (customer -> server)
            yield self._ingress_latency_s

            # Look up customer data in local cache (on miss, fetches from remote datastore)
            customer_id = event.context.get("customer_id", 0)
            key = f"customer:{customer_id}"
            # Cache.get() yields cache latency on hit, or datastore latency on miss
            _customer_data = yield from self._cache.get(key)

            # Simulate some processing time
            yield 0.001  # 1ms processing
        finally:
            self._in_flight -= 1

        if self.downstream is None:
            return []

        return [
            Event(
                time=self.now,
                event_type="Response",
                target=self.downstream,
                context=event.context,
            )
        ]


class LatencyTrackingSink(Entity):
    """Sink that records end-to-end latency from event context."""

    def __init__(self, name: str):
        super().__init__(name)
        self.events_received: int = 0
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.events_received += 1

        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)

        return []

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)


# =============================================================================
# Simulation Setup
# =============================================================================


@dataclass
class SimulationResult:
    """Results from cold start simulation."""

    sink: LatencyTrackingSink
    server: CachedServer
    datastore: KVStore
    hit_rate_data: Data
    miss_rate_data: Data
    cache_size_data: Data
    datastore_reads_data: Data
    queue_depth_data: Data
    in_flight_data: Data
    config: ColdStartConfig


def run_cold_start_simulation(config: ColdStartConfig) -> SimulationResult:
    """Run the cold start simulation.

    Args:
        config: Simulation configuration.

    Returns:
        SimulationResult with all collected metrics.
    """
    if config.seed is not None:
        random.seed(config.seed)

    # Create external datastore (separate entity representing remote DB)
    # Total latency = network RTT + DB processing time
    datastore = KVStore(
        name="Datastore",
        read_latency=config.db_network_latency_s + config.datastore_read_latency_s,
        write_latency=config.db_network_latency_s + config.datastore_read_latency_s,
    )

    # Pre-populate datastore with customer data
    for customer_id in range(config.num_customers):
        datastore.put_sync(
            f"customer:{customer_id}",
            {"id": customer_id, "balance": 100.0 + customer_id},
        )

    # Create sink and server
    sink = LatencyTrackingSink(name="Sink")
    server = CachedServer(
        name="Server",
        datastore=datastore,
        cache_capacity=config.cache_capacity,
        cache_read_latency_s=config.cache_read_latency_s,
        ingress_latency_s=config.ingress_latency_s,
        downstream=sink,
    )

    # Create customer ID distribution
    customer_ids = list(range(config.num_customers))
    if config.distribution_type == "zipf":
        customer_dist = ZipfDistribution(customer_ids, s=config.zipf_s, seed=config.seed)
    else:
        customer_dist = UniformDistribution(customer_ids, seed=config.seed)

    # Create event provider
    provider = DistributedFieldProvider(
        target=server,
        event_type="Request",
        field_distributions={"customer_id": customer_dist},
        stop_after=Instant.from_seconds(config.duration_s),
    )

    # Create arrival time provider
    profile = ConstantRateProfile(rate=config.arrival_rate)
    if config.use_poisson:
        arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
    else:
        arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    # Create probes for metrics
    hit_rate_data = Data()
    miss_rate_data = Data()
    cache_size_data = Data()
    datastore_reads_data = Data()
    queue_depth_data = Data()
    in_flight_data = Data()

    probes = [
        Probe(
            target=server,
            metric="hit_rate",
            data=hit_rate_data,
            interval=config.probe_interval_s,
            start_time=Instant.Epoch,
        ),
        Probe(
            target=server,
            metric="miss_rate",
            data=miss_rate_data,
            interval=config.probe_interval_s,
            start_time=Instant.Epoch,
        ),
        Probe(
            target=server,
            metric="cache_size",
            data=cache_size_data,
            interval=config.probe_interval_s,
            start_time=Instant.Epoch,
        ),
        Probe(
            target=server,
            metric="datastore_reads",
            data=datastore_reads_data,
            interval=config.probe_interval_s,
            start_time=Instant.Epoch,
        ),
        Probe(
            target=server,
            metric="depth",
            data=queue_depth_data,
            interval=config.probe_interval_s,
            start_time=Instant.Epoch,
        ),
        Probe(
            target=server,
            metric="in_flight",
            data=in_flight_data,
            interval=config.probe_interval_s,
            start_time=Instant.Epoch,
        ),
    ]

    # Create cache reset event if configured
    extra_events: list[Event] = []
    if config.cold_start_time_s is not None:

        def reset_callback(_e: Event) -> list[Event]:
            server.reset_cache()
            return []

        reset_event = Event(
            time=Instant.from_seconds(config.cold_start_time_s),
            event_type="CacheReset",
            callback=reset_callback,
        )
        extra_events.append(reset_event)

    # Run simulation - datastore is a separate entity
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(config.duration_s + 10.0),  # Extra drain time
        sources=[source],
        entities=[datastore, server, sink],  # Datastore is a separate entity
        probes=probes,
    )

    # Schedule the reset event
    for event in extra_events:
        sim.schedule(event)

    sim.run()

    return SimulationResult(
        sink=sink,
        server=server,
        datastore=datastore,
        hit_rate_data=hit_rate_data,
        miss_rate_data=miss_rate_data,
        cache_size_data=cache_size_data,
        datastore_reads_data=datastore_reads_data,
        queue_depth_data=queue_depth_data,
        in_flight_data=in_flight_data,
        config=config,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def percentile_sorted(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from sorted values (p in [0, 1])."""
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def bucket_latencies(
    times_s: list[float],
    latencies_s: list[float],
    bucket_size_s: float = 1.0,
) -> dict[str, list[float]]:
    """Bucket latencies by time and compute statistics."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for t_s, latency_s in zip(times_s, latencies_s, strict=False):
        bucket = int(math.floor(t_s / bucket_size_s))
        buckets[bucket].append(latency_s)

    result: dict[str, list[float]] = {
        "time_s": [],
        "avg": [],
        "p50": [],
        "p99": [],
        "p100": [],
        "count": [],
    }

    for bucket in sorted(buckets.keys()):
        vals_sorted = sorted(buckets[bucket])
        bucket_start = bucket * bucket_size_s

        result["time_s"].append(bucket_start)
        result["avg"].append(sum(vals_sorted) / len(vals_sorted))
        result["p50"].append(percentile_sorted(vals_sorted, 0.50))
        result["p99"].append(percentile_sorted(vals_sorted, 0.99))
        result["p100"].append(percentile_sorted(vals_sorted, 1.0))
        result["count"].append(float(len(vals_sorted)))

    return result


def compute_datastore_read_rate(
    datastore_reads_data: Data, window_s: float = 1.0
) -> tuple[list[float], list[float]]:
    """Compute rate of datastore reads per second."""
    values = datastore_reads_data.values
    if len(values) < 2:
        return [], []

    times: list[float] = []
    rates: list[float] = []

    for i in range(1, len(values)):
        t_prev, reads_prev = values[i - 1]
        t_curr, reads_curr = values[i]
        dt = t_curr - t_prev
        if dt > 0:
            rate = (reads_curr - reads_prev) / dt
            times.append(t_curr)
            rates.append(rate)

    return times, rates


# =============================================================================
# Visualization
# =============================================================================


def _plot_all_charts(
    axes: list,
    result: SimulationResult,
    reset_time: float | None,
    xlim: tuple[float, float] | None,
) -> None:
    """Plot all 5 charts on the given axes.

    Args:
        axes: List of 5 matplotlib axes.
        result: Simulation results.
        reset_time: Time of cache reset (or None).
        xlim: Optional (xmin, xmax) to zoom the view.
    """
    config = result.config

    # Extract data
    hr_times = [t for (t, _) in result.hit_rate_data.values]
    hr_values = [v for (_, v) in result.hit_rate_data.values]

    mr_times = [t for (t, _) in result.miss_rate_data.values]
    mr_values = [v for (_, v) in result.miss_rate_data.values]

    cs_times = [t for (t, _) in result.cache_size_data.values]
    cs_values = [v for (_, v) in result.cache_size_data.values]

    dr_rate_times, dr_rate_values = compute_datastore_read_rate(result.datastore_reads_data)

    if_times = [t for (t, _) in result.in_flight_data.values]
    if_values = [v for (_, v) in result.in_flight_data.values]

    times_s, latencies_s = result.sink.latency_time_series_seconds()
    latency_buckets = bucket_latencies(times_s, latencies_s, bucket_size_s=1.0)

    # Chart 1: Hit/Miss Rate
    ax = axes[0]
    ax.plot(hr_times, hr_values, "g-", linewidth=2, label="Hit Rate")
    ax.plot(mr_times, mr_values, "r-", linewidth=1.5, alpha=0.7, label="Miss Rate")
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate")
    ax.set_title("Cache Hit/Miss Rate")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    if xlim:
        ax.set_xlim(xlim)

    # Chart 2: Cache Size
    ax = axes[1]
    ax.plot(cs_times, cs_values, "b-", linewidth=2)
    ax.axhline(
        y=config.cache_capacity,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Cap ({config.cache_capacity})",
    )
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Entries")
    ax.set_title("Cache Size")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)

    # Chart 3: Datastore Read Rate
    ax = axes[2]
    ax.plot(dr_rate_times, dr_rate_values, "orange", linewidth=2)
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reads/s")
    ax.set_title("Datastore Read Rate")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)

    # Chart 4: Server Concurrency (In-Flight)
    ax = axes[3]
    ax.plot(if_times, if_values, "m-", linewidth=2)
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("In-Flight")
    ax.set_title("Server Concurrency")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)

    # Chart 5: Client Latency (in milliseconds)
    ax = axes[4]
    if latency_buckets["time_s"]:
        avg_ms = [v * 1000 for v in latency_buckets["avg"]]
        ax.plot(latency_buckets["time_s"], avg_ms, "b-", linewidth=2, label="Avg")
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Client Latency")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    if xlim:
        ax.set_xlim(xlim)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations of simulation results.

    Creates a 2x5 figure:
    - Row 1: Full timeline view
    - Row 2: Zoomed view around cold start time (+/- 5 seconds)
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    config = result.config
    reset_time = config.cold_start_time_s

    # Figure: 2x5 grid (full timeline + zoomed)
    fig, axes = plt.subplots(2, 5, figsize=(25, 8))

    # Row 1: Full timeline (trimmed: skip first/last 5 seconds)
    full_xlim = (5, config.duration_s - 5)
    _plot_all_charts(list(axes[0]), result, reset_time, xlim=full_xlim)

    # Row 2: Zoomed to cold start (+/- 5 seconds)
    if reset_time is not None:
        zoom_xlim = (reset_time - 5, reset_time + 5)
        _plot_all_charts(list(axes[1]), result, reset_time, xlim=zoom_xlim)
        # Add row labels
        axes[0, 0].set_ylabel("Full Timeline\n\nRate", fontsize=10)
        axes[1, 0].set_ylabel(f"Zoomed (t={reset_time-5:.0f}s-{reset_time+5:.0f}s)\n\nRate", fontsize=10)
    else:
        # No reset time - just duplicate full timeline
        _plot_all_charts(list(axes[1]), result, reset_time, xlim=full_xlim)

    fig.suptitle(
        f"Cold Start Simulation "
        f"({config.arrival_rate:.0f} req/s, {config.num_customers} customers, cache={config.cache_capacity})",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cold_start_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'cold_start_overview.png'}")


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("COLD START SIMULATION RESULTS")
    print("=" * 70)

    config = result.config
    times_s, latencies_s = result.sink.latency_time_series_seconds()

    print(f"\nConfiguration:")
    print(f"  Arrival rate: {config.arrival_rate} req/s")
    print(f"  Customers: {config.num_customers}")
    print(f"  Distribution: {config.distribution_type}" + (f" (s={config.zipf_s})" if config.distribution_type == "zipf" else ""))
    print(f"  Cache capacity: {config.cache_capacity}")
    print(f"  DB network latency: {config.db_network_latency_s * 1000:.1f}ms")
    print(f"  Datastore latency: {config.datastore_read_latency_s * 1000:.1f}ms")
    print(f"  Cache reset time: {config.cold_start_time_s}s" if config.cold_start_time_s else "  Cache reset: None")
    print(f"  Duration: {config.duration_s}s")

    print(f"\nResults:")
    print(f"  Requests completed: {result.sink.events_received}")
    print(f"  Final hit rate: {result.server.hit_rate:.1%}")
    print(f"  Final cache size: {result.server.cache_size}")
    print(f"  Total datastore reads: {result.datastore.stats.reads}")

    # Analyze phases
    reset_time = config.cold_start_time_s

    def avg_in_range(times: list[float], values: list[float], start: float, end: float) -> float:
        vals = [v for t, v in zip(times, values) if start <= t < end]
        return sum(vals) / len(vals) if vals else 0.0

    hr_times = [t for (t, _) in result.hit_rate_data.values]
    hr_values = [v for (_, v) in result.hit_rate_data.values]

    if reset_time is not None:
        print(f"\nHit Rate Analysis:")
        # Early warmup (first 10 seconds)
        early_hr = avg_in_range(hr_times, hr_values, 0, 10)
        # Late warmup / steady state (before reset)
        steady_hr = avg_in_range(hr_times, hr_values, reset_time - 30, reset_time)
        # Immediately after reset
        post_reset_hr = avg_in_range(hr_times, hr_values, reset_time, reset_time + 10)
        # Recovery (30s after reset)
        recovery_hr = avg_in_range(hr_times, hr_values, reset_time + 20, reset_time + 30)

        print(f"  Early warmup (0-10s):        {early_hr:.1%}")
        print(f"  Steady state (before reset): {steady_hr:.1%}")
        print(f"  Immediately after reset:     {post_reset_hr:.1%}")
        print(f"  Recovery (20-30s post):      {recovery_hr:.1%}")

    # Latency analysis
    if latencies_s:
        sorted_latencies = sorted(latencies_s)
        print(f"\nLatency (overall):")
        print(f"  Average: {sum(latencies_s) / len(latencies_s) * 1000:.2f}ms")
        print(f"  p50:     {percentile_sorted(sorted_latencies, 0.50) * 1000:.2f}ms")
        print(f"  p99:     {percentile_sorted(sorted_latencies, 0.99) * 1000:.2f}ms")
        print(f"  Max:     {max(latencies_s) * 1000:.2f}ms")

        if reset_time is not None:
            steady_lats = sorted([lat for t, lat in zip(times_s, latencies_s) if reset_time - 30 <= t < reset_time])
            post_lats = sorted([lat for t, lat in zip(times_s, latencies_s) if reset_time <= t < reset_time + 30])

            if steady_lats and post_lats:
                print(f"\nLatency comparison:")
                print(f"  Steady state avg: {sum(steady_lats) / len(steady_lats) * 1000:.2f}ms")
                print(f"  Post-reset avg:   {sum(post_lats) / len(post_lats) * 1000:.2f}ms")
                print(f"  Increase:         {(sum(post_lats) / len(post_lats)) / (sum(steady_lats) / len(steady_lats)):.1f}x")

    print("\n" + "=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cold start simulation demonstrating cache warmup and reset behavior"
    )
    parser.add_argument("--rate", type=float, default=1000.0, help="Arrival rate (req/s)")
    parser.add_argument("--customers", type=int, default=200, help="Number of unique customers")
    parser.add_argument("--cache-size", type=int, default=150, help="Cache capacity")
    parser.add_argument(
        "--distribution", choices=["zipf", "uniform"], default="zipf", help="Customer ID distribution"
    )
    parser.add_argument("--zipf-s", type=float, default=1.5, help="Zipf exponent (if using zipf)")
    parser.add_argument(
        "--db-network-latency", type=float, default=0.010, help="DB network RTT in seconds"
    )
    parser.add_argument(
        "--datastore-latency", type=float, default=0.001, help="Datastore processing latency in seconds"
    )
    parser.add_argument(
        "--reset-time", type=float, default=90.0, help="When to reset cache (use -1 for no reset)"
    )
    parser.add_argument("--duration", type=float, default=180.0, help="Simulation duration (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (use -1 for random)")
    parser.add_argument("--output", type=str, default="output/cold_start", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    reset_time = None if args.reset_time < 0 else args.reset_time

    config = ColdStartConfig(
        arrival_rate=args.rate,
        num_customers=args.customers,
        distribution_type=args.distribution,
        zipf_s=args.zipf_s,
        cache_capacity=args.cache_size,
        db_network_latency_s=args.db_network_latency,
        datastore_read_latency_s=args.datastore_latency,
        cold_start_time_s=reset_time,
        duration_s=args.duration,
        seed=seed,
    )

    print("Running cold start simulation...")
    print(f"  Arrival rate: {config.arrival_rate} req/s")
    print(f"  Customers: {config.num_customers}")
    print(f"  Cache capacity: {config.cache_capacity}")
    print(f"  Distribution: {config.distribution_type}")
    print(f"  DB network latency: {config.db_network_latency_s * 1000:.1f}ms")
    print(f"  Datastore latency: {config.datastore_read_latency_s * 1000:.1f}ms")
    print(f"  Reset time: {config.cold_start_time_s}s" if config.cold_start_time_s else "  Reset: disabled")
    print(f"  Duration: {config.duration_s}s")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    result = run_cold_start_simulation(config)

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
