"""Cold start simulation demonstrating cache behavior during warmup and reset.

This example shows:
1. Cache warmup from empty state - hit rate gradually improves
2. Mid-simulation cache reset - triggers cold start recovery
3. Visualization of hit rate, datastore load spikes, and latency impact

## Architecture Diagram

```
Customer Traffic -> [network delay] -> CachedServer -> [cache hit] -> Response
                                            |
                                            v [cache miss]
                                       Datastore (KVStore)
                                            |
                                            v [hydrate cache]
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
        datastore_read_latency_s: Latency for datastore reads (includes network).
        cold_start_time_s: When to reset the cache (None = no reset).
        duration_s: Total simulation duration.
        probe_interval_s: Metric sampling interval.
        seed: Random seed for reproducibility.
        use_poisson: Use Poisson arrivals (True) or constant rate (False).
    """

    arrival_rate: float = 50.0
    num_customers: int = 1000
    distribution_type: str = "zipf"
    zipf_s: float = 1.0
    cache_capacity: int = 100
    cache_read_latency_s: float = 0.0001  # 100 microseconds
    ingress_latency_s: float = 0.005  # 5ms network delay
    datastore_read_latency_s: float = 0.010  # 10ms (includes network)
    cold_start_time_s: float | None = 90.0
    duration_s: float = 180.0
    probe_interval_s: float = 1.0
    seed: int | None = 42
    use_poisson: bool = True


# =============================================================================
# Entities
# =============================================================================


class CachedServer(QueuedResource):
    """Server with an internal cache backed by a datastore.

    Composes a KVStore (datastore) with a CachedStore (cache layer).
    Processes requests by looking up customer data, using cache when possible.

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
        num_customers: int,
        cache_capacity: int,
        cache_read_latency_s: float,
        datastore_read_latency_s: float,
        ingress_latency_s: float,
        downstream: Entity | None = None,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self._ingress_latency_s = ingress_latency_s

        # Create backing datastore
        self._datastore = KVStore(
            name=f"{name}_datastore",
            read_latency=datastore_read_latency_s,
            write_latency=datastore_read_latency_s,
        )

        # Pre-populate datastore with customer data
        for customer_id in range(num_customers):
            self._datastore.put_sync(
                f"customer:{customer_id}",
                {"id": customer_id, "balance": 100.0 + customer_id},
            )

        # Create cache layer with LRU eviction
        self._cache = CachedStore(
            name=f"{name}_cache",
            backing_store=self._datastore,
            cache_capacity=cache_capacity,
            eviction_policy=LRUEviction(),
            cache_read_latency=cache_read_latency_s,
            write_through=True,
        )

        # Track windowed hit rate (reset when cache is cleared)
        self._window_hits = 0
        self._window_misses = 0
        self._last_cache_hits = 0
        self._last_cache_misses = 0

    @property
    def hit_rate(self) -> float:
        """Windowed cache hit rate since last reset (0.0 to 1.0)."""
        # Calculate hits/misses since window start
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
        """Total reads from backing datastore."""
        return self._datastore.stats.reads

    def reset_cache(self) -> None:
        """Clear the cache, triggering cold start behavior."""
        self._cache.invalidate_all()
        # Reset windowed stats tracking
        self._last_cache_hits = self._cache.stats.hits
        self._last_cache_misses = self._cache.stats.misses

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a customer request."""
        # Simulate network ingress delay
        yield self._ingress_latency_s

        # Look up customer data (cache or datastore)
        customer_id = event.context.get("customer_id", 0)
        key = f"customer:{customer_id}"
        _customer_data = yield from self._cache.get(key)

        # Simulate some processing time
        yield 0.001  # 1ms processing

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
    hit_rate_data: Data
    miss_rate_data: Data
    cache_size_data: Data
    datastore_reads_data: Data
    queue_depth_data: Data
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

    # Create sink and server
    sink = LatencyTrackingSink(name="Sink")
    server = CachedServer(
        name="Server",
        num_customers=config.num_customers,
        cache_capacity=config.cache_capacity,
        cache_read_latency_s=config.cache_read_latency_s,
        datastore_read_latency_s=config.datastore_read_latency_s,
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
    ]

    # Create cache reset event if configured
    extra_events: list[Event] = []
    if config.cold_start_time_s is not None:
        reset_event = Event(
            time=Instant.from_seconds(config.cold_start_time_s),
            event_type="CacheReset",
            callback=lambda _e: (server.reset_cache(), [])[1],
        )
        extra_events.append(reset_event)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(config.duration_s + 10.0),  # Extra drain time
        sources=[source],
        entities=[server, sink],
        probes=probes,
    )

    # Schedule the reset event
    for event in extra_events:
        sim.schedule(event)

    sim.run()

    return SimulationResult(
        sink=sink,
        server=server,
        hit_rate_data=hit_rate_data,
        miss_rate_data=miss_rate_data,
        cache_size_data=cache_size_data,
        datastore_reads_data=datastore_reads_data,
        queue_depth_data=queue_depth_data,
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


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations of simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    config = result.config
    reset_time = config.cold_start_time_s

    # Extract data
    hr_times = [t for (t, _) in result.hit_rate_data.values]
    hr_values = [v for (_, v) in result.hit_rate_data.values]

    mr_times = [t for (t, _) in result.miss_rate_data.values]
    mr_values = [v for (_, v) in result.miss_rate_data.values]

    cs_times = [t for (t, _) in result.cache_size_data.values]
    cs_values = [v for (_, v) in result.cache_size_data.values]

    dr_rate_times, dr_rate_values = compute_datastore_read_rate(result.datastore_reads_data)

    qd_times = [t for (t, _) in result.queue_depth_data.values]
    qd_values = [v for (_, v) in result.queue_depth_data.values]

    # Figure 1: Overview (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Hit/Miss rate
    ax = axes[0, 0]
    ax.plot(hr_times, hr_values, "g-", linewidth=2, label="Hit Rate")
    ax.plot(mr_times, mr_values, "r-", linewidth=1.5, alpha=0.7, label="Miss Rate")
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Cache Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rate")
    ax.set_title("Cache Hit/Miss Rate Over Time")
    ax.legend(loc="right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Top-right: Cache size
    ax = axes[0, 1]
    ax.plot(cs_times, cs_values, "b-", linewidth=2)
    ax.axhline(
        y=config.cache_capacity,
        color="r",
        linestyle="--",
        alpha=0.7,
        label=f"Capacity ({config.cache_capacity})",
    )
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Cache Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Entries")
    ax.set_title("Cache Size Over Time")
    ax.legend(loc="right")
    ax.grid(True, alpha=0.3)

    # Bottom-left: Datastore load rate
    ax = axes[1, 0]
    ax.plot(dr_rate_times, dr_rate_values, "orange", linewidth=2)
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Cache Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reads/second")
    ax.set_title("Datastore Read Rate (Load)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Queue depth
    ax = axes[1, 1]
    ax.plot(qd_times, qd_values, "m-", linewidth=2)
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Cache Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Depth")
    ax.set_title("Server Queue Depth")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Cold Start Simulation Overview\n"
        f"({config.arrival_rate} req/s, {config.num_customers} customers, "
        f"cache capacity {config.cache_capacity}, {config.distribution_type} distribution)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "cold_start_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'cold_start_overview.png'}")

    # Figure 2: Latency analysis (1x2)
    times_s, latencies_s = result.sink.latency_time_series_seconds()
    latency_buckets = bucket_latencies(times_s, latencies_s, bucket_size_s=1.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Latency over time
    ax = axes[0]
    if latency_buckets["time_s"]:
        ax.plot(latency_buckets["time_s"], latency_buckets["avg"], "b-", linewidth=2, label="Avg")
        ax.plot(latency_buckets["time_s"], latency_buckets["p99"], "r-", linewidth=1.5, label="p99")
        ax.fill_between(
            latency_buckets["time_s"],
            latency_buckets["p50"],
            latency_buckets["p99"],
            alpha=0.2,
            color="blue",
        )
    if reset_time is not None:
        ax.axvline(x=reset_time, color="purple", linestyle="--", alpha=0.7, label="Cache Reset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("End-to-End Latency Over Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Right: Latency histogram comparison
    ax = axes[1]
    if reset_time is not None and times_s:
        # Compare steady state vs post-reset
        warmup_end = min(30.0, reset_time - 10.0) if reset_time > 40.0 else reset_time / 2
        steady_latencies = [
            lat for t, lat in zip(times_s, latencies_s) if warmup_end <= t < reset_time
        ]
        post_reset_latencies = [
            lat for t, lat in zip(times_s, latencies_s) if reset_time <= t < reset_time + 30.0
        ]

        if steady_latencies:
            ax.hist(
                steady_latencies,
                bins=30,
                alpha=0.5,
                label=f"Steady state ({warmup_end:.0f}s-{reset_time:.0f}s)",
                color="green",
            )
        if post_reset_latencies:
            ax.hist(
                post_reset_latencies,
                bins=30,
                alpha=0.5,
                label=f"Post-reset ({reset_time:.0f}s-{reset_time + 30:.0f}s)",
                color="red",
            )
        ax.legend()
    else:
        # No reset - just show overall distribution
        if latencies_s:
            ax.hist(latencies_s, bins=50, alpha=0.7, color="blue")

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution Comparison")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Cold Start Latency Analysis", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / "cold_start_latency.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'cold_start_latency.png'}")


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
    print(f"  Cache reset time: {config.cold_start_time_s}s" if config.cold_start_time_s else "  Cache reset: None")
    print(f"  Duration: {config.duration_s}s")

    print(f"\nResults:")
    print(f"  Requests completed: {result.sink.events_received}")
    print(f"  Final hit rate: {result.server.hit_rate:.1%}")
    print(f"  Final cache size: {result.server.cache_size}")
    print(f"  Total datastore reads: {result.server.datastore_reads}")

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
    parser.add_argument("--rate", type=float, default=50.0, help="Arrival rate (req/s)")
    parser.add_argument("--customers", type=int, default=1000, help="Number of unique customers")
    parser.add_argument("--cache-size", type=int, default=100, help="Cache capacity")
    parser.add_argument(
        "--distribution", choices=["zipf", "uniform"], default="zipf", help="Customer ID distribution"
    )
    parser.add_argument("--zipf-s", type=float, default=1.0, help="Zipf exponent (if using zipf)")
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
        cold_start_time_s=reset_time,
        duration_s=args.duration,
        seed=seed,
    )

    print("Running cold start simulation...")
    print(f"  Arrival rate: {config.arrival_rate} req/s")
    print(f"  Customers: {config.num_customers}")
    print(f"  Cache capacity: {config.cache_capacity}")
    print(f"  Distribution: {config.distribution_type}")
    print(f"  Reset time: {config.cold_start_time_s}s" if config.cold_start_time_s else "  Reset: disabled")
    print(f"  Duration: {config.duration_s}s")
    print(f"  Seed: {seed if seed is not None else 'random'}")

    result = run_cold_start_simulation(config)

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
