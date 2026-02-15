"""Zipf-distributed traffic and cache hit rates by customer cohort.

Demonstrates the "heavy hitter" effect: a small fraction of customers generate
the majority of requests. When a cache with limited capacity sits in front of a
datastore, these heavy hitters stay in cache (high hit rate) while the long tail
gets evicted before their next access (low hit rate).

This example measures per-cohort cache statistics to quantify the effect.

## Architecture

```
Source (100 req/s, constant rate)
  -> DistributedFieldProvider (samples customer_id from ZipfDistribution)
    -> CacheClient (Entity -- tracks per-customer hit/miss)
      -> SoftTTLCache (capacity=200, LRU eviction)
        -> KVStore (pre-populated with 1000 customer records)
```

## Expected Results (default config, s=1.0, 1000 customers, cache=200)

| Cohort          | Traffic Share | Hit Rate |
|-----------------|---------------|----------|
| Top 1% (10)     | ~27%          | ~95%+    |
| Top 10% (100)   | ~52%          | ~85%+    |
| Bottom 50% (500)| ~9%           | ~5-15%   |
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
    ZipfDistribution,
    DistributedFieldProvider,
)
from happysimulator.components.datastore import KVStore, SoftTTLCache


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class CohortConfig:
    """Parameters for the Zipf cache cohort simulation."""

    num_customers: int = 1000
    zipf_s: float = 1.0
    arrival_rate: float = 100.0
    duration_s: float = 60.0
    cache_capacity: int = 200
    soft_ttl: float = 300.0
    hard_ttl: float = 3600.0
    db_read_latency: float = 0.010
    cache_read_latency: float = 0.0001
    seed: int = 42


# =============================================================================
# CacheClient Entity
# =============================================================================


class CacheClient(Entity):
    """Receives request events and reads from cache, tracking per-customer hits/misses."""

    def __init__(self, name: str, cache: SoftTTLCache):
        super().__init__(name)
        self._cache = cache
        self.hits: dict[int, int] = defaultdict(int)
        self.misses: dict[int, int] = defaultdict(int)
        self.total_requests: int = 0

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        customer_id = event.context["customer_id"]
        self.total_requests += 1

        key = f"customer:{customer_id}"

        # Check cache state before the get (synchronous, reliable)
        is_cached = self._cache.contains_cached(key)

        # Perform the actual cache read
        yield from self._cache.get(key)

        if is_cached:
            self.hits[customer_id] += 1
        else:
            self.misses[customer_id] += 1

        return None


# =============================================================================
# Cohort Analysis
# =============================================================================


@dataclass
class CohortResult:
    """Statistics for a single customer cohort."""

    name: str
    start_rank: int
    end_rank: int
    num_customers: int
    expected_traffic_share: float
    actual_requests: int
    actual_traffic_share: float
    hits: int
    misses: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


def analyze_cohorts(
    client: CacheClient,
    zipf: ZipfDistribution,
    num_customers: int,
) -> list[CohortResult]:
    """Group customers by Zipf rank into cohorts and compute per-cohort stats."""
    cohort_defs = [
        ("Top 1%", 1, max(1, int(num_customers * 0.01))),
        ("Top 5%", 1, max(1, int(num_customers * 0.05))),
        ("Top 10%", 1, max(1, int(num_customers * 0.10))),
        ("Top 20%", 1, max(1, int(num_customers * 0.20))),
        ("Middle 30%", int(num_customers * 0.20) + 1, int(num_customers * 0.50)),
        ("Bottom 50%", int(num_customers * 0.50) + 1, num_customers),
    ]

    total_requests = client.total_requests
    results = []

    for name, start_rank, end_rank in cohort_defs:
        # Customer IDs are 0-indexed but ranks are 1-indexed
        customer_ids = list(range(start_rank - 1, end_rank))
        n = len(customer_ids)

        expected_share = zipf.top_n_probability(end_rank)
        if start_rank > 1:
            expected_share -= zipf.top_n_probability(start_rank - 1)

        hits = sum(client.hits[cid] for cid in customer_ids)
        misses = sum(client.misses[cid] for cid in customer_ids)
        actual_reqs = hits + misses
        actual_share = actual_reqs / total_requests if total_requests > 0 else 0.0

        results.append(CohortResult(
            name=name,
            start_rank=start_rank,
            end_rank=end_rank,
            num_customers=n,
            expected_traffic_share=expected_share,
            actual_requests=actual_reqs,
            actual_traffic_share=actual_share,
            hits=hits,
            misses=misses,
        ))

    return results


# =============================================================================
# Simulation Setup
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the Zipf cache cohort simulation."""

    config: CohortConfig
    client: CacheClient
    cache: SoftTTLCache
    zipf: ZipfDistribution
    cohorts: list[CohortResult]


def run_simulation(config: CohortConfig = CohortConfig()) -> SimulationResult:
    """Wire up components and run the simulation."""
    # Backing store pre-populated with customer records
    db = KVStore(name="db", read_latency=config.db_read_latency)
    for i in range(config.num_customers):
        db.put_sync(f"customer:{i}", {"id": i, "name": f"Customer-{i}"})

    # Cache in front of the store
    cache = SoftTTLCache(
        name="cache",
        backing_store=db,
        soft_ttl=config.soft_ttl,
        hard_ttl=config.hard_ttl,
        cache_capacity=config.cache_capacity,
        cache_read_latency=config.cache_read_latency,
    )

    # Client that tracks hits/misses
    client = CacheClient(name="CacheClient", cache=cache)

    # Zipf distribution for customer selection
    zipf = ZipfDistribution(
        values=list(range(config.num_customers)),
        s=config.zipf_s,
        seed=config.seed,
    )

    # Load generation
    provider = DistributedFieldProvider(
        target=client,
        event_type="CacheRead",
        field_distributions={"customer_id": zipf},
        stop_after=Instant.from_seconds(config.duration_s),
    )

    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=config.arrival_rate),
        start_time=Instant.Epoch,
    )

    source = Source(
        name="Traffic",
        event_provider=provider,
        arrival_time_provider=arrival,
    )

    # Run
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 1.0,
        sources=[source],
        entities=[client, cache, db],
    )
    sim.run()

    cohorts = analyze_cohorts(client, zipf, config.num_customers)

    return SimulationResult(
        config=config,
        client=client,
        cache=cache,
        zipf=zipf,
        cohorts=cohorts,
    )


# =============================================================================
# Output
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print configuration and cohort results table."""
    cfg = result.config
    cache = result.cache

    print("\n" + "=" * 72)
    print("ZIPF CACHE COHORT ANALYSIS")
    print("=" * 72)

    print(f"\nConfiguration:")
    print(f"  Customers:       {cfg.num_customers}")
    print(f"  Zipf exponent:   s={cfg.zipf_s}")
    print(f"  Arrival rate:    {cfg.arrival_rate} req/s")
    print(f"  Duration:        {cfg.duration_s}s")
    print(f"  Cache capacity:  {cfg.cache_capacity}")
    print(f"  Seed:            {cfg.seed}")

    print(f"\nCache Stats:")
    print(f"  Total reads:     {cache.stats.reads}")
    print(f"  Fresh hits:      {cache.stats.fresh_hits}")
    print(f"  Stale hits:      {cache.stats.stale_hits}")
    print(f"  Hard misses:     {cache.stats.hard_misses}")
    print(f"  Evictions:       {cache.stats.evictions}")
    print(f"  Hit rate:        {cache.stats.total_hit_rate:.1%}")

    print(f"\n{'Cohort':<16} {'Customers':>10} {'Requests':>10} {'Traffic':>10} "
          f"{'Expected':>10} {'Hits':>8} {'Misses':>8} {'Hit Rate':>10}")
    print("-" * 92)

    for c in result.cohorts:
        print(f"{c.name:<16} {c.num_customers:>10} {c.actual_requests:>10} "
              f"{c.actual_traffic_share:>9.1%} {c.expected_traffic_share:>9.1%} "
              f"{c.hits:>8} {c.misses:>8} {c.hit_rate:>9.1%}")

    print("=" * 72)


def visualize_results(
    result: SimulationResult,
    output_dir: Path,
) -> None:
    """Generate a 3-subplot figure with cohort analysis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    cohorts = result.cohorts
    names = [c.name for c in cohorts]

    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # --- Subplot 1: Hit rate by cohort ---
    ax1 = axes[0]
    hit_rates = [c.hit_rate * 100 for c in cohorts]
    bars = ax1.bar(names, hit_rates, color="steelblue", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Hit Rate (%)")
    ax1.set_title("Cache Hit Rate by Customer Cohort")
    ax1.set_ylim(0, 105)
    ax1.grid(axis="y", alpha=0.3)
    for bar, rate in zip(bars, hit_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                 f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    # --- Subplot 2: Expected vs actual traffic share ---
    ax2 = axes[1]
    x = range(len(names))
    width = 0.35
    expected = [c.expected_traffic_share * 100 for c in cohorts]
    actual = [c.actual_traffic_share * 100 for c in cohorts]
    ax2.bar([i - width / 2 for i in x], expected, width, label="Expected (Zipf)", color="lightcoral", edgecolor="black", linewidth=0.5)
    ax2.bar([i + width / 2 for i in x], actual, width, label="Actual", color="steelblue", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Traffic Share (%)")
    ax2.set_title("Expected vs Actual Traffic Share by Cohort")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(names)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # --- Subplot 3: Per-customer hit rate for top 100 by rank ---
    ax3 = axes[2]
    top_n = min(100, result.config.num_customers)
    ranks = list(range(1, top_n + 1))
    per_customer_hit_rates = []
    for cid in range(top_n):
        h = result.client.hits[cid]
        m = result.client.misses[cid]
        total = h + m
        per_customer_hit_rates.append(h / total * 100 if total > 0 else 0.0)

    ax3.bar(ranks, per_customer_hit_rates, color="steelblue", edgecolor="none", width=1.0)
    ax3.set_xlabel("Customer Rank")
    ax3.set_ylabel("Hit Rate (%)")
    ax3.set_title(f"Per-Customer Hit Rate (Top {top_n} by Zipf Rank)")
    ax3.set_ylim(0, 105)
    ax3.grid(axis="y", alpha=0.3)

    # Mark the cache capacity boundary
    cap = result.config.cache_capacity
    if cap < top_n:
        ax3.axvline(x=cap, color="red", linestyle="--", linewidth=1.5, label=f"Cache capacity ({cap})")
        ax3.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "zipf_cache_cohorts.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved: {output_dir / 'zipf_cache_cohorts.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Zipf-distributed traffic and cache hit rates by customer cohort",
    )
    parser.add_argument("--customers", type=int, default=1000, help="Number of customers (default: 1000)")
    parser.add_argument("--zipf-s", type=float, default=1.0, help="Zipf exponent (default: 1.0)")
    parser.add_argument("--rate", type=float, default=100.0, help="Arrival rate in req/s (default: 100)")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds (default: 60)")
    parser.add_argument("--cache-size", type=int, default=200, help="Cache capacity (default: 200)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42, use -1 for random)")
    parser.add_argument("--output", type=str, default="output/zipf_cache_cohorts", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = args.seed if args.seed != -1 else None

    config = CohortConfig(
        num_customers=args.customers,
        zipf_s=args.zipf_s,
        arrival_rate=args.rate,
        duration_s=args.duration,
        cache_capacity=args.cache_size,
        seed=seed if seed is not None else 42,
    )

    print("Running Zipf cache cohort simulation...")
    print(f"  Customers: {config.num_customers}, Zipf s={config.zipf_s}")
    print(f"  Rate: {config.arrival_rate} req/s, Duration: {config.duration_s}s")
    print(f"  Cache: {config.cache_capacity} entries")

    result = run_simulation(config)
    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
