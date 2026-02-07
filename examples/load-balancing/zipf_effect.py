"""Zipf Distribution Effect on Load Balancing.

This example demonstrates how Zipf-distributed access patterns cause uneven
server load even with perfect consistent hashing. The problem isn't key
assignment uniformity - it's that some keys are accessed much more frequently.

## Concept

```
    ZIPF DISTRIBUTION: "The Rich Get Richer"
    ═══════════════════════════════════════════════════════════════

    Access Frequency (log scale)
    │
    │ ████
    │ ████
    │ ████ ███
    │ ████ ███
    │ ████ ███ ██
    │ ████ ███ ██ █
    │ ████ ███ ██ █ █ █ · · · · · · · · · · · · · · · · · · · ·
    └─────────────────────────────────────────────────────────────
      C1   C2  C3 C4 C5 C6 ...                              C1000

    With Zipf (s=1.5):
    - Customer 1 (rank 1): ~10% of all requests
    - Top 10 customers: ~40% of all requests
    - Top 100 customers: ~80% of all requests


    CONSISTENT HASHING + ZIPF = UNEVEN LOAD
    ═══════════════════════════════════════════════════════════════

    Keys are uniformly assigned to servers:
    Server 0: Customers 3, 8, 15, 22, ...   (200 customers)
    Server 1: Customers 1, 5, 12, 18, ...   (200 customers)  ← Has #1!
    Server 2: Customers 2, 7, 14, 21, ...   (200 customers)  ← Has #2!
    Server 3: Customers 4, 9, 16, 23, ...   (200 customers)
    Server 4: Customers 6, 11, 17, 24, ...  (200 customers)

    But request VOLUME is NOT uniform:
    Server 1 gets 10% of traffic (just from Customer 1!)
    Server 2 gets 5% of traffic (just from Customer 2!)
    Other servers share the remaining 85% more evenly

    This is the "hot shard" problem in distributed systems.
```

## Expected Results

- Uniform distribution: All servers get ~20% of requests (5 servers)
- Zipf distribution (s=1.5): Server with top customer gets 2-3x more requests
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)
from happysimulator.components.datastore.kv_store import KVStore
from happysimulator.components.load_balancer.load_balancer import LoadBalancer
from happysimulator.distributions.uniform import UniformDistribution
from happysimulator.distributions.zipf import ZipfDistribution

# Import from common (sibling module)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    CachingServer,
    CustomerRequestProvider,
    collect_aggregate_metrics,
    create_customer_consistent_hash,
    customer_id_key_extractor,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ZipfConfig:
    """Configuration for Zipf effect analysis."""

    arrival_rate: float = 500.0
    num_customers: int = 1000
    duration_s: float = 30.0
    num_servers: int = 5
    cache_capacity: int = 200  # Larger cache for this demo
    cache_ttl_s: float = 60.0
    zipf_s: float = 1.5  # Zipf exponent (higher = more skewed)
    seed: int = 42


# =============================================================================
# Analysis
# =============================================================================


@dataclass
class DistributionResult:
    """Results from a single distribution simulation."""

    distribution_type: str
    per_server_requests: dict[str, int]
    per_server_hit_rates: dict[str, float]
    aggregate_hit_rate: float
    total_requests: int
    load_imbalance: float  # max/min ratio
    top_server: str
    top_server_pct: float


def run_scenario(
    distribution_type: str,
    distribution,
    config: ZipfConfig,
) -> DistributionResult:
    """Run simulation with a specific customer distribution."""
    random.seed(config.seed)

    # Create shared datastore
    datastore = KVStore(
        name="SharedDatastore",
        read_latency=0.005,
        write_latency=0.010,
    )

    # Create servers
    servers: list[CachingServer] = []
    for i in range(config.num_servers):
        server = CachingServer(
            name=f"Server_{i}",
            server_id=i,
            datastore=datastore,
            cache_capacity=config.cache_capacity,
            cache_ttl_s=config.cache_ttl_s,
        )
        servers.append(server)

    # Create load balancer with consistent hashing
    lb = LoadBalancer(
        name=f"LB_{distribution_type}",
        backends=servers,
        strategy=create_customer_consistent_hash(virtual_nodes=100),
    )

    # Create event provider with the given distribution
    stop_after = Instant.from_seconds(config.duration_s)
    provider = CustomerRequestProvider(
        target=lb,
        customer_distribution=distribution,
        stop_after=stop_after,
    )

    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=config.arrival_rate),
        start_time=Instant.Epoch,
    )
    source = Source(
        name=f"Source_{distribution_type}",
        event_provider=provider,
        arrival_time_provider=arrival,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(config.duration_s + 1.0),
        sources=[source],
        entities=[lb, *servers],
    )
    sim.run()

    # Collect metrics
    metrics = collect_aggregate_metrics(servers)

    # Calculate load imbalance
    requests = list(metrics.per_server_request_counts.values())
    load_imbalance = max(requests) / min(requests) if min(requests) > 0 else float('inf')

    top_server = max(metrics.per_server_request_counts,
                     key=metrics.per_server_request_counts.get)
    top_server_pct = metrics.per_server_request_counts[top_server] / metrics.total_requests \
        if metrics.total_requests > 0 else 0

    return DistributionResult(
        distribution_type=distribution_type,
        per_server_requests=metrics.per_server_request_counts,
        per_server_hit_rates=metrics.per_server_hit_rates,
        aggregate_hit_rate=metrics.aggregate_hit_rate,
        total_requests=metrics.total_requests,
        load_imbalance=load_imbalance,
        top_server=top_server,
        top_server_pct=top_server_pct,
    )


@dataclass
class ComparisonResult:
    """Results comparing uniform vs Zipf distributions."""

    uniform: DistributionResult
    zipf: DistributionResult
    config: ZipfConfig
    customer_to_server: dict[int, str]  # Which server each customer routes to


def run_comparison(config: ZipfConfig) -> ComparisonResult:
    """Run both distributions and compare."""
    customers = list(range(config.num_customers))

    # First, determine customer-to-server mapping using our custom strategy
    strategy = create_customer_consistent_hash(virtual_nodes=100)

    class DummyBackend(Entity):
        def __init__(self, name: str):
            super().__init__(name)
        def handle_event(self, event: Event) -> list[Event]:
            return []

    backends = [DummyBackend(f"Server_{i}") for i in range(config.num_servers)]
    for b in backends:
        strategy.add_backend(b)

    customer_to_server: dict[int, str] = {}
    for customer_id in customers:
        mock_event = Event(
            time=Instant.Epoch,
            event_type="Mock",
            target=backends[0],
            context={"metadata": {"customer_id": customer_id}},
        )
        selected = strategy.select(backends, mock_event)
        if selected:
            customer_to_server[customer_id] = selected.name

    print("Running Uniform distribution scenario...")
    uniform_dist = UniformDistribution(customers)
    uniform_result = run_scenario("Uniform", uniform_dist, config)
    print(f"  Load imbalance: {uniform_result.load_imbalance:.2f}x")

    print(f"Running Zipf distribution (s={config.zipf_s}) scenario...")
    zipf_dist = ZipfDistribution(customers, s=config.zipf_s, seed=config.seed)
    zipf_result = run_scenario("Zipf", zipf_dist, config)
    print(f"  Load imbalance: {zipf_result.load_imbalance:.2f}x")

    return ComparisonResult(
        uniform=uniform_result,
        zipf=zipf_result,
        config=config,
        customer_to_server=customer_to_server,
    )


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: ComparisonResult, output_dir: Path) -> None:
    """Generate Zipf effect visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Request distribution comparison
    ax = axes[0, 0]
    servers = sorted(result.uniform.per_server_requests.keys())

    x = list(range(len(servers)))
    width = 0.35

    uniform_counts = [result.uniform.per_server_requests[s] for s in servers]
    zipf_counts = [result.zipf.per_server_requests[s] for s in servers]

    ax.bar([i - width/2 for i in x], uniform_counts, width, label='Uniform',
           color='steelblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], zipf_counts, width, label='Zipf',
           color='coral', alpha=0.8)

    ax.axhline(y=result.uniform.total_requests / len(servers), color='blue',
               linestyle='--', alpha=0.5, label='Expected (uniform)')

    ax.set_xlabel("Server")
    ax.set_ylabel("Request Count")
    ax.set_title("Request Distribution: Uniform vs Zipf")
    ax.set_xticks(x)
    ax.set_xticklabels(servers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Top-right: Hit rate comparison
    ax = axes[0, 1]
    uniform_rates = [result.uniform.per_server_hit_rates[s] for s in servers]
    zipf_rates = [result.zipf.per_server_hit_rates[s] for s in servers]

    ax.bar([i - width/2 for i in x], uniform_rates, width, label='Uniform',
           color='steelblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], zipf_rates, width, label='Zipf',
           color='coral', alpha=0.8)

    ax.set_xlabel("Server")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Per-Server Cache Hit Rate: Uniform vs Zipf")
    ax.set_xticks(x)
    ax.set_xticklabels(servers)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Bottom-left: Load imbalance visualization
    ax = axes[1, 0]

    # Calculate server load percentages
    total_uniform = sum(uniform_counts)
    total_zipf = sum(zipf_counts)
    uniform_pcts = [c / total_uniform * 100 for c in uniform_counts]
    zipf_pcts = [c / total_zipf * 100 for c in zipf_counts]

    ax.bar([i - width/2 for i in x], uniform_pcts, width, label='Uniform',
           color='steelblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], zipf_pcts, width, label='Zipf',
           color='coral', alpha=0.8)

    expected_pct = 100 / len(servers)
    ax.axhline(y=expected_pct, color='green', linestyle='--',
               label=f'Expected ({expected_pct:.0f}%)')

    ax.set_xlabel("Server")
    ax.set_ylabel("% of Total Requests")
    ax.set_title("Load Distribution (% of Total)")
    ax.set_xticks(x)
    ax.set_xticklabels(servers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Annotate the "hot" server
    hot_idx = servers.index(result.zipf.top_server)
    ax.annotate(
        f"HOT!\n{result.zipf.top_server_pct:.1%}",
        xy=(hot_idx + width/2, zipf_pcts[hot_idx]),
        xytext=(hot_idx + 1, zipf_pcts[hot_idx] + 5),
        arrowprops=dict(arrowstyle='->', color='red'),
        fontsize=10,
        color='red',
        fontweight='bold',
    )

    # Bottom-right: Summary metrics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    SUMMARY COMPARISON
    ══════════════════════════════════════

    Configuration:
      Customers: {result.config.num_customers}
      Servers: {result.config.num_servers}
      Zipf exponent: {result.config.zipf_s}

    UNIFORM DISTRIBUTION:
      Load imbalance: {result.uniform.load_imbalance:.2f}x
      Top server ({result.uniform.top_server}): {result.uniform.top_server_pct:.1%}
      Aggregate hit rate: {result.uniform.aggregate_hit_rate:.1%}

    ZIPF DISTRIBUTION:
      Load imbalance: {result.zipf.load_imbalance:.2f}x
      Top server ({result.zipf.top_server}): {result.zipf.top_server_pct:.1%}
      Aggregate hit rate: {result.zipf.aggregate_hit_rate:.1%}

    KEY INSIGHT:
      With Zipf, {result.zipf.top_server} receives
      {result.zipf.top_server_pct / (1/result.config.num_servers):.1f}x more load
      than expected, even with perfect consistent hashing!
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    fig.savefig(output_dir / "zipf_effect.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'zipf_effect.png'}")

    # Additional figure: Zipf distribution visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Zipf probability distribution (top 50 customers)
    ax = axes[0]
    zipf_dist = ZipfDistribution(list(range(result.config.num_customers)),
                                  s=result.config.zipf_s, seed=result.config.seed)

    top_n = 50
    probs = [zipf_dist.probability(rank) for rank in range(1, top_n + 1)]

    ax.bar(range(1, top_n + 1), probs, color='coral', alpha=0.8)
    ax.set_xlabel("Customer Rank")
    ax.set_ylabel("Access Probability")
    ax.set_title(f"Zipf Distribution (s={result.config.zipf_s}): Top {top_n} Customers")
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate key statistics
    top10_prob = sum(probs[:10])
    ax.annotate(
        f"Top 10: {top10_prob:.1%}\nof all traffic",
        xy=(5, probs[4]),
        xytext=(15, probs[4] * 0.8),
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
    )

    # Right: Cumulative distribution
    ax = axes[1]
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)

    ax.plot(range(1, top_n + 1), cumulative, 'b-', linewidth=2)
    ax.fill_between(range(1, top_n + 1), cumulative, alpha=0.3)

    ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.7)
    ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7)

    # Find 50% and 80% points
    for threshold, color, label in [(0.5, 'green', '50%'), (0.8, 'orange', '80%')]:
        for i, c in enumerate(cumulative):
            if c >= threshold:
                ax.axvline(x=i+1, color=color, linestyle=':', alpha=0.7)
                ax.annotate(
                    f"Top {i+1}:\n{label} of traffic",
                    xy=(i+1, threshold),
                    xytext=(i+10, threshold - 0.1),
                    arrowprops=dict(arrowstyle='->', color=color),
                    fontsize=9,
                    color=color,
                )
                break

    ax.set_xlabel("Top N Customers")
    ax.set_ylabel("Cumulative Traffic Share")
    ax.set_title("Cumulative Traffic Distribution")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(output_dir / "zipf_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'zipf_distribution.png'}")


def print_summary(result: ComparisonResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("ZIPF DISTRIBUTION EFFECT ON LOAD BALANCING")
    print("=" * 70)

    config = result.config
    print(f"\nConfiguration:")
    print(f"  Customers: {config.num_customers}")
    print(f"  Servers: {config.num_servers}")
    print(f"  Zipf exponent: {config.zipf_s}")

    print(f"\nUniform Distribution:")
    print(f"  Load imbalance (max/min): {result.uniform.load_imbalance:.2f}x")
    print(f"  Top server: {result.uniform.top_server} ({result.uniform.top_server_pct:.1%})")
    print(f"  Aggregate hit rate: {result.uniform.aggregate_hit_rate:.1%}")

    print(f"\nZipf Distribution:")
    print(f"  Load imbalance (max/min): {result.zipf.load_imbalance:.2f}x")
    print(f"  Top server: {result.zipf.top_server} ({result.zipf.top_server_pct:.1%})")
    print(f"  Aggregate hit rate: {result.zipf.aggregate_hit_rate:.1%}")

    print(f"\nPer-Server Request Counts:")
    servers = sorted(result.uniform.per_server_requests.keys())
    print(f"  {'Server':<10} | {'Uniform':>10} | {'Zipf':>10} | {'Zipf Diff':>10}")
    print(f"  {'-' * 10} | {'-' * 10} | {'-' * 10} | {'-' * 10}")
    for s in servers:
        uniform_count = result.uniform.per_server_requests[s]
        zipf_count = result.zipf.per_server_requests[s]
        delta_pct = (zipf_count - uniform_count) / uniform_count * 100 if uniform_count > 0 else 0
        marker = "HOT" if s == result.zipf.top_server else ""
        print(f"  {s:<10} | {uniform_count:>10} | {zipf_count:>10} | {delta_pct:>+9.1f}% {marker}")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print(f"\n1. CONSISTENT HASHING distributes KEYS uniformly")
    print(f"   (each server has ~{100/config.num_servers:.0f}% of customer IDs)")
    print(f"\n2. But with ZIPF, TRAFFIC is NOT uniform")
    print(f"   (a few customers generate most of the traffic)")
    print(f"\n3. The server assigned to the hottest customers becomes a \"HOT SHARD\"")
    print(f"   ({result.zipf.top_server} handles {result.zipf.top_server_pct:.1%} of traffic)")
    print(f"\n4. Solutions for hot shards:")
    print(f"   - Key replication (cache hot keys on multiple servers)")
    print(f"   - Request splitting (break hot keys into sub-keys)")
    print(f"   - Weighted routing (reduce traffic to hot servers)")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze Zipf distribution effect on load balancing"
    )
    parser.add_argument("--rate", type=float, default=500.0, help="Arrival rate (req/s)")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration (s)")
    parser.add_argument("--customers", type=int, default=1000, help="Number of unique customers")
    parser.add_argument("--servers", type=int, default=5, help="Number of servers")
    parser.add_argument("--zipf-s", type=float, default=1.5, help="Zipf exponent (higher=more skewed)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/load-balancing",
                        help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    config = ZipfConfig(
        arrival_rate=args.rate,
        duration_s=args.duration,
        num_customers=args.customers,
        num_servers=args.servers,
        zipf_s=args.zipf_s,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running Zipf effect analysis...")
    result = run_comparison(config)

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualization saved to: {output_dir.absolute()}")
