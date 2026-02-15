"""Consistent Hashing vs Round Robin comparison.

This example demonstrates the cache hit rate benefits of consistent hashing
in a distributed caching scenario. With consistent hashing, the same customer
always routes to the same server, maximizing cache hits. With round robin,
requests spread randomly, causing frequent cache misses.

## Architecture Diagram

```
                    LOAD BALANCER COMPARISON
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   Source ──> LoadBalancer ──┬──> Server 0 (Cache 0)      ║
    ║   (customer_ids)            ├──> Server 1 (Cache 1)      ║
    ║                             ├──> Server 2 (Cache 2)      ║
    ║                             ├──> Server 3 (Cache 3)      ║
    ║                             └──> Server 4 (Cache 4)      ║
    ║                                        │                  ║
    ║                                        ▼                  ║
    ║                              Shared Datastore             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝

    Consistent Hashing: customer_id → same server → high cache hit rate
    Round Robin:        customer_id → random server → low cache hit rate
```

## Expected Results

- Consistent Hashing: ~80-90% cache hit rate (after warmup)
- Round Robin: ~10-20% cache hit rate (roughly 1/N for N servers)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    Entity,
    Event,
    Instant,
    Probe,
    Simulation,
    Source,
)
from happysimulator.components.datastore.kv_store import KVStore
from happysimulator.components.load_balancer.load_balancer import LoadBalancer
from happysimulator.components.load_balancer.strategies import ConsistentHash, RoundRobin
from happysimulator.distributions.uniform import UniformDistribution

# Import from common (sibling module)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import (
    CachingServer,
    CustomerRequestProvider,
    AggregateMetrics,
    collect_aggregate_metrics,
    create_customer_consistent_hash,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BasicConfig:
    """Configuration for basic consistent hashing comparison."""

    arrival_rate: float = 500.0  # Requests per second
    num_customers: int = 1000  # Distinct customer IDs
    duration_s: float = 60.0  # Simulation duration
    warmup_s: float = 5.0  # Warmup period before measuring
    num_servers: int = 5  # Number of backend servers
    cache_capacity: int = 100  # Cache entries per server
    cache_ttl_s: float = 30.0  # Cache TTL in seconds
    seed: int = 42  # Random seed for reproducibility


# =============================================================================
# Result Collection
# =============================================================================


@dataclass
class ScenarioResult:
    """Results from a single strategy simulation."""

    strategy_name: str
    servers: list[CachingServer]
    hit_rate_over_time: list[tuple[float, float]]  # (time_s, hit_rate)
    final_metrics: AggregateMetrics
    datastore_reads: int


@dataclass
class ComparisonResult:
    """Results comparing both strategies."""

    consistent_hash: ScenarioResult
    round_robin: ScenarioResult
    config: BasicConfig


# =============================================================================
# Hit Rate Tracking Probe
# =============================================================================


class HitRateProbe(Entity):
    """Probes aggregate hit rate across all servers at intervals."""

    def __init__(
        self,
        name: str,
        servers: list[CachingServer],
        interval_s: float = 1.0,
    ):
        super().__init__(name)
        self._servers = servers
        self._interval_s = interval_s
        self.hit_rate_history: list[tuple[float, float]] = []

    def start(self) -> list[Event]:
        """Schedule first probe."""
        return [
            Event(
                time=Instant.from_seconds(self._interval_s),
                event_type="Probe",
                target=self,
            )
        ]

    def handle_event(self, event: Event) -> list[Event]:
        """Record current hit rate and schedule next probe."""
        metrics = collect_aggregate_metrics(self._servers)
        current_time = self.now.to_seconds()
        self.hit_rate_history.append((current_time, metrics.aggregate_hit_rate))

        # Schedule next probe
        return [
            Event(
                time=Instant.from_seconds(current_time + self._interval_s),
                event_type="Probe",
                target=self,
            )
        ]


# =============================================================================
# Simulation Runner
# =============================================================================


def run_scenario(
    strategy_name: str,
    strategy: RoundRobin | ConsistentHash,
    config: BasicConfig,
) -> ScenarioResult:
    """Run simulation with a specific load balancing strategy.

    Args:
        strategy_name: Name for identifying this strategy.
        strategy: The load balancing strategy to use.
        config: Simulation configuration.

    Returns:
        ScenarioResult with metrics and time series data.
    """
    random.seed(config.seed)

    # Create shared datastore
    datastore = KVStore(
        name="SharedDatastore",
        read_latency=0.005,  # 5ms read from datastore
        write_latency=0.010,
    )

    # Create caching servers
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

    # Create load balancer with strategy
    lb = LoadBalancer(
        name=f"LB_{strategy_name}",
        backends=servers,
        strategy=strategy,
    )

    # Create customer distribution (uniform for this scenario)
    customer_dist = UniformDistribution(list(range(config.num_customers)))

    # Create event provider and source
    stop_after = Instant.from_seconds(config.duration_s)
    provider = CustomerRequestProvider(
        target=lb,
        customer_distribution=customer_dist,
        stop_after=stop_after,
    )

    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=config.arrival_rate),
        start_time=Instant.Epoch,
    )
    source = Source(
        name=f"Source_{strategy_name}",
        event_provider=provider,
        arrival_time_provider=arrival,
    )

    # Create hit rate probe
    probe = HitRateProbe(
        name=f"HitRateProbe_{strategy_name}",
        servers=servers,
        interval_s=1.0,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 1.0,
        sources=[source],
        entities=[lb, *servers, probe],
    )

    # Schedule probe start
    for event in probe.start():
        sim.schedule(event)

    sim.run()

    # Collect final metrics
    final_metrics = collect_aggregate_metrics(servers)

    return ScenarioResult(
        strategy_name=strategy_name,
        servers=servers,
        hit_rate_over_time=probe.hit_rate_history,
        final_metrics=final_metrics,
        datastore_reads=final_metrics.total_misses,
    )


def run_comparison(config: BasicConfig) -> ComparisonResult:
    """Run both strategies and return comparison results."""
    print(f"Running Consistent Hash scenario...")
    consistent_result = run_scenario(
        strategy_name="ConsistentHash",
        strategy=create_customer_consistent_hash(virtual_nodes=100),
        config=config,
    )
    print(f"  Hit rate: {consistent_result.final_metrics.aggregate_hit_rate:.1%}")

    print(f"Running Round Robin scenario...")
    round_robin_result = run_scenario(
        strategy_name="RoundRobin",
        strategy=RoundRobin(),
        config=config,
    )
    print(f"  Hit rate: {round_robin_result.final_metrics.aggregate_hit_rate:.1%}")

    return ComparisonResult(
        consistent_hash=consistent_result,
        round_robin=round_robin_result,
        config=config,
    )


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: ComparisonResult, output_dir: Path) -> None:
    """Generate comparison visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Hit rate over time comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Hit rate over time
    ax = axes[0, 0]
    ch_times = [t for t, _ in result.consistent_hash.hit_rate_over_time]
    ch_rates = [r for _, r in result.consistent_hash.hit_rate_over_time]
    rr_times = [t for t, _ in result.round_robin.hit_rate_over_time]
    rr_rates = [r for _, r in result.round_robin.hit_rate_over_time]

    ax.plot(ch_times, ch_rates, 'b-', linewidth=2, label='Consistent Hash')
    ax.plot(rr_times, rr_rates, 'r--', linewidth=2, label='Round Robin')
    ax.axhline(y=1/result.config.num_servers, color='gray', linestyle=':',
               label=f'Theoretical RR ({1/result.config.num_servers:.1%})')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Aggregate Cache Hit Rate")
    ax.set_title("Cache Hit Rate Over Time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Top-right: Final hit rate comparison (bar chart)
    ax = axes[0, 1]
    strategies = ['Consistent Hash', 'Round Robin']
    hit_rates = [
        result.consistent_hash.final_metrics.aggregate_hit_rate,
        result.round_robin.final_metrics.aggregate_hit_rate,
    ]
    colors = ['steelblue', 'coral']
    bars = ax.bar(strategies, hit_rates, color=colors, alpha=0.8)

    for bar, rate in zip(bars, hit_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold',
        )

    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Final Cache Hit Rate Comparison")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    # Bottom-left: Per-server hit rates (Consistent Hash)
    ax = axes[1, 0]
    ch_server_rates = result.consistent_hash.final_metrics.per_server_hit_rates
    servers = list(ch_server_rates.keys())
    rates = list(ch_server_rates.values())
    ax.bar(servers, rates, color='steelblue', alpha=0.8)
    ax.axhline(y=result.consistent_hash.final_metrics.aggregate_hit_rate,
               color='blue', linestyle='--', label='Aggregate')
    ax.set_xlabel("Server")
    ax.set_ylabel("Hit Rate")
    ax.set_title("Per-Server Hit Rate (Consistent Hash)")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    # Bottom-right: Per-server request counts
    ax = axes[1, 1]
    ch_counts = result.consistent_hash.final_metrics.per_server_request_counts
    rr_counts = result.round_robin.final_metrics.per_server_request_counts

    x = list(range(len(servers)))
    width = 0.35

    ax.bar([i - width/2 for i in x], [ch_counts[s] for s in servers],
           width, label='Consistent Hash', color='steelblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], [rr_counts[s] for s in servers],
           width, label='Round Robin', color='coral', alpha=0.8)

    ax.set_xlabel("Server")
    ax.set_ylabel("Request Count")
    ax.set_title("Per-Server Request Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(servers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / "consistent_hashing_basics.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'consistent_hashing_basics.png'}")


def print_summary(result: ComparisonResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("CONSISTENT HASHING vs ROUND ROBIN COMPARISON")
    print("=" * 70)

    config = result.config
    print(f"\nConfiguration:")
    print(f"  Arrival rate: {config.arrival_rate} req/s")
    print(f"  Duration: {config.duration_s}s")
    print(f"  Unique customers: {config.num_customers}")
    print(f"  Servers: {config.num_servers}")
    print(f"  Cache capacity per server: {config.cache_capacity}")
    print(f"  Cache TTL: {config.cache_ttl_s}s")

    ch = result.consistent_hash.final_metrics
    rr = result.round_robin.final_metrics

    print(f"\nConsistent Hash Results:")
    print(f"  Total requests: {ch.total_requests}")
    print(f"  Cache hits: {ch.total_hits} ({ch.aggregate_hit_rate:.1%})")
    print(f"  Cache misses: {ch.total_misses}")
    print(f"  Datastore reads: {result.consistent_hash.datastore_reads}")

    print(f"\nRound Robin Results:")
    print(f"  Total requests: {rr.total_requests}")
    print(f"  Cache hits: {rr.total_hits} ({rr.aggregate_hit_rate:.1%})")
    print(f"  Cache misses: {rr.total_misses}")
    print(f"  Datastore reads: {result.round_robin.datastore_reads}")

    print(f"\nComparison:")
    improvement = ch.aggregate_hit_rate / rr.aggregate_hit_rate if rr.aggregate_hit_rate > 0 else float('inf')
    print(f"  Hit rate improvement: {improvement:.1f}x")
    datastore_reduction = 1 - (result.consistent_hash.datastore_reads / result.round_robin.datastore_reads) \
        if result.round_robin.datastore_reads > 0 else 0
    print(f"  Datastore read reduction: {datastore_reduction:.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("-" * 70)
    print(f"\nConsistent hashing routes each customer to the same server,")
    print(f"resulting in a {ch.aggregate_hit_rate:.1%} cache hit rate.")
    print(f"\nRound robin spreads requests randomly across {config.num_servers} servers,")
    print(f"resulting in only a {rr.aggregate_hit_rate:.1%} hit rate (~1/{config.num_servers}).")
    print(f"\nThis {improvement:.1f}x improvement translates to {datastore_reduction:.1%} fewer")
    print(f"datastore reads, reducing backend load and latency.")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare consistent hashing vs round robin load balancing"
    )
    parser.add_argument("--rate", type=float, default=500.0, help="Arrival rate (req/s)")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    parser.add_argument("--customers", type=int, default=1000, help="Number of unique customers")
    parser.add_argument("--servers", type=int, default=5, help="Number of backend servers")
    parser.add_argument("--cache-capacity", type=int, default=100, help="Cache capacity per server")
    parser.add_argument("--cache-ttl", type=float, default=30.0, help="Cache TTL (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/load-balancing",
                        help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    config = BasicConfig(
        arrival_rate=args.rate,
        duration_s=args.duration,
        num_customers=args.customers,
        num_servers=args.servers,
        cache_capacity=args.cache_capacity,
        cache_ttl_s=args.cache_ttl,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running consistent hashing basics comparison...")
    result = run_comparison(config)

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualization saved to: {output_dir.absolute()}")
