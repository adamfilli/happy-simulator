"""Fleet Change Impact: Consistent Hashing vs Modulo Hashing.

This example demonstrates that modulo hashing causes catastrophic cache
invalidation when servers are added/removed, while consistent hashing
only shifts ~1/N keys.

## Architecture Diagram

```
    FLEET CHANGE TIMELINE
    ════════════════════════════════════════════════════════════════

    t=0                           t=30s                        t=60s
    │                             │                            │
    │◄─── 5 servers ────────────►│◄─── Add 6th server ──────►│
    │    (warmup & steady)        │    (observe impact)        │
    │                             │                            │
    ════════════════════════════════════════════════════════════════

    MODULO HASHING (IPHash):
    ┌─────────────────────────────────────────────────────────────┐
    │  hash(key) % 5  →  hash(key) % 6                           │
    │  Customer 10:  10 % 5 = 0  →  10 % 6 = 4  (CHANGED!)       │
    │  Customer 11:  11 % 5 = 1  →  11 % 6 = 5  (CHANGED!)       │
    │  Customer 12:  12 % 5 = 2  →  12 % 6 = 0  (CHANGED!)       │
    │                                                             │
    │  ~80% of keys change servers → cache invalidation storm!   │
    └─────────────────────────────────────────────────────────────┘

    CONSISTENT HASHING:
    ┌─────────────────────────────────────────────────────────────┐
    │  Virtual nodes on hash ring - add server takes ~1/N keys   │
    │                                                             │
    │  Before: ─────○─────○─────○─────○─────○─────               │
    │                s0    s1    s2    s3    s4                   │
    │                                                             │
    │  After:  ─────○───○─○─────○─────○─────○─────               │
    │                s0 s5 s1    s2    s3    s4                   │
    │                    ↑                                        │
    │                Only keys between s0 and s5 move!           │
    │                ~1/6 = 17% of keys shift (not 80%!)         │
    └─────────────────────────────────────────────────────────────┘
```

## Expected Results

- Modulo Hashing: Hit rate drops to ~0% after fleet change, slow recovery
- Consistent Hashing: Hit rate drops briefly by ~1/N, fast recovery
"""

from __future__ import annotations

import random

# Import from common (sibling module)
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from typing import TYPE_CHECKING

from common import (
    CachingServer,
    CustomerRequestProvider,
    collect_aggregate_metrics,
    create_customer_consistent_hash,
    create_customer_ip_hash,
)

if TYPE_CHECKING:
    from happysimulator.components.load_balancer.strategies import ConsistentHash, IPHash

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class FleetChangeConfig:
    """Configuration for fleet change comparison."""

    arrival_rate: float = 500.0
    num_customers: int = 1000
    duration_s: float = 60.0
    fleet_change_time_s: float = 30.0  # When to add the new server
    initial_servers: int = 5
    cache_capacity: int = 100
    cache_ttl_s: float = 60.0  # Longer TTL to see impact
    probe_interval_s: float = 0.5  # Finer granularity around change
    seed: int = 42


# =============================================================================
# Hit Rate Tracker
# =============================================================================


class PeriodicHitRateTracker(Entity):
    """Tracks aggregate hit rate over time with configurable interval."""

    def __init__(
        self,
        name: str,
        servers: list[CachingServer],
        interval_s: float = 0.5,
        end_time_s: float = 60.0,
    ):
        super().__init__(name)
        self._servers = servers
        self._interval_s = interval_s
        self._end_time_s = end_time_s
        self.history: list[tuple[float, float]] = []
        self._last_hits: int = 0
        self._last_total: int = 0

    def start(self) -> list[Event]:
        return [
            Event(
                time=Instant.from_seconds(self._interval_s),
                event_type="Track",
                target=self,
            )
        ]

    def handle_event(self, event: Event) -> list[Event]:
        current_time = self.now.to_seconds()

        # Calculate incremental hit rate (this interval)
        metrics = collect_aggregate_metrics(self._servers)
        total = metrics.total_hits + metrics.total_misses
        hits = metrics.total_hits

        interval_total = total - self._last_total
        interval_hits = hits - self._last_hits

        if interval_total > 0:
            interval_rate = interval_hits / interval_total
        else:
            interval_rate = 0.0

        self.history.append((current_time, interval_rate))

        self._last_total = total
        self._last_hits = hits

        # Schedule next
        if current_time + self._interval_s <= self._end_time_s:
            return [
                Event(
                    time=Instant.from_seconds(current_time + self._interval_s),
                    event_type="Track",
                    target=self,
                )
            ]
        return []


# =============================================================================
# Fleet Change Event
# =============================================================================


def create_fleet_change_callback(
    lb: LoadBalancer,
    new_server: CachingServer,
    tracker_for_key_shift: dict,
) -> Event:
    """Create an event that adds a server at the specified time."""

    def add_server(event: Event) -> list[Event]:
        # Record key distribution before change
        before_dist = _compute_key_routing(lb, tracker_for_key_shift["customers"])
        tracker_for_key_shift["before"] = before_dist

        # Add the new server
        lb.add_backend(new_server)

        # Record key distribution after change
        after_dist = _compute_key_routing(lb, tracker_for_key_shift["customers"])
        tracker_for_key_shift["after"] = after_dist

        # Calculate shift
        shifted = sum(
            1 for c in tracker_for_key_shift["customers"] if before_dist.get(c) != after_dist.get(c)
        )
        tracker_for_key_shift["shifted"] = shifted
        tracker_for_key_shift["shift_pct"] = shifted / len(tracker_for_key_shift["customers"])

        return []

    return add_server


def _compute_key_routing(lb: LoadBalancer, customers: list[int]) -> dict[int, str]:
    """Compute which server each customer routes to."""
    routing: dict[int, str] = {}

    for customer_id in customers:
        mock_event = Event(
            time=Instant.Epoch,
            event_type="Mock",
            target=lb,
            context={"metadata": {"customer_id": customer_id}},
        )
        # Use the strategy directly with healthy backends
        healthy_backends = lb.healthy_backends
        selected = lb._strategy.select(healthy_backends, mock_event)
        if selected:
            routing[customer_id] = selected.name

    return routing


# =============================================================================
# Simulation Runner
# =============================================================================


@dataclass
class FleetChangeResult:
    """Results from a fleet change scenario."""

    strategy_name: str
    hit_rate_history: list[tuple[float, float]]
    keys_shifted: int
    keys_shifted_pct: float
    final_hit_rate: float


def run_scenario(
    strategy_name: str,
    strategy: ConsistentHash | IPHash,
    config: FleetChangeConfig,
) -> FleetChangeResult:
    """Run fleet change simulation with a specific strategy."""
    random.seed(config.seed)

    # Create shared datastore
    datastore = KVStore(
        name="SharedDatastore",
        read_latency=0.005,
        write_latency=0.010,
    )

    # Create initial servers
    servers: list[CachingServer] = []
    for i in range(config.initial_servers):
        server = CachingServer(
            name=f"Server_{i}",
            server_id=i,
            datastore=datastore,
            cache_capacity=config.cache_capacity,
            cache_ttl_s=config.cache_ttl_s,
        )
        servers.append(server)

    # Create server to add later
    new_server = CachingServer(
        name=f"Server_{config.initial_servers}",
        server_id=config.initial_servers,
        datastore=datastore,
        cache_capacity=config.cache_capacity,
        cache_ttl_s=config.cache_ttl_s,
    )

    # Create load balancer
    lb = LoadBalancer(
        name=f"LB_{strategy_name}",
        backends=servers,
        strategy=strategy,
    )

    # Track key shift
    customers = list(range(config.num_customers))
    key_shift_tracker: dict = {"customers": customers}

    # Create fleet change callback
    fleet_change_callback = create_fleet_change_callback(lb, new_server, key_shift_tracker)

    # Create customer distribution
    customer_dist = UniformDistribution(customers)

    # Create source
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

    # Create hit rate tracker
    tracker = PeriodicHitRateTracker(
        name=f"Tracker_{strategy_name}",
        servers=[*servers, new_server],  # Include new server for post-change tracking
        interval_s=config.probe_interval_s,
        end_time_s=config.duration_s,
    )

    # Create fleet change event
    fleet_change_event = Event.once(
        time=Instant.from_seconds(config.fleet_change_time_s),
        event_type="FleetChange",
        fn=fleet_change_callback,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 1.0,
        sources=[source],
        entities=[lb, *servers, new_server, tracker],
    )

    # Schedule fleet change and tracker start
    sim.schedule(fleet_change_event)
    for event in tracker.start():
        sim.schedule(event)

    sim.run()

    # Calculate final metrics
    all_servers = [*servers, new_server]
    final_metrics = collect_aggregate_metrics(all_servers)

    return FleetChangeResult(
        strategy_name=strategy_name,
        hit_rate_history=tracker.history,
        keys_shifted=key_shift_tracker.get("shifted", 0),
        keys_shifted_pct=key_shift_tracker.get("shift_pct", 0.0),
        final_hit_rate=final_metrics.aggregate_hit_rate,
    )


@dataclass
class ComparisonResult:
    """Results comparing both strategies."""

    consistent_hash: FleetChangeResult
    modulo_hash: FleetChangeResult
    config: FleetChangeConfig


def run_comparison(config: FleetChangeConfig) -> ComparisonResult:
    """Run both strategies and compare."""
    print("Running Consistent Hash scenario...")
    ch_result = run_scenario(
        strategy_name="ConsistentHash",
        strategy=create_customer_consistent_hash(virtual_nodes=100),
        config=config,
    )
    print(f"  Keys shifted: {ch_result.keys_shifted_pct:.1%}")

    print("Running Modulo Hash (IPHash) scenario...")
    modulo_result = run_scenario(
        strategy_name="ModuloHash",
        strategy=create_customer_ip_hash(),
        config=config,
    )
    print(f"  Keys shifted: {modulo_result.keys_shifted_pct:.1%}")

    return ComparisonResult(
        consistent_hash=ch_result,
        modulo_hash=modulo_result,
        config=config,
    )


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: ComparisonResult, output_dir: Path) -> None:
    """Generate fleet change comparison visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    change_time = result.config.fleet_change_time_s

    # Top-left: Hit rate over time
    ax = axes[0, 0]
    ch_times = [t for t, _ in result.consistent_hash.hit_rate_history]
    ch_rates = [r for _, r in result.consistent_hash.hit_rate_history]
    mod_times = [t for t, _ in result.modulo_hash.hit_rate_history]
    mod_rates = [r for _, r in result.modulo_hash.hit_rate_history]

    ax.plot(ch_times, ch_rates, "b-", linewidth=2, label="Consistent Hash")
    ax.plot(mod_times, mod_rates, "r-", linewidth=2, label="Modulo Hash (IPHash)")
    ax.axvline(
        x=change_time,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Add Server (t={change_time}s)",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Interval Hit Rate")
    ax.set_title("Cache Hit Rate Over Time (Fleet Change)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Annotate the impact
    ax.annotate(
        "Modulo: ALL keys\nreshuffle!",
        xy=(change_time + 2, 0.1),
        fontsize=9,
        color="red",
        bbox={"boxstyle": "round", "facecolor": "mistyrose", "alpha": 0.8},
    )

    ax.annotate(
        f"Consistent: ~{1 / (result.config.initial_servers + 1):.0%}\nkeys shift",
        xy=(change_time + 2, 0.7),
        fontsize=9,
        color="blue",
        bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
    )

    # Top-right: Keys shifted comparison
    ax = axes[0, 1]
    strategies = ["Consistent Hash", "Modulo Hash"]
    shifted_pcts = [
        result.consistent_hash.keys_shifted_pct * 100,
        result.modulo_hash.keys_shifted_pct * 100,
    ]
    colors = ["steelblue", "coral"]
    bars = ax.bar(strategies, shifted_pcts, color=colors, alpha=0.8)

    for bar, pct in zip(bars, shifted_pcts, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Add theoretical line for consistent hash
    theoretical_ch = 100 / (result.config.initial_servers + 1)
    ax.axhline(
        y=theoretical_ch,
        color="blue",
        linestyle=":",
        label=f"Theoretical CH ({theoretical_ch:.1f}%)",
    )

    ax.set_ylabel("Keys Shifted (%)")
    ax.set_title("Key Redistribution on Fleet Change")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Zoomed view around fleet change
    ax = axes[1, 0]
    zoom_start = change_time - 5
    zoom_end = change_time + 5

    ch_zoom = [
        (t, r) for t, r in result.consistent_hash.hit_rate_history if zoom_start <= t <= zoom_end
    ]
    mod_zoom = [
        (t, r) for t, r in result.modulo_hash.hit_rate_history if zoom_start <= t <= zoom_end
    ]

    if ch_zoom:
        ax.plot(
            [t for t, _ in ch_zoom],
            [r for _, r in ch_zoom],
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Consistent Hash",
        )
    if mod_zoom:
        ax.plot(
            [t for t, _ in mod_zoom],
            [r for _, r in mod_zoom],
            "r-",
            linewidth=2,
            marker="s",
            markersize=4,
            label="Modulo Hash",
        )

    ax.axvline(x=change_time, color="green", linestyle="--", linewidth=2)
    ax.fill_between(
        [change_time, change_time + 2], 0, 1, alpha=0.1, color="yellow", label="Recovery period"
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Interval Hit Rate")
    ax.set_title(f"Zoomed View: Fleet Change Impact (t={zoom_start}s to {zoom_end}s)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Bottom-right: Recovery comparison
    ax = axes[1, 1]

    # Calculate recovery metrics
    def get_post_change_stats(history: list[tuple[float, float]]) -> dict:
        pre = [r for t, r in history if change_time - 10 <= t < change_time]
        post_immediate = [r for t, r in history if change_time <= t < change_time + 2]
        post_recovery = [r for t, r in history if change_time + 10 <= t < change_time + 20]

        return {
            "pre_avg": sum(pre) / len(pre) if pre else 0,
            "immediate_avg": sum(post_immediate) / len(post_immediate) if post_immediate else 0,
            "recovery_avg": sum(post_recovery) / len(post_recovery) if post_recovery else 0,
        }

    ch_stats = get_post_change_stats(result.consistent_hash.hit_rate_history)
    mod_stats = get_post_change_stats(result.modulo_hash.hit_rate_history)

    x = [0, 1, 2]
    labels = ["Pre-Change\n(t=20-30s)", "Immediate\n(t=30-32s)", "Recovered\n(t=40-50s)"]
    width = 0.35

    ch_vals = [ch_stats["pre_avg"], ch_stats["immediate_avg"], ch_stats["recovery_avg"]]
    mod_vals = [mod_stats["pre_avg"], mod_stats["immediate_avg"], mod_stats["recovery_avg"]]

    ax.bar(
        [i - width / 2 for i in x],
        ch_vals,
        width,
        label="Consistent Hash",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        [i + width / 2 for i in x], mod_vals, width, label="Modulo Hash", color="coral", alpha=0.8
    )

    ax.set_ylabel("Average Hit Rate")
    ax.set_title("Hit Rate: Before, During, and After Fleet Change")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "fleet_change_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'fleet_change_comparison.png'}")


def print_summary(result: ComparisonResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("FLEET CHANGE IMPACT: CONSISTENT HASH vs MODULO HASH")
    print("=" * 70)

    config = result.config
    print("\nConfiguration:")
    print(f"  Initial servers: {config.initial_servers}")
    print(f"  Fleet change: Add 1 server at t={config.fleet_change_time_s}s")
    print(f"  Total customers: {config.num_customers}")

    ch = result.consistent_hash
    mod = result.modulo_hash

    print("\nKey Redistribution on Fleet Change:")
    print(f"  Consistent Hash: {ch.keys_shifted} keys ({ch.keys_shifted_pct:.1%})")
    print(f"  Modulo Hash:     {mod.keys_shifted} keys ({mod.keys_shifted_pct:.1%})")

    theoretical = 1 / (config.initial_servers + 1)
    print(f"\n  Theoretical (Consistent Hash): ~{theoretical:.1%} keys should shift")
    print(
        f"  Theoretical (Modulo Hash): ~{1 - config.initial_servers / (config.initial_servers + 1):.1%} to ~80% keys shift"
    )

    print("\nFinal Hit Rates:")
    print(f"  Consistent Hash: {ch.final_hit_rate:.1%}")
    print(f"  Modulo Hash:     {mod.final_hit_rate:.1%}")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("-" * 70)
    print("\nModulo hashing (hash % N) causes catastrophic cache invalidation")
    print(f"when N changes. With {config.initial_servers} -> {config.initial_servers + 1} servers:")
    print(f"  - Modulo: {mod.keys_shifted_pct:.1%} of keys changed servers (cache miss storm)")
    print(f"  - Consistent: {ch.keys_shifted_pct:.1%} of keys changed (minimal impact)")
    print("\nThis is why consistent hashing is essential for distributed caches,")
    print("especially in auto-scaling environments where fleet size changes frequently.")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare fleet change impact: consistent hashing vs modulo hashing"
    )
    parser.add_argument("--rate", type=float, default=500.0, help="Arrival rate (req/s)")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    parser.add_argument(
        "--change-time", type=float, default=30.0, help="Time to add new server (s)"
    )
    parser.add_argument("--customers", type=int, default=1000, help="Number of unique customers")
    parser.add_argument("--servers", type=int, default=5, help="Initial number of servers")
    parser.add_argument("--cache-capacity", type=int, default=100, help="Cache capacity per server")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument(
        "--output", type=str, default="output/load-balancing", help="Output directory"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    config = FleetChangeConfig(
        arrival_rate=args.rate,
        duration_s=args.duration,
        fleet_change_time_s=args.change_time,
        num_customers=args.customers,
        initial_servers=args.servers,
        cache_capacity=args.cache_capacity,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running fleet change comparison...")
    result = run_comparison(config)

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualization saved to: {output_dir.absolute()}")
