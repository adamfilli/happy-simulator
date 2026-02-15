r"""Virtual Nodes Analysis for Consistent Hashing.

This example analyzes how the number of virtual nodes affects key distribution
uniformity in consistent hashing. More virtual nodes = more uniform distribution,
but at the cost of memory and lookup time.

```

## Expected Results

| V-nodes | CoV (lower=better) | Max/Min Ratio |
|---------|-------------------|---------------|
| 1       | ~0.6-0.9          | ~3-5x         |
| 5       | ~0.3-0.5          | ~2-3x         |
| 10      | ~0.2-0.3          | ~1.5-2x       |
| 50      | ~0.1-0.15         | ~1.2-1.4x     |
| 100     | ~0.05-0.10        | ~1.1-1.2x     |
| 200     | ~0.03-0.07        | ~1.05-1.15x   |
| 500     | ~0.02-0.04        | ~1.03-1.08x   |
"""

from __future__ import annotations

# Import from common (sibling module)
import sys
from dataclasses import dataclass
from pathlib import Path

from happysimulator import Entity, Event, Instant
from happysimulator.components.load_balancer.strategies import ConsistentHash

sys.path.insert(0, str(Path(__file__).parent))
from common import customer_id_key_extractor

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class VNodesConfig:
    """Configuration for virtual nodes analysis."""

    num_servers: int = 5
    num_keys: int = 10000
    vnode_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100, 200, 500)
    num_trials: int = 5  # Run multiple trials for statistical stability


# =============================================================================
# Analysis
# =============================================================================


class MockBackend(Entity):
    """Minimal entity for backend simulation."""

    def __init__(self, name: str):
        super().__init__(name)

    def handle_event(self, event: Event) -> list[Event]:
        return []


@dataclass
class DistributionStats:
    """Statistics for a key distribution."""

    vnode_count: int
    counts: dict[str, int]  # server name -> key count
    mean: float
    std: float
    cov: float  # coefficient of variation
    max_min_ratio: float
    max_server: str
    min_server: str


def analyze_distribution(
    vnode_count: int,
    num_servers: int,
    keys: list[int],
) -> DistributionStats:
    """Analyze key distribution for a given vnode count.

    Args:
        vnode_count: Number of virtual nodes per server.
        num_servers: Number of backend servers.
        keys: List of keys to distribute.

    Returns:
        DistributionStats with uniformity metrics.
    """
    # Create backends
    backends = [MockBackend(f"Server_{i}") for i in range(num_servers)]

    # Create consistent hash strategy with customer_id key extraction
    strategy = ConsistentHash(
        virtual_nodes=vnode_count,
        get_key=customer_id_key_extractor,
    )

    # Force initialization of the ring
    for backend in backends:
        strategy.add_backend(backend)

    # Count keys per server
    counts: dict[str, int] = {b.name: 0 for b in backends}

    for key in keys:
        mock_event = Event(
            time=Instant.Epoch,
            event_type="Mock",
            target=backends[0],
            context={"metadata": {"customer_id": key}},
        )
        selected = strategy.select(backends, mock_event)
        if selected:
            counts[selected.name] += 1

    # Calculate statistics
    values = list(counts.values())
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance**0.5
    cov = std / mean if mean > 0 else 0

    max_count = max(values)
    min_count = min(values)
    max_min_ratio = max_count / min_count if min_count > 0 else float("inf")

    max_server = max(counts, key=counts.get)
    min_server = min(counts, key=counts.get)

    return DistributionStats(
        vnode_count=vnode_count,
        counts=counts,
        mean=mean,
        std=std,
        cov=cov,
        max_min_ratio=max_min_ratio,
        max_server=max_server,
        min_server=min_server,
    )


def run_analysis(config: VNodesConfig) -> list[DistributionStats]:
    """Run distribution analysis for all vnode counts.

    Returns averaged statistics across multiple trials.
    """
    results: list[DistributionStats] = []

    for vnode_count in config.vnode_counts:
        print(f"  Analyzing {vnode_count} virtual nodes...")

        # Run multiple trials and average
        trial_covs: list[float] = []
        trial_ratios: list[float] = []

        for trial in range(config.num_trials):
            # Use different key ranges for each trial
            keys = list(range(trial * config.num_keys, (trial + 1) * config.num_keys))
            stats = analyze_distribution(vnode_count, config.num_servers, keys)
            trial_covs.append(stats.cov)
            trial_ratios.append(stats.max_min_ratio)

        # Average across trials
        avg_cov = sum(trial_covs) / len(trial_covs)
        avg_ratio = sum(trial_ratios) / len(trial_ratios)

        # Run one more time to get representative counts
        final_keys = list(range(config.num_keys))
        final_stats = analyze_distribution(vnode_count, config.num_servers, final_keys)

        results.append(
            DistributionStats(
                vnode_count=vnode_count,
                counts=final_stats.counts,
                mean=final_stats.mean,
                std=final_stats.std,
                cov=avg_cov,
                max_min_ratio=avg_ratio,
                max_server=final_stats.max_server,
                min_server=final_stats.min_server,
            )
        )

    return results


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(
    results: list[DistributionStats], config: VNodesConfig, output_dir: Path
) -> None:
    """Generate virtual nodes analysis visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Coefficient of Variation vs V-nodes
    ax = axes[0, 0]
    vnodes = [r.vnode_count for r in results]
    covs = [r.cov for r in results]

    ax.plot(vnodes, covs, "b-o", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Virtual Nodes per Server")
    ax.set_ylabel("Coefficient of Variation (lower = better)")
    ax.set_title("Distribution Uniformity vs Virtual Nodes")
    ax.grid(True, alpha=0.3)

    # Add threshold line for "good enough"
    ax.axhline(y=0.1, color="green", linestyle="--", alpha=0.7, label="Good (CoV < 0.1)")
    ax.legend(loc="upper right")

    # Top-right: Max/Min ratio vs V-nodes
    ax = axes[0, 1]
    ratios = [r.max_min_ratio for r in results]

    ax.plot(vnodes, ratios, "r-o", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Virtual Nodes per Server")
    ax.set_ylabel("Max/Min Key Count Ratio")
    ax.set_title("Load Imbalance vs Virtual Nodes")
    ax.grid(True, alpha=0.3)

    # Add threshold line
    ax.axhline(y=1.2, color="green", linestyle="--", alpha=0.7, label="Good (ratio < 1.2)")
    ax.legend(loc="upper right")

    # Bottom-left: Key distribution for low vnode count
    ax = axes[1, 0]
    low_vnode = results[0]  # First (lowest) vnode count
    servers = list(low_vnode.counts.keys())
    counts = list(low_vnode.counts.values())

    bars = ax.bar(servers, counts, color="coral", alpha=0.8)
    ax.axhline(y=low_vnode.mean, color="blue", linestyle="--", label=f"Mean ({low_vnode.mean:.0f})")

    ax.set_xlabel("Server")
    ax.set_ylabel("Number of Keys")
    ax.set_title(
        f"Key Distribution with {low_vnode.vnode_count} Virtual Node(s)\n"
        f"CoV={low_vnode.cov:.3f}, Max/Min={low_vnode.max_min_ratio:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

    # Highlight imbalance
    for bar, count in zip(bars, counts, strict=False):
        deviation = (count - low_vnode.mean) / low_vnode.mean * 100
        color = "red" if abs(deviation) > 20 else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + low_vnode.mean * 0.02,
            f"{deviation:+.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color=color,
        )

    # Bottom-right: Key distribution for high vnode count
    ax = axes[1, 1]
    high_vnode = results[-1]  # Last (highest) vnode count
    servers = list(high_vnode.counts.keys())
    counts = list(high_vnode.counts.values())

    bars = ax.bar(servers, counts, color="steelblue", alpha=0.8)
    ax.axhline(
        y=high_vnode.mean, color="blue", linestyle="--", label=f"Mean ({high_vnode.mean:.0f})"
    )

    ax.set_xlabel("Server")
    ax.set_ylabel("Number of Keys")
    ax.set_title(
        f"Key Distribution with {high_vnode.vnode_count} Virtual Nodes\n"
        f"CoV={high_vnode.cov:.3f}, Max/Min={high_vnode.max_min_ratio:.2f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="x", rotation=45)

    # Highlight (much smaller) imbalance
    for bar, count in zip(bars, counts, strict=False):
        deviation = (count - high_vnode.mean) / high_vnode.mean * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + high_vnode.mean * 0.02,
            f"{deviation:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_dir / "vnodes_analysis.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'vnodes_analysis.png'}")

    # Additional figure: Comparative bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = list(range(len(results)))
    width = 0.35

    ax.bar([i - width / 2 for i in x], covs, width, label="CoV", color="steelblue", alpha=0.8)
    ax.bar(
        [i + width / 2 for i in x],
        [r - 1 for r in ratios],
        width,
        label="Max/Min - 1",
        color="coral",
        alpha=0.8,
    )

    ax.set_xlabel("Virtual Nodes per Server")
    ax.set_ylabel("Value (lower = better)")
    ax.set_title("Uniformity Metrics vs Virtual Node Count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in vnodes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "vnodes_metrics_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'vnodes_metrics_comparison.png'}")


def print_summary(results: list[DistributionStats], config: VNodesConfig) -> None:
    """Print summary table."""
    print("\n" + "=" * 70)
    print("VIRTUAL NODES ANALYSIS FOR CONSISTENT HASHING")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Servers: {config.num_servers}")
    print(f"  Keys: {config.num_keys}")
    print(f"  Trials per vnode count: {config.num_trials}")

    print("\nResults:")
    print(f"  {'V-Nodes':>10} | {'CoV':>8} | {'Max/Min':>8} | {'Recommendation':>20}")
    print(f"  {'-' * 10} | {'-' * 8} | {'-' * 8} | {'-' * 20}")

    for r in results:
        if r.cov < 0.05:
            rec = "Excellent"
        elif r.cov < 0.10:
            rec = "Good"
        elif r.cov < 0.20:
            rec = "Acceptable"
        elif r.cov < 0.40:
            rec = "Poor"
        else:
            rec = "Very Poor"

        print(f"  {r.vnode_count:>10} | {r.cov:>8.4f} | {r.max_min_ratio:>8.3f} | {rec:>20}")

    # Find the "sweet spot"
    good_vnodes = [r for r in results if r.cov < 0.10]
    if good_vnodes:
        sweet_spot = min(good_vnodes, key=lambda r: r.vnode_count)
        print(f"\nSweet Spot: {sweet_spot.vnode_count} virtual nodes")
        print("  - Achieves CoV < 0.10 (good uniformity)")
        print("  - Minimal memory overhead")

    print("\n" + "=" * 70)
    print("KEY INSIGHTS:")
    print("-" * 70)
    print("\n1. With 1 virtual node per server, distribution is highly uneven")
    print(
        f"   ({results[0].max_server} has {results[0].max_min_ratio:.1f}x more keys than {results[0].min_server})"
    )
    print(f"\n2. With {results[-1].vnode_count} virtual nodes, distribution is nearly uniform")
    print(f"   (variance is only {results[-1].cov:.1%} of the mean)")
    print("\n3. Industry standard: 100-200 virtual nodes provides good balance")
    print("   of uniformity vs memory/computation overhead")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze virtual node impact on consistent hashing uniformity"
    )
    parser.add_argument("--servers", type=int, default=5, help="Number of servers")
    parser.add_argument("--keys", type=int, default=10000, help="Number of keys to distribute")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per vnode count")
    parser.add_argument(
        "--output", type=str, default="output/load-balancing", help="Output directory"
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")

    args = parser.parse_args()

    config = VNodesConfig(
        num_servers=args.servers,
        num_keys=args.keys,
        num_trials=args.trials,
    )

    print("Running virtual nodes analysis...")
    results = run_analysis(config)

    print_summary(results, config)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(results, config, output_dir)
        print(f"\nVisualization saved to: {output_dir.absolute()}")
