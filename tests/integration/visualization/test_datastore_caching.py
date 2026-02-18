"""Visual tests for multi-tier cache and cache warming behavior."""

from __future__ import annotations

import numpy as np

from happysimulator.components.datastore import (
    CachedStore,
    CacheWarmer,
    KVStore,
    LRUEviction,
    MultiTierCache,
)


class TestMultiTierCacheVisualization:
    """Visual tests for multi-tier cache behavior."""

    def test_tier_hit_distribution(self, test_output_dir):
        """Visualize hit distribution across cache tiers."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        backing = KVStore(name="backing", read_latency=0.010)

        l1 = CachedStore(
            name="l1", backing_store=backing, cache_capacity=10,
            cache_read_latency=0.0001, eviction_policy=LRUEviction(),
        )

        l2 = CachedStore(
            name="l2", backing_store=backing, cache_capacity=100,
            cache_read_latency=0.001, eviction_policy=LRUEviction(),
        )

        cache = MultiTierCache(name="multi", tiers=[l1, l2], backing_store=backing)

        num_keys = 200
        for i in range(num_keys):
            backing.put_sync(f"key_{i}", f"value_{i}")

        np.random.seed(42)
        num_accesses = 1000
        tier_hits_over_time = {"L1": [], "L2": [], "Backing": []}

        for i in range(num_accesses):
            rank = int(np.random.zipf(1.5))
            key = f"key_{min(rank, num_keys) - 1}"

            gen = cache.get(key)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

            if (i + 1) % 50 == 0:
                tier_hits_over_time["L1"].append(cache.stats.tier_hits.get(0, 0))
                tier_hits_over_time["L2"].append(cache.stats.tier_hits.get(1, 0))
                tier_hits_over_time["Backing"].append(cache.stats.backing_store_hits)

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        tier_totals = [
            cache.stats.tier_hits.get(0, 0),
            cache.stats.tier_hits.get(1, 0),
            cache.stats.backing_store_hits,
        ]
        labels = ["L1 Cache", "L2 Cache", "Backing Store"]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        explode = (0.05, 0, 0)
        ax1.pie(
            tier_totals, explode=explode, labels=labels, colors=colors,
            autopct="%1.1f%%", shadow=True, startangle=90,
        )
        ax1.set_title("Request Distribution by Tier")

        ax2 = axes[0, 1]
        x = [i * 50 for i in range(1, len(tier_hits_over_time["L1"]) + 1)]
        total_accesses = [i * 50 for i in range(1, len(x) + 1)]
        l1_rates = [tier_hits_over_time["L1"][i] / total_accesses[i] * 100 for i in range(len(x))]

        ax2.plot(x, l1_rates, "g-", linewidth=2, label="L1 Hit Rate")
        ax2.plot(x, [cache.hit_rate * 100] * len(x), "b--", label=f"Final: {cache.hit_rate * 100:.1f}%")
        ax2.set_xlabel("Requests Processed")
        ax2.set_ylabel("L1 Hit Rate (%)")
        ax2.set_title("L1 Cache Hit Rate Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        latencies = [0.0001, 0.001, 0.010]
        [latencies[2] / lat for lat in latencies]

        bars = ax3.bar(labels, [lat * 1000 for lat in latencies], color=colors, alpha=0.7)
        ax3.set_ylabel("Latency (ms)")
        ax3.set_title("Access Latency by Tier")
        ax3.set_yscale("log")
        ax3.grid(True, alpha=0.3, axis="y")

        for bar, lat in zip(bars, latencies, strict=False):
            ax3.annotate(
                f"{lat * 1000:.2f}ms",
                xy=(bar.get_x() + bar.get_width() / 2, lat * 1000 * 1.2),
                ha="center", fontsize=9,
            )

        ax4 = axes[1, 1]
        ax4.axis("off")

        avg_latency = (
            tier_totals[0] * latencies[0]
            + tier_totals[1] * latencies[1]
            + tier_totals[2] * latencies[2]
        ) / sum(tier_totals)

        summary = f"""
Multi-Tier Cache Statistics

Configuration:
  - L1 Capacity: 10 entries (0.1ms latency)
  - L2 Capacity: 100 entries (1ms latency)
  - Backing Store: {num_keys} keys (10ms latency)

Results:
  - Total Accesses: {num_accesses}
  - L1 Hits: {tier_totals[0]} ({tier_totals[0] / num_accesses * 100:.1f}%)
  - L2 Hits: {tier_totals[1]} ({tier_totals[1] / num_accesses * 100:.1f}%)
  - Backing Hits: {tier_totals[2]} ({tier_totals[2] / num_accesses * 100:.1f}%)

Performance:
  - Overall Hit Rate: {cache.hit_rate * 100:.1f}%
  - Avg Latency: {avg_latency * 1000:.3f}ms
  - Without cache: 10ms
  - Speedup: {0.010 / avg_latency:.1f}x
"""
        ax4.text(
            0.1, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
        )

        plt.suptitle("Multi-Tier Cache Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "multi_tier_cache.png", dpi=150)
        plt.close()

        assert cache.hit_rate > 0


class TestCacheWarmingVisualization:
    """Visual tests for cache warming behavior."""

    def test_cache_warming_progress(self, test_output_dir):
        """Visualize cache warming progress and impact on hit rate."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_keys = 100
        cache_capacity = 50
        num_accesses = 500

        np.random.seed(42)
        hot_keys = [f"key_{int(np.random.zipf(1.5)) % num_keys}" for _ in range(num_accesses)]
        unique_hot_keys = list(set(hot_keys[:cache_capacity]))

        # Cold start scenario
        backing_cold = KVStore(name="backing_cold")
        cache_cold = CachedStore(
            name="cache_cold", backing_store=backing_cold,
            cache_capacity=cache_capacity, eviction_policy=LRUEviction(),
        )

        for i in range(num_keys):
            backing_cold.put_sync(f"key_{i}", f"value_{i}")

        cold_hit_rates = []
        for i, key in enumerate(hot_keys):
            list(cache_cold.get(key))
            if (i + 1) % 25 == 0:
                cold_hit_rates.append(cache_cold.hit_rate)

        # Warmed cache scenario
        backing_warm = KVStore(name="backing_warm")
        cache_warm = CachedStore(
            name="cache_warm", backing_store=backing_warm,
            cache_capacity=cache_capacity, eviction_policy=LRUEviction(),
        )

        for i in range(num_keys):
            backing_warm.put_sync(f"key_{i}", f"value_{i}")

        CacheWarmer(name="warmer", cache=cache_warm, keys_to_warm=unique_hot_keys)

        for key in unique_hot_keys:
            list(cache_warm.get(key))

        cache_warm._hits = 0
        cache_warm._misses = 0
        cache_warm._reads = 0

        warm_hit_rates = []
        for i, key in enumerate(hot_keys):
            list(cache_warm.get(key))
            if (i + 1) % 25 == 0:
                warm_hit_rates.append(cache_warm.hit_rate)

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        x = [i * 25 for i in range(1, len(cold_hit_rates) + 1)]
        ax1.plot(x, [r * 100 for r in cold_hit_rates], "b-", linewidth=2, label="Cold Start")
        ax1.plot(x, [r * 100 for r in warm_hit_rates], "g-", linewidth=2, label="Pre-warmed")
        ax1.set_xlabel("Requests Processed")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("Hit Rate: Cold Start vs Pre-warmed Cache")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        final_rates = [cold_hit_rates[-1] * 100, warm_hit_rates[-1] * 100]
        bars = ax2.bar(["Cold Start", "Pre-warmed"], final_rates, color=["#3498db", "#2ecc71"], alpha=0.7)
        ax2.set_ylabel("Final Hit Rate (%)")
        ax2.set_title("Final Hit Rate Comparison")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, rate in zip(bars, final_rates, strict=False):
            ax2.annotate(f"{rate:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, rate + 1), ha="center")

        ax3 = axes[1, 0]
        benefit = [(w - c) * 100 for w, c in zip(warm_hit_rates, cold_hit_rates, strict=False)]
        ax3.fill_between(x, benefit, alpha=0.3, color="green")
        ax3.plot(x, benefit, "g-", linewidth=2)
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax3.set_xlabel("Requests Processed")
        ax3.set_ylabel("Hit Rate Advantage (%)")
        ax3.set_title("Warming Benefit (Warm - Cold)")
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.axis("off")
        summary = f"""
Cache Warming Analysis

Configuration:
  - Total Keys: {num_keys}
  - Cache Capacity: {cache_capacity}
  - Keys Warmed: {len(unique_hot_keys)}
  - Total Accesses: {num_accesses}

Results:
  - Cold Start Final Hit Rate: {cold_hit_rates[-1] * 100:.1f}%
  - Warmed Cache Final Hit Rate: {warm_hit_rates[-1] * 100:.1f}%
  - Hit Rate Improvement: {(warm_hit_rates[-1] - cold_hit_rates[-1]) * 100:.1f}%

Key Insights:
  - Warming eliminates initial miss penalty
  - Benefit highest in early requests
  - Converges as cold cache warms up
  - Most valuable for predictable workloads
"""
        ax4.text(
            0.1, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Cache Warming Impact Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "cache_warming.png", dpi=150)
        plt.close()

        assert warm_hit_rates[0] > cold_hit_rates[0]
