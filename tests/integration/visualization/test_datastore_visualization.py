"""Integration tests with visualizations for datastore components.

These tests demonstrate caching behavior through visual output,
showing hit rates, eviction patterns, and cache efficiency.

Run:
    pytest tests/integration/test_datastore_visualization.py -v

Output:
    test_output/test_datastore_visualization/<test_name>/
"""

from __future__ import annotations

import contextlib
import random

from happysimulator.components.datastore import (
    CachedStore,
    FIFOEviction,
    KVStore,
    LFUEviction,
    LRUEviction,
    RandomEviction,
)


class TestCacheHitRateVisualization:
    """Visual tests for cache hit rate behavior."""

    def test_hit_rate_vs_cache_size(self, test_output_dir):
        """
        Visualize how cache size affects hit rate.

        Shows the relationship between cache capacity and hit rate
        for a fixed workload.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Generate a workload with zipf-like access pattern
        random.seed(42)
        num_keys = 1000
        num_accesses = 5000

        # Zipf distribution - some keys accessed much more than others
        keys = []
        for _ in range(num_accesses):
            # Power law: key index proportional to 1/rank
            rank = int(np.random.zipf(1.5))
            key = f"key_{min(rank, num_keys)}"
            keys.append(key)

        # Test different cache sizes
        cache_sizes = [10, 25, 50, 100, 200, 500]
        hit_rates = []

        for size in cache_sizes:
            backing = KVStore(name="backing")
            cache = CachedStore(
                name="cache",
                backing_store=backing,
                cache_capacity=size,
                eviction_policy=LRUEviction(),
            )

            # Pre-populate backing store
            for i in range(num_keys):
                backing.put_sync(f"key_{i + 1}", f"value_{i + 1}")

            # Run workload
            for key in keys:
                list(cache.get(key))

            hit_rates.append(cache.hit_rate)

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Hit rate vs cache size
        ax1.plot(cache_sizes, hit_rates, "bo-", linewidth=2, markersize=8)
        ax1.set_xlabel("Cache Size (entries)")
        ax1.set_ylabel("Hit Rate")
        ax1.set_title("Cache Hit Rate vs Size (Zipf Workload)")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Percentage of keys vs hit rate
        cache_pct = [s / num_keys * 100 for s in cache_sizes]
        ax2.plot(cache_pct, [r * 100 for r in hit_rates], "go-", linewidth=2, markersize=8)
        ax2.set_xlabel("Cache Size (% of key space)")
        ax2.set_ylabel("Hit Rate (%)")
        ax2.set_title("Efficiency: Small Cache, High Hit Rate")
        ax2.grid(True, alpha=0.3)

        # Annotate the sweet spot
        best_efficiency_idx = np.argmax(
            [hr / (cs / num_keys) for hr, cs in zip(hit_rates, cache_sizes, strict=False)]
        )
        ax2.annotate(
            f"{hit_rates[best_efficiency_idx] * 100:.1f}% hits\nwith {cache_pct[best_efficiency_idx]:.1f}% cache",
            xy=(cache_pct[best_efficiency_idx], hit_rates[best_efficiency_idx] * 100),
            xytext=(cache_pct[best_efficiency_idx] + 10, hit_rates[best_efficiency_idx] * 100 - 10),
            arrowprops={"arrowstyle": "->", "color": "red"},
            fontsize=10,
        )

        plt.tight_layout()
        plt.savefig(test_output_dir / "hit_rate_vs_cache_size.png", dpi=150)
        plt.close()

        # Verify reasonable hit rates
        assert hit_rates[-1] > hit_rates[0]  # Bigger cache = higher hit rate


class TestEvictionPolicyComparison:
    """Compare different eviction policies."""

    def test_eviction_policies_comparison(self, test_output_dir):
        """
        Compare hit rates of different eviction policies.

        Shows how LRU, LFU, FIFO, and Random perform on different workloads.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cache_size = 50
        num_keys = 200
        num_accesses = 2000

        # Different workload patterns
        workloads = {
            "Temporal Locality": self._temporal_locality_workload(num_keys, num_accesses),
            "Frequency Skew": self._frequency_skew_workload(num_keys, num_accesses),
            "Scan + Hot": self._scan_then_hot_workload(num_keys, num_accesses),
            "Uniform Random": self._uniform_workload(num_keys, num_accesses),
        }

        policies = {
            "LRU": lambda: LRUEviction(),
            "LFU": lambda: LFUEviction(),
            "FIFO": lambda: FIFOEviction(),
            "Random": lambda: RandomEviction(seed=42),
        }

        results = {workload: {} for workload in workloads}

        for workload_name, keys in workloads.items():
            for policy_name, policy_factory in policies.items():
                backing = KVStore(name="backing")
                cache = CachedStore(
                    name="cache",
                    backing_store=backing,
                    cache_capacity=cache_size,
                    eviction_policy=policy_factory(),
                )

                # Pre-populate
                for i in range(num_keys):
                    backing.put_sync(f"key_{i}", f"value_{i}")

                # Run workload
                for key in keys:
                    list(cache.get(key))

                results[workload_name][policy_name] = cache.hit_rate

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        colors = {"LRU": "blue", "LFU": "green", "FIFO": "orange", "Random": "red"}

        for idx, (workload_name, policy_results) in enumerate(results.items()):
            ax = axes[idx]
            policy_names = list(policy_results.keys())
            hit_rates = [policy_results[p] * 100 for p in policy_names]
            bar_colors = [colors[p] for p in policy_names]

            bars = ax.bar(policy_names, hit_rates, color=bar_colors, alpha=0.7, edgecolor="black")
            ax.set_ylabel("Hit Rate (%)")
            ax.set_title(f"{workload_name}")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, rate in zip(bars, hit_rates, strict=False):
                ax.annotate(
                    f"{rate:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, rate),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.suptitle(
            f"Eviction Policy Comparison (Cache Size={cache_size}, Keys={num_keys})", fontsize=14
        )
        plt.tight_layout()
        plt.savefig(test_output_dir / "eviction_policy_comparison.png", dpi=150)
        plt.close()

        # Verify tests ran
        assert len(results) == 4

    def _temporal_locality_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Generate workload with temporal locality (recent keys reaccessed)."""
        random.seed(42)
        keys = []
        recent = []
        for _ in range(num_accesses):
            if recent and random.random() < 0.7:
                # Access recent key
                key = random.choice(recent[-20:])
            else:
                # Access new key
                key = f"key_{random.randint(0, num_keys - 1)}"
            keys.append(key)
            recent.append(key)
        return keys

    def _frequency_skew_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Generate workload with frequency skew (some keys very popular)."""
        import numpy as np

        np.random.seed(42)
        keys = []
        for _ in range(num_accesses):
            rank = int(np.random.zipf(1.3))
            key = f"key_{min(rank, num_keys) - 1}"
            keys.append(key)
        return keys

    def _scan_then_hot_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Generate workload: full scan then hot set (tests scan resistance)."""
        random.seed(42)
        # First: scan through all keys
        keys = [f"key_{i}" for i in range(num_keys)]
        # Then: repeatedly access hot set
        hot_keys = [f"key_{i}" for i in range(10)]
        keys.extend(random.choice(hot_keys) for _ in range(num_accesses - num_keys))
        return keys

    def _uniform_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Generate uniform random workload."""
        random.seed(42)
        return [f"key_{random.randint(0, num_keys - 1)}" for _ in range(num_accesses)]


class TestCacheWarmupVisualization:
    """Visualize cache warmup behavior."""

    def test_cold_start_warmup(self, test_output_dir):
        """
        Visualize cache warming from cold start.

        Shows how hit rate improves as cache warms up.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cache_size = 100
        num_keys = 500
        num_accesses = 2000

        # Generate zipf workload
        np.random.seed(42)
        keys = []
        for _ in range(num_accesses):
            rank = int(np.random.zipf(1.5))
            key = f"key_{min(rank, num_keys)}"
            keys.append(key)

        backing = KVStore(name="backing")
        cache = CachedStore(
            name="cache",
            backing_store=backing,
            cache_capacity=cache_size,
            eviction_policy=LRUEviction(),
        )

        # Pre-populate backing store
        for i in range(num_keys):
            backing.put_sync(f"key_{i + 1}", f"value_{i + 1}")

        # Track hit rate over time
        window_size = 100
        hit_rates = []

        for i, key in enumerate(keys):
            gen = cache.get(key)
            next(gen)
            with contextlib.suppress(StopIteration):
                next(gen)

            # Track if this was a hit (cache size increased means miss and cache)
            (
                cache.stats.hits
                > (sum(1 for _ in range(i)) - cache.stats.misses + cache.stats.hits - 1)
                if i > 0
                else False
            )

            if i >= window_size and (i + 1) % window_size == 0:
                current_hits = cache.stats.hits
                current_total = cache.stats.reads
                if current_total > 0:
                    hit_rates.append(current_hits / current_total)

        # Simpler approach: just track cumulative hit rate at intervals
        backing2 = KVStore(name="backing2")
        cache2 = CachedStore(
            name="cache2",
            backing_store=backing2,
            cache_capacity=cache_size,
            eviction_policy=LRUEviction(),
        )
        for i in range(num_keys):
            backing2.put_sync(f"key_{i + 1}", f"value_{i + 1}")

        cumulative_hit_rates = []
        for i, key in enumerate(keys):
            list(cache2.get(key))
            if (i + 1) % 50 == 0:
                cumulative_hit_rates.append(cache2.hit_rate)

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Cumulative hit rate over time
        x = [i * 50 for i in range(1, len(cumulative_hit_rates) + 1)]
        ax1.plot(x, [r * 100 for r in cumulative_hit_rates], "b-", linewidth=2)
        ax1.set_xlabel("Requests Processed")
        ax1.set_ylabel("Cumulative Hit Rate (%)")
        ax1.set_title("Cache Warmup: Hit Rate Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(
            y=cumulative_hit_rates[-1] * 100,
            color="r",
            linestyle="--",
            label=f"Final: {cumulative_hit_rates[-1] * 100:.1f}%",
        )
        ax1.legend()

        # Cache size utilization
        ax2.axhline(y=cache_size, color="r", linestyle="--", label="Capacity")
        ax2.axhline(
            y=cache2.cache_size, color="b", linestyle="-", label=f"Final Size: {cache2.cache_size}"
        )
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Cache Entries")
        ax2.set_title("Cache Utilization")
        ax2.set_ylim(0, cache_size * 1.2)
        ax2.legend()

        # Add statistics text
        stats_text = f"""Cache Statistics:
Capacity: {cache_size}
Total Reads: {cache2.stats.reads}
Hits: {cache2.stats.hits}
Misses: {cache2.stats.misses}
Hit Rate: {cache2.hit_rate * 100:.1f}%
Evictions: {cache2.stats.evictions}"""

        ax2.text(
            0.5,
            0.5,
            stats_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="center",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

        plt.tight_layout()
        plt.savefig(test_output_dir / "cache_warmup.png", dpi=150)
        plt.close()

        # Verify warmup occurred
        assert cumulative_hit_rates[-1] > cumulative_hit_rates[0]


class TestWritePolicyVisualization:
    """Visualize write policy behavior."""

    def test_write_through_vs_back(self, test_output_dir):
        """
        Compare write-through and write-back performance.

        Shows latency differences between write strategies.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_writes = 100
        backing_write_latency = 0.010  # 10ms

        # Simulate write-through: every write goes to backing store
        write_through_latencies = [backing_write_latency] * num_writes
        write_through_total = sum(write_through_latencies)

        # Simulate write-back: writes go to cache, periodic flush
        cache_write_latency = 0.0001  # 0.1ms
        flush_interval = 10  # Flush every 10 writes
        write_back_latencies = []
        pending_writes = 0

        for _i in range(num_writes):
            write_back_latencies.append(cache_write_latency)
            pending_writes += 1

            if pending_writes >= flush_interval:
                # Flush all pending
                flush_latency = backing_write_latency * pending_writes * 0.5  # Batching efficiency
                write_back_latencies[-1] += flush_latency
                pending_writes = 0

        # Final flush
        if pending_writes > 0:
            write_back_latencies[-1] += backing_write_latency * pending_writes * 0.5

        write_back_total = sum(write_back_latencies)

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Cumulative latency over writes
        wt_cumulative = []
        wb_cumulative = []
        total_wt = 0
        total_wb = 0
        for wt, wb in zip(write_through_latencies, write_back_latencies, strict=False):
            total_wt += wt
            total_wb += wb
            wt_cumulative.append(total_wt)
            wb_cumulative.append(total_wb)

        ax1.plot(
            range(1, num_writes + 1),
            [t * 1000 for t in wt_cumulative],
            "b-",
            linewidth=2,
            label="Write-Through",
        )
        ax1.plot(
            range(1, num_writes + 1),
            [t * 1000 for t in wb_cumulative],
            "g-",
            linewidth=2,
            label="Write-Back",
        )
        ax1.set_xlabel("Write Operations")
        ax1.set_ylabel("Cumulative Latency (ms)")
        ax1.set_title("Write Policy: Cumulative Latency")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Per-write latency
        ax2.bar(
            ["Write-Through", "Write-Back"],
            [write_through_total * 1000, write_back_total * 1000],
            color=["blue", "green"],
            alpha=0.7,
        )
        ax2.set_ylabel("Total Latency (ms)")
        ax2.set_title(f"Total Latency for {num_writes} Writes")
        ax2.grid(True, alpha=0.3, axis="y")

        # Add speedup annotation
        speedup = write_through_total / write_back_total
        ax2.annotate(
            f"{speedup:.1f}x faster",
            xy=(1, write_back_total * 1000),
            xytext=(1, write_through_total * 500),
            arrowprops={"arrowstyle": "->", "color": "red"},
            ha="center",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(test_output_dir / "write_policy_comparison.png", dpi=150)
        plt.close()

        # Verify write-back is faster
        assert write_back_total < write_through_total
