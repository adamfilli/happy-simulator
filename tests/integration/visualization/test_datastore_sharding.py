"""Visual tests for sharding strategies."""

from __future__ import annotations

import numpy as np

from happysimulator.components.datastore import (
    ConsistentHashSharding,
    HashSharding,
    KVStore,
    RangeSharding,
    ShardedStore,
)


class TestShardingVisualization:
    """Visual tests for sharding strategies."""

    def test_sharding_strategy_comparison(self, test_output_dir):
        """Compare key distribution across different sharding strategies."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_shards = 8
        num_keys = 1000

        strategies = {
            "Hash": HashSharding(),
            "Range": RangeSharding(),
            "ConsistentHash": ConsistentHashSharding(seed=42, virtual_nodes=100),
        }

        keys = [f"user_{i}" for i in range(num_keys)]

        results = {}
        for name, strategy in strategies.items():
            distribution = [0] * num_shards
            for key in keys:
                shard = strategy.get_shard(key, num_shards)
                distribution[shard] += 1
            results[name] = distribution

        range_strat = RangeSharding()
        alpha_keys = [f"{chr(97 + i % 26)}_{i}" for i in range(num_keys)]
        range_distribution = [0] * num_shards
        for key in alpha_keys:
            shard = range_strat.get_shard(key, num_shards)
            range_distribution[shard] += 1
        results["Range (alpha)"] = range_distribution

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6"]

        ax1 = axes[0, 0]
        x = np.arange(num_shards)
        width = 0.25
        main_strategies = ["Hash", "Range", "ConsistentHash"]
        for i, name in enumerate(main_strategies):
            dist = results[name]
            ax1.bar(x + i * width, dist, width, label=name, color=colors[i], alpha=0.7)

        ax1.set_xlabel("Shard Index")
        ax1.set_ylabel("Key Count")
        ax1.set_title("Key Distribution by Sharding Strategy")
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f"S{i}" for i in range(num_shards)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axhline(y=num_keys / num_shards, color="red", linestyle="--", label="Ideal")

        ax2 = axes[0, 1]
        std_devs = [np.std(results[name]) for name in main_strategies]
        bars = ax2.bar(main_strategies, std_devs, color=colors[:3], alpha=0.7)
        ax2.set_ylabel("Standard Deviation")
        ax2.set_title("Distribution Uniformity (lower is better)")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, std in zip(bars, std_devs, strict=False):
            ax2.annotate(f"{std:.1f}", xy=(bar.get_x() + bar.get_width() / 2, std + 1), ha="center")

        ax3 = axes[1, 0]
        strategy_ch = ConsistentHashSharding(seed=42, virtual_nodes=100)
        strategy_hash = HashSharding()

        original_ch = {key: strategy_ch.get_shard(key, 8) for key in keys}
        original_hash = {key: strategy_hash.get_shard(key, 8) for key in keys}
        new_ch = {key: strategy_ch.get_shard(key, 9) for key in keys}
        new_hash = {key: strategy_hash.get_shard(key, 9) for key in keys}

        moved_ch = sum(1 for key in keys if original_ch[key] != new_ch.get(key, -1))
        moved_hash = sum(1 for key in keys if original_hash[key] != new_hash.get(key, -1))

        bars = ax3.bar(
            ["Hash", "ConsistentHash"],
            [moved_hash / num_keys * 100, moved_ch / num_keys * 100],
            color=["#e74c3c", "#2ecc71"], alpha=0.7,
        )
        ax3.set_ylabel("Keys Moved (%)")
        ax3.set_title("Keys Moved When Adding 1 Shard (8 → 9)")
        ax3.grid(True, alpha=0.3, axis="y")

        for bar, pct in zip(
            bars, [moved_hash / num_keys * 100, moved_ch / num_keys * 100], strict=False
        ):
            ax3.annotate(f"{pct:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, pct + 2), ha="center")

        ax4 = axes[1, 1]
        ax4.axis("off")
        explanation = f"""
Sharding Strategy Comparison

Keys tested: {num_keys}
Shards: {num_shards}
Ideal keys/shard: {num_keys // num_shards}

Hash Sharding:
  - Simple MD5-based hashing
  - Good distribution
  - All keys move on shard count change

Range Sharding:
  - Alphabetical/lexicographic ranges
  - May have hot spots with skewed data
  - Good for range queries

Consistent Hashing:
  - Uses virtual nodes for balance
  - Only ~1/N keys move on shard change
  - Best for dynamic scaling
"""
        ax4.text(
            0.1, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Sharding Strategy Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "sharding_comparison.png", dpi=150)
        plt.close()

        assert min(results["Hash"]) > 0, "Hash has empty shard"
        assert min(results["ConsistentHash"]) > 0, "ConsistentHash has empty shard"

    def test_sharded_store_operations(self, test_output_dir):
        """Visualize operations on a sharded store."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_shards = 4
        shards = [KVStore(name=f"shard{i}") for i in range(num_shards)]
        store = ShardedStore(name="sharded", shards=shards)

        np.random.seed(42)
        num_operations = 500

        for i in range(num_operations):
            rank = int(np.random.zipf(1.5))
            key = f"key_{min(rank, 100)}"
            value = f"value_{i}"

            if np.random.random() < 0.7:
                gen = store.get(key)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass
            else:
                gen = store.put(key, value)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        sizes = store.get_shard_sizes()
        shard_names = [f"Shard {i}" for i in range(num_shards)]
        ax1.bar(shard_names, [sizes[i] for i in range(num_shards)], color="steelblue", alpha=0.7)
        ax1.set_ylabel("Keys Stored")
        ax1.set_title("Data Distribution Across Shards")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = axes[0, 1]
        read_dist = [store.stats.shard_reads.get(i, 0) for i in range(num_shards)]
        ax2.bar(shard_names, read_dist, color="#2ecc71", alpha=0.7)
        ax2.set_ylabel("Read Operations")
        ax2.set_title("Read Distribution Across Shards")
        ax2.grid(True, alpha=0.3, axis="y")

        ax3 = axes[1, 0]
        write_dist = [store.stats.shard_writes.get(i, 0) for i in range(num_shards)]
        ax3.bar(shard_names, write_dist, color="#e74c3c", alpha=0.7)
        ax3.set_ylabel("Write Operations")
        ax3.set_title("Write Distribution Across Shards")
        ax3.grid(True, alpha=0.3, axis="y")

        ax4 = axes[1, 1]
        ax4.axis("off")
        summary = f"""
Sharded Store Statistics

Configuration:
  - Shards: {num_shards}
  - Sharding Strategy: {type(store.sharding_strategy).__name__}

Operations:
  - Total Reads: {store.stats.reads}
  - Total Writes: {store.stats.writes}

Distribution:
  - Keys per shard: {[sizes[i] for i in range(num_shards)]}
  - Reads per shard: {read_dist}
  - Writes per shard: {write_dist}

Balance:
  - Key balance (std): {np.std([sizes[i] for i in range(num_shards)]):.1f}
  - Read balance (std): {np.std(read_dist):.1f}
  - Write balance (std): {np.std(write_dist):.1f}
"""
        ax4.text(
            0.1, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
        )

        plt.suptitle("Sharded Store Operations", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "sharded_store_operations.png", dpi=150)
        plt.close()

        assert store.stats.reads + store.stats.writes == num_operations
