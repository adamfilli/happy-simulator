"""Integration tests with visualizations for advanced datastore components.

These tests demonstrate advanced datastore patterns through visual output,
including sharding, replication, multi-tier caching, cache warming, and database operations.

Run:
    pytest tests/integration/test_advanced_datastore_visualization.py -v

Output:
    test_output/test_advanced_datastore_visualization/<test_name>/
"""

from __future__ import annotations

import random
from typing import Generator, Any

import pytest

from happysimulator.components.datastore import (
    KVStore,
    CachedStore,
    LRUEviction,
    ShardedStore,
    HashSharding,
    RangeSharding,
    ConsistentHashSharding,
    ReplicatedStore,
    ConsistencyLevel,
    MultiTierCache,
    PromotionPolicy,
    CacheWarmer,
    Database,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class TestShardingVisualization:
    """Visual tests for sharding strategies."""

    def test_sharding_strategy_comparison(self, test_output_dir):
        """
        Compare key distribution across different sharding strategies.

        Shows how Hash, Range, and ConsistentHash distribute keys.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        num_shards = 8
        num_keys = 1000

        strategies = {
            "Hash": HashSharding(),
            "Range": RangeSharding(),
            "ConsistentHash": ConsistentHashSharding(seed=42, virtual_nodes=100),
        }

        # Generate keys with various patterns
        keys = [f"user_{i}" for i in range(num_keys)]

        results = {}
        for name, strategy in strategies.items():
            distribution = [0] * num_shards
            for key in keys:
                shard = strategy.get_shard(key, num_shards)
                distribution[shard] += 1
            results[name] = distribution

        # Ensure RangeSharding has some distribution (may be skewed for numeric keys)
        # Range sharding works best with alphabetically varied keys
        range_strat = RangeSharding()
        alpha_keys = [f"{chr(97 + i % 26)}_{i}" for i in range(num_keys)]  # a_0, b_1, c_2...
        range_distribution = [0] * num_shards
        for key in alpha_keys:
            shard = range_strat.get_shard(key, num_shards)
            range_distribution[shard] += 1
        results["Range (alpha)"] = range_distribution

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

        # Bar chart comparison (only show main 3 strategies)
        ax1 = axes[0, 0]
        x = np.arange(num_shards)
        width = 0.25
        main_strategies = ["Hash", "Range", "ConsistentHash"]
        for i, name in enumerate(main_strategies):
            dist = results[name]
            ax1.bar(x + i * width, dist, width, label=name, color=colors[i], alpha=0.7)

        ax1.set_xlabel('Shard Index')
        ax1.set_ylabel('Key Count')
        ax1.set_title('Key Distribution by Sharding Strategy')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f'S{i}' for i in range(num_shards)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=num_keys / num_shards, color='red', linestyle='--', label='Ideal')

        # Distribution uniformity (std dev)
        ax2 = axes[0, 1]
        std_devs = [np.std(results[name]) for name in main_strategies]
        bars = ax2.bar(main_strategies, std_devs, color=colors[:3], alpha=0.7)
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Distribution Uniformity (lower is better)')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, std in zip(bars, std_devs):
            ax2.annotate(f'{std:.1f}', xy=(bar.get_x() + bar.get_width() / 2, std + 1),
                        ha='center')

        # Consistent hash - show impact of adding/removing shard
        ax3 = axes[1, 0]
        strategy_ch = ConsistentHashSharding(seed=42, virtual_nodes=100)
        strategy_hash = HashSharding()

        # Original assignment
        original_ch = {key: strategy_ch.get_shard(key, 8) for key in keys}
        original_hash = {key: strategy_hash.get_shard(key, 8) for key in keys}

        # New assignment with one more shard
        new_ch = {key: strategy_ch.get_shard(key, 9) for key in keys}
        new_hash = {key: strategy_hash.get_shard(key, 9) for key in keys}

        # Count keys that moved
        moved_ch = sum(1 for key in keys if original_ch[key] != new_ch.get(key, -1))
        moved_hash = sum(1 for key in keys if original_hash[key] != new_hash.get(key, -1))

        bars = ax3.bar(['Hash', 'ConsistentHash'],
                      [moved_hash / num_keys * 100, moved_ch / num_keys * 100],
                      color=['#e74c3c', '#2ecc71'], alpha=0.7)
        ax3.set_ylabel('Keys Moved (%)')
        ax3.set_title('Keys Moved When Adding 1 Shard (8 → 9)')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, pct in zip(bars, [moved_hash / num_keys * 100, moved_ch / num_keys * 100]):
            ax3.annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, pct + 2),
                        ha='center')

        # Explanation
        ax4 = axes[1, 1]
        ax4.axis('off')
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
        ax4.text(0.1, 0.95, explanation, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Sharding Strategy Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'sharding_comparison.png', dpi=150)
        plt.close()

        # Verify Hash and ConsistentHash have reasonable distribution
        # (Range may be skewed for certain key patterns)
        assert min(results["Hash"]) > 0, "Hash has empty shard"
        assert min(results["ConsistentHash"]) > 0, "ConsistentHash has empty shard"

    def test_sharded_store_operations(self, test_output_dir):
        """
        Visualize operations on a sharded store.

        Shows read/write distribution and shard utilization.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        num_shards = 4
        shards = [KVStore(name=f"shard{i}") for i in range(num_shards)]
        store = ShardedStore(name="sharded", shards=shards)

        # Generate workload with zipf distribution
        np.random.seed(42)
        num_operations = 500

        for i in range(num_operations):
            # Zipf-distributed key access
            rank = int(np.random.zipf(1.5))
            key = f"key_{min(rank, 100)}"
            value = f"value_{i}"

            if np.random.random() < 0.7:
                # Read
                gen = store.get(key)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass
            else:
                # Write
                gen = store.put(key, value)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Shard sizes
        ax1 = axes[0, 0]
        sizes = store.get_shard_sizes()
        shard_names = [f'Shard {i}' for i in range(num_shards)]
        ax1.bar(shard_names, [sizes[i] for i in range(num_shards)], color='steelblue', alpha=0.7)
        ax1.set_ylabel('Keys Stored')
        ax1.set_title('Data Distribution Across Shards')
        ax1.grid(True, alpha=0.3, axis='y')

        # Read distribution
        ax2 = axes[0, 1]
        read_dist = [store.stats.shard_reads.get(i, 0) for i in range(num_shards)]
        ax2.bar(shard_names, read_dist, color='#2ecc71', alpha=0.7)
        ax2.set_ylabel('Read Operations')
        ax2.set_title('Read Distribution Across Shards')
        ax2.grid(True, alpha=0.3, axis='y')

        # Write distribution
        ax3 = axes[1, 0]
        write_dist = [store.stats.shard_writes.get(i, 0) for i in range(num_shards)]
        ax3.bar(shard_names, write_dist, color='#e74c3c', alpha=0.7)
        ax3.set_ylabel('Write Operations')
        ax3.set_title('Write Distribution Across Shards')
        ax3.grid(True, alpha=0.3, axis='y')

        # Summary stats
        ax4 = axes[1, 1]
        ax4.axis('off')
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
        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.suptitle('Sharded Store Operations', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'sharded_store_operations.png', dpi=150)
        plt.close()

        assert store.stats.reads + store.stats.writes == num_operations


class TestReplicationVisualization:
    """Visual tests for replicated store behavior."""

    def test_consistency_levels_comparison(self, test_output_dir):
        """
        Compare read latency and consistency for different consistency levels.

        Shows tradeoff between latency and consistency guarantees.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        num_replicas = 5
        num_reads = 100

        # Simulate replicas with varying latencies
        latencies = {
            ConsistencyLevel.ONE: [],
            ConsistencyLevel.QUORUM: [],
            ConsistencyLevel.ALL: [],
        }

        random.seed(42)

        for level in latencies.keys():
            replicas = [
                KVStore(name=f"node{i}", read_latency=0.001 + random.random() * 0.009)
                for i in range(num_replicas)
            ]

            # Pre-populate
            for r in replicas:
                r.put_sync("key1", "value1")

            store = ReplicatedStore(
                name="distributed",
                replicas=replicas,
                read_consistency=level,
            )

            for _ in range(num_reads):
                gen = store.get("key1")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            latencies[level] = store.stats.read_latencies.copy()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {'ONE': '#2ecc71', 'QUORUM': '#3498db', 'ALL': '#e74c3c'}

        # Latency distribution boxplot
        ax1 = axes[0, 0]
        data = [latencies[level] for level in latencies.keys()]
        bp = ax1.boxplot(data, tick_labels=[level.name for level in latencies.keys()], patch_artist=True)
        for patch, level in zip(bp['boxes'], latencies.keys()):
            patch.set_facecolor(colors[level.name])
            patch.set_alpha(0.7)
        ax1.set_ylabel('Latency (s)')
        ax1.set_title('Read Latency by Consistency Level')
        ax1.grid(True, alpha=0.3, axis='y')

        # Average latency bar chart
        ax2 = axes[0, 1]
        avg_latencies = [np.mean(latencies[level]) * 1000 for level in latencies.keys()]
        bars = ax2.bar([level.name for level in latencies.keys()], avg_latencies,
                       color=[colors[level.name] for level in latencies.keys()], alpha=0.7)
        ax2.set_ylabel('Average Latency (ms)')
        ax2.set_title('Average Read Latency')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, lat in zip(bars, avg_latencies):
            ax2.annotate(f'{lat:.2f}ms', xy=(bar.get_x() + bar.get_width() / 2, lat + 0.1),
                        ha='center')

        # Quorum sizes
        ax3 = axes[1, 0]
        replica_counts = [3, 5, 7, 9, 11]
        quorum_sizes = [(n // 2) + 1 for n in replica_counts]

        ax3.plot(replica_counts, quorum_sizes, 'bo-', linewidth=2, markersize=8)
        ax3.plot(replica_counts, replica_counts, 'r--', label='ALL (n)', alpha=0.5)
        ax3.plot(replica_counts, [1] * len(replica_counts), 'g--', label='ONE (1)', alpha=0.5)
        ax3.fill_between(replica_counts, quorum_sizes, replica_counts, alpha=0.1, color='blue')
        ax3.set_xlabel('Number of Replicas')
        ax3.set_ylabel('Nodes to Contact')
        ax3.set_title('Quorum Size vs Replica Count')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Consistency explanation
        ax4 = axes[1, 1]
        ax4.axis('off')
        explanation = """
Consistency Level Tradeoffs

ONE:
  - Lowest latency
  - May read stale data
  - Best for: Read-heavy, eventual consistency OK

QUORUM:
  - Balanced latency/consistency
  - Strong consistency with R + W > N
  - Best for: Most applications

ALL:
  - Highest latency (wait for slowest)
  - Strongest consistency
  - Best for: Critical data, infrequent reads

Quorum Formula:
  - R + W > N ensures consistency
  - R = W = (N/2) + 1 is common choice
  - Example: 5 nodes, quorum = 3
    Read 3 + Write 3 = 6 > 5 = consistent
"""
        ax4.text(0.1, 0.95, explanation, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Replicated Store: Consistency Level Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'consistency_levels.png', dpi=150)
        plt.close()

        # Verify all consistency levels were tested
        assert len(latencies[ConsistencyLevel.ONE]) == num_reads
        assert len(latencies[ConsistencyLevel.QUORUM]) == num_reads
        assert len(latencies[ConsistencyLevel.ALL]) == num_reads


class TestMultiTierCacheVisualization:
    """Visual tests for multi-tier cache behavior."""

    def test_tier_hit_distribution(self, test_output_dir):
        """
        Visualize hit distribution across cache tiers.

        Shows how requests are served from L1, L2, and backing store.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Create tiered cache setup
        backing = KVStore(name="backing", read_latency=0.010)

        l1 = CachedStore(
            name="l1",
            backing_store=backing,
            cache_capacity=10,
            cache_read_latency=0.0001,
            eviction_policy=LRUEviction(),
        )

        l2 = CachedStore(
            name="l2",
            backing_store=backing,
            cache_capacity=100,
            cache_read_latency=0.001,
            eviction_policy=LRUEviction(),
        )

        cache = MultiTierCache(
            name="multi",
            tiers=[l1, l2],
            backing_store=backing,
        )

        # Pre-populate backing store
        num_keys = 200
        for i in range(num_keys):
            backing.put_sync(f"key_{i}", f"value_{i}")

        # Generate workload with zipf distribution
        np.random.seed(42)
        num_accesses = 1000
        tier_hits_over_time = {'L1': [], 'L2': [], 'Backing': []}

        for i in range(num_accesses):
            rank = int(np.random.zipf(1.5))
            key = f"key_{min(rank, num_keys) - 1}"

            gen = cache.get(key)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

            # Record cumulative hits
            if (i + 1) % 50 == 0:
                tier_hits_over_time['L1'].append(cache.stats.tier_hits.get(0, 0))
                tier_hits_over_time['L2'].append(cache.stats.tier_hits.get(1, 0))
                tier_hits_over_time['Backing'].append(cache.stats.backing_store_hits)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Tier hit distribution pie chart
        ax1 = axes[0, 0]
        tier_totals = [
            cache.stats.tier_hits.get(0, 0),
            cache.stats.tier_hits.get(1, 0),
            cache.stats.backing_store_hits,
        ]
        labels = ['L1 Cache', 'L2 Cache', 'Backing Store']
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        explode = (0.05, 0, 0)
        ax1.pie(tier_totals, explode=explode, labels=labels, colors=colors,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax1.set_title('Request Distribution by Tier')

        # Hit rate over time
        ax2 = axes[0, 1]
        x = [i * 50 for i in range(1, len(tier_hits_over_time['L1']) + 1)]

        # Calculate cumulative hit rate
        total_accesses = [i * 50 for i in range(1, len(x) + 1)]
        l1_rates = [tier_hits_over_time['L1'][i] / total_accesses[i] * 100 for i in range(len(x))]

        ax2.plot(x, l1_rates, 'g-', linewidth=2, label='L1 Hit Rate')
        ax2.plot(x, [cache.hit_rate * 100] * len(x), 'b--', label=f'Final: {cache.hit_rate*100:.1f}%')
        ax2.set_xlabel('Requests Processed')
        ax2.set_ylabel('L1 Hit Rate (%)')
        ax2.set_title('L1 Cache Hit Rate Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Latency by tier
        ax3 = axes[1, 0]
        latencies = [0.0001, 0.001, 0.010]  # L1, L2, Backing
        speedups = [latencies[2] / lat for lat in latencies]

        bars = ax3.bar(labels, [lat * 1000 for lat in latencies], color=colors, alpha=0.7)
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Access Latency by Tier')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, lat in zip(bars, latencies):
            ax3.annotate(f'{lat*1000:.2f}ms', xy=(bar.get_x() + bar.get_width() / 2, lat * 1000 * 1.2),
                        ha='center', fontsize=9)

        # Summary stats
        ax4 = axes[1, 1]
        ax4.axis('off')

        avg_latency = (
            tier_totals[0] * latencies[0] +
            tier_totals[1] * latencies[1] +
            tier_totals[2] * latencies[2]
        ) / sum(tier_totals)

        summary = f"""
Multi-Tier Cache Statistics

Configuration:
  - L1 Capacity: 10 entries (0.1ms latency)
  - L2 Capacity: 100 entries (1ms latency)
  - Backing Store: {num_keys} keys (10ms latency)

Results:
  - Total Accesses: {num_accesses}
  - L1 Hits: {tier_totals[0]} ({tier_totals[0]/num_accesses*100:.1f}%)
  - L2 Hits: {tier_totals[1]} ({tier_totals[1]/num_accesses*100:.1f}%)
  - Backing Hits: {tier_totals[2]} ({tier_totals[2]/num_accesses*100:.1f}%)

Performance:
  - Overall Hit Rate: {cache.hit_rate*100:.1f}%
  - Avg Latency: {avg_latency*1000:.3f}ms
  - Without cache: 10ms
  - Speedup: {0.010/avg_latency:.1f}x
"""
        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.suptitle('Multi-Tier Cache Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'multi_tier_cache.png', dpi=150)
        plt.close()

        assert cache.hit_rate > 0


class TestCacheWarmingVisualization:
    """Visual tests for cache warming behavior."""

    def test_cache_warming_progress(self, test_output_dir):
        """
        Visualize cache warming progress and impact on hit rate.

        Compares cold start vs warmed cache performance.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Setup
        num_keys = 100
        cache_capacity = 50
        num_accesses = 500

        # Pre-determine hot keys (keys that will be accessed most)
        np.random.seed(42)
        hot_keys = [f"key_{int(np.random.zipf(1.5)) % num_keys}" for _ in range(num_accesses)]
        unique_hot_keys = list(set(hot_keys[:cache_capacity]))  # Keys to warm

        # Cold start scenario
        backing_cold = KVStore(name="backing_cold")
        cache_cold = CachedStore(
            name="cache_cold",
            backing_store=backing_cold,
            cache_capacity=cache_capacity,
            eviction_policy=LRUEviction(),
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
            name="cache_warm",
            backing_store=backing_warm,
            cache_capacity=cache_capacity,
            eviction_policy=LRUEviction(),
        )

        for i in range(num_keys):
            backing_warm.put_sync(f"key_{i}", f"value_{i}")

        # Create warmer
        warmer = CacheWarmer(
            name="warmer",
            cache=cache_warm,
            keys_to_warm=unique_hot_keys,
        )

        # Simulate warming (just pre-populate the cache)
        for key in unique_hot_keys:
            list(cache_warm.get(key))

        # Reset stats after warming
        cache_warm._hits = 0
        cache_warm._misses = 0
        cache_warm._reads = 0

        warm_hit_rates = []
        for i, key in enumerate(hot_keys):
            list(cache_warm.get(key))
            if (i + 1) % 25 == 0:
                warm_hit_rates.append(cache_warm.hit_rate)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Hit rate over time comparison
        ax1 = axes[0, 0]
        x = [i * 25 for i in range(1, len(cold_hit_rates) + 1)]
        ax1.plot(x, [r * 100 for r in cold_hit_rates], 'b-', linewidth=2, label='Cold Start')
        ax1.plot(x, [r * 100 for r in warm_hit_rates], 'g-', linewidth=2, label='Pre-warmed')
        ax1.set_xlabel('Requests Processed')
        ax1.set_ylabel('Hit Rate (%)')
        ax1.set_title('Hit Rate: Cold Start vs Pre-warmed Cache')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Final hit rate comparison
        ax2 = axes[0, 1]
        final_rates = [cold_hit_rates[-1] * 100, warm_hit_rates[-1] * 100]
        bars = ax2.bar(['Cold Start', 'Pre-warmed'], final_rates,
                       color=['#3498db', '#2ecc71'], alpha=0.7)
        ax2.set_ylabel('Final Hit Rate (%)')
        ax2.set_title('Final Hit Rate Comparison')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, final_rates):
            ax2.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, rate + 1),
                        ha='center')

        # Warming benefit over time
        ax3 = axes[1, 0]
        benefit = [(w - c) * 100 for w, c in zip(warm_hit_rates, cold_hit_rates)]
        ax3.fill_between(x, benefit, alpha=0.3, color='green')
        ax3.plot(x, benefit, 'g-', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Requests Processed')
        ax3.set_ylabel('Hit Rate Advantage (%)')
        ax3.set_title('Warming Benefit (Warm - Cold)')
        ax3.grid(True, alpha=0.3)

        # Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"""
Cache Warming Analysis

Configuration:
  - Total Keys: {num_keys}
  - Cache Capacity: {cache_capacity}
  - Keys Warmed: {len(unique_hot_keys)}
  - Total Accesses: {num_accesses}

Results:
  - Cold Start Final Hit Rate: {cold_hit_rates[-1]*100:.1f}%
  - Warmed Cache Final Hit Rate: {warm_hit_rates[-1]*100:.1f}%
  - Hit Rate Improvement: {(warm_hit_rates[-1] - cold_hit_rates[-1])*100:.1f}%

Key Insights:
  - Warming eliminates initial miss penalty
  - Benefit highest in early requests
  - Converges as cold cache warms up
  - Most valuable for predictable workloads
"""
        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Cache Warming Impact Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'cache_warming.png', dpi=150)
        plt.close()

        # Verify warming helped
        assert warm_hit_rates[0] > cold_hit_rates[0]


class TestDatabaseVisualization:
    """Visual tests for database behavior."""

    def test_connection_pool_utilization(self, test_output_dir):
        """
        Visualize database connection pool behavior.

        Shows connection utilization under varying load.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Test different pool sizes
        pool_sizes = [5, 10, 20, 50]
        num_queries = 200

        results = {}

        for pool_size in pool_sizes:
            db = Database(
                name="postgres",
                max_connections=pool_size,
                query_latency=0.005,
                connection_latency=0.001,
            )

            # Simulate concurrent queries
            for i in range(num_queries):
                gen = db.execute(f"SELECT * FROM users WHERE id = {i}")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            results[pool_size] = {
                'connections_created': db.stats.connections_created,
                'queries': db.stats.queries_executed,
                'avg_latency': db.stats.avg_query_latency,
            }

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Connections created vs pool size
        ax1 = axes[0, 0]
        ax1.bar([str(s) for s in pool_sizes],
                [results[s]['connections_created'] for s in pool_sizes],
                color='steelblue', alpha=0.7)
        ax1.set_xlabel('Pool Size')
        ax1.set_ylabel('Connections Created')
        ax1.set_title('Connections Created by Pool Size')
        ax1.grid(True, alpha=0.3, axis='y')

        # Simulate transaction workload
        db_tx = Database(name="db_tx", max_connections=10)
        tx_count = 50

        commit_latencies = []
        for i in range(tx_count):
            gen = db_tx.begin_transaction()
            tx = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                tx = e.value

            # Execute some queries
            for j in range(3):
                gen = tx.execute(f"UPDATE users SET name = 'user{i}_{j}' WHERE id = {j}")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            # Commit
            gen = tx.commit()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # Transaction statistics
        ax2 = axes[0, 1]
        tx_stats = [
            db_tx.stats.transactions_started,
            db_tx.stats.transactions_committed,
            db_tx.stats.transactions_rolled_back,
        ]
        labels = ['Started', 'Committed', 'Rolled Back']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        ax2.bar(labels, tx_stats, color=colors, alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Transaction Statistics')
        ax2.grid(True, alpha=0.3, axis='y')

        # Query latency distribution
        ax3 = axes[1, 0]
        if db_tx.stats.query_latencies:
            ax3.hist([l * 1000 for l in db_tx.stats.query_latencies], bins=20,
                    color='purple', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Latency (ms)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Query Latency Distribution')
        ax3.grid(True, alpha=0.3)

        # Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"""
Database Connection Pool Analysis

Test 1: Pool Size Impact
  - Pool sizes tested: {pool_sizes}
  - Queries per test: {num_queries}
  - Connections created varies with pool size

Test 2: Transaction Workload
  - Transactions: {tx_count}
  - Queries per transaction: 3
  - Started: {db_tx.stats.transactions_started}
  - Committed: {db_tx.stats.transactions_committed}
  - Rolled back: {db_tx.stats.transactions_rolled_back}

Query Statistics:
  - Total queries: {db_tx.stats.queries_executed}
  - Avg latency: {db_tx.stats.avg_query_latency*1000:.2f}ms
  - P95 latency: {db_tx.stats.query_latency_p95*1000:.2f}ms

Connection Pool Benefits:
  - Reuses connections (avoids setup cost)
  - Limits concurrent connections
  - Queues requests when pool exhausted
"""
        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.suptitle('Database Connection Pool & Transaction Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'database_pool.png', dpi=150)
        plt.close()

        assert db_tx.stats.transactions_committed == tx_count

    def test_query_type_latency(self, test_output_dir):
        """
        Visualize latency differences by query type.

        Shows how different operations have different performance characteristics.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Custom latency function based on query type
        def query_latency(query: str) -> float:
            query_upper = query.upper()
            if query_upper.startswith("SELECT"):
                # SELECTs are faster on average
                return 0.002 + np.random.exponential(0.001)
            elif query_upper.startswith("INSERT"):
                return 0.003 + np.random.exponential(0.002)
            elif query_upper.startswith("UPDATE"):
                return 0.004 + np.random.exponential(0.002)
            elif query_upper.startswith("DELETE"):
                return 0.003 + np.random.exponential(0.001)
            return 0.005

        np.random.seed(42)
        db = Database(
            name="postgres",
            max_connections=20,
            query_latency=query_latency,
        )

        # Execute various query types
        query_types = {
            'SELECT': [],
            'INSERT': [],
            'UPDATE': [],
            'DELETE': [],
        }

        for _ in range(100):
            for qtype in query_types.keys():
                if qtype == 'SELECT':
                    query = "SELECT * FROM users WHERE id = 1"
                elif qtype == 'INSERT':
                    query = "INSERT INTO users (name) VALUES ('test')"
                elif qtype == 'UPDATE':
                    query = "UPDATE users SET name = 'new' WHERE id = 1"
                else:
                    query = "DELETE FROM users WHERE id = 1"

                latency_before = len(db.stats.query_latencies)
                gen = db.execute(query)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass
                latency_after = len(db.stats.query_latencies)
                if latency_after > latency_before:
                    query_types[qtype].append(db.stats.query_latencies[-1])

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {'SELECT': '#2ecc71', 'INSERT': '#3498db', 'UPDATE': '#e74c3c', 'DELETE': '#9b59b6'}

        # Latency boxplot
        ax1 = axes[0, 0]
        data = [query_types[qt] for qt in query_types.keys()]
        bp = ax1.boxplot(data, tick_labels=list(query_types.keys()), patch_artist=True)
        for patch, qt in zip(bp['boxes'], query_types.keys()):
            patch.set_facecolor(colors[qt])
            patch.set_alpha(0.7)
        ax1.set_ylabel('Latency (s)')
        ax1.set_title('Query Latency Distribution by Type')
        ax1.grid(True, alpha=0.3, axis='y')

        # Average latency bar chart
        ax2 = axes[0, 1]
        avg_latencies = [np.mean(query_types[qt]) * 1000 for qt in query_types.keys()]
        bars = ax2.bar(query_types.keys(), avg_latencies,
                       color=[colors[qt] for qt in query_types.keys()], alpha=0.7)
        ax2.set_ylabel('Average Latency (ms)')
        ax2.set_title('Average Query Latency by Type')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, lat in zip(bars, avg_latencies):
            ax2.annotate(f'{lat:.2f}ms', xy=(bar.get_x() + bar.get_width() / 2, lat + 0.1),
                        ha='center', fontsize=9)

        # Throughput simulation
        ax3 = axes[1, 0]
        throughputs = [1.0 / np.mean(query_types[qt]) for qt in query_types.keys()]
        ax3.bar(query_types.keys(), throughputs,
               color=[colors[qt] for qt in query_types.keys()], alpha=0.7)
        ax3.set_ylabel('Throughput (queries/s)')
        ax3.set_title('Theoretical Max Throughput by Type')
        ax3.grid(True, alpha=0.3, axis='y')

        # Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        summary = f"""
Query Performance Analysis

Queries Executed: {db.stats.queries_executed}
Queries per type: {len(query_types['SELECT'])} each

Latency Statistics (ms):
"""
        for qt in query_types.keys():
            lats = [l * 1000 for l in query_types[qt]]
            summary += f"  {qt}: avg={np.mean(lats):.2f}, p50={np.percentile(lats, 50):.2f}, p99={np.percentile(lats, 99):.2f}\n"

        summary += """
Key Observations:
  - SELECT typically fastest (read-only)
  - UPDATE slowest (lock + write)
  - INSERT/DELETE similar (single write)
  - Variance depends on query complexity
"""
        ax4.text(0.1, 0.95, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Database Query Performance by Type', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'query_type_latency.png', dpi=150)
        plt.close()

        assert db.stats.queries_executed == 400  # 100 * 4 query types
