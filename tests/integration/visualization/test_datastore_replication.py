"""Visual tests for replicated store behavior."""

from __future__ import annotations

import random

from happysimulator.components.datastore import ConsistencyLevel, KVStore, ReplicatedStore


class TestReplicationVisualization:
    """Visual tests for replicated store behavior."""

    def test_consistency_levels_comparison(self, test_output_dir):
        """Compare read latency and consistency for different consistency levels."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        num_replicas = 5
        num_reads = 100

        latencies = {
            ConsistencyLevel.ONE: [],
            ConsistencyLevel.QUORUM: [],
            ConsistencyLevel.ALL: [],
        }

        random.seed(42)

        for level in latencies:
            replicas = [
                KVStore(name=f"node{i}", read_latency=0.001 + random.random() * 0.009)
                for i in range(num_replicas)
            ]

            for r in replicas:
                r.put_sync("key1", "value1")

            store = ReplicatedStore(
                name="distributed", replicas=replicas, read_consistency=level,
            )

            for _ in range(num_reads):
                gen = store.get("key1")
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            latencies[level] = store.stats.read_latencies.copy()

        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = {"ONE": "#2ecc71", "QUORUM": "#3498db", "ALL": "#e74c3c"}

        ax1 = axes[0, 0]
        data = [latencies[level] for level in latencies]
        bp = ax1.boxplot(data, tick_labels=[level.name for level in latencies], patch_artist=True)
        for patch, level in zip(bp["boxes"], latencies.keys(), strict=False):
            patch.set_facecolor(colors[level.name])
            patch.set_alpha(0.7)
        ax1.set_ylabel("Latency (s)")
        ax1.set_title("Read Latency by Consistency Level")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2 = axes[0, 1]
        avg_latencies = [np.mean(latencies[level]) * 1000 for level in latencies]
        bars = ax2.bar(
            [level.name for level in latencies], avg_latencies,
            color=[colors[level.name] for level in latencies], alpha=0.7,
        )
        ax2.set_ylabel("Average Latency (ms)")
        ax2.set_title("Average Read Latency")
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, lat in zip(bars, avg_latencies, strict=False):
            ax2.annotate(f"{lat:.2f}ms", xy=(bar.get_x() + bar.get_width() / 2, lat + 0.1), ha="center")

        ax3 = axes[1, 0]
        replica_counts = [3, 5, 7, 9, 11]
        quorum_sizes = [(n // 2) + 1 for n in replica_counts]

        ax3.plot(replica_counts, quorum_sizes, "bo-", linewidth=2, markersize=8)
        ax3.plot(replica_counts, replica_counts, "r--", label="ALL (n)", alpha=0.5)
        ax3.plot(replica_counts, [1] * len(replica_counts), "g--", label="ONE (1)", alpha=0.5)
        ax3.fill_between(replica_counts, quorum_sizes, replica_counts, alpha=0.1, color="blue")
        ax3.set_xlabel("Number of Replicas")
        ax3.set_ylabel("Nodes to Contact")
        ax3.set_title("Quorum Size vs Replica Count")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        ax4.axis("off")
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
        ax4.text(
            0.1, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Replicated Store: Consistency Level Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "consistency_levels.png", dpi=150)
        plt.close()

        assert len(latencies[ConsistencyLevel.ONE]) == num_reads
        assert len(latencies[ConsistencyLevel.QUORUM]) == num_reads
        assert len(latencies[ConsistencyLevel.ALL]) == num_reads
