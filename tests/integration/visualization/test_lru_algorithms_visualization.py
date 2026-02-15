"""Integration tests with visualizations for LRU algorithm variants.

These tests demonstrate and compare different LRU-based eviction policies:
- LRU (Least Recently Used)
- SLRU (Segmented LRU - scan resistant)
- SampledLRU (Probabilistic LRU - memory efficient)
- Clock (Second-Chance - CPU efficient)
- 2Q (Two Queue - adaptive)

Run:
    pytest tests/integration/test_lru_algorithms_visualization.py -v

Output:
    test_output/test_lru_algorithms_visualization/<test_name>/
"""

from __future__ import annotations

import random

from happysimulator.components.datastore import (
    CachedStore,
    ClockEviction,
    KVStore,
    LRUEviction,
    SampledLRUEviction,
    SLRUEviction,
    TwoQueueEviction,
)


class TestLRUAlgorithmComparison:
    """Compare LRU algorithm variants across different workloads."""

    def test_scan_resistance_comparison(self, test_output_dir):
        """
        Compare scan resistance of LRU variants.

        Demonstrates how SLRU and 2Q protect hot items from being evicted
        during sequential scans, while standard LRU does not.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Use parameters that highlight scan resistance differences:
        # - Cache smaller than hot set forces evictions during hot phase
        # - Large key space means scan fully fills cache
        cache_size = 30
        num_keys = 500
        hot_set_size = 50
        num_accesses = 4000

        # Generate scan-then-hot workload
        keys = self._scan_then_hot_workload(num_keys, hot_set_size, num_accesses)

        policies = {
            "LRU": lambda: LRUEviction(),
            "SLRU": lambda: SLRUEviction(protected_ratio=0.8),
            "Clock": lambda: ClockEviction(),
            "2Q": lambda: TwoQueueEviction(kin_ratio=0.25),
            "SampledLRU": lambda: SampledLRUEviction(sample_size=5, seed=42),
        }

        results = {}
        hit_rate_over_time = {}

        for policy_name, policy_factory in policies.items():
            backing = KVStore(name=f"backing_{policy_name}")
            cache = CachedStore(
                name=f"cache_{policy_name}",
                backing_store=backing,
                cache_capacity=cache_size,
                eviction_policy=policy_factory(),
            )

            # Pre-populate backing store
            for i in range(num_keys):
                backing.put_sync(f"key_{i}", f"value_{i}")

            # Track hit rate over time
            hit_rates = []
            for i, key in enumerate(keys):
                list(cache.get(key))
                if (i + 1) % 100 == 0:
                    hit_rates.append(cache.hit_rate)

            results[policy_name] = cache.hit_rate
            hit_rate_over_time[policy_name] = hit_rates

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Final hit rates bar chart
        ax1 = axes[0, 0]
        policy_names = list(results.keys())
        hit_rates = [results[p] * 100 for p in policy_names]
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]
        bars = ax1.bar(policy_names, hit_rates, color=colors, alpha=0.8, edgecolor="black")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("Scan Resistance: Final Hit Rates")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, rate in zip(bars, hit_rates, strict=False):
            ax1.annotate(
                f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, rate + 1),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # 2. Hit rate over time
        ax2 = axes[0, 1]
        x_points = list(range(100, num_accesses + 1, 100))
        for policy_name, hr_list in hit_rate_over_time.items():
            color_idx = policy_names.index(policy_name)
            ax2.plot(
                x_points[: len(hr_list)],
                [r * 100 for r in hr_list],
                label=policy_name,
                linewidth=2,
                color=colors[color_idx],
            )

        # Mark scan phase
        ax2.axvline(x=num_keys, color="gray", linestyle="--", alpha=0.7, label="Scan ends")
        ax2.set_xlabel("Requests Processed")
        ax2.set_ylabel("Cumulative Hit Rate (%)")
        ax2.set_title("Hit Rate During Scan → Hot Access Transition")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        # 3. Workload explanation
        ax3 = axes[1, 0]
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        ax3.axis("off")
        explanation = f"""
Workload Pattern: Scan → Hot Set

Phase 1 (Scan): Sequential access to all {num_keys} keys
  - Simulates database table scan or backup operation
  - Cache gets filled with scan items

Phase 2 (Hot Access): Repeated access to {hot_set_size} hot keys
  - Simulates normal workload returning
  - {num_accesses - num_keys} accesses to hot set

Cache Size: {cache_size} entries ({cache_size / num_keys * 100:.0f}% of key space)

Challenge: Can the algorithm retain hot items
          during the scan phase?
"""
        ax3.text(
            0.1,
            0.9,
            explanation,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        # 4. Algorithm characteristics
        ax4 = axes[1, 1]
        ax4.axis("off")
        characteristics = """
Algorithm Characteristics:

LRU:       Pure recency-based. Scan evicts all hot items.

SLRU:      Two segments (probationary + protected).
           Items promoted on re-access. Scan-resistant.

Clock:     Circular buffer with reference bits.
           Second chance for recently accessed items.

2Q:        FIFO queue + LRU queue + ghost queue.
           Items go to LRU only on re-access after eviction.

SampledLRU: Random sample of N keys, evict oldest in sample.
           Probabilistic approximation. Memory efficient.
"""
        ax4.text(
            0.05,
            0.95,
            characteristics,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
        )

        plt.suptitle("LRU Algorithm Comparison: Scan Resistance", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "scan_resistance_comparison.png", dpi=150)
        plt.close()

        # Verify all algorithms ran successfully
        assert len(results) == 5
        # Algorithms should have reasonable hit rates for this workload
        for policy_name, hit_rate in results.items():
            assert hit_rate >= 0.0 and hit_rate <= 1.0, f"{policy_name} hit rate out of range"

    def test_temporal_locality_comparison(self, test_output_dir):
        """
        Compare LRU variants on temporal locality workload.

        Shows how algorithms handle bursty, recency-based access patterns.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cache_size = 50
        num_keys = 200
        num_accesses = 3000

        keys = self._temporal_locality_workload(num_keys, num_accesses)

        policies = {
            "LRU": lambda: LRUEviction(),
            "SLRU": lambda: SLRUEviction(protected_ratio=0.8),
            "Clock": lambda: ClockEviction(),
            "2Q": lambda: TwoQueueEviction(kin_ratio=0.25),
            "SampledLRU": lambda: SampledLRUEviction(sample_size=5, seed=42),
        }

        results = {}
        eviction_counts = {}

        for policy_name, policy_factory in policies.items():
            backing = KVStore(name=f"backing_{policy_name}")
            cache = CachedStore(
                name=f"cache_{policy_name}",
                backing_store=backing,
                cache_capacity=cache_size,
                eviction_policy=policy_factory(),
            )

            for i in range(num_keys):
                backing.put_sync(f"key_{i}", f"value_{i}")

            for key in keys:
                list(cache.get(key))

            results[policy_name] = cache.hit_rate
            eviction_counts[policy_name] = cache.stats.evictions

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        policy_names = list(results.keys())
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        # Hit rate comparison
        hit_rates = [results[p] * 100 for p in policy_names]
        bars = ax1.bar(policy_names, hit_rates, color=colors, alpha=0.8, edgecolor="black")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("Temporal Locality Workload: Hit Rates")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3, axis="y")
        for bar, rate in zip(bars, hit_rates, strict=False):
            ax1.annotate(
                f"{rate:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, rate + 1),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Eviction count comparison
        evictions = [eviction_counts[p] for p in policy_names]
        bars2 = ax2.bar(policy_names, evictions, color=colors, alpha=0.8, edgecolor="black")
        ax2.set_ylabel("Number of Evictions")
        ax2.set_title("Eviction Activity (Lower = Better Cache Stability)")
        ax2.grid(True, alpha=0.3, axis="y")
        for bar, count in zip(bars2, evictions, strict=False):
            ax2.annotate(
                f"{count}",
                xy=(bar.get_x() + bar.get_width() / 2, count + 10),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.suptitle(
            f"Temporal Locality Workload (Cache={cache_size}, Keys={num_keys})", fontsize=14
        )
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(test_output_dir / "temporal_locality_comparison.png", dpi=150)
        plt.close()

        assert len(results) == 5

    def test_working_set_shift_comparison(self, test_output_dir):
        """
        Compare LRU variants when the working set shifts over time.

        Shows algorithm adaptability to changing access patterns.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        cache_size = 30
        num_keys = 100
        phase_length = 500
        num_phases = 4

        # Generate shifting workload
        keys, phase_boundaries = self._shifting_workload(num_keys, phase_length, num_phases)

        policies = {
            "LRU": lambda: LRUEviction(),
            "SLRU": lambda: SLRUEviction(protected_ratio=0.8),
            "Clock": lambda: ClockEviction(),
            "2Q": lambda: TwoQueueEviction(kin_ratio=0.25),
            "SampledLRU": lambda: SampledLRUEviction(sample_size=5, seed=42),
        }

        phase_hit_rates = {name: [] for name in policies}

        for policy_name, policy_factory in policies.items():
            backing = KVStore(name=f"backing_{policy_name}")
            cache = CachedStore(
                name=f"cache_{policy_name}",
                backing_store=backing,
                cache_capacity=cache_size,
                eviction_policy=policy_factory(),
            )

            for i in range(num_keys):
                backing.put_sync(f"key_{i}", f"value_{i}")

            # Track per-phase performance
            last_hits = 0
            last_reads = 0

            for i, key in enumerate(keys):
                list(cache.get(key))

                if i + 1 in phase_boundaries:
                    current_hits = cache.stats.hits
                    current_reads = cache.stats.reads
                    phase_hits = current_hits - last_hits
                    phase_reads = current_reads - last_reads
                    phase_hit_rate = phase_hits / phase_reads if phase_reads > 0 else 0
                    phase_hit_rates[policy_name].append(phase_hit_rate)
                    last_hits = current_hits
                    last_reads = current_reads

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        # Per-phase hit rates
        x = np.arange(num_phases)
        width = 0.15
        policy_names = list(policies.keys())

        for i, policy_name in enumerate(policy_names):
            offset = (i - len(policy_names) / 2 + 0.5) * width
            rates = [r * 100 for r in phase_hit_rates[policy_name]]
            ax1.bar(x + offset, rates, width, label=policy_name, color=colors[i], alpha=0.8)

        ax1.set_xlabel("Phase")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("Hit Rate by Phase (Working Set Shifts Each Phase)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Phase {i + 1}" for i in range(num_phases)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.set_ylim(0, 100)

        # Average adaptation speed (hit rate improvement from phase 1 to later phases)
        adaptation_scores = {}
        for policy_name in policy_names:
            rates = phase_hit_rates[policy_name]
            # Average hit rate after first phase (adaptation)
            if len(rates) > 1:
                adaptation_scores[policy_name] = sum(rates[1:]) / len(rates[1:])
            else:
                adaptation_scores[policy_name] = rates[0] if rates else 0

        ax2.bar(
            policy_names,
            [adaptation_scores[p] * 100 for p in policy_names],
            color=colors,
            alpha=0.8,
            edgecolor="black",
        )
        ax2.set_ylabel("Avg Hit Rate After Phase 1 (%)")
        ax2.set_title("Adaptation: Performance After First Working Set")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.set_ylim(0, 100)

        plt.suptitle("Working Set Shift Adaptation", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(test_output_dir / "working_set_shift_comparison.png", dpi=150)
        plt.close()

        assert len(phase_hit_rates) == 5

    def test_sample_size_impact(self, test_output_dir):
        """
        Analyze the impact of sample size on SampledLRU accuracy.

        Shows the tradeoff between sample size (memory/CPU) and eviction quality.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cache_size = 50
        num_keys = 200
        num_accesses = 3000

        # Generate zipf workload
        keys = self._frequency_skew_workload(num_keys, num_accesses)

        sample_sizes = [1, 2, 3, 5, 10, 20, 50]
        sampled_hit_rates = []

        # Get baseline LRU hit rate
        backing = KVStore(name="backing_lru")
        cache = CachedStore(
            name="cache_lru",
            backing_store=backing,
            cache_capacity=cache_size,
            eviction_policy=LRUEviction(),
        )
        for i in range(num_keys):
            backing.put_sync(f"key_{i}", f"value_{i}")
        for key in keys:
            list(cache.get(key))
        lru_hit_rate = cache.hit_rate

        for sample_size in sample_sizes:
            backing = KVStore(name=f"backing_s{sample_size}")
            cache = CachedStore(
                name=f"cache_s{sample_size}",
                backing_store=backing,
                cache_capacity=cache_size,
                eviction_policy=SampledLRUEviction(sample_size=sample_size, seed=42),
            )

            for i in range(num_keys):
                backing.put_sync(f"key_{i}", f"value_{i}")

            for key in keys:
                list(cache.get(key))

            sampled_hit_rates.append(cache.hit_rate)

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Hit rate vs sample size
        ax1.plot(
            sample_sizes, [r * 100 for r in sampled_hit_rates], "bo-", linewidth=2, markersize=8
        )
        ax1.axhline(
            y=lru_hit_rate * 100,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"True LRU ({lru_hit_rate * 100:.1f}%)",
        )
        ax1.set_xlabel("Sample Size")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("SampledLRU: Hit Rate vs Sample Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("log")

        # Accuracy (% of LRU performance achieved)
        accuracy = [(r / lru_hit_rate) * 100 if lru_hit_rate > 0 else 0 for r in sampled_hit_rates]
        ax2.bar(
            range(len(sample_sizes)),
            accuracy,
            tick_label=[str(s) for s in sample_sizes],
            color="steelblue",
            alpha=0.8,
            edgecolor="black",
        )
        ax2.axhline(y=100, color="red", linestyle="--", label="100% = True LRU")
        ax2.set_xlabel("Sample Size")
        ax2.set_ylabel("Accuracy (% of LRU Performance)")
        ax2.set_title("SampledLRU Accuracy Relative to True LRU")
        ax2.set_ylim(0, 110)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Add annotations for memory savings
        for i, (size, acc) in enumerate(zip(sample_sizes, accuracy, strict=False)):
            if size <= 10:
                savings = (1 - size / cache_size) * 100
                ax2.annotate(
                    f"{acc:.0f}%\n({savings:.0f}% mem saved)",
                    xy=(i, acc + 2),
                    ha="center",
                    fontsize=8,
                )

        plt.suptitle(f"SampledLRU Analysis (Cache={cache_size}, Keys={num_keys})", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(test_output_dir / "sampled_lru_analysis.png", dpi=150)
        plt.close()

        # Larger sample sizes should approach LRU
        assert sampled_hit_rates[-1] >= sampled_hit_rates[0] * 0.95

    def test_slru_segment_ratio_impact(self, test_output_dir):
        """
        Analyze the impact of protected ratio on SLRU performance.

        Shows how the balance between probationary and protected segments
        affects cache efficiency.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cache_size = 50
        num_keys = 200

        # Test with scan workload (where SLRU shines)
        keys_scan = self._scan_then_hot_workload(num_keys, 20, 2000)
        # Test with random workload (baseline)
        keys_random = self._frequency_skew_workload(num_keys, 2000)

        protected_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        scan_hit_rates = []
        random_hit_rates = []

        for ratio in protected_ratios:
            # Scan workload
            backing = KVStore(name=f"backing_scan_{ratio}")
            cache = CachedStore(
                name=f"cache_scan_{ratio}",
                backing_store=backing,
                cache_capacity=cache_size,
                eviction_policy=SLRUEviction(protected_ratio=ratio),
            )
            for i in range(num_keys):
                backing.put_sync(f"key_{i}", f"value_{i}")
            for key in keys_scan:
                list(cache.get(key))
            scan_hit_rates.append(cache.hit_rate)

            # Random workload
            backing2 = KVStore(name=f"backing_random_{ratio}")
            cache2 = CachedStore(
                name=f"cache_random_{ratio}",
                backing_store=backing2,
                cache_capacity=cache_size,
                eviction_policy=SLRUEviction(protected_ratio=ratio),
            )
            for i in range(num_keys):
                backing2.put_sync(f"key_{i}", f"value_{i}")
            for key in keys_random:
                list(cache2.get(key))
            random_hit_rates.append(cache2.hit_rate)

        # Create visualization
        _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Hit rate vs protected ratio
        ax1.plot(
            protected_ratios,
            [r * 100 for r in scan_hit_rates],
            "b-o",
            linewidth=2,
            markersize=8,
            label="Scan + Hot Workload",
        )
        ax1.plot(
            protected_ratios,
            [r * 100 for r in random_hit_rates],
            "g-s",
            linewidth=2,
            markersize=8,
            label="Frequency Skew Workload",
        )
        ax1.set_xlabel("Protected Segment Ratio")
        ax1.set_ylabel("Hit Rate (%)")
        ax1.set_title("SLRU: Protected Ratio Impact")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)

        # Best ratio annotation
        best_scan_idx = scan_hit_rates.index(max(scan_hit_rates))
        random_hit_rates.index(max(random_hit_rates))
        ax1.annotate(
            f"Best: {protected_ratios[best_scan_idx]}",
            xy=(protected_ratios[best_scan_idx], scan_hit_rates[best_scan_idx] * 100),
            xytext=(protected_ratios[best_scan_idx] + 0.1, scan_hit_rates[best_scan_idx] * 100 + 3),
            arrowprops={"arrowstyle": "->", "color": "blue"},
            color="blue",
            fontsize=10,
        )

        # Segment size visualization
        ax2.stackplot(
            protected_ratios,
            [[r * 100 for r in protected_ratios], [(1 - r) * 100 for r in protected_ratios]],
            labels=["Protected (LRU)", "Probationary (FIFO)"],
            colors=["steelblue", "lightcoral"],
            alpha=0.8,
        )
        ax2.set_xlabel("Protected Ratio")
        ax2.set_ylabel("Segment Size (%)")
        ax2.set_title("SLRU Segment Proportions")
        ax2.legend(loc="center right")
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 100)

        plt.suptitle("SLRU Protected Ratio Analysis", fontsize=14)
        plt.subplots_adjust(top=0.92)
        plt.savefig(test_output_dir / "slru_ratio_analysis.png", dpi=150)
        plt.close()

        assert len(scan_hit_rates) == len(protected_ratios)

    def test_algorithm_characteristics_summary(self, test_output_dir):
        """
        Create a comprehensive summary comparing all LRU variants.

        Generates a visual reference showing algorithm tradeoffs.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cache_size = 50
        num_keys = 200
        num_accesses = 2000

        # Test all workloads
        workloads = {
            "Temporal Locality": self._temporal_locality_workload(num_keys, num_accesses),
            "Frequency Skew": self._frequency_skew_workload(num_keys, num_accesses),
            "Scan + Hot": self._scan_then_hot_workload(num_keys, 20, num_accesses),
            "Uniform Random": self._uniform_workload(num_keys, num_accesses),
        }

        policies = {
            "LRU": lambda: LRUEviction(),
            "SLRU": lambda: SLRUEviction(protected_ratio=0.8),
            "Clock": lambda: ClockEviction(),
            "2Q": lambda: TwoQueueEviction(kin_ratio=0.25),
            "SampledLRU-5": lambda: SampledLRUEviction(sample_size=5, seed=42),
        }

        results = {workload: {} for workload in workloads}

        for workload_name, keys in workloads.items():
            for policy_name, policy_factory in policies.items():
                backing = KVStore(name=f"backing_{workload_name}_{policy_name}")
                cache = CachedStore(
                    name=f"cache_{workload_name}_{policy_name}",
                    backing_store=backing,
                    cache_capacity=cache_size,
                    eviction_policy=policy_factory(),
                )

                for i in range(num_keys):
                    backing.put_sync(f"key_{i}", f"value_{i}")

                for key in keys:
                    list(cache.get(key))

                results[workload_name][policy_name] = cache.hit_rate

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

        for idx, (workload_name, policy_results) in enumerate(results.items()):
            ax = axes[idx]
            policy_names = list(policy_results.keys())
            hit_rates = [policy_results[p] * 100 for p in policy_names]

            bars = ax.bar(policy_names, hit_rates, color=colors, alpha=0.8, edgecolor="black")
            ax.set_ylabel("Hit Rate (%)")
            ax.set_title(workload_name)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3, axis="y")

            # Highlight best
            best_idx = hit_rates.index(max(hit_rates))
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)

            for bar, rate in zip(bars, hit_rates, strict=False):
                ax.annotate(
                    f"{rate:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, rate + 1),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        plt.suptitle("LRU Algorithm Comparison Across Workloads", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(test_output_dir / "lru_algorithm_summary.png", dpi=150)
        plt.close()

        # Create heatmap
        _fig, ax = plt.subplots(figsize=(10, 6))

        workload_names = list(results.keys())
        policy_names = list(next(iter(results.values())).keys())
        data = [[results[w][p] * 100 for p in policy_names] for w in workload_names]

        im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

        ax.set_xticks(range(len(policy_names)))
        ax.set_xticklabels(policy_names, rotation=45, ha="right")
        ax.set_yticks(range(len(workload_names)))
        ax.set_yticklabels(workload_names)

        # Add text annotations
        for i in range(len(workload_names)):
            for j in range(len(policy_names)):
                ax.text(
                    j,
                    i,
                    f"{data[i][j]:.1f}%",
                    ha="center",
                    va="center",
                    color="black",
                    fontweight="bold",
                )

        plt.colorbar(im, label="Hit Rate (%)")
        plt.title("LRU Algorithm Performance Heatmap")
        plt.tight_layout()
        plt.savefig(test_output_dir / "lru_algorithm_heatmap.png", dpi=150)
        plt.close()

        assert len(results) == 4

    # Helper methods for generating workloads

    def _scan_then_hot_workload(
        self, num_keys: int, hot_set_size: int, num_accesses: int
    ) -> list[str]:
        """Full scan followed by hot set access."""
        random.seed(42)
        # Phase 1: Full scan
        keys = [f"key_{i}" for i in range(num_keys)]
        # Phase 2: Hot set access
        hot_keys = [f"key_{i}" for i in range(hot_set_size)]
        keys.extend(random.choice(hot_keys) for _ in range(num_accesses - num_keys))
        return keys

    def _temporal_locality_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Workload with temporal locality (recent keys reaccessed)."""
        random.seed(42)
        keys = []
        recent = []
        for _ in range(num_accesses):
            if recent and random.random() < 0.7:
                key = random.choice(recent[-30:])
            else:
                key = f"key_{random.randint(0, num_keys - 1)}"
            keys.append(key)
            recent.append(key)
        return keys

    def _frequency_skew_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Workload with zipf-distributed frequency skew."""
        import numpy as np

        np.random.seed(42)
        keys = []
        for _ in range(num_accesses):
            rank = int(np.random.zipf(1.3))
            key = f"key_{min(rank, num_keys) - 1}"
            keys.append(key)
        return keys

    def _uniform_workload(self, num_keys: int, num_accesses: int) -> list[str]:
        """Uniform random access pattern."""
        random.seed(42)
        return [f"key_{random.randint(0, num_keys - 1)}" for _ in range(num_accesses)]

    def _shifting_workload(self, num_keys: int, phase_length: int, num_phases: int):
        """Workload where hot set shifts each phase."""
        random.seed(42)
        keys = []
        phase_boundaries = []
        keys_per_phase = num_keys // num_phases

        for phase in range(num_phases):
            # Each phase has a different hot set
            hot_start = phase * keys_per_phase
            hot_end = hot_start + keys_per_phase
            hot_keys = [f"key_{i}" for i in range(hot_start, min(hot_end, num_keys))]

            for _ in range(phase_length):
                if random.random() < 0.8 and hot_keys:
                    keys.append(random.choice(hot_keys))
                else:
                    keys.append(f"key_{random.randint(0, num_keys - 1)}")

            phase_boundaries.append(len(keys))

        return keys, phase_boundaries
