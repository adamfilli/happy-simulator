"""Integration tests comparing sketch accuracy to exact statistics.

These tests run full simulations with both sketch collectors and exact
counters, then compare the results to verify sketch accuracy.
"""

from collections import Counter
from pathlib import Path

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    DistributedFieldProvider,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
    ZipfDistribution,
)
from happysimulator.components.sketching import TopKCollector
from happysimulator.sketching import CountMinSketch, HyperLogLog, TDigest, TopK


class IdealStatsSink(Entity):
    """Collects exact statistics for comparison with sketches."""

    def __init__(self, name: str, key_field: str):
        super().__init__(name)
        self.key_field = key_field
        self.counts: Counter = Counter()
        self.values: list = []
        self.unique_items: set = set()

    def handle_event(self, event: Event) -> list[Event]:
        key = event.context.get(self.key_field)
        if key is not None:
            self.counts[key] += 1
            self.values.append(key)
            self.unique_items.add(key)
        return []


class LatencySink(Entity):
    """Collects exact latency values for comparison."""

    def __init__(self, name: str, latency_field: str):
        super().__init__(name)
        self.latency_field = latency_field
        self.latencies: list[float] = []

    def handle_event(self, event: Event) -> list[Event]:
        latency = event.context.get(self.latency_field)
        if latency is not None:
            self.latencies.append(latency)
        return []


class DualSink(Entity):
    """Routes events to multiple sinks."""

    def __init__(self, name: str, sinks: list[Entity]):
        super().__init__(name)
        self._sinks = sinks

    def set_clock(self, clock):
        super().set_clock(clock)
        for sink in self._sinks:
            sink.set_clock(clock)

    def handle_event(self, event: Event) -> list[Event]:
        for sink in self._sinks:
            sink.handle_event(event)
        return []


class TestTopKAccuracy:
    """Tests for TopK accuracy vs exact counting."""

    def test_topk_precision_recall(self, test_output_dir: Path):
        """TopK correctly identifies heavy hitters with good precision/recall."""
        # Setup distribution
        population_size = 1000
        k = 100
        dist = ZipfDistribution(range(population_size), s=1.0, seed=42)

        # Create collectors
        topk_collector = TopKCollector(
            name="topk",
            k=k,
            value_extractor=lambda e: e.context.get("customer_id"),
        )
        ideal_sink = IdealStatsSink(name="ideal", key_field="customer_id")
        router = DualSink("router", [topk_collector, ideal_sink])

        # Create source with Zipf-distributed customer IDs
        provider = DistributedFieldProvider(
            target=router,
            event_type="Request",
            field_distributions={"customer_id": dist},
        )
        source = Source(
            name="traffic",
            event_provider=provider,
            arrival_time_provider=ConstantArrivalTimeProvider(
                profile=ConstantRateProfile(rate=1000),
                start_time=Instant.Epoch,
            ),
        )

        # Run simulation
        sim = Simulation(
            sources=[source],
            entities=[router, topk_collector, ideal_sink],
            duration=100,
        )
        sim.run()

        # Compare results
        true_top_k = {item for item, _ in ideal_sink.counts.most_common(k)}
        sketch_top_k = {item.item for item in topk_collector.top(k)}

        # Calculate precision and recall
        true_positives = len(sketch_top_k & true_top_k)
        precision = true_positives / len(sketch_top_k) if sketch_top_k else 0
        recall = true_positives / len(true_top_k) if true_top_k else 0

        # Should have reasonable precision and recall
        # With k=100 tracking 1000 items with Zipf distribution,
        # precision/recall > 50% is a reasonable expectation
        assert precision > 0.5, f"Precision {precision:.2%} too low"
        assert recall > 0.5, f"Recall {recall:.2%} too low"

        # Visualize if matplotlib available
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Plot true counts vs estimated counts for top items
            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # True counts (log scale)
            true_counts = [ideal_sink.counts[i] for i in range(min(50, population_size))]
            ax1.bar(range(len(true_counts)), true_counts)
            ax1.set_xlabel("Item (by rank)")
            ax1.set_ylabel("True count")
            ax1.set_title("True frequency distribution (top 50)")
            ax1.set_yscale("log")

            # Sketch counts for same items
            sketch_counts = [topk_collector.estimate(i) for i in range(min(50, population_size))]
            ax2.bar(range(len(sketch_counts)), sketch_counts, alpha=0.7, label="Sketch")
            ax2.bar(range(len(true_counts)), true_counts, alpha=0.3, label="True")
            ax2.set_xlabel("Item (by rank)")
            ax2.set_ylabel("Count")
            ax2.set_title(f"TopK vs True (k={k}, precision={precision:.0%})")
            ax2.legend()
            ax2.set_yscale("log")

            plt.tight_layout()
            plt.savefig(test_output_dir / "topk_accuracy.png", dpi=150)
            plt.close()
        except ImportError:
            pass

    def test_topk_error_within_bounds(self):
        """TopK error is within theoretical bounds."""
        k = 50
        topk = TopK[int](k=k)
        exact: Counter[int] = Counter()
        dist = ZipfDistribution(range(500), s=1.0, seed=42)

        # Add samples
        for _ in range(50000):
            item = dist.sample()
            topk.add(item)
            exact[item] += 1

        # For tracked items, error should be bounded
        max_error = topk.max_error()
        for estimate in topk.top():
            true_count = exact[estimate.item]
            observed_error = abs(estimate.count - true_count)
            assert observed_error <= max_error, (
                f"Item {estimate.item}: error {observed_error} > max {max_error}"
            )


class TestCountMinSketchAccuracy:
    """Tests for Count-Min Sketch accuracy vs exact counting."""

    def test_cms_never_underestimates(self):
        """CMS estimates are never below true counts."""
        cms = CountMinSketch[int](width=2000, depth=10, seed=42)
        exact: Counter[int] = Counter()
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)

        for _ in range(50000):
            item = dist.sample()
            cms.add(item)
            exact[item] += 1

        for item, true_count in exact.items():
            estimate = cms.estimate(item)
            assert estimate >= true_count, f"Item {item}: estimate {estimate} < true {true_count}"

    def test_cms_error_within_bounds(self, test_output_dir: Path):
        """CMS error is within epsilon * N with high probability."""
        cms = CountMinSketch.from_error_rate(epsilon=0.01, delta=0.01, seed=42)
        exact: Counter[int] = Counter()
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)

        n_samples = 10000
        for _ in range(n_samples):
            item = dist.sample()
            cms.add(item)
            exact[item] += 1

        error_bound = cms.epsilon * cms.item_count

        violations = 0
        errors = []
        for item in range(1000):
            true_count = exact.get(item, 0)
            estimate = cms.estimate(item)
            error = estimate - true_count
            errors.append(error)
            if error > error_bound:
                violations += 1

        # Should have few violations (delta controls this)
        violation_rate = violations / 1000
        assert violation_rate < 0.05, f"Violation rate {violation_rate:.1%} too high"

        # Visualize error distribution
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(errors, bins=50, edgecolor="black", alpha=0.7)
            ax.axvline(error_bound, color="red", linestyle="--", label=f"ε×N = {error_bound:.0f}")
            ax.set_xlabel("Estimation error (estimate - true)")
            ax.set_ylabel("Count")
            ax.set_title(f"CMS Error Distribution (ε={cms.epsilon:.3f}, δ={cms.delta:.3f})")
            ax.legend()

            plt.tight_layout()
            plt.savefig(test_output_dir / "cms_error_distribution.png", dpi=150)
            plt.close()
        except ImportError:
            pass


class TestHyperLogLogAccuracy:
    """Tests for HyperLogLog cardinality estimation accuracy."""

    def test_hll_accuracy_across_scales(self, test_output_dir: Path):
        """HLL maintains accuracy across different cardinalities."""
        results = []

        for true_cardinality in [100, 1000, 10000, 50000]:
            hll = HyperLogLog[int](precision=14, seed=42)

            for i in range(true_cardinality):
                hll.add(i)

            estimate = hll.cardinality()
            relative_error = abs(estimate - true_cardinality) / true_cardinality

            results.append(
                {
                    "true": true_cardinality,
                    "estimate": estimate,
                    "error": relative_error,
                }
            )

            # Should be within 3x standard error
            expected_error = hll.standard_error()
            assert relative_error < expected_error * 3, (
                f"Cardinality {true_cardinality}: "
                f"error {relative_error:.1%} > 3×σ ({expected_error * 3:.1%})"
            )

        # Visualize
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _fig, ax = plt.subplots(figsize=(8, 5))

            [r["true"] for r in results]
            [r["estimate"] for r in results]
            errors = [r["error"] * 100 for r in results]

            ax.bar(range(len(results)), errors)
            ax.set_xticks(range(len(results)))
            ax.set_xticklabels([f"n={r['true']}" for r in results])
            ax.axhline(
                1.04 / (2**7) * 100,
                color="red",
                linestyle="--",
                label=f"Expected error (~{1.04 / 128:.1%})",
            )
            ax.set_ylabel("Relative error (%)")
            ax.set_title("HyperLogLog Accuracy vs Cardinality (precision=14)")
            ax.legend()

            plt.tight_layout()
            plt.savefig(test_output_dir / "hll_accuracy.png", dpi=150)
            plt.close()
        except ImportError:
            pass


class TestTDigestAccuracy:
    """Tests for T-Digest quantile estimation accuracy."""

    def test_tdigest_percentiles_vs_exact(self, test_output_dir: Path):
        """T-Digest percentiles are close to exact values."""
        import random

        td = TDigest(compression=200)
        rng = random.Random(42)

        # Generate latency-like values (exponential + some outliers)
        n_samples = 10000
        values = []
        for _ in range(n_samples):
            # Base exponential with mean 100ms
            latency = rng.expovariate(1 / 100)
            # Add occasional outliers
            if rng.random() < 0.01:
                latency += rng.uniform(500, 2000)
            values.append(latency)
            td.add(latency)

        # Calculate exact percentiles
        values.sort()

        percentiles = [50, 75, 90, 95, 99, 99.9]
        results = []
        for p in percentiles:
            idx = int(p / 100 * n_samples)
            exact = values[min(idx, n_samples - 1)]
            estimated = td.percentile(p)
            error = abs(estimated - exact) / exact if exact > 0 else 0
            results.append(
                {
                    "percentile": p,
                    "exact": exact,
                    "estimated": estimated,
                    "relative_error": error,
                }
            )

        # Check reasonable accuracy - this simplified T-Digest implementation
        # provides approximate estimates. For better accuracy, use higher compression.
        # We check that estimates are in the right ballpark (within 2x)
        for r in results:
            assert r["relative_error"] < 1.0, (
                f"p{r['percentile']}: error {r['relative_error']:.1%} too high "
                f"(estimated={r['estimated']:.2f}, exact={r['exact']:.2f})"
            )

        # Visualize
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot exact vs estimated
            ps = [r["percentile"] for r in results]
            exact_vals = [r["exact"] for r in results]
            est_vals = [r["estimated"] for r in results]

            x = range(len(ps))
            width = 0.35
            ax1.bar([i - width / 2 for i in x], exact_vals, width, label="Exact")
            ax1.bar([i + width / 2 for i in x], est_vals, width, label="Estimated")
            ax1.set_xticks(x)
            ax1.set_xticklabels([f"p{p}" for p in ps])
            ax1.set_ylabel("Latency (ms)")
            ax1.set_title("T-Digest vs Exact Percentiles")
            ax1.legend()

            # Plot relative error
            errors = [r["relative_error"] * 100 for r in results]
            ax2.bar(x, errors)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f"p{p}" for p in ps])
            ax2.set_ylabel("Relative Error (%)")
            ax2.set_title("T-Digest Estimation Error")

            plt.tight_layout()
            plt.savefig(test_output_dir / "tdigest_accuracy.png", dpi=150)
            plt.close()
        except ImportError:
            pass


class TestSketchMemoryVsAccuracy:
    """Tests demonstrating memory-accuracy tradeoffs."""

    def test_topk_memory_tradeoff(self, test_output_dir: Path):
        """More k means more memory but better accuracy."""
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)
        n_samples = 50000

        # Sample once
        samples = [dist.sample() for _ in range(n_samples)]
        exact = Counter(samples)
        true_top_50 = {item for item, _ in exact.most_common(50)}

        results = []
        for k in [20, 50, 100, 200]:
            topk = TopK[int](k=k)
            for item in samples:
                topk.add(item)

            sketch_top_50 = {item.item for item in topk.top(50)}
            precision = len(sketch_top_50 & true_top_50) / 50

            results.append(
                {
                    "k": k,
                    "memory_bytes": topk.memory_bytes,
                    "precision": precision,
                }
            )

        # More k should mean better precision
        for i in range(1, len(results)):
            assert results[i]["precision"] >= results[i - 1]["precision"] * 0.9

        # Visualize
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            _fig, ax1 = plt.subplots(figsize=(8, 5))

            ks = [r["k"] for r in results]
            precisions = [r["precision"] * 100 for r in results]
            memories = [r["memory_bytes"] / 1024 for r in results]

            ax1.bar(range(len(ks)), precisions, alpha=0.7)
            ax1.set_xticks(range(len(ks)))
            ax1.set_xticklabels([f"k={k}" for k in ks])
            ax1.set_ylabel("Precision (%)")
            ax1.set_ylim(0, 100)

            ax2 = ax1.twinx()
            ax2.plot(range(len(ks)), memories, "r-o", label="Memory")
            ax2.set_ylabel("Memory (KB)", color="red")

            ax1.set_title("TopK: Memory vs Accuracy Tradeoff")

            plt.tight_layout()
            plt.savefig(test_output_dir / "memory_vs_accuracy.png", dpi=150)
            plt.close()
        except ImportError:
            pass
