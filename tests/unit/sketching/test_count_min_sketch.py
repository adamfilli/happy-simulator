"""Tests for Count-Min Sketch."""

import math
from collections import Counter

import pytest

from happysimulator.distributions import ZipfDistribution
from happysimulator.sketching import CountMinSketch, FrequencyEstimate


class TestCountMinSketchCreation:
    """Tests for CountMinSketch creation and configuration."""

    def test_creates_with_dimensions(self):
        """CountMinSketch is created with specified dimensions."""
        cms = CountMinSketch[int](width=100, depth=5)

        assert cms.width == 100
        assert cms.depth == 5
        assert cms.item_count == 0

    def test_creates_with_seed(self):
        """CountMinSketch accepts seed for reproducibility."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)

        assert cms.width == 100

    def test_rejects_zero_width(self):
        """Rejects width=0."""
        with pytest.raises(ValueError, match="must be positive"):
            CountMinSketch[int](width=0, depth=5)

    def test_rejects_negative_width(self):
        """Rejects negative width."""
        with pytest.raises(ValueError, match="must be positive"):
            CountMinSketch[int](width=-10, depth=5)

    def test_rejects_zero_depth(self):
        """Rejects depth=0."""
        with pytest.raises(ValueError, match="must be positive"):
            CountMinSketch[int](width=100, depth=0)

    def test_rejects_negative_depth(self):
        """Rejects negative depth."""
        with pytest.raises(ValueError, match="must be positive"):
            CountMinSketch[int](width=100, depth=-3)

    def test_works_with_strings(self):
        """Works with string items."""
        cms = CountMinSketch[str](width=100, depth=5)
        cms.add("hello")
        cms.add("world")

        assert cms.estimate("hello") >= 1
        assert cms.estimate("world") >= 1

    def test_works_with_tuples(self):
        """Works with tuple items."""
        cms = CountMinSketch[tuple](width=100, depth=5)
        cms.add((1, "a"))
        cms.add((2, "b"))

        assert cms.estimate((1, "a")) >= 1


class TestCountMinSketchFromErrorRate:
    """Tests for from_error_rate factory method."""

    def test_creates_with_error_bounds(self):
        """Creates sketch with specified error bounds."""
        cms = CountMinSketch.from_error_rate(epsilon=0.01, delta=0.01)

        # Should have appropriate dimensions
        assert cms.width >= 100  # ceil(e / 0.01) = 272
        assert cms.depth >= 5  # ceil(ln(1/0.01)) = 5

    def test_epsilon_determines_width(self):
        """Smaller epsilon means larger width."""
        cms_coarse = CountMinSketch.from_error_rate(epsilon=0.1, delta=0.1)
        cms_fine = CountMinSketch.from_error_rate(epsilon=0.01, delta=0.1)

        assert cms_fine.width > cms_coarse.width

    def test_delta_determines_depth(self):
        """Smaller delta means larger depth."""
        cms_low_conf = CountMinSketch.from_error_rate(epsilon=0.1, delta=0.1)
        cms_high_conf = CountMinSketch.from_error_rate(epsilon=0.1, delta=0.001)

        assert cms_high_conf.depth > cms_low_conf.depth

    def test_rejects_invalid_epsilon(self):
        """Rejects epsilon outside (0, 1)."""
        with pytest.raises(ValueError, match="epsilon"):
            CountMinSketch.from_error_rate(epsilon=0.0, delta=0.1)
        with pytest.raises(ValueError, match="epsilon"):
            CountMinSketch.from_error_rate(epsilon=1.0, delta=0.1)
        with pytest.raises(ValueError, match="epsilon"):
            CountMinSketch.from_error_rate(epsilon=-0.1, delta=0.1)

    def test_rejects_invalid_delta(self):
        """Rejects delta outside (0, 1)."""
        with pytest.raises(ValueError, match="delta"):
            CountMinSketch.from_error_rate(epsilon=0.1, delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            CountMinSketch.from_error_rate(epsilon=0.1, delta=1.0)


class TestCountMinSketchAddAndEstimate:
    """Tests for adding items and estimating frequencies."""

    def test_add_single_item(self):
        """Adding a single item allows estimation."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)
        cms.add(42)

        assert cms.estimate(42) >= 1
        assert cms.item_count == 1

    def test_add_same_item_multiple_times(self):
        """Adding same item increases estimate."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)
        cms.add(42)
        cms.add(42)
        cms.add(42)

        assert cms.estimate(42) >= 3
        assert cms.item_count == 3

    def test_add_with_count(self):
        """Adding with count > 1 works correctly."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)
        cms.add(42, count=5)

        assert cms.estimate(42) >= 5
        assert cms.item_count == 5

    def test_add_zero_count_no_effect(self):
        """Adding with count=0 has no effect."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)
        cms.add(42, count=0)

        assert cms.item_count == 0

    def test_rejects_negative_count(self):
        """Rejects negative count."""
        cms = CountMinSketch[int](width=100, depth=5)
        with pytest.raises(ValueError, match="non-negative"):
            cms.add(42, count=-1)

    def test_estimate_never_underestimates(self):
        """Estimates are never below true count."""
        cms = CountMinSketch[int](width=1000, depth=10, seed=42)
        exact: Counter[int] = Counter()

        # Add items with varying counts
        for i in range(100):
            count = (i % 10) + 1
            cms.add(i, count=count)
            exact[i] = count

        # All estimates should be >= true count
        for item, true_count in exact.items():
            assert cms.estimate(item) >= true_count

    def test_estimate_unseen_item(self):
        """Estimating unseen item may return > 0 due to hash collisions."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)

        # Add some items
        for i in range(50):
            cms.add(i, count=10)

        # Unseen item estimate may be > 0 but is bounded
        estimate = cms.estimate(999)
        assert estimate >= 0
        # But with high probability, estimate is bounded by epsilon * N
        assert estimate <= int(cms.epsilon * cms.item_count) + 1


class TestCountMinSketchDeterminism:
    """Tests for deterministic behavior with seed."""

    def test_same_seed_same_estimates(self):
        """Same seed produces same estimates."""
        cms1 = CountMinSketch[int](width=100, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=100, depth=5, seed=42)

        for i in range(100):
            cms1.add(i)
            cms2.add(i)

        for i in range(100):
            assert cms1.estimate(i) == cms2.estimate(i)

    def test_different_seeds_different_hashing(self):
        """Different seeds use different hash functions."""
        cms1 = CountMinSketch[int](width=100, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=100, depth=5, seed=123)

        for i in range(100):
            cms1.add(i)
            cms2.add(i)

        # With different hashes, estimates may differ (especially for collisions)
        # But true counts should still be estimated correctly
        for i in range(100):
            assert cms1.estimate(i) >= 1
            assert cms2.estimate(i) >= 1


class TestCountMinSketchErrorBounds:
    """Tests for error bound properties."""

    def test_epsilon_calculation(self):
        """epsilon property returns e/width."""
        cms = CountMinSketch[int](width=272, depth=5)

        assert cms.epsilon == pytest.approx(math.e / 272)

    def test_delta_calculation(self):
        """delta property returns e^(-depth)."""
        cms = CountMinSketch[int](width=100, depth=5)

        assert cms.delta == pytest.approx(math.exp(-5))

    def test_estimate_with_error(self):
        """estimate_with_error returns FrequencyEstimate."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)
        cms.add(42, count=100)

        estimate = cms.estimate_with_error(42)

        assert isinstance(estimate, FrequencyEstimate)
        assert estimate.item == 42
        assert estimate.count >= 100
        # Error bound is epsilon * total_count
        expected_error = math.ceil(cms.epsilon * cms.item_count)
        assert estimate.error == expected_error


class TestCountMinSketchTop:
    """Tests for top() method (which raises NotImplementedError)."""

    def test_top_raises_not_implemented(self):
        """top() raises NotImplementedError."""
        cms = CountMinSketch[int](width=100, depth=5)
        cms.add(42)

        with pytest.raises(NotImplementedError, match="cannot enumerate"):
            cms.top(10)


class TestCountMinSketchInnerProduct:
    """Tests for inner product estimation."""

    def test_inner_product_identical_sketches(self):
        """Inner product of identical sketches equals sum of squares."""
        cms1 = CountMinSketch[int](width=200, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=200, depth=5, seed=42)

        for i in range(10):
            count = (i + 1) * 10
            cms1.add(i, count=count)
            cms2.add(i, count=count)

        # Inner product should approximate sum of count^2
        expected = sum((i + 1) * 10 * (i + 1) * 10 for i in range(10))
        actual = cms1.inner_product(cms2)

        # Allow some error due to hash collisions
        assert actual >= expected * 0.9
        assert actual <= expected * 1.2

    def test_inner_product_disjoint_sketches(self):
        """Inner product of disjoint sketches is small relative to overlap case."""
        cms1 = CountMinSketch[int](width=500, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=500, depth=5, seed=42)

        for i in range(100):
            cms1.add(i, count=10)
        for i in range(100, 200):
            cms2.add(i, count=10)

        # Inner product should be small (only collision noise)
        disjoint_product = cms1.inner_product(cms2)

        # Compare to overlapping case
        cms3 = CountMinSketch[int](width=500, depth=5, seed=42)
        cms4 = CountMinSketch[int](width=500, depth=5, seed=42)
        for i in range(100):
            cms3.add(i, count=10)
            cms4.add(i, count=10)
        overlap_product = cms3.inner_product(cms4)

        # Disjoint product should be much smaller than overlap product
        assert disjoint_product < overlap_product * 0.2

    def test_inner_product_rejects_different_dimensions(self):
        """inner_product rejects sketches with different dimensions."""
        cms1 = CountMinSketch[int](width=100, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=200, depth=5, seed=42)

        with pytest.raises(ValueError, match="dimensions differ"):
            cms1.inner_product(cms2)


class TestCountMinSketchMerge:
    """Tests for merging sketches."""

    def test_merge_combines_counts(self):
        """Merging combines counts correctly."""
        cms1 = CountMinSketch[int](width=100, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=100, depth=5, seed=42)

        cms1.add(1, count=10)
        cms2.add(1, count=5)
        cms2.add(2, count=3)

        cms1.merge(cms2)

        assert cms1.estimate(1) >= 15
        assert cms1.estimate(2) >= 3
        assert cms1.item_count == 18

    def test_merge_rejects_different_dimensions(self):
        """Cannot merge sketches with different dimensions."""
        cms1 = CountMinSketch[int](width=100, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=200, depth=5, seed=42)

        with pytest.raises(ValueError, match="dimensions differ"):
            cms1.merge(cms2)

    def test_merge_rejects_different_seeds(self):
        """Cannot merge sketches with different seeds."""
        cms1 = CountMinSketch[int](width=100, depth=5, seed=42)
        cms2 = CountMinSketch[int](width=100, depth=5, seed=123)

        with pytest.raises(ValueError, match="seeds differ"):
            cms1.merge(cms2)

    def test_merge_rejects_wrong_type(self):
        """Cannot merge with non-CountMinSketch."""
        cms = CountMinSketch[int](width=100, depth=5)

        with pytest.raises(TypeError, match="Can only merge"):
            cms.merge("not a cms")  # type: ignore


class TestCountMinSketchClear:
    """Tests for clear() method."""

    def test_clear_resets_state(self):
        """clear() resets all counters."""
        cms = CountMinSketch[int](width=100, depth=5, seed=42)
        cms.add(1, count=100)
        cms.add(2, count=50)

        cms.clear()

        assert cms.item_count == 0
        assert cms.estimate(1) == 0
        assert cms.estimate(2) == 0


class TestCountMinSketchMemory:
    """Tests for memory estimation."""

    def test_memory_bytes_proportional_to_dimensions(self):
        """memory_bytes is proportional to width * depth."""
        cms_small = CountMinSketch[int](width=100, depth=5)
        cms_large = CountMinSketch[int](width=1000, depth=10)

        # Large sketch should use more memory
        assert cms_large.memory_bytes > cms_small.memory_bytes

    def test_memory_bytes_constant_with_items(self):
        """memory_bytes doesn't grow with item count."""
        cms = CountMinSketch[int](width=100, depth=5)
        initial_size = cms.memory_bytes

        for i in range(10000):
            cms.add(i)

        # Memory should be same (sketch has fixed size)
        assert cms.memory_bytes == initial_size


class TestCountMinSketchRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self):
        """repr includes dimensions and error bounds."""
        cms = CountMinSketch[int](width=100, depth=5)
        cms.add(1, count=50)

        r = repr(cms)

        assert "width=100" in r
        assert "depth=5" in r
        assert "total=50" in r


class TestCountMinSketchWithZipfDistribution:
    """Integration tests with Zipf distribution."""

    def test_estimates_heavy_hitters_accurately(self):
        """Heavy hitters have accurate estimates."""
        cms = CountMinSketch[int](width=2000, depth=10, seed=42)
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)
        exact: Counter[int] = Counter()

        # Add 100k samples
        for _ in range(100_000):
            item = dist.sample()
            cms.add(item)
            exact[item] += 1

        # Check top 10 items have good estimates
        top_10 = exact.most_common(10)
        for item, true_count in top_10:
            estimate = cms.estimate(item)
            # Never underestimates
            assert estimate >= true_count
            # Within error bound with high probability
            error_bound = cms.epsilon * cms.item_count
            assert estimate <= true_count + error_bound * 2

    def test_error_within_theoretical_bounds(self):
        """Error is within theoretical epsilon * N with high probability."""
        cms = CountMinSketch.from_error_rate(epsilon=0.01, delta=0.01, seed=42)
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)
        exact: Counter[int] = Counter()

        for _ in range(10_000):
            item = dist.sample()
            cms.add(item)
            exact[item] += 1

        # Check error for all items
        error_bound = cms.epsilon * cms.item_count
        violations = 0
        for item in range(1000):
            true_count = exact.get(item, 0)
            estimate = cms.estimate(item)
            if estimate - true_count > error_bound:
                violations += 1

        # With delta=0.01, expect <= 1% violations
        # But be lenient for test stability
        assert violations / 1000 < 0.05, f"Too many violations: {violations}/1000"
