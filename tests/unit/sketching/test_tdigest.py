"""Tests for T-Digest quantile estimation."""

import random

import pytest

from happysimulator.sketching import TDigest


class TestTDigestCreation:
    """Tests for TDigest creation and configuration."""

    def test_creates_with_defaults(self):
        """TDigest is created with default compression."""
        td = TDigest()

        assert td.compression == 100.0
        assert td.item_count == 0

    def test_creates_with_custom_compression(self):
        """TDigest is created with custom compression."""
        td = TDigest(compression=200.0)

        assert td.compression == 200.0

    def test_creates_with_seed(self):
        """TDigest accepts seed parameter (for API consistency)."""
        td = TDigest(compression=100, seed=42)

        assert td.compression == 100

    def test_rejects_zero_compression(self):
        """Rejects compression=0."""
        with pytest.raises(ValueError, match="must be positive"):
            TDigest(compression=0)

    def test_rejects_negative_compression(self):
        """Rejects negative compression."""
        with pytest.raises(ValueError, match="must be positive"):
            TDigest(compression=-50)


class TestTDigestAdd:
    """Tests for adding values."""

    def test_add_single_value(self):
        """Adding a single value tracks it."""
        td = TDigest()
        td.add(42.0)

        assert td.item_count == 1
        assert td.min == 42.0
        assert td.max == 42.0

    def test_add_multiple_values(self):
        """Adding multiple values tracks them."""
        td = TDigest()
        td.add(10.0)
        td.add(20.0)
        td.add(30.0)

        assert td.item_count == 3
        assert td.min == 10.0
        assert td.max == 30.0

    def test_add_with_count(self):
        """Adding with count > 1 works correctly."""
        td = TDigest()
        td.add(42.0, count=5)

        assert td.item_count == 5

    def test_add_zero_count_no_effect(self):
        """Adding with count=0 has no effect."""
        td = TDigest()
        td.add(42.0, count=0)

        assert td.item_count == 0

    def test_rejects_negative_count(self):
        """Rejects negative count."""
        td = TDigest()
        with pytest.raises(ValueError, match="non-negative"):
            td.add(42.0, count=-1)


class TestTDigestQuantile:
    """Tests for quantile estimation."""

    def test_quantile_single_value(self):
        """Quantile of single value returns that value."""
        td = TDigest()
        td.add(42.0)

        assert td.quantile(0.0) == 42.0
        assert td.quantile(0.5) == 42.0
        assert td.quantile(1.0) == 42.0

    def test_quantile_two_values(self):
        """Quantile of two values interpolates."""
        td = TDigest()
        td.add(0.0)
        td.add(100.0)

        assert td.quantile(0.0) == 0.0
        assert td.quantile(1.0) == 100.0
        # Middle should be somewhere between
        p50 = td.quantile(0.5)
        assert 0.0 <= p50 <= 100.0

    def test_quantile_uniform_distribution(self):
        """Quantile estimates are reasonable for uniform data."""
        td = TDigest(compression=100)
        rng = random.Random(42)

        # Add 10000 uniform values in [0, 100]
        for _ in range(10000):
            td.add(rng.uniform(0, 100))

        # Check key quantiles
        p10 = td.quantile(0.10)
        p50 = td.quantile(0.50)
        p90 = td.quantile(0.90)

        # Should be roughly at expected positions
        assert 5 < p10 < 20, f"p10={p10} not in expected range"
        assert 40 < p50 < 60, f"p50={p50} not in expected range"
        assert 80 < p90 < 95, f"p90={p90} not in expected range"

    def test_quantile_rejects_invalid_q(self):
        """quantile rejects q outside [0, 1]."""
        td = TDigest()
        td.add(42.0)

        with pytest.raises(ValueError, match="must be in"):
            td.quantile(-0.1)

        with pytest.raises(ValueError, match="must be in"):
            td.quantile(1.1)

    def test_quantile_empty_raises(self):
        """quantile raises on empty digest."""
        td = TDigest()

        with pytest.raises(ValueError, match="empty"):
            td.quantile(0.5)


class TestTDigestPercentile:
    """Tests for percentile convenience method."""

    def test_percentile_matches_quantile(self):
        """percentile(p) equals quantile(p/100)."""
        td = TDigest()
        for i in range(100):
            td.add(float(i))

        assert td.percentile(50) == td.quantile(0.5)
        assert td.percentile(95) == td.quantile(0.95)
        assert td.percentile(99) == td.quantile(0.99)

    def test_percentile_rejects_invalid_p(self):
        """percentile rejects p outside [0, 100]."""
        td = TDigest()
        td.add(42.0)

        with pytest.raises(ValueError, match="must be in"):
            td.percentile(-1)

        with pytest.raises(ValueError, match="must be in"):
            td.percentile(101)


class TestTDigestCDF:
    """Tests for CDF estimation."""

    def test_cdf_boundaries(self):
        """CDF returns 0 below min, 1 above max."""
        td = TDigest()
        for i in range(100):
            td.add(float(i))

        assert td.cdf(-1) == 0.0
        assert td.cdf(100) == 1.0

    def test_cdf_monotonic(self):
        """CDF is monotonically increasing."""
        td = TDigest()
        rng = random.Random(42)
        for _ in range(1000):
            td.add(rng.uniform(0, 100))

        prev_cdf = 0.0
        for x in range(0, 100, 5):
            current_cdf = td.cdf(float(x))
            assert current_cdf >= prev_cdf
            prev_cdf = current_cdf


class TestTDigestMerge:
    """Tests for merging T-Digests."""

    def test_merge_combines_data(self):
        """Merging combines data from both digests."""
        td1 = TDigest()
        td2 = TDigest()

        for i in range(100):
            td1.add(float(i))
        for i in range(100, 200):
            td2.add(float(i))

        td1.merge(td2)

        assert td1.item_count == 200
        assert td1.min == 0.0
        assert td1.max == 199.0

    def test_merge_preserves_quantiles(self):
        """Merged digest has reasonable quantiles."""
        td1 = TDigest()
        td2 = TDigest()

        # First digest: [0, 50)
        for i in range(50):
            td1.add(float(i))
        # Second digest: [50, 100)
        for i in range(50, 100):
            td2.add(float(i))

        td1.merge(td2)

        # Median should be around 50
        p50 = td1.quantile(0.5)
        assert 40 < p50 < 60

    def test_merge_rejects_wrong_type(self):
        """Cannot merge with non-TDigest."""
        td = TDigest()

        with pytest.raises(TypeError, match="Can only merge"):
            td.merge("not a tdigest")  # type: ignore


class TestTDigestClear:
    """Tests for clear() method."""

    def test_clear_resets_state(self):
        """clear() resets all state."""
        td = TDigest()
        for i in range(100):
            td.add(float(i))

        td.clear()

        assert td.item_count == 0
        assert td.min is None
        assert td.max is None


class TestTDigestMemory:
    """Tests for memory and centroid count."""

    def test_centroid_count_bounded(self):
        """Centroid count is bounded by compression."""
        td = TDigest(compression=50)

        for i in range(10000):
            td.add(float(i))

        # Centroid count should be O(compression)
        assert td.centroid_count < td.compression * 3

    def test_memory_bytes_increases_with_compression(self):
        """Higher compression uses more memory."""
        td_small = TDigest(compression=50)
        td_large = TDigest(compression=200)

        for i in range(1000):
            td_small.add(float(i))
            td_large.add(float(i))

        assert td_large.memory_bytes > td_small.memory_bytes


class TestTDigestRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self):
        """repr includes compression and total."""
        td = TDigest(compression=100)
        td.add(1.0)
        td.add(2.0)

        r = repr(td)

        assert "compression=100" in r
        assert "total=2" in r


class TestTDigestAccuracy:
    """Tests for quantile estimation accuracy."""

    def test_tail_accuracy(self):
        """T-Digest provides reasonable accuracy at tails."""
        td = TDigest(compression=200)  # Higher compression for better accuracy
        rng = random.Random(42)

        # Add 10000 exponential values (common for latencies)
        values = [rng.expovariate(1.0) for _ in range(10000)]
        for v in values:
            td.add(v)

        # Sort for exact percentiles
        values.sort()
        exact_p99 = values[9899]
        exact_p999 = values[9989]

        estimated_p99 = td.percentile(99)
        estimated_p999 = td.percentile(99.9)

        # Should be in the right ballpark - this simplified implementation
        # provides reasonable estimates but not as tight as production T-Digest
        assert estimated_p99 > exact_p99 * 0.5, f"p99 too low: {estimated_p99} vs {exact_p99}"
        assert estimated_p99 < exact_p99 * 2.0, f"p99 too high: {estimated_p99} vs {exact_p99}"
        assert estimated_p999 > exact_p999 * 0.5, f"p999 too low: {estimated_p999} vs {exact_p999}"
        assert estimated_p999 < exact_p999 * 2.0, f"p999 too high: {estimated_p999} vs {exact_p999}"
