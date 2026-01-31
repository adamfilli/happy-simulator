"""Tests for PercentileFittedLatency distribution."""

import math
import random

import pytest

from happysimulator.core.temporal import Instant
from happysimulator.distributions import PercentileFittedLatency


class TestPercentileFittedConstruction:
    """Test construction and validation of PercentileFittedLatency."""

    def test_requires_at_least_one_percentile(self):
        """Raises ValueError when no percentiles provided."""
        with pytest.raises(ValueError, match="At least one percentile must be provided"):
            PercentileFittedLatency()

    def test_constructs_with_single_p50(self):
        """Can construct with only p50."""
        dist = PercentileFittedLatency(p50=0.1)
        assert dist._lambda > 0
        assert dist._mean_latency > 0

    def test_constructs_with_single_p99(self):
        """Can construct with only p99."""
        dist = PercentileFittedLatency(p99=0.5)
        assert dist._lambda > 0

    def test_constructs_with_all_percentiles(self):
        """Can construct with all percentiles provided."""
        dist = PercentileFittedLatency(
            p50=0.07,
            p90=0.23,
            p99=0.46,
            p999=0.69,
            p9999=0.92,
        )
        assert dist._lambda > 0

    def test_single_percentile_exact_fit(self):
        """Single percentile should be matched exactly."""
        target_p50 = 0.1
        dist = PercentileFittedLatency(p50=target_p50)

        fitted_p50 = dist.get_percentile(0.50).to_seconds()
        assert abs(fitted_p50 - target_p50) < 1e-9

    def test_get_percentile_method(self):
        """get_percentile returns correct values for fitted distribution."""
        dist = PercentileFittedLatency(p50=0.1)

        # For exponential: Q(p) = -ln(1-p) / λ
        # p50 should be exactly 0.1 (what we fitted to)
        assert abs(dist.get_percentile(0.50).to_seconds() - 0.1) < 1e-9

        # p99 should be higher than p50
        assert dist.get_percentile(0.99).to_seconds() > dist.get_percentile(0.50).to_seconds()

        # p10 should be lower than p50
        assert dist.get_percentile(0.10).to_seconds() < dist.get_percentile(0.50).to_seconds()


class TestPercentileFittedSampling:
    """Test that sampled values match expected percentile distribution."""

    @pytest.fixture(autouse=True)
    def set_seed(self):
        """Set random seed for reproducible tests."""
        random.seed(42)

    def test_single_p50_sampling(self):
        """Samples from p50-fitted distribution have correct median."""
        target_p50 = 0.1
        dist = PercentileFittedLatency(p50=target_p50)

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(10000)
        ]
        samples.sort()

        observed_p50 = samples[len(samples) // 2]

        # Allow 5% relative tolerance for statistical variation
        assert abs(observed_p50 - target_p50) / target_p50 < 0.05

    def test_single_p99_sampling(self):
        """Samples from p99-fitted distribution have correct 99th percentile."""
        target_p99 = 0.5
        dist = PercentileFittedLatency(p99=target_p99)

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(10000)
        ]
        samples.sort()

        observed_p99 = samples[int(len(samples) * 0.99)]

        # Allow 10% relative tolerance (tail percentiles have more variance)
        assert abs(observed_p99 - target_p99) / target_p99 < 0.10

    def test_multiple_percentiles_sampling(self):
        """Samples from multi-percentile fit approximate all targets."""
        # Use percentiles that are consistent with exponential distribution
        # For exponential with mean=1: p50=0.693, p90=2.303, p99=4.605
        # Scale by 0.1 to get reasonable latencies
        scale = 0.1
        target_p50 = 0.693 * scale
        target_p90 = 2.303 * scale
        target_p99 = 4.605 * scale

        dist = PercentileFittedLatency(
            p50=target_p50,
            p90=target_p90,
            p99=target_p99,
        )

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(20000)
        ]
        samples.sort()

        observed_p50 = samples[int(len(samples) * 0.50)]
        observed_p90 = samples[int(len(samples) * 0.90)]
        observed_p99 = samples[int(len(samples) * 0.99)]

        # All should be close since targets are consistent with exponential
        assert abs(observed_p50 - target_p50) / target_p50 < 0.05
        assert abs(observed_p90 - target_p90) / target_p90 < 0.08
        assert abs(observed_p99 - target_p99) / target_p99 < 0.15

    def test_mean_matches_expected(self):
        """Sampled mean should match 1/lambda."""
        dist = PercentileFittedLatency(p50=0.1)
        expected_mean = dist._mean_latency

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(10000)
        ]

        observed_mean = sum(samples) / len(samples)

        # Allow 5% tolerance
        assert abs(observed_mean - expected_mean) / expected_mean < 0.05

    def test_all_samples_positive(self):
        """All sampled latencies should be positive."""
        dist = PercentileFittedLatency(p50=0.1)

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(1000)
        ]

        assert all(s > 0 for s in samples)


class TestPercentileFittedWithInconsistentTargets:
    """Test behavior when percentile targets don't match exponential exactly."""

    @pytest.fixture(autouse=True)
    def set_seed(self):
        """Set random seed for reproducible tests."""
        random.seed(123)

    def test_inconsistent_percentiles_finds_compromise(self):
        """When targets don't match exponential, fitting finds a compromise."""
        # These targets are NOT consistent with any exponential distribution
        # (p99 is too close to p50 for exponential)
        dist = PercentileFittedLatency(
            p50=0.1,
            p99=0.2,  # Would be ~0.66 for true exponential
        )

        # The fit should still produce a valid distribution
        assert dist._lambda > 0

        # Sample and verify we get a reasonable distribution
        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(10000)
        ]
        samples.sort()

        # The fitted values won't match targets exactly, but should be reasonable
        fitted_p50 = dist.get_percentile(0.50).to_seconds()
        fitted_p99 = dist.get_percentile(0.99).to_seconds()

        # Fitted p50 will be lower than target (compromise)
        # Fitted p99 will be higher than target (compromise)
        assert fitted_p50 < 0.1
        assert fitted_p99 > 0.2

    def test_heavily_weighted_toward_high_percentiles(self):
        """Larger percentile values have more influence in least-squares fit."""
        # p999 value is much larger, so it will dominate the fit
        dist = PercentileFittedLatency(
            p50=0.01,
            p999=1.0,
        )

        # The p999 should be closer to target than p50
        fitted_p50 = dist.get_percentile(0.50).to_seconds()
        fitted_p999 = dist.get_percentile(0.999).to_seconds()

        p50_error = abs(fitted_p50 - 0.01) / 0.01
        p999_error = abs(fitted_p999 - 1.0) / 1.0

        # p999 should have lower relative error due to weighting
        assert p999_error < p50_error


class TestPercentileFittedArithmetic:
    """Test arithmetic operations inherited from LatencyDistribution."""

    def test_add_increases_mean(self):
        """Adding to distribution increases mean latency."""
        dist = PercentileFittedLatency(p50=0.1)
        original_mean = dist._mean_latency

        new_dist = dist + 0.05

        assert new_dist._mean_latency == original_mean + 0.05
        assert dist._mean_latency == original_mean  # Original unchanged

    def test_subtract_decreases_mean(self):
        """Subtracting from distribution decreases mean latency."""
        dist = PercentileFittedLatency(p50=0.1)
        original_mean = dist._mean_latency

        new_dist = dist - 0.02

        assert new_dist._mean_latency == original_mean - 0.02
        assert dist._mean_latency == original_mean  # Original unchanged


class TestPercentileFittedStatisticalProperties:
    """Test statistical properties of the fitted exponential distribution."""

    @pytest.fixture(autouse=True)
    def set_seed(self):
        """Set random seed for reproducible tests."""
        random.seed(999)

    def test_coefficient_of_variation_is_one(self):
        """Exponential distribution has CV = 1 (std dev equals mean)."""
        dist = PercentileFittedLatency(p50=0.1)

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(20000)
        ]

        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean

        # CV should be close to 1.0 for exponential
        assert abs(cv - 1.0) < 0.05

    def test_memoryless_property_approximation(self):
        """Samples above median should follow same relative distribution."""
        dist = PercentileFittedLatency(p50=0.1)

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(20000)
        ]

        median = sorted(samples)[len(samples) // 2]

        # Filter samples above median, then subtract median
        # These should follow same exponential distribution
        above_median = [s - median for s in samples if s > median]
        above_median.sort()

        # The median of (samples - median | samples > median) should be close to original median
        # due to memoryless property
        conditional_median = above_median[len(above_median) // 2]

        # Allow 10% tolerance
        assert abs(conditional_median - median) / median < 0.10

    def test_large_sample_percentile_accuracy(self):
        """With many samples, observed percentiles converge to theoretical."""
        dist = PercentileFittedLatency(p90=0.5)

        samples = [
            dist.get_latency(Instant.Epoch).to_seconds()
            for _ in range(50000)
        ]
        samples.sort()

        # Check multiple percentiles
        percentiles_to_check = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        for p in percentiles_to_check:
            theoretical = dist.get_percentile(p).to_seconds()
            observed = samples[int(len(samples) * p)]

            # Tighter tolerance with more samples
            tolerance = 0.03 if p < 0.95 else 0.08
            relative_error = abs(observed - theoretical) / theoretical

            assert relative_error < tolerance, (
                f"p{int(p*100)}: theoretical={theoretical:.4f}, "
                f"observed={observed:.4f}, error={relative_error:.2%}"
            )
