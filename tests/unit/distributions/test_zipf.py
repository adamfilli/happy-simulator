"""Tests for ZipfDistribution."""

from collections import Counter

import pytest

from happysimulator.distributions import UniformDistribution, ZipfDistribution


class TestZipfDistributionCreation:
    """Tests for ZipfDistribution creation."""

    def test_creates_with_defaults(self):
        """ZipfDistribution is created with default s=1.0."""
        dist = ZipfDistribution(range(100))

        assert dist.size == 100
        assert dist.s == 1.0

    def test_creates_with_custom_s(self):
        """ZipfDistribution is created with custom s value."""
        dist = ZipfDistribution(range(50), s=1.5)

        assert dist.size == 50
        assert dist.s == 1.5

    def test_creates_with_seed(self):
        """ZipfDistribution is created with seed for reproducibility."""
        dist = ZipfDistribution(range(100), s=1.0, seed=42)

        assert dist.size == 100

    def test_rejects_empty_values(self):
        """Rejects empty values sequence."""
        with pytest.raises(ValueError, match="must not be empty"):
            ZipfDistribution([])

    def test_rejects_negative_s(self):
        """Rejects negative s value."""
        with pytest.raises(ValueError, match="non-negative"):
            ZipfDistribution(range(10), s=-0.5)

    def test_accepts_s_zero(self):
        """Accepts s=0 (uniform distribution)."""
        dist = ZipfDistribution(range(10), s=0.0)

        assert dist.s == 0.0

    def test_works_with_strings(self):
        """Works with string values."""
        dist = ZipfDistribution(["hot", "warm", "cool", "cold"], s=1.0)

        assert dist.size == 4
        assert "hot" in dist.population

    def test_works_with_tuples(self):
        """Works with tuple values."""
        values = [(1, "a"), (2, "b"), (3, "c")]
        dist = ZipfDistribution(values, s=1.0)

        assert dist.size == 3


class TestZipfDistributionSampling:
    """Tests for ZipfDistribution sampling."""

    def test_sample_returns_value_from_population(self):
        """Sample returns a value from the population."""
        dist = ZipfDistribution(range(100), s=1.0, seed=42)

        sample = dist.sample()

        assert sample in range(100)

    def test_sample_n_returns_correct_count(self):
        """sample_n returns the requested number of samples."""
        dist = ZipfDistribution(range(100), s=1.0, seed=42)

        samples = dist.sample_n(50)

        assert len(samples) == 50
        assert all(s in range(100) for s in samples)

    def test_deterministic_with_seed(self):
        """Same seed produces same samples."""
        dist1 = ZipfDistribution(range(100), s=1.0, seed=42)
        dist2 = ZipfDistribution(range(100), s=1.0, seed=42)

        samples1 = dist1.sample_n(100)
        samples2 = dist2.sample_n(100)

        assert samples1 == samples2

    def test_different_seeds_produce_different_samples(self):
        """Different seeds produce different samples."""
        dist1 = ZipfDistribution(range(100), s=1.0, seed=42)
        dist2 = ZipfDistribution(range(100), s=1.0, seed=123)

        samples1 = dist1.sample_n(100)
        samples2 = dist2.sample_n(100)

        assert samples1 != samples2


class TestZipfDistributionCharacteristics:
    """Tests for Zipf distribution statistical characteristics."""

    def test_s_zero_is_approximately_uniform(self):
        """With s=0, distribution should be approximately uniform."""
        dist = ZipfDistribution(range(10), s=0.0, seed=42)
        samples = dist.sample_n(10000)
        counts = Counter(samples)

        # Each value should appear ~1000 times (±15% for statistical tolerance)
        for v in range(10):
            assert 850 < counts[v] < 1150, f"Value {v} count {counts[v]} outside expected range"

    def test_s_one_classic_zipf_ratio(self):
        """With s=1, rank 1 should appear ~2x as often as rank 2."""
        dist = ZipfDistribution(range(100), s=1.0, seed=42)
        samples = dist.sample_n(100000)
        counts = Counter(samples)

        # Rank 1 (value 0) vs rank 2 (value 1)
        ratio = counts[0] / counts[1]
        # Should be ~2.0, allow 1.7-2.3 for statistical variance
        assert 1.7 < ratio < 2.3, f"Rank 1/2 ratio {ratio:.2f} not close to 2.0"

    def test_higher_s_more_skewed(self):
        """Higher s should concentrate more on top values."""
        dist_mild = ZipfDistribution(range(100), s=0.5, seed=42)
        dist_extreme = ZipfDistribution(range(100), s=2.0, seed=42)

        mild_samples = dist_mild.sample_n(10000)
        extreme_samples = dist_extreme.sample_n(10000)

        mild_top10 = sum(1 for s in mild_samples if s < 10) / 10000
        extreme_top10 = sum(1 for s in extreme_samples if s < 10) / 10000

        assert extreme_top10 > mild_top10, (
            f"s=2.0 top10% ({extreme_top10:.2f}) should be > s=0.5 top10% ({mild_top10:.2f})"
        )

    def test_top_n_probability_increases_with_n(self):
        """top_n_probability should increase with n."""
        dist = ZipfDistribution(range(100), s=1.0)

        top_5 = dist.top_n_probability(5)
        top_10 = dist.top_n_probability(10)
        top_20 = dist.top_n_probability(20)

        assert top_5 < top_10 < top_20

    def test_top_n_probability_sums_to_one(self):
        """top_n_probability(n) for all n should equal 1.0."""
        dist = ZipfDistribution(range(50), s=1.0)

        total = dist.top_n_probability(50)

        assert total == pytest.approx(1.0)


class TestZipfDistributionProbabilities:
    """Tests for probability calculations."""

    def test_probability_rank_one_highest(self):
        """Rank 1 should have highest probability."""
        dist = ZipfDistribution(range(100), s=1.0)

        p1 = dist.probability(1)
        p2 = dist.probability(2)
        p10 = dist.probability(10)

        assert p1 > p2 > p10

    def test_probability_sums_to_one(self):
        """Probabilities for all ranks should sum to 1."""
        dist = ZipfDistribution(range(50), s=1.0)

        total = sum(dist.probability(r) for r in range(1, 51))

        assert total == pytest.approx(1.0)

    def test_probability_invalid_rank(self):
        """probability() raises for invalid rank."""
        dist = ZipfDistribution(range(10), s=1.0)

        with pytest.raises(ValueError):
            dist.probability(0)  # Ranks are 1-indexed

        with pytest.raises(ValueError):
            dist.probability(11)  # Out of range

    def test_probability_for_value(self):
        """Can get probability for specific value."""
        dist = ZipfDistribution(["a", "b", "c"], s=1.0)

        p_a = dist.probability_for_value("a")
        p_b = dist.probability_for_value("b")

        assert p_a > p_b
        assert p_a > 0

    def test_probability_for_value_not_found(self):
        """probability_for_value raises for unknown value."""
        dist = ZipfDistribution(["a", "b", "c"], s=1.0)

        with pytest.raises(ValueError, match="not in population"):
            dist.probability_for_value("z")

    def test_expected_frequency(self):
        """expected_frequency returns probability * n_samples."""
        dist = ZipfDistribution(range(100), s=1.0)

        expected = dist.expected_frequency(1, 10000)
        probability = dist.probability(1)

        assert expected == pytest.approx(probability * 10000)


class TestZipfDistributionProperties:
    """Tests for property accessors."""

    def test_population_returns_copy(self):
        """population property returns the values."""
        original = [1, 2, 3, 4, 5]
        dist = ZipfDistribution(original, s=1.0)

        pop = dist.population

        assert list(pop) == original

    def test_size_matches_population(self):
        """size matches population length."""
        dist = ZipfDistribution(range(42), s=1.0)

        assert dist.size == 42
        assert dist.size == len(dist.population)

    def test_repr(self):
        """repr shows size and s parameter."""
        dist = ZipfDistribution(range(100), s=1.5)

        r = repr(dist)

        assert "100" in r
        assert "1.5" in r


class TestUniformDistributionCreation:
    """Tests for UniformDistribution creation."""

    def test_creates_with_values(self):
        """UniformDistribution is created with values."""
        dist = UniformDistribution(range(100))

        assert dist.size == 100

    def test_creates_with_seed(self):
        """UniformDistribution is created with seed."""
        dist = UniformDistribution(range(100), seed=42)

        assert dist.size == 100

    def test_rejects_empty_values(self):
        """Rejects empty values sequence."""
        with pytest.raises(ValueError, match="must not be empty"):
            UniformDistribution([])

    def test_works_with_strings(self):
        """Works with string values."""
        dist = UniformDistribution(["a", "b", "c"])

        assert dist.size == 3


class TestUniformDistributionSampling:
    """Tests for UniformDistribution sampling."""

    def test_sample_returns_value_from_population(self):
        """Sample returns a value from the population."""
        dist = UniformDistribution(range(100), seed=42)

        sample = dist.sample()

        assert sample in range(100)

    def test_sample_n_returns_correct_count(self):
        """sample_n returns the requested number of samples."""
        dist = UniformDistribution(range(100), seed=42)

        samples = dist.sample_n(50)

        assert len(samples) == 50

    def test_deterministic_with_seed(self):
        """Same seed produces same samples."""
        dist1 = UniformDistribution(range(100), seed=42)
        dist2 = UniformDistribution(range(100), seed=42)

        assert dist1.sample_n(100) == dist2.sample_n(100)

    def test_approximately_uniform(self):
        """Sampling should be approximately uniform."""
        dist = UniformDistribution(range(10), seed=42)
        samples = dist.sample_n(10000)
        counts = Counter(samples)

        # Each value should appear ~1000 times (±15%)
        for v in range(10):
            assert 850 < counts[v] < 1150

    def test_probability(self):
        """probability returns 1/n."""
        dist = UniformDistribution(range(100))

        assert dist.probability() == pytest.approx(0.01)

    def test_repr(self):
        """repr shows size."""
        dist = UniformDistribution(range(50))

        r = repr(dist)

        assert "50" in r
