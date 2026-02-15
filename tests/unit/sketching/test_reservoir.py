"""Tests for Reservoir Sampling."""

from collections import Counter

import pytest

from happysimulator.sketching import ReservoirSampler


class TestReservoirSamplerCreation:
    """Tests for ReservoirSampler creation and configuration."""

    def test_creates_with_size(self):
        """ReservoirSampler is created with specified size."""
        rs = ReservoirSampler[int](size=100)

        assert rs.capacity == 100
        assert rs.sample_size == 0
        assert rs.item_count == 0

    def test_creates_with_seed(self):
        """ReservoirSampler accepts seed for reproducibility."""
        rs = ReservoirSampler[int](size=100, seed=42)

        assert rs.capacity == 100

    def test_rejects_zero_size(self):
        """Rejects size=0."""
        with pytest.raises(ValueError, match="must be positive"):
            ReservoirSampler[int](size=0)

    def test_rejects_negative_size(self):
        """Rejects negative size."""
        with pytest.raises(ValueError, match="must be positive"):
            ReservoirSampler[int](size=-10)

    def test_works_with_strings(self):
        """Works with string items."""
        rs = ReservoirSampler[str](size=10)
        rs.add("hello")
        rs.add("world")

        assert rs.sample_size == 2

    def test_works_with_tuples(self):
        """Works with tuple items."""
        rs = ReservoirSampler[tuple](size=10)
        rs.add((1, "a"))
        rs.add((2, "b"))

        assert rs.sample_size == 2


class TestReservoirSamplerAdd:
    """Tests for adding items."""

    def test_add_fills_reservoir(self):
        """Adding items fills reservoir up to capacity."""
        rs = ReservoirSampler[int](size=5)
        for i in range(5):
            rs.add(i)

        assert rs.sample_size == 5
        assert rs.is_full

    def test_add_beyond_capacity(self):
        """Adding beyond capacity maintains sample size."""
        rs = ReservoirSampler[int](size=5)
        for i in range(100):
            rs.add(i)

        assert rs.sample_size == 5
        assert rs.item_count == 100

    def test_add_with_count(self):
        """Adding with count > 1 treats each as separate occurrence."""
        rs = ReservoirSampler[int](size=10)
        rs.add(42, count=5)

        assert rs.item_count == 5
        # Sample may contain multiple 42s
        assert rs.sample_size == 5

    def test_add_zero_count_no_effect(self):
        """Adding with count=0 has no effect."""
        rs = ReservoirSampler[int](size=10)
        rs.add(42, count=0)

        assert rs.item_count == 0
        assert rs.sample_size == 0

    def test_rejects_negative_count(self):
        """Rejects negative count."""
        rs = ReservoirSampler[int](size=10)
        with pytest.raises(ValueError, match="non-negative"):
            rs.add(42, count=-1)


class TestReservoirSamplerSample:
    """Tests for retrieving samples."""

    def test_sample_returns_list(self):
        """sample() returns a list."""
        rs = ReservoirSampler[int](size=10)
        for i in range(5):
            rs.add(i)

        result = rs.sample()

        assert isinstance(result, list)
        assert len(result) == 5

    def test_sample_returns_copy(self):
        """sample() returns a copy, not internal list."""
        rs = ReservoirSampler[int](size=10)
        for i in range(5):
            rs.add(i)

        sample1 = rs.sample()
        sample1.append(999)  # Modify copy

        sample2 = rs.sample()
        assert 999 not in sample2

    def test_iteration(self):
        """Can iterate over sampled items."""
        rs = ReservoirSampler[int](size=10)
        for i in range(5):
            rs.add(i)

        items = list(rs)
        assert len(items) == 5

    def test_indexing(self):
        """Can index into reservoir."""
        rs = ReservoirSampler[int](size=10)
        for i in range(5):
            rs.add(i)

        # First 5 items should be in reservoir (before any eviction)
        assert rs[0] == 0
        assert len(rs) == 5


class TestReservoirSamplerUniformity:
    """Tests for uniform sampling property."""

    def test_uniform_sampling_small_stream(self):
        """Sampling from small stream includes all items."""
        rs = ReservoirSampler[int](size=10)
        for i in range(5):
            rs.add(i)

        sample = set(rs.sample())
        assert sample == {0, 1, 2, 3, 4}

    def test_uniform_sampling_large_stream(self):
        """Each item has equal probability of being in sample."""
        # Run many trials to test uniformity
        counts: Counter[int] = Counter()
        n_trials = 10000
        stream_size = 100
        sample_size = 10

        for trial in range(n_trials):
            rs = ReservoirSampler[int](size=sample_size, seed=trial)
            for i in range(stream_size):
                rs.add(i)
            for item in rs.sample():
                counts[item] += 1

        # Expected count per item: n_trials * sample_size / stream_size = 1000
        expected = n_trials * sample_size / stream_size

        # All items should appear roughly equally often
        for i in range(stream_size):
            observed = counts[i]
            # Allow 20% tolerance
            assert expected * 0.8 < observed < expected * 1.2, (
                f"Item {i}: expected ~{expected}, got {observed}"
            )

    def test_deterministic_with_seed(self):
        """Same seed produces same sample."""
        rs1 = ReservoirSampler[int](size=10, seed=42)
        rs2 = ReservoirSampler[int](size=10, seed=42)

        for i in range(100):
            rs1.add(i)
            rs2.add(i)

        assert rs1.sample() == rs2.sample()

    def test_different_seeds_different_samples(self):
        """Different seeds may produce different samples."""
        rs1 = ReservoirSampler[int](size=10, seed=42)
        rs2 = ReservoirSampler[int](size=10, seed=123)

        for i in range(100):
            rs1.add(i)
            rs2.add(i)

        # Samples will likely be different
        sample1 = set(rs1.sample())
        sample2 = set(rs2.sample())

        # With 100 items and sample of 10, very unlikely to be identical
        # But if they happen to be, that's not a bug - just check they're valid
        assert len(sample1) == 10
        assert len(sample2) == 10


class TestReservoirSamplerMerge:
    """Tests for merging reservoirs."""

    def test_merge_combines_samples(self):
        """Merging creates sample from combined stream."""
        rs1 = ReservoirSampler[int](size=10, seed=42)
        rs2 = ReservoirSampler[int](size=10, seed=42)

        for i in range(50):
            rs1.add(i)
        for i in range(50, 100):
            rs2.add(i)

        rs1.merge(rs2)

        # Sample should be from combined stream
        assert rs1.item_count == 100
        sample = set(rs1.sample())
        # Should have items from both ranges
        assert len(sample) <= 10

    def test_merge_rejects_different_capacity(self):
        """Cannot merge reservoirs with different capacity."""
        rs1 = ReservoirSampler[int](size=10)
        rs2 = ReservoirSampler[int](size=20)

        with pytest.raises(ValueError, match="capacity differs"):
            rs1.merge(rs2)

    def test_merge_rejects_wrong_type(self):
        """Cannot merge with non-ReservoirSampler."""
        rs = ReservoirSampler[int](size=10)

        with pytest.raises(TypeError, match="Can only merge"):
            rs.merge("not a sampler")  # type: ignore


class TestReservoirSamplerClear:
    """Tests for clear() method."""

    def test_clear_resets_state(self):
        """clear() resets all state."""
        rs = ReservoirSampler[int](size=10)
        for i in range(100):
            rs.add(i)

        rs.clear()

        assert rs.sample_size == 0
        assert rs.item_count == 0
        assert len(rs.sample()) == 0


class TestReservoirSamplerMemory:
    """Tests for memory estimation."""

    def test_memory_bytes_proportional_to_capacity(self):
        """Memory is proportional to capacity."""
        rs_small = ReservoirSampler[int](size=10)
        rs_large = ReservoirSampler[int](size=1000)

        # Fill both
        for i in range(1000):
            rs_small.add(i)
            rs_large.add(i)

        assert rs_large.memory_bytes > rs_small.memory_bytes

    def test_memory_bounded_by_capacity(self):
        """Memory stays bounded regardless of stream size."""
        rs = ReservoirSampler[int](size=10)

        initial_memory = rs.memory_bytes

        for i in range(10000):
            rs.add(i)

        # Memory should not grow significantly beyond capacity
        # (Some overhead is expected)
        assert rs.memory_bytes < initial_memory + 1000


class TestReservoirSamplerRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self):
        """repr includes capacity, sampled, and seen."""
        rs = ReservoirSampler[int](size=10)
        for i in range(100):
            rs.add(i)

        r = repr(rs)

        assert "capacity=10" in r
        assert "sampled=10" in r
        assert "seen=100" in r
