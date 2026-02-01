"""Tests for Bloom Filter membership testing."""

import random

import pytest

from happysimulator.sketching import BloomFilter


class TestBloomFilterCreation:
    """Tests for BloomFilter creation and configuration."""

    def test_creates_with_size_and_hashes(self):
        """BloomFilter is created with specified parameters."""
        bf = BloomFilter[int](size_bits=1000, num_hashes=5)

        assert bf.size_bits == 1000
        assert bf.num_hashes == 5

    def test_creates_with_default_hashes(self):
        """BloomFilter uses default num_hashes if not specified."""
        bf = BloomFilter[int](size_bits=1000)

        assert bf.size_bits == 1000
        assert bf.num_hashes == 7  # Default

    def test_creates_with_seed(self):
        """BloomFilter accepts seed for reproducibility."""
        bf = BloomFilter[int](size_bits=1000, seed=42)

        assert bf.size_bits == 1000

    def test_rejects_zero_size(self):
        """Rejects size_bits=0."""
        with pytest.raises(ValueError, match="must be positive"):
            BloomFilter[int](size_bits=0)

    def test_rejects_negative_size(self):
        """Rejects negative size_bits."""
        with pytest.raises(ValueError, match="must be positive"):
            BloomFilter[int](size_bits=-100)

    def test_rejects_zero_hashes(self):
        """Rejects num_hashes=0."""
        with pytest.raises(ValueError, match="must be positive"):
            BloomFilter[int](size_bits=1000, num_hashes=0)

    def test_works_with_strings(self):
        """Works with string items."""
        bf = BloomFilter[str](size_bits=1000)
        bf.add("hello")
        bf.add("world")

        assert bf.contains("hello")
        assert bf.contains("world")

    def test_works_with_tuples(self):
        """Works with tuple items."""
        bf = BloomFilter[tuple](size_bits=1000)
        bf.add((1, "a"))

        assert bf.contains((1, "a"))


class TestBloomFilterFromExpectedItems:
    """Tests for from_expected_items factory method."""

    def test_creates_with_expected_items(self):
        """Creates filter sized for expected items."""
        bf = BloomFilter.from_expected_items(n=1000, fp_rate=0.01)

        # Should have appropriate size
        assert bf.size_bits > 1000  # Needs more bits than items
        assert bf.num_hashes >= 1

    def test_smaller_fp_rate_means_larger_filter(self):
        """Smaller fp_rate requires larger filter."""
        bf_coarse = BloomFilter.from_expected_items(n=1000, fp_rate=0.1)
        bf_fine = BloomFilter.from_expected_items(n=1000, fp_rate=0.01)

        assert bf_fine.size_bits > bf_coarse.size_bits

    def test_more_items_means_larger_filter(self):
        """More expected items requires larger filter."""
        bf_small = BloomFilter.from_expected_items(n=100, fp_rate=0.01)
        bf_large = BloomFilter.from_expected_items(n=10000, fp_rate=0.01)

        assert bf_large.size_bits > bf_small.size_bits

    def test_rejects_invalid_n(self):
        """Rejects negative n."""
        with pytest.raises(ValueError, match="non-negative"):
            BloomFilter.from_expected_items(n=-1, fp_rate=0.01)

    def test_rejects_invalid_fp_rate(self):
        """Rejects fp_rate outside (0, 1)."""
        with pytest.raises(ValueError, match="fp_rate"):
            BloomFilter.from_expected_items(n=100, fp_rate=0.0)
        with pytest.raises(ValueError, match="fp_rate"):
            BloomFilter.from_expected_items(n=100, fp_rate=1.0)


class TestBloomFilterAddAndContains:
    """Tests for adding items and membership testing."""

    def test_add_makes_item_contained(self):
        """Adding an item makes contains() return True."""
        bf = BloomFilter[int](size_bits=10000, num_hashes=7)
        bf.add(42)

        assert bf.contains(42)
        assert 42 in bf  # Test __contains__

    def test_contains_returns_false_for_unseen(self):
        """contains() returns False for definitely unseen items."""
        bf = BloomFilter[int](size_bits=10000, num_hashes=7)
        bf.add(1)
        bf.add(2)
        bf.add(3)

        # Most unseen items should return False
        false_count = sum(1 for i in range(1000, 2000) if not bf.contains(i))
        assert false_count > 900  # Expect very few false positives with this size

    def test_no_false_negatives(self):
        """Never returns False for items that were added."""
        bf = BloomFilter[int](size_bits=10000, num_hashes=7, seed=42)

        # Add items
        added = list(range(500))
        for item in added:
            bf.add(item)

        # All added items must return True
        for item in added:
            assert bf.contains(item), f"False negative for {item}"

    def test_add_with_count(self):
        """Adding with count > 1 is same as count=1."""
        bf = BloomFilter[int](size_bits=1000)
        bf.add(42, count=5)

        assert bf.contains(42)
        assert bf.item_count == 5

    def test_add_zero_count_no_effect(self):
        """Adding with count=0 has no effect."""
        bf = BloomFilter[int](size_bits=1000)
        bf.add(42, count=0)

        assert bf.item_count == 0
        # May or may not contain due to empty filter

    def test_rejects_negative_count(self):
        """Rejects negative count."""
        bf = BloomFilter[int](size_bits=1000)
        with pytest.raises(ValueError, match="non-negative"):
            bf.add(42, count=-1)


class TestBloomFilterFalsePositiveRate:
    """Tests for false positive rate."""

    def test_fp_rate_starts_at_zero(self):
        """Empty filter has 0 false positive rate."""
        bf = BloomFilter[int](size_bits=1000)

        assert bf.false_positive_rate == 0.0

    def test_fp_rate_increases_with_items(self):
        """FP rate increases as more items are added."""
        bf = BloomFilter[int](size_bits=1000, num_hashes=5)

        fp_rates = []
        for i in range(100):
            bf.add(i)
            fp_rates.append(bf.false_positive_rate)

        # Should be monotonically increasing
        for i in range(1, len(fp_rates)):
            assert fp_rates[i] >= fp_rates[i - 1]

    def test_actual_fp_rate_matches_estimate(self):
        """Actual false positive rate is close to estimated."""
        bf = BloomFilter.from_expected_items(n=1000, fp_rate=0.05, seed=42)

        # Add 1000 items
        for i in range(1000):
            bf.add(i)

        # Test against items we didn't add
        false_positives = sum(
            1 for i in range(10000, 20000) if bf.contains(i)
        )
        actual_fp_rate = false_positives / 10000

        # Should be within 2x of target
        assert actual_fp_rate < 0.10, f"FP rate {actual_fp_rate:.2%} too high"


class TestBloomFilterFillRatio:
    """Tests for fill ratio."""

    def test_fill_ratio_starts_at_zero(self):
        """Empty filter has 0 fill ratio."""
        bf = BloomFilter[int](size_bits=1000)

        assert bf.fill_ratio == 0.0

    def test_fill_ratio_increases_with_items(self):
        """Fill ratio increases as items are added."""
        bf = BloomFilter[int](size_bits=1000, num_hashes=5)

        fill_ratios = []
        for i in range(100):
            bf.add(i)
            fill_ratios.append(bf.fill_ratio)

        # Should be monotonically increasing
        for i in range(1, len(fill_ratios)):
            assert fill_ratios[i] >= fill_ratios[i - 1]


class TestBloomFilterDeterminism:
    """Tests for deterministic behavior with seed."""

    def test_same_seed_same_behavior(self):
        """Same seed produces same membership results."""
        bf1 = BloomFilter[int](size_bits=1000, seed=42)
        bf2 = BloomFilter[int](size_bits=1000, seed=42)

        for i in range(100):
            bf1.add(i)
            bf2.add(i)

        # Should have same behavior for test items
        for i in range(200):
            assert bf1.contains(i) == bf2.contains(i)


class TestBloomFilterMerge:
    """Tests for merging Bloom Filters."""

    def test_merge_combines_sets(self):
        """Merging combines membership from both filters."""
        bf1 = BloomFilter[int](size_bits=10000, num_hashes=5, seed=42)
        bf2 = BloomFilter[int](size_bits=10000, num_hashes=5, seed=42)

        for i in range(100):
            bf1.add(i)
        for i in range(100, 200):
            bf2.add(i)

        bf1.merge(bf2)

        # Should contain items from both
        for i in range(200):
            assert bf1.contains(i)

    def test_merge_rejects_different_size(self):
        """Cannot merge filters with different size."""
        bf1 = BloomFilter[int](size_bits=1000, seed=42)
        bf2 = BloomFilter[int](size_bits=2000, seed=42)

        with pytest.raises(ValueError, match="size_bits differs"):
            bf1.merge(bf2)

    def test_merge_rejects_different_hashes(self):
        """Cannot merge filters with different num_hashes."""
        bf1 = BloomFilter[int](size_bits=1000, num_hashes=5, seed=42)
        bf2 = BloomFilter[int](size_bits=1000, num_hashes=7, seed=42)

        with pytest.raises(ValueError, match="num_hashes differs"):
            bf1.merge(bf2)

    def test_merge_rejects_different_seed(self):
        """Cannot merge filters with different seeds."""
        bf1 = BloomFilter[int](size_bits=1000, seed=42)
        bf2 = BloomFilter[int](size_bits=1000, seed=123)

        with pytest.raises(ValueError, match="seeds differ"):
            bf1.merge(bf2)

    def test_merge_rejects_wrong_type(self):
        """Cannot merge with non-BloomFilter."""
        bf = BloomFilter[int](size_bits=1000)

        with pytest.raises(TypeError, match="Can only merge"):
            bf.merge("not a bloom filter")  # type: ignore


class TestBloomFilterClear:
    """Tests for clear() method."""

    def test_clear_resets_state(self):
        """clear() resets all state."""
        bf = BloomFilter[int](size_bits=1000)
        for i in range(100):
            bf.add(i)

        bf.clear()

        assert bf.item_count == 0
        assert bf.fill_ratio == 0.0
        assert bf.false_positive_rate == 0.0


class TestBloomFilterMemory:
    """Tests for memory estimation."""

    def test_memory_bytes_proportional_to_size(self):
        """Memory is proportional to size_bits."""
        bf_small = BloomFilter[int](size_bits=1000)
        bf_large = BloomFilter[int](size_bits=100000)

        assert bf_large.memory_bytes > bf_small.memory_bytes

    def test_memory_constant_with_items(self):
        """Memory doesn't grow with item count."""
        bf = BloomFilter[int](size_bits=10000)
        initial_memory = bf.memory_bytes

        for i in range(1000):
            bf.add(i)

        # Memory should be the same
        assert bf.memory_bytes == initial_memory


class TestBloomFilterRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self):
        """repr includes size, hashes, fill, and fp_rate."""
        bf = BloomFilter[int](size_bits=1000, num_hashes=5)
        for i in range(10):
            bf.add(i)

        r = repr(bf)

        assert "size_bits=1000" in r
        assert "num_hashes=5" in r
        assert "fill=" in r
        assert "fp_rate" in r
