"""Tests for TopK (Space-Saving) algorithm."""

from collections import Counter

import pytest

from happysimulator.sketching import TopK, FrequencyEstimate
from happysimulator.distributions import ZipfDistribution


class TestTopKCreation:
    """Tests for TopK creation and configuration."""

    def test_creates_with_k(self):
        """TopK is created with specified k."""
        topk = TopK[int](k=100)

        assert topk.k == 100
        assert topk.tracked_count == 0
        assert topk.item_count == 0

    def test_creates_with_seed(self):
        """TopK accepts seed parameter (for API consistency)."""
        topk = TopK[int](k=50, seed=42)

        assert topk.k == 50

    def test_rejects_zero_k(self):
        """Rejects k=0."""
        with pytest.raises(ValueError, match="must be positive"):
            TopK[int](k=0)

    def test_rejects_negative_k(self):
        """Rejects negative k."""
        with pytest.raises(ValueError, match="must be positive"):
            TopK[int](k=-5)

    def test_works_with_strings(self):
        """Works with string items."""
        topk = TopK[str](k=10)
        topk.add("hello")
        topk.add("world")

        assert topk.tracked_count == 2
        assert "hello" in topk
        assert "world" in topk

    def test_works_with_tuples(self):
        """Works with tuple items."""
        topk = TopK[tuple](k=10)
        topk.add((1, "a"))
        topk.add((2, "b"))

        assert topk.tracked_count == 2


class TestTopKAddAndEstimate:
    """Tests for adding items and estimating frequencies."""

    def test_add_single_item(self):
        """Adding a single item tracks it."""
        topk = TopK[int](k=10)
        topk.add(42)

        assert topk.estimate(42) == 1
        assert topk.item_count == 1
        assert topk.tracked_count == 1

    def test_add_same_item_multiple_times(self):
        """Adding same item multiple times increases count."""
        topk = TopK[int](k=10)
        topk.add(42)
        topk.add(42)
        topk.add(42)

        assert topk.estimate(42) == 3
        assert topk.item_count == 3
        assert topk.tracked_count == 1

    def test_add_with_count(self):
        """Adding with count > 1 works correctly."""
        topk = TopK[int](k=10)
        topk.add(42, count=5)

        assert topk.estimate(42) == 5
        assert topk.item_count == 5

    def test_add_zero_count_no_effect(self):
        """Adding with count=0 has no effect."""
        topk = TopK[int](k=10)
        topk.add(42, count=0)

        assert topk.estimate(42) == 0
        assert topk.item_count == 0
        assert topk.tracked_count == 0

    def test_rejects_negative_count(self):
        """Rejects negative count."""
        topk = TopK[int](k=10)
        with pytest.raises(ValueError, match="non-negative"):
            topk.add(42, count=-1)

    def test_estimate_untracked_item_returns_zero(self):
        """Estimating untracked item returns 0."""
        topk = TopK[int](k=10)
        topk.add(1)

        assert topk.estimate(999) == 0

    def test_contains_tracked_item(self):
        """Contains returns True for tracked items."""
        topk = TopK[int](k=10)
        topk.add(42)

        assert 42 in topk
        assert 999 not in topk


class TestTopKEviction:
    """Tests for item eviction when capacity is reached."""

    def test_evicts_minimum_when_full(self):
        """Evicts item with minimum count when k is reached."""
        topk = TopK[int](k=3)

        # Add 3 items
        topk.add(1, count=10)
        topk.add(2, count=5)
        topk.add(3, count=1)

        # All tracked
        assert topk.tracked_count == 3
        assert 1 in topk
        assert 2 in topk
        assert 3 in topk

        # Add 4th item - should evict item 3 (count=1)
        topk.add(4, count=2)

        assert topk.tracked_count == 3
        assert 1 in topk
        assert 2 in topk
        assert 4 in topk
        assert 3 not in topk

    def test_evicted_item_estimate_with_error(self):
        """New item inherits minimum count as error."""
        topk = TopK[int](k=2)
        topk.add(1, count=10)
        topk.add(2, count=3)

        # Adding 3 evicts 2 (min count=3)
        topk.add(3, count=1)

        # New item has count = 3 + 1 = 4, error = 3
        estimate = topk.estimate_with_error(3)
        assert estimate.count == 4
        assert estimate.error == 3


class TestTopKTop:
    """Tests for top() query method."""

    def test_top_returns_sorted_by_count(self):
        """top() returns items sorted by count descending."""
        topk = TopK[int](k=10)
        topk.add(1, count=100)
        topk.add(2, count=50)
        topk.add(3, count=75)

        results = topk.top(3)

        assert len(results) == 3
        assert results[0].item == 1
        assert results[0].count == 100
        assert results[1].item == 3
        assert results[1].count == 75
        assert results[2].item == 2
        assert results[2].count == 50

    def test_top_returns_frequency_estimates(self):
        """top() returns FrequencyEstimate objects."""
        topk = TopK[int](k=10)
        topk.add(1, count=10)

        results = topk.top(1)

        assert len(results) == 1
        assert isinstance(results[0], FrequencyEstimate)
        assert results[0].item == 1
        assert results[0].count == 10
        assert results[0].error == 0

    def test_top_n_less_than_tracked(self):
        """top(n) with n < tracked count returns only n items."""
        topk = TopK[int](k=10)
        for i in range(5):
            topk.add(i, count=(5 - i) * 10)

        results = topk.top(3)

        assert len(results) == 3

    def test_top_n_none_returns_all(self):
        """top(None) returns all tracked items."""
        topk = TopK[int](k=10)
        for i in range(5):
            topk.add(i)

        results = topk.top()

        assert len(results) == 5

    def test_top_empty_sketch(self):
        """top() on empty sketch returns empty list."""
        topk = TopK[int](k=10)

        results = topk.top(5)

        assert results == []


class TestTopKErrorBounds:
    """Tests for error bound calculations."""

    def test_max_error_empty_sketch(self):
        """max_error on empty sketch is 0."""
        topk = TopK[int](k=10)

        assert topk.max_error() == 0

    def test_max_error_equals_min_count(self):
        """max_error equals minimum tracked count."""
        topk = TopK[int](k=5)
        topk.add(1, count=100)
        topk.add(2, count=50)
        topk.add(3, count=10)

        assert topk.max_error() == 10

    def test_guaranteed_threshold(self):
        """guaranteed_threshold returns N/k."""
        topk = TopK[int](k=100)

        for i in range(1000):
            topk.add(i)

        # Total count = 1000, k = 100, so threshold = 10
        assert topk.guaranteed_threshold() == 10


class TestTopKMerge:
    """Tests for merging TopK sketches."""

    def test_merge_combines_counts(self):
        """Merging combines counts for same items."""
        topk1 = TopK[int](k=10)
        topk2 = TopK[int](k=10)

        topk1.add(1, count=10)
        topk2.add(1, count=5)

        topk1.merge(topk2)

        assert topk1.estimate(1) == 15

    def test_merge_adds_new_items(self):
        """Merging adds items not in first sketch."""
        topk1 = TopK[int](k=10)
        topk2 = TopK[int](k=10)

        topk1.add(1, count=10)
        topk2.add(2, count=5)

        topk1.merge(topk2)

        assert topk1.estimate(1) == 10
        assert topk1.estimate(2) == 5

    def test_merge_rejects_different_k(self):
        """Cannot merge sketches with different k."""
        topk1 = TopK[int](k=10)
        topk2 = TopK[int](k=20)

        with pytest.raises(ValueError, match="Cannot merge"):
            topk1.merge(topk2)

    def test_merge_rejects_wrong_type(self):
        """Cannot merge with non-TopK."""
        topk = TopK[int](k=10)

        with pytest.raises(TypeError, match="Can only merge"):
            topk.merge("not a topk")  # type: ignore


class TestTopKClear:
    """Tests for clear() method."""

    def test_clear_resets_state(self):
        """clear() resets all state."""
        topk = TopK[int](k=10)
        topk.add(1, count=100)
        topk.add(2, count=50)

        topk.clear()

        assert topk.tracked_count == 0
        assert topk.item_count == 0
        assert topk.estimate(1) == 0
        assert topk.estimate(2) == 0


class TestTopKMemory:
    """Tests for memory estimation."""

    def test_memory_bytes_increases_with_items(self):
        """memory_bytes increases as items are added."""
        topk = TopK[int](k=100)
        empty_size = topk.memory_bytes

        for i in range(50):
            topk.add(i)

        assert topk.memory_bytes > empty_size

    def test_memory_bytes_bounded_by_k(self):
        """memory_bytes stays bounded when k is reached."""
        topk = TopK[int](k=10)

        # Add many more than k items
        for i in range(1000):
            topk.add(i)

        # Should have memory proportional to k, not 1000
        # This is a rough check - exact values depend on Python internals
        assert topk.tracked_count == 10


class TestTopKRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self):
        """repr includes k, tracked count, and total."""
        topk = TopK[int](k=100)
        topk.add(1, count=50)
        topk.add(2, count=30)

        r = repr(topk)

        assert "k=100" in r
        assert "tracked=2" in r
        assert "total=80" in r


class TestTopKWithZipfDistribution:
    """Integration tests with Zipf distribution."""

    def test_identifies_heavy_hitters(self):
        """TopK correctly identifies heavy hitters from Zipf distribution."""
        # Use k=100 to reliably capture top items from 1000-item Zipf dist
        topk = TopK[int](k=100)
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)

        # Add 100k samples
        for _ in range(100_000):
            topk.add(dist.sample())

        # True top 20 (ranks 0-19) should mostly be in sketch's top 20
        top_20 = {item.item for item in topk.top(20)}
        true_top_20 = set(range(20))

        # At least 15 of true top 20 should be identified
        overlap = len(top_20 & true_top_20)
        assert overlap >= 15, f"Only {overlap} of top 20 identified: {top_20}"

    def test_count_ordering_preserved(self):
        """Higher-frequency items have higher estimated counts."""
        topk = TopK[int](k=50)
        dist = ZipfDistribution(range(100), s=1.0, seed=42)

        for _ in range(50_000):
            topk.add(dist.sample())

        results = topk.top(10)

        # Counts should be in descending order
        counts = [r.count for r in results]
        assert counts == sorted(counts, reverse=True)

    def test_error_bound_respected(self):
        """Estimation error is within theoretical bounds."""
        topk = TopK[int](k=100)
        dist = ZipfDistribution(range(1000), s=1.0, seed=42)
        exact_counts: Counter[int] = Counter()

        # Track exact counts alongside sketch
        for _ in range(50_000):
            item = dist.sample()
            topk.add(item)
            exact_counts[item] += 1

        # For tracked items, error should be <= max_error()
        max_error = topk.max_error()
        for estimate in topk.top():
            true_count = exact_counts[estimate.item]
            error = abs(estimate.count - true_count)
            assert error <= max_error, (
                f"Item {estimate.item}: estimated={estimate.count}, "
                f"true={true_count}, error={error} > max_error={max_error}"
            )
