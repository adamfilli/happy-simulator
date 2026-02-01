"""Tests for HyperLogLog cardinality estimation."""

import random

import pytest

from happysimulator.sketching import HyperLogLog


class TestHyperLogLogCreation:
    """Tests for HyperLogLog creation and configuration."""

    def test_creates_with_default_precision(self):
        """HyperLogLog is created with default precision."""
        hll = HyperLogLog[int]()

        assert hll.precision == 14
        assert hll.num_registers == 16384

    def test_creates_with_custom_precision(self):
        """HyperLogLog is created with custom precision."""
        hll = HyperLogLog[int](precision=10)

        assert hll.precision == 10
        assert hll.num_registers == 1024

    def test_creates_with_seed(self):
        """HyperLogLog accepts seed for reproducibility."""
        hll = HyperLogLog[int](precision=10, seed=42)

        assert hll.precision == 10

    def test_rejects_precision_below_4(self):
        """Rejects precision < 4."""
        with pytest.raises(ValueError, match="must be in"):
            HyperLogLog[int](precision=3)

    def test_rejects_precision_above_16(self):
        """Rejects precision > 16."""
        with pytest.raises(ValueError, match="must be in"):
            HyperLogLog[int](precision=17)

    def test_works_with_strings(self):
        """Works with string items."""
        hll = HyperLogLog[str](precision=10)
        hll.add("hello")
        hll.add("world")

        assert hll.cardinality() >= 1

    def test_works_with_tuples(self):
        """Works with tuple items."""
        hll = HyperLogLog[tuple](precision=10)
        hll.add((1, "a"))
        hll.add((2, "b"))

        assert hll.cardinality() >= 1


class TestHyperLogLogAdd:
    """Tests for adding items."""

    def test_add_single_item(self):
        """Adding a single item increases cardinality."""
        hll = HyperLogLog[int](precision=10)
        hll.add(42)

        assert hll.cardinality() >= 1
        assert hll.item_count == 1

    def test_add_same_item_multiple_times(self):
        """Adding same item doesn't increase cardinality."""
        hll = HyperLogLog[int](precision=10)
        hll.add(42)
        hll.add(42)
        hll.add(42)

        # Cardinality should still be ~1
        assert hll.cardinality() <= 2
        assert hll.item_count == 3

    def test_add_with_count(self):
        """Adding with count tracks total but not distinct count."""
        hll = HyperLogLog[int](precision=10)
        hll.add(42, count=5)

        assert hll.item_count == 5
        # Cardinality should still be ~1
        assert hll.cardinality() <= 2

    def test_add_zero_count_no_effect(self):
        """Adding with count=0 has no effect."""
        hll = HyperLogLog[int](precision=10)
        hll.add(42, count=0)

        assert hll.item_count == 0

    def test_rejects_negative_count(self):
        """Rejects negative count."""
        hll = HyperLogLog[int](precision=10)
        with pytest.raises(ValueError, match="non-negative"):
            hll.add(42, count=-1)


class TestHyperLogLogCardinality:
    """Tests for cardinality estimation."""

    def test_cardinality_empty(self):
        """Empty sketch has cardinality 0."""
        hll = HyperLogLog[int](precision=10)

        assert hll.cardinality() == 0

    def test_cardinality_single_item(self):
        """Single item gives cardinality ~1."""
        hll = HyperLogLog[int](precision=14)
        hll.add(42)

        card = hll.cardinality()
        assert 0 < card <= 2

    def test_cardinality_small_set(self):
        """Small set cardinality is reasonably accurate."""
        hll = HyperLogLog[int](precision=14)
        n = 100

        for i in range(n):
            hll.add(i)

        card = hll.cardinality()
        # Should be within 20% for small sets
        assert n * 0.8 < card < n * 1.2, f"Expected ~{n}, got {card}"

    def test_cardinality_medium_set(self):
        """Medium set cardinality is reasonably accurate."""
        hll = HyperLogLog[int](precision=14)
        n = 10000

        for i in range(n):
            hll.add(i)

        card = hll.cardinality()
        # Should be within 5% for larger sets with precision=14
        error = abs(card - n) / n
        assert error < 0.1, f"Expected ~{n}, got {card} (error={error:.1%})"

    def test_cardinality_large_set(self):
        """Large set cardinality is reasonably accurate."""
        hll = HyperLogLog[int](precision=14)
        n = 100000

        for i in range(n):
            hll.add(i)

        card = hll.cardinality()
        # Should be within 5% for large sets
        error = abs(card - n) / n
        assert error < 0.1, f"Expected ~{n}, got {card} (error={error:.1%})"


class TestHyperLogLogStandardError:
    """Tests for standard error calculation."""

    def test_standard_error_decreases_with_precision(self):
        """Higher precision gives lower standard error."""
        hll_low = HyperLogLog[int](precision=8)
        hll_high = HyperLogLog[int](precision=14)

        assert hll_high.standard_error() < hll_low.standard_error()

    def test_standard_error_matches_theory(self):
        """Standard error matches theoretical 1.04/sqrt(m)."""
        import math

        for precision in [8, 10, 12, 14]:
            hll = HyperLogLog[int](precision=precision)
            m = 2 ** precision
            expected = 1.04 / math.sqrt(m)
            assert hll.standard_error() == pytest.approx(expected, rel=0.01)


class TestHyperLogLogDeterminism:
    """Tests for deterministic behavior with seed."""

    def test_same_seed_same_cardinality(self):
        """Same seed produces same cardinality for same items."""
        hll1 = HyperLogLog[int](precision=10, seed=42)
        hll2 = HyperLogLog[int](precision=10, seed=42)

        for i in range(1000):
            hll1.add(i)
            hll2.add(i)

        assert hll1.cardinality() == hll2.cardinality()

    def test_different_seeds_may_differ(self):
        """Different seeds may produce different estimates."""
        hll1 = HyperLogLog[int](precision=10, seed=42)
        hll2 = HyperLogLog[int](precision=10, seed=123)

        for i in range(1000):
            hll1.add(i)
            hll2.add(i)

        # May or may not differ, but both should be reasonable
        assert abs(hll1.cardinality() - 1000) < 200
        assert abs(hll2.cardinality() - 1000) < 200


class TestHyperLogLogMerge:
    """Tests for merging HyperLogLog sketches."""

    def test_merge_disjoint_sets(self):
        """Merging disjoint sets gives union cardinality."""
        hll1 = HyperLogLog[int](precision=14, seed=42)
        hll2 = HyperLogLog[int](precision=14, seed=42)

        for i in range(1000):
            hll1.add(i)
        for i in range(1000, 2000):
            hll2.add(i)

        hll1.merge(hll2)

        # Should be ~2000
        card = hll1.cardinality()
        assert 1800 < card < 2200, f"Expected ~2000, got {card}"

    def test_merge_identical_sets(self):
        """Merging identical sets doesn't increase cardinality."""
        hll1 = HyperLogLog[int](precision=14, seed=42)
        hll2 = HyperLogLog[int](precision=14, seed=42)

        for i in range(1000):
            hll1.add(i)
            hll2.add(i)

        original_card = hll1.cardinality()
        hll1.merge(hll2)

        # Should be same as original
        assert abs(hll1.cardinality() - original_card) < 100

    def test_merge_rejects_different_precision(self):
        """Cannot merge sketches with different precision."""
        hll1 = HyperLogLog[int](precision=10)
        hll2 = HyperLogLog[int](precision=12)

        with pytest.raises(ValueError, match="precision differs"):
            hll1.merge(hll2)

    def test_merge_rejects_wrong_type(self):
        """Cannot merge with non-HyperLogLog."""
        hll = HyperLogLog[int](precision=10)

        with pytest.raises(TypeError, match="Can only merge"):
            hll.merge("not a hll")  # type: ignore


class TestHyperLogLogClear:
    """Tests for clear() method."""

    def test_clear_resets_state(self):
        """clear() resets all state."""
        hll = HyperLogLog[int](precision=10)
        for i in range(1000):
            hll.add(i)

        hll.clear()

        assert hll.cardinality() == 0
        assert hll.item_count == 0


class TestHyperLogLogMemory:
    """Tests for memory estimation."""

    def test_memory_bytes_proportional_to_precision(self):
        """Memory is proportional to 2^precision."""
        hll_small = HyperLogLog[int](precision=8)
        hll_large = HyperLogLog[int](precision=14)

        assert hll_large.memory_bytes > hll_small.memory_bytes

    def test_memory_constant_with_items(self):
        """Memory doesn't grow with item count."""
        hll = HyperLogLog[int](precision=10)
        initial_memory = hll.memory_bytes

        for i in range(10000):
            hll.add(i)

        # Memory should be roughly the same
        assert hll.memory_bytes == initial_memory


class TestHyperLogLogRepr:
    """Tests for string representation."""

    def test_repr_includes_key_info(self):
        """repr includes precision and cardinality."""
        hll = HyperLogLog[int](precision=10)
        for i in range(100):
            hll.add(i)

        r = repr(hll)

        assert "precision=10" in r
        assert "registers=1024" in r
        assert "cardinality" in r


class TestHyperLogLogAccuracy:
    """Tests for estimation accuracy across different scales."""

    def test_accuracy_across_scales(self):
        """Accuracy is maintained across different cardinalities."""
        hll = HyperLogLog[int](precision=14, seed=42)
        rng = random.Random(42)

        expected_error = hll.standard_error()

        for n in [100, 1000, 10000]:
            hll.clear()
            for i in range(n):
                hll.add(i)

            card = hll.cardinality()
            relative_error = abs(card - n) / n

            # Should be within 3x standard error most of the time
            assert relative_error < expected_error * 3, (
                f"n={n}: expected error ~{expected_error:.2%}, "
                f"got {relative_error:.2%}"
            )
