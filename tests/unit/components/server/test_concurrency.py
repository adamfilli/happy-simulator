"""Tests for concurrency control models."""

from __future__ import annotations

import pytest

from happysimulator.components.server.concurrency import (
    ConcurrencyModel,
    DynamicConcurrency,
    FixedConcurrency,
    WeightedConcurrency,
)


class TestFixedConcurrency:
    """Tests for FixedConcurrency model."""

    def test_creates_with_limit(self):
        """FixedConcurrency is created with specified limit."""
        model = FixedConcurrency(max_concurrent=4)
        assert model.limit == 4
        assert model.active == 0
        assert model.available == 4

    def test_rejects_zero_limit(self):
        """FixedConcurrency rejects max_concurrent < 1."""
        with pytest.raises(ValueError):
            FixedConcurrency(max_concurrent=0)

        with pytest.raises(ValueError):
            FixedConcurrency(max_concurrent=-1)

    def test_acquire_succeeds_when_available(self):
        """acquire() returns True when capacity available."""
        model = FixedConcurrency(max_concurrent=2)

        assert model.acquire() is True
        assert model.active == 1
        assert model.available == 1

        assert model.acquire() is True
        assert model.active == 2
        assert model.available == 0

    def test_acquire_fails_when_full(self):
        """acquire() returns False when at capacity."""
        model = FixedConcurrency(max_concurrent=1)

        assert model.acquire() is True
        assert model.acquire() is False
        assert model.active == 1

    def test_release_frees_capacity(self):
        """release() makes capacity available."""
        model = FixedConcurrency(max_concurrent=1)

        model.acquire()
        assert model.available == 0

        model.release()
        assert model.available == 1
        assert model.active == 0

    def test_has_capacity_reflects_state(self):
        """has_capacity() returns correct state."""
        model = FixedConcurrency(max_concurrent=1)

        assert model.has_capacity() is True
        model.acquire()
        assert model.has_capacity() is False
        model.release()
        assert model.has_capacity() is True

    def test_release_does_not_go_negative(self):
        """release() without acquire doesn't go below 0."""
        model = FixedConcurrency(max_concurrent=2)

        model.release()
        assert model.active == 0
        assert model.available == 2


class TestDynamicConcurrency:
    """Tests for DynamicConcurrency model."""

    def test_creates_with_bounds(self):
        """DynamicConcurrency is created with initial and bounds."""
        model = DynamicConcurrency(initial=4, min_limit=1, max_limit=10)
        assert model.current_limit == 4
        assert model.min_limit == 1
        assert model.max_limit == 10
        assert model.limit == 4

    def test_rejects_invalid_bounds(self):
        """DynamicConcurrency rejects invalid bound configurations."""
        # min_limit < 1
        with pytest.raises(ValueError):
            DynamicConcurrency(initial=4, min_limit=0, max_limit=10)

        # initial < min_limit
        with pytest.raises(ValueError):
            DynamicConcurrency(initial=0, min_limit=1, max_limit=10)

        # initial > max_limit
        with pytest.raises(ValueError):
            DynamicConcurrency(initial=15, min_limit=1, max_limit=10)

        # max_limit < min_limit
        with pytest.raises(ValueError):
            DynamicConcurrency(initial=5, min_limit=10, max_limit=5)

    def test_set_limit_changes_capacity(self):
        """set_limit() adjusts the concurrency limit."""
        model = DynamicConcurrency(initial=4, min_limit=1, max_limit=10)

        model.set_limit(8)
        assert model.current_limit == 8
        assert model.available == 8

    def test_set_limit_clamps_to_bounds(self):
        """set_limit() clamps to min/max bounds."""
        model = DynamicConcurrency(initial=5, min_limit=2, max_limit=8)

        model.set_limit(1)  # Below min
        assert model.current_limit == 2

        model.set_limit(100)  # Above max
        assert model.current_limit == 8

    def test_scale_up_and_down(self):
        """scale_up() and scale_down() adjust limit."""
        model = DynamicConcurrency(initial=5, min_limit=1, max_limit=10)

        model.scale_up(2)
        assert model.current_limit == 7

        model.scale_down(3)
        assert model.current_limit == 4

    def test_active_requests_continue_after_scale_down(self):
        """Active requests continue even if limit is reduced below active count."""
        model = DynamicConcurrency(initial=4, min_limit=1, max_limit=10)

        # Acquire 4 slots
        for _ in range(4):
            model.acquire()
        assert model.active == 4

        # Scale down below active count
        model.set_limit(2)
        assert model.current_limit == 2
        assert model.active == 4  # Still 4 active
        assert model.available == 0  # No new capacity
        assert model.has_capacity() is False

        # New requests rejected
        assert model.acquire() is False

        # Release 3
        for _ in range(3):
            model.release()
        assert model.active == 1
        assert model.available == 1
        assert model.has_capacity() is True

    def test_unlimited_max(self):
        """DynamicConcurrency works with unlimited max."""
        model = DynamicConcurrency(initial=10, min_limit=1, max_limit=None)

        model.set_limit(1000)
        assert model.current_limit == 1000

        model.set_limit(1)  # Clamped to min
        assert model.current_limit == 1


class TestWeightedConcurrency:
    """Tests for WeightedConcurrency model."""

    def test_creates_with_capacity(self):
        """WeightedConcurrency is created with total capacity."""
        model = WeightedConcurrency(total_capacity=100)
        assert model.total_capacity == 100
        assert model.available == 100
        assert model.active == 0
        assert model.limit == 100

    def test_rejects_zero_capacity(self):
        """WeightedConcurrency rejects total_capacity < 1."""
        with pytest.raises(ValueError):
            WeightedConcurrency(total_capacity=0)

    def test_acquire_with_weight(self):
        """acquire() consumes specified weight."""
        model = WeightedConcurrency(total_capacity=100)

        assert model.acquire(weight=10) is True
        assert model.active == 10
        assert model.available == 90

        assert model.acquire(weight=25) is True
        assert model.active == 35
        assert model.available == 65

    def test_acquire_fails_when_insufficient(self):
        """acquire() fails when insufficient capacity."""
        model = WeightedConcurrency(total_capacity=100)

        model.acquire(weight=80)
        assert model.acquire(weight=30) is False
        assert model.active == 80

    def test_release_with_weight(self):
        """release() frees specified weight."""
        model = WeightedConcurrency(total_capacity=100)

        model.acquire(weight=50)
        model.release(weight=20)

        assert model.active == 30
        assert model.available == 70

    def test_has_capacity_with_weight(self):
        """has_capacity() checks for specific weight."""
        model = WeightedConcurrency(total_capacity=100)

        model.acquire(weight=90)

        assert model.has_capacity(weight=10) is True
        assert model.has_capacity(weight=11) is False

    def test_utilization(self):
        """utilization property calculates correctly."""
        model = WeightedConcurrency(total_capacity=100)

        assert model.utilization == 0.0

        model.acquire(weight=50)
        assert model.utilization == 0.5

        model.acquire(weight=25)
        assert model.utilization == 0.75

    def test_rejects_invalid_weight(self):
        """acquire/release reject weight < 1."""
        model = WeightedConcurrency(total_capacity=100)

        with pytest.raises(ValueError):
            model.acquire(weight=0)

        with pytest.raises(ValueError):
            model.release(weight=0)

    def test_mixed_weights(self):
        """Handles multiple requests with different weights."""
        model = WeightedConcurrency(total_capacity=100)

        # Light requests (weight=1)
        for _ in range(10):
            assert model.acquire(weight=1) is True
        assert model.active == 10

        # Heavy request (weight=50)
        assert model.acquire(weight=50) is True
        assert model.active == 60

        # Medium request (weight=20)
        assert model.acquire(weight=20) is True
        assert model.active == 80

        # Too heavy (weight=30)
        assert model.acquire(weight=30) is False

        # Exact fit (weight=20)
        assert model.acquire(weight=20) is True
        assert model.active == 100
        assert model.available == 0


class TestConcurrencyModelProtocol:
    """Tests that all models satisfy the ConcurrencyModel protocol."""

    def test_fixed_is_concurrency_model(self):
        """FixedConcurrency satisfies ConcurrencyModel protocol."""
        model = FixedConcurrency(4)
        assert isinstance(model, ConcurrencyModel)

    def test_dynamic_is_concurrency_model(self):
        """DynamicConcurrency satisfies ConcurrencyModel protocol."""
        model = DynamicConcurrency(4, 1, 10)
        assert isinstance(model, ConcurrencyModel)

    def test_weighted_is_concurrency_model(self):
        """WeightedConcurrency satisfies ConcurrencyModel protocol."""
        model = WeightedConcurrency(100)
        assert isinstance(model, ConcurrencyModel)
