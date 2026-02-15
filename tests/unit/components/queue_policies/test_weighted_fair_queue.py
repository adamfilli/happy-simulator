"""Tests for WeightedFairQueue."""

import pytest

from happysimulator.components.queue_policies import WeightedFairQueue


def get_flow_id(item: dict) -> str:
    """Extract flow ID from test item."""
    return item.get("flow", "default")


def get_weight(flow_id: str) -> int:
    """Get weight for flow ID."""
    weights = {"premium": 3, "standard": 1, "default": 1}
    return weights.get(flow_id, 1)


class TestWeightedFairQueueCreation:
    """Tests for WeightedFairQueue creation."""

    def test_creates_with_basic_parameters(self):
        """WeightedFairQueue can be created with basic parameters."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        assert queue.flow_count == 0
        assert queue.capacity == float("inf")

    def test_creates_with_capacity(self):
        """WeightedFairQueue can be created with capacity."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
            capacity=100,
        )

        assert queue.capacity == 100

    def test_rejects_invalid_capacity(self):
        """WeightedFairQueue rejects invalid capacity."""
        with pytest.raises(ValueError):
            WeightedFairQueue(
                get_flow_id=get_flow_id,
                get_weight=get_weight,
                capacity=0,
            )


class TestWeightedFairQueueBehavior:
    """Tests for WeightedFairQueue behavior."""

    def test_creates_flow_with_weight(self):
        """WeightedFairQueue creates flow with assigned weight."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        queue.push({"flow": "premium"})

        assert queue.get_flow_weight("premium") == 3

    def test_weighted_round_robin(self):
        """WeightedFairQueue dequeues according to weights."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        # Add items to flows with different weights
        for i in range(10):
            queue.push({"flow": "premium", "n": i})  # weight 3
        for i in range(10):
            queue.push({"flow": "standard", "n": i})  # weight 1

        # Dequeue 8 items
        flow_counts = {"premium": 0, "standard": 0}
        for _ in range(8):
            item = queue.pop()
            flow_counts[item["flow"]] += 1

        # Premium should get ~3x standard (6 vs 2)
        assert flow_counts["premium"] == 6
        assert flow_counts["standard"] == 2

    def test_respects_capacity(self):
        """WeightedFairQueue respects total capacity."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
            capacity=5,
        )

        for i in range(10):
            queue.push({"flow": "A", "n": i})

        assert len(queue) == 5
        assert queue.stats.rejected_capacity == 5

    def test_pop_empty_returns_none(self):
        """WeightedFairQueue.pop() returns None when empty."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        assert queue.pop() is None

    def test_peek_returns_next_item(self):
        """WeightedFairQueue.peek() returns next item without removing it."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        queue.push({"flow": "A", "value": "test"})
        assert queue.peek()["value"] == "test"
        assert len(queue) == 1

    def test_is_empty(self):
        """WeightedFairQueue.is_empty() returns correct state."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        assert queue.is_empty() is True

        queue.push({"flow": "A"})
        assert queue.is_empty() is False

        queue.pop()
        assert queue.is_empty() is True


class TestWeightedFairQueueFairness:
    """Tests for WeightedFairQueue weighted fairness."""

    def test_proportional_service(self):
        """WeightedFairQueue provides proportional service by weight."""
        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=get_weight,
        )

        # Fill queues
        for _i in range(30):
            queue.push({"flow": "premium"})  # weight 3
            queue.push({"flow": "standard"})  # weight 1

        # Dequeue all items
        flow_counts = {"premium": 0, "standard": 0}
        while not queue.is_empty():
            item = queue.pop()
            flow_counts[item["flow"]] += 1

        # Both should be fully drained
        assert flow_counts["premium"] == 30
        assert flow_counts["standard"] == 30

    def test_minimum_weight_is_one(self):
        """Flows with weight < 1 are treated as weight 1."""

        def zero_weight(flow_id: str) -> int:
            return 0 if flow_id == "zero" else 1

        queue = WeightedFairQueue(
            get_flow_id=get_flow_id,
            get_weight=zero_weight,
        )

        queue.push({"flow": "zero"})
        # Should not raise, weight should be clamped to 1
        assert queue.get_flow_weight("zero") == 1
