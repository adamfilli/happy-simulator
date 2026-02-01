"""Tests for FairQueue."""

import pytest

from happysimulator.components.queue_policies import FairQueue


def get_flow_id(item: dict) -> str:
    """Extract flow ID from test item."""
    return item.get("flow", "default")


class TestFairQueueCreation:
    """Tests for FairQueue creation."""

    def test_creates_with_basic_parameters(self):
        """FairQueue can be created with basic parameters."""
        queue = FairQueue(get_flow_id=get_flow_id)

        assert queue.flow_count == 0
        assert queue.max_flows is None
        assert queue.per_flow_capacity == float("inf")

    def test_creates_with_limits(self):
        """FairQueue can be created with flow limits."""
        queue = FairQueue(
            get_flow_id=get_flow_id,
            max_flows=10,
            per_flow_capacity=5,
        )

        assert queue.max_flows == 10
        assert queue.per_flow_capacity == 5

    def test_rejects_invalid_max_flows(self):
        """FairQueue rejects invalid max_flows."""
        with pytest.raises(ValueError):
            FairQueue(get_flow_id=get_flow_id, max_flows=0)

    def test_rejects_invalid_per_flow_capacity(self):
        """FairQueue rejects invalid per_flow_capacity."""
        with pytest.raises(ValueError):
            FairQueue(get_flow_id=get_flow_id, per_flow_capacity=0)


class TestFairQueueBehavior:
    """Tests for FairQueue behavior."""

    def test_creates_flow_on_first_item(self):
        """FairQueue creates flow queue on first item for that flow."""
        queue = FairQueue(get_flow_id=get_flow_id)

        queue.push({"flow": "A", "value": 1})

        assert queue.flow_count == 1
        assert queue.get_flow_depth("A") == 1
        assert queue.stats.flows_created == 1

    def test_round_robin_across_flows(self):
        """FairQueue dequeues round-robin across flows."""
        queue = FairQueue(get_flow_id=get_flow_id)

        # Add items to different flows
        queue.push({"flow": "A", "value": 1})
        queue.push({"flow": "A", "value": 2})
        queue.push({"flow": "B", "value": 3})
        queue.push({"flow": "B", "value": 4})

        # Should alternate between flows
        items = [queue.pop() for _ in range(4)]
        flows = [item["flow"] for item in items]

        # First dequeue from A (created first), then B, then A, then B
        assert flows == ["A", "B", "A", "B"]

    def test_respects_max_flows(self):
        """FairQueue rejects items when max_flows reached."""
        queue = FairQueue(get_flow_id=get_flow_id, max_flows=2)

        queue.push({"flow": "A"})
        queue.push({"flow": "B"})
        result = queue.push({"flow": "C"})

        assert result is False
        assert queue.flow_count == 2
        assert queue.stats.rejected_max_flows == 1

    def test_respects_per_flow_capacity(self):
        """FairQueue rejects items when per-flow capacity reached."""
        queue = FairQueue(get_flow_id=get_flow_id, per_flow_capacity=2)

        queue.push({"flow": "A", "n": 1})
        queue.push({"flow": "A", "n": 2})
        result = queue.push({"flow": "A", "n": 3})

        assert result is False
        assert queue.get_flow_depth("A") == 2
        assert queue.stats.rejected_flow_capacity == 1

    def test_removes_empty_flows(self):
        """FairQueue removes flows when they become empty."""
        queue = FairQueue(get_flow_id=get_flow_id)

        queue.push({"flow": "A"})
        assert queue.flow_count == 1

        queue.pop()
        assert queue.flow_count == 0
        assert queue.stats.flows_removed == 1

    def test_pop_empty_returns_none(self):
        """FairQueue.pop() returns None when empty."""
        queue = FairQueue(get_flow_id=get_flow_id)

        assert queue.pop() is None

    def test_peek_returns_next_item(self):
        """FairQueue.peek() returns next item without removing it."""
        queue = FairQueue(get_flow_id=get_flow_id)

        queue.push({"flow": "A", "value": "test"})
        assert queue.peek()["value"] == "test"
        assert len(queue) == 1

    def test_is_empty(self):
        """FairQueue.is_empty() returns correct state."""
        queue = FairQueue(get_flow_id=get_flow_id)

        assert queue.is_empty() is True

        queue.push({"flow": "A"})
        assert queue.is_empty() is False

        queue.pop()
        assert queue.is_empty() is True

    def test_len_counts_all_items(self):
        """FairQueue.__len__() returns total items across flows."""
        queue = FairQueue(get_flow_id=get_flow_id)

        queue.push({"flow": "A"})
        queue.push({"flow": "A"})
        queue.push({"flow": "B"})

        assert len(queue) == 3


class TestFairQueueFairness:
    """Tests for FairQueue fairness properties."""

    def test_equal_service_across_flows(self):
        """FairQueue provides equal service to all flows."""
        queue = FairQueue(get_flow_id=get_flow_id)

        # Add 10 items to each of 3 flows
        for i in range(10):
            queue.push({"flow": "A", "n": i})
            queue.push({"flow": "B", "n": i})
            queue.push({"flow": "C", "n": i})

        # Dequeue 15 items
        flow_counts = {"A": 0, "B": 0, "C": 0}
        for _ in range(15):
            item = queue.pop()
            flow_counts[item["flow"]] += 1

        # Each flow should get 5 items
        assert flow_counts["A"] == 5
        assert flow_counts["B"] == 5
        assert flow_counts["C"] == 5

    def test_new_flow_gets_immediate_service(self):
        """New flows get service without waiting for existing flows to drain."""
        queue = FairQueue(get_flow_id=get_flow_id)

        # Add many items from flow A
        for i in range(100):
            queue.push({"flow": "A", "n": i})

        # Add one item from flow B
        queue.push({"flow": "B", "n": 0})

        # Flow B should get served within 2 dequeues
        results = [queue.pop() for _ in range(3)]
        flows = [r["flow"] for r in results]

        assert "B" in flows[:2]
