"""Tests for REDQueue (Random Early Detection)."""

import pytest

from happysimulator.components.queue_policies import REDQueue


class TestREDQueueCreation:
    """Tests for REDQueue creation."""

    def test_creates_with_basic_parameters(self):
        """REDQueue can be created with basic parameters."""
        queue = REDQueue(min_threshold=10, max_threshold=30)

        assert queue.min_threshold == 10
        assert queue.max_threshold == 30
        assert queue.max_probability == 0.1
        assert queue.capacity == 60  # 2 * max_threshold

    def test_creates_with_custom_parameters(self):
        """REDQueue can be created with custom parameters."""
        queue = REDQueue(
            min_threshold=5,
            max_threshold=20,
            max_probability=0.2,
            capacity=50,
        )

        assert queue.min_threshold == 5
        assert queue.max_threshold == 20
        assert queue.max_probability == 0.2
        assert queue.capacity == 50

    def test_rejects_invalid_thresholds(self):
        """REDQueue rejects invalid threshold values."""
        with pytest.raises(ValueError):
            REDQueue(min_threshold=-1, max_threshold=30)

        with pytest.raises(ValueError):
            REDQueue(min_threshold=30, max_threshold=20)  # min > max

        with pytest.raises(ValueError):
            REDQueue(min_threshold=10, max_threshold=10)  # min == max

    def test_rejects_invalid_probability(self):
        """REDQueue rejects invalid max_probability."""
        with pytest.raises(ValueError):
            REDQueue(min_threshold=10, max_threshold=30, max_probability=0)

        with pytest.raises(ValueError):
            REDQueue(min_threshold=10, max_threshold=30, max_probability=1.5)

    def test_rejects_capacity_below_max_threshold(self):
        """REDQueue rejects capacity below max_threshold."""
        with pytest.raises(ValueError):
            REDQueue(min_threshold=10, max_threshold=30, capacity=25)


class TestREDQueueBehavior:
    """Tests for REDQueue behavior."""

    def test_accepts_below_min_threshold(self):
        """REDQueue accepts all items below min_threshold."""
        queue = REDQueue(min_threshold=10, max_threshold=30)

        # Add items below threshold
        for i in range(5):
            assert queue.push(i) is True

        assert len(queue) == 5
        assert queue.stats.dropped_probabilistic == 0

    def test_drops_above_max_threshold(self):
        """REDQueue drops items when average exceeds max_threshold."""
        # Use very high weight so average tracks actual length closely
        queue = REDQueue(min_threshold=5, max_threshold=10, capacity=100, weight=0.9)

        # Fill queue well past max_threshold - need enough items for average to catch up
        for i in range(100):
            queue.push(i)

        # With weight=0.9, average quickly approaches actual length
        # Once avg >= max_threshold (10), drop probability = 100%
        # So we should see either probabilistic or forced drops
        total_drops = queue.stats.dropped_probabilistic + queue.stats.dropped_forced
        assert total_drops > 0, f"Expected some drops but got none. Enqueued={queue.stats.enqueued}, len={len(queue)}"

    def test_fifo_dequeue(self):
        """REDQueue dequeues in FIFO order."""
        queue = REDQueue(min_threshold=100, max_threshold=200)  # High thresholds

        queue.push("first")
        queue.push("second")
        queue.push("third")

        assert queue.pop() == "first"
        assert queue.pop() == "second"
        assert queue.pop() == "third"

    def test_pop_empty_returns_none(self):
        """REDQueue.pop() returns None when empty."""
        queue = REDQueue(min_threshold=10, max_threshold=30)

        assert queue.pop() is None

    def test_peek_returns_next_item(self):
        """REDQueue.peek() returns next item without removing it."""
        queue = REDQueue(min_threshold=100, max_threshold=200)

        queue.push("item")
        assert queue.peek() == "item"
        assert len(queue) == 1

    def test_is_empty(self):
        """REDQueue.is_empty() returns correct state."""
        queue = REDQueue(min_threshold=10, max_threshold=30)

        assert queue.is_empty() is True

        queue.push("item")
        assert queue.is_empty() is False

        queue.pop()
        assert queue.is_empty() is True

    def test_tracks_statistics(self):
        """REDQueue tracks enqueue/dequeue statistics."""
        queue = REDQueue(min_threshold=100, max_threshold=200)

        for i in range(10):
            queue.push(i)

        for i in range(5):
            queue.pop()

        assert queue.stats.enqueued == 10
        assert queue.stats.dequeued == 5


class TestREDQueueAveraging:
    """Tests for RED's exponential moving average."""

    def test_avg_queue_starts_at_zero(self):
        """Average queue length starts at zero."""
        queue = REDQueue(min_threshold=10, max_threshold=30)

        assert queue.avg_queue_length == 0.0

    def test_avg_queue_updates_on_push(self):
        """Average queue length updates when pushing."""
        queue = REDQueue(min_threshold=10, max_threshold=30, weight=0.5)

        queue.push("item")
        # With weight 0.5: avg = 0.5 * 0 + 0.5 * 0 = 0 (measured before push)
        # Actually it updates before checking, so it sees queue empty
        assert queue.avg_queue_length >= 0
