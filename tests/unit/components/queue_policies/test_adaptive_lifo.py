"""Tests for AdaptiveLIFO."""

import pytest

from happysimulator.components.queue_policies import AdaptiveLIFO


class TestAdaptiveLIFOCreation:
    """Tests for AdaptiveLIFO creation."""

    def test_creates_with_basic_parameters(self):
        """AdaptiveLIFO can be created with basic parameters."""
        queue = AdaptiveLIFO(congestion_threshold=10)

        assert queue.congestion_threshold == 10
        assert queue.capacity == float("inf")
        assert queue.mode == "FIFO"

    def test_creates_with_capacity(self):
        """AdaptiveLIFO can be created with capacity."""
        queue = AdaptiveLIFO(congestion_threshold=5, capacity=100)

        assert queue.congestion_threshold == 5
        assert queue.capacity == 100

    def test_rejects_invalid_threshold(self):
        """AdaptiveLIFO rejects invalid congestion_threshold."""
        with pytest.raises(ValueError):
            AdaptiveLIFO(congestion_threshold=0)

        with pytest.raises(ValueError):
            AdaptiveLIFO(congestion_threshold=-1)

    def test_rejects_invalid_capacity(self):
        """AdaptiveLIFO rejects invalid capacity."""
        with pytest.raises(ValueError):
            AdaptiveLIFO(congestion_threshold=5, capacity=0)

        with pytest.raises(ValueError):
            AdaptiveLIFO(congestion_threshold=5, capacity=-1)


class TestAdaptiveLIFOFIFOMode:
    """Tests for FIFO mode (below congestion threshold)."""

    def test_fifo_order_below_threshold(self):
        """Below threshold, items are dequeued in FIFO order."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        queue.push("first")
        queue.push("second")
        queue.push("third")

        assert queue.mode == "FIFO"
        assert queue.pop() == "first"
        assert queue.pop() == "second"
        assert queue.pop() == "third"

    def test_peek_in_fifo_mode(self):
        """Peek returns oldest item in FIFO mode."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        queue.push("first")
        queue.push("second")

        assert queue.peek() == "first"
        assert len(queue) == 2  # Not removed

    def test_tracks_fifo_dequeues(self):
        """Statistics track FIFO dequeues."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        queue.push("a")
        queue.push("b")
        queue.pop()
        queue.pop()

        assert queue.stats.dequeued_fifo == 2
        assert queue.stats.dequeued_lifo == 0


class TestAdaptiveLIFOLIFOMode:
    """Tests for LIFO mode (at or above congestion threshold)."""

    def test_lifo_order_at_threshold(self):
        """At threshold, items are dequeued in LIFO order."""
        queue = AdaptiveLIFO(congestion_threshold=3)

        queue.push("first")
        queue.push("second")
        queue.push("third")  # Now at threshold

        assert queue.mode == "LIFO"
        assert queue.pop() == "third"  # Most recent first

    def test_lifo_order_above_threshold(self):
        """Above threshold, items are dequeued in LIFO order."""
        queue = AdaptiveLIFO(congestion_threshold=3)

        queue.push("first")
        queue.push("second")
        queue.push("third")
        queue.push("fourth")

        assert queue.mode == "LIFO"
        assert queue.pop() == "fourth"
        assert queue.pop() == "third"  # Still above threshold

    def test_peek_in_lifo_mode(self):
        """Peek returns newest item in LIFO mode."""
        queue = AdaptiveLIFO(congestion_threshold=2)

        queue.push("first")
        queue.push("second")

        assert queue.is_congested is True
        assert queue.peek() == "second"  # Most recent

    def test_tracks_lifo_dequeues(self):
        """Statistics track LIFO dequeues."""
        queue = AdaptiveLIFO(congestion_threshold=2)

        queue.push("a")
        queue.push("b")
        queue.pop()  # LIFO mode

        assert queue.stats.dequeued_lifo == 1
        assert queue.stats.dequeued_fifo == 0


class TestAdaptiveLIFOModeSwitch:
    """Tests for mode switching behavior."""

    def test_switches_from_fifo_to_lifo(self):
        """Queue switches from FIFO to LIFO when threshold reached."""
        queue = AdaptiveLIFO(congestion_threshold=3)

        queue.push("a")
        queue.push("b")
        assert queue.mode == "FIFO"

        queue.push("c")  # Now at threshold
        assert queue.mode == "LIFO"

    def test_switches_from_lifo_to_fifo(self):
        """Queue switches from LIFO to FIFO when below threshold."""
        queue = AdaptiveLIFO(congestion_threshold=3)

        queue.push("a")
        queue.push("b")
        queue.push("c")
        assert queue.mode == "LIFO"

        queue.pop()  # Now below threshold
        assert queue.mode == "FIFO"

    def test_tracks_mode_switches(self):
        """Statistics track mode switches."""
        queue = AdaptiveLIFO(congestion_threshold=2)

        # Start FIFO
        queue.push("a")
        queue.pop()  # FIFO dequeue, mode didn't switch yet

        # Push to threshold - switch to LIFO
        queue.push("a")
        queue.push("b")
        queue.pop()  # LIFO dequeue - first switch

        # Now below threshold - switch to FIFO
        queue.pop()  # FIFO dequeue - second switch

        assert queue.stats.mode_switches == 2

    def test_mixed_mode_behavior(self):
        """Queue correctly mixes FIFO and LIFO as load varies."""
        queue = AdaptiveLIFO(congestion_threshold=3)

        # Add items
        queue.push(1)
        queue.push(2)
        result1 = queue.pop()  # FIFO
        assert result1 == 1

        # Add more to reach threshold
        queue.push(3)
        queue.push(4)
        queue.push(5)  # Now: 2, 3, 4, 5 (4 items >= threshold 3)

        result2 = queue.pop()  # LIFO
        assert result2 == 5

        result3 = queue.pop()  # Still LIFO (3 items)
        assert result3 == 4

        result4 = queue.pop()  # Now FIFO (2 items)
        assert result4 == 2


class TestAdaptiveLIFOCapacity:
    """Tests for capacity handling."""

    def test_respects_capacity(self):
        """AdaptiveLIFO respects capacity limit."""
        queue = AdaptiveLIFO(congestion_threshold=2, capacity=3)

        assert queue.push("a") is True
        assert queue.push("b") is True
        assert queue.push("c") is True
        assert queue.push("d") is False

        assert len(queue) == 3
        assert queue.stats.capacity_rejected == 1

    def test_capacity_rejection_tracked(self):
        """Capacity rejections are tracked in statistics."""
        queue = AdaptiveLIFO(congestion_threshold=2, capacity=2)

        queue.push("a")
        queue.push("b")
        queue.push("c")  # Rejected
        queue.push("d")  # Rejected

        assert queue.stats.capacity_rejected == 2


class TestAdaptiveLIFOBasicOperations:
    """Tests for basic queue operations."""

    def test_pop_empty_returns_none(self):
        """pop() returns None when queue is empty."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        assert queue.pop() is None

    def test_peek_empty_returns_none(self):
        """peek() returns None when queue is empty."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        assert queue.peek() is None

    def test_is_empty(self):
        """is_empty() returns correct state."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        assert queue.is_empty() is True

        queue.push("item")
        assert queue.is_empty() is False

        queue.pop()
        assert queue.is_empty() is True

    def test_len(self):
        """__len__() returns correct count."""
        queue = AdaptiveLIFO(congestion_threshold=5)

        assert len(queue) == 0

        queue.push("a")
        queue.push("b")
        assert len(queue) == 2

        queue.pop()
        assert len(queue) == 1

    def test_is_congested_property(self):
        """is_congested property reflects queue state."""
        queue = AdaptiveLIFO(congestion_threshold=2)

        assert queue.is_congested is False

        queue.push("a")
        assert queue.is_congested is False

        queue.push("b")
        assert queue.is_congested is True

        queue.pop()
        assert queue.is_congested is False
