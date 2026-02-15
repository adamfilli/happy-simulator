"""Tests for CoDelQueue (Controlled Delay)."""

import pytest

from happysimulator.components.queue_policies import CoDelQueue
from happysimulator.core.temporal import Instant


class TestCoDelQueueCreation:
    """Tests for CoDelQueue creation."""

    def test_creates_with_default_parameters(self):
        """CoDelQueue can be created with default parameters."""
        queue = CoDelQueue()

        assert queue.target_delay == 0.005  # 5ms
        assert queue.interval == 0.100  # 100ms
        assert queue.capacity == float("inf")

    def test_creates_with_custom_parameters(self):
        """CoDelQueue can be created with custom parameters."""
        queue = CoDelQueue(
            target_delay=0.010,
            interval=0.050,
            capacity=100,
        )

        assert queue.target_delay == 0.010
        assert queue.interval == 0.050
        assert queue.capacity == 100

    def test_rejects_invalid_target_delay(self):
        """CoDelQueue rejects invalid target_delay."""
        with pytest.raises(ValueError):
            CoDelQueue(target_delay=0)

        with pytest.raises(ValueError):
            CoDelQueue(target_delay=-1)

    def test_rejects_invalid_interval(self):
        """CoDelQueue rejects invalid interval."""
        with pytest.raises(ValueError):
            CoDelQueue(interval=0)

        with pytest.raises(ValueError):
            CoDelQueue(interval=-1)

    def test_rejects_invalid_capacity(self):
        """CoDelQueue rejects invalid capacity."""
        with pytest.raises(ValueError):
            CoDelQueue(capacity=0)

        with pytest.raises(ValueError):
            CoDelQueue(capacity=-1)


class MockClock:
    """Mock clock for testing time-dependent behavior."""

    def __init__(self, start_seconds: float = 0.0):
        self._time = Instant.from_seconds(start_seconds)

    def __call__(self) -> Instant:
        return self._time

    def advance(self, seconds: float):
        """Advance time by given seconds."""
        from happysimulator.core.temporal import Duration

        self._time = self._time + Duration.from_seconds(seconds)


class TestCoDelQueueBasicOperations:
    """Tests for basic CoDelQueue operations."""

    def test_push_and_pop(self):
        """CoDelQueue can push and pop items."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        queue.push("item")
        result = queue.pop()

        assert result == "item"

    def test_fifo_order(self):
        """CoDelQueue maintains FIFO order."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        queue.push("first")
        queue.push("second")
        queue.push("third")

        assert queue.pop() == "first"
        assert queue.pop() == "second"
        assert queue.pop() == "third"

    def test_pop_empty_returns_none(self):
        """pop() returns None when queue is empty."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        assert queue.pop() is None

    def test_peek_returns_next_item(self):
        """peek() returns next item without removing it."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        queue.push("item")

        assert queue.peek() == "item"
        assert len(queue) == 1

    def test_peek_empty_returns_none(self):
        """peek() returns None when queue is empty."""
        queue = CoDelQueue()

        assert queue.peek() is None

    def test_is_empty(self):
        """is_empty() returns correct state."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        assert queue.is_empty() is True

        queue.push("item")
        assert queue.is_empty() is False

        queue.pop()
        assert queue.is_empty() is True

    def test_len(self):
        """__len__() returns correct count."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        assert len(queue) == 0

        queue.push("a")
        queue.push("b")
        assert len(queue) == 2


class TestCoDelQueueCapacity:
    """Tests for capacity handling."""

    def test_respects_capacity(self):
        """CoDelQueue respects capacity limit."""
        clock = MockClock()
        queue = CoDelQueue(capacity=3, clock_func=clock)

        assert queue.push("a") is True
        assert queue.push("b") is True
        assert queue.push("c") is True
        assert queue.push("d") is False

        assert len(queue) == 3

    def test_tracks_capacity_rejections(self):
        """Capacity rejections are tracked in statistics."""
        clock = MockClock()
        queue = CoDelQueue(capacity=2, clock_func=clock)

        queue.push("a")
        queue.push("b")
        queue.push("c")  # Rejected

        assert queue.stats.capacity_rejected == 1


class TestCoDelQueueClockRequirement:
    """Tests for clock function requirement."""

    def test_push_requires_clock(self):
        """push() requires clock function to be set."""
        queue = CoDelQueue()

        with pytest.raises(RuntimeError):
            queue.push("item")

    def test_set_clock(self):
        """set_clock() allows setting clock after creation."""
        queue = CoDelQueue()
        clock = MockClock()

        queue.set_clock(clock)
        queue.push("item")  # Should not raise

        assert len(queue) == 1


class TestCoDelQueueDelayTracking:
    """Tests for CoDel's delay-based dropping."""

    def test_no_drop_below_target(self):
        """No drops when delay is below target."""
        clock = MockClock()
        queue = CoDelQueue(
            target_delay=0.005,  # 5ms
            interval=0.100,
            clock_func=clock,
        )

        # Add items
        for i in range(10):
            queue.push(i)

        # Advance time less than target delay
        clock.advance(0.003)

        # Dequeue all - none should be dropped
        results = []
        while not queue.is_empty():
            result = queue.pop()
            if result is not None:
                results.append(result)

        assert queue.stats.dropped == 0
        assert len(results) == 10

    def test_starts_dropping_on_persistent_delay(self):
        """CoDel starts dropping when delay exceeds target for full interval."""
        clock = MockClock()
        queue = CoDelQueue(
            target_delay=0.005,  # 5ms
            interval=0.050,  # 50ms interval
            clock_func=clock,
        )

        # Add items
        for i in range(20):
            queue.push(i)

        # First pop after delay sets first_above_time
        clock.advance(0.010)  # 10ms - above target
        queue.pop()  # Sets first_above_time to now + interval = 0.060

        # Advance past first_above_time to trigger drop state
        clock.advance(0.100)  # Now at 110ms, well past first_above_time

        # Dequeue more items - should enter drop state and drop
        for _ in range(10):
            queue.pop()

        # Should have dropped some items and/or entered drop intervals
        assert queue.stats.dropped > 0 or queue.stats.drop_intervals > 0

    def test_exits_dropping_when_delay_acceptable(self):
        """CoDel exits dropping state when delay becomes acceptable."""
        clock = MockClock()
        queue = CoDelQueue(
            target_delay=0.005,
            interval=0.100,
            clock_func=clock,
        )

        # Fill queue
        for i in range(10):
            queue.push(i)

        # Advance past interval
        clock.advance(0.200)

        # Drain most of queue
        for _ in range(9):
            queue.pop()

        # Add fresh item
        queue.push("new")

        # Small time advance - new item has low delay
        clock.advance(0.001)

        # Should not be in dropping state for fresh item
        assert queue.dropping is False or queue.is_empty()

    def test_dropping_property(self):
        """dropping property reflects queue state."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        assert queue.dropping is False


class TestCoDelQueueStatistics:
    """Tests for CoDel statistics tracking."""

    def test_tracks_enqueued(self):
        """Statistics track enqueued items."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        queue.push("a")
        queue.push("b")
        queue.push("c")

        assert queue.stats.enqueued == 3

    def test_tracks_dequeued(self):
        """Statistics track dequeued items."""
        clock = MockClock()
        queue = CoDelQueue(clock_func=clock)

        queue.push("a")
        queue.push("b")
        queue.pop()
        queue.pop()

        assert queue.stats.dequeued == 2
