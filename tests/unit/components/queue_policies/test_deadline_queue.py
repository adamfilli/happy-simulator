"""Tests for DeadlineQueue."""

import pytest

from happysimulator.components.queue_policies import DeadlineQueue
from happysimulator.core.temporal import Instant


class TestDeadlineQueueCreation:
    """Tests for DeadlineQueue creation."""

    def test_creates_with_basic_parameters(self):
        """DeadlineQueue can be created with basic parameters."""
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
        )

        assert queue.capacity == float("inf")

    def test_creates_with_capacity(self):
        """DeadlineQueue can be created with capacity."""
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
            capacity=100,
        )

        assert queue.capacity == 100

    def test_rejects_invalid_capacity(self):
        """DeadlineQueue rejects invalid capacity."""
        with pytest.raises(ValueError):
            DeadlineQueue(
                get_deadline=lambda item: item["deadline"],
                capacity=0,
            )


class TestDeadlineQueueBehavior:
    """Tests for DeadlineQueue behavior."""

    def test_dequeues_by_deadline_order(self):
        """DeadlineQueue dequeues items by earliest deadline first."""
        queue = DeadlineQueue(get_deadline=lambda item: item["deadline"])

        # Add items with different deadlines (not in order)
        queue.push({"name": "late", "deadline": Instant.from_seconds(3.0)})
        queue.push({"name": "early", "deadline": Instant.from_seconds(1.0)})
        queue.push({"name": "middle", "deadline": Instant.from_seconds(2.0)})

        # Should dequeue in deadline order
        assert queue.pop()["name"] == "early"
        assert queue.pop()["name"] == "middle"
        assert queue.pop()["name"] == "late"

    def test_stable_ordering_same_deadline(self):
        """Items with same deadline are dequeued in insertion order."""
        deadline = Instant.from_seconds(1.0)
        queue = DeadlineQueue(get_deadline=lambda item: item["deadline"])

        queue.push({"name": "first", "deadline": deadline})
        queue.push({"name": "second", "deadline": deadline})
        queue.push({"name": "third", "deadline": deadline})

        assert queue.pop()["name"] == "first"
        assert queue.pop()["name"] == "second"
        assert queue.pop()["name"] == "third"

    def test_respects_capacity(self):
        """DeadlineQueue respects capacity limit."""
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
            capacity=3,
        )

        for i in range(5):
            queue.push({"deadline": Instant.from_seconds(float(i))})

        assert len(queue) == 3
        assert queue.stats.capacity_rejected == 2

    def test_pop_empty_returns_none(self):
        """DeadlineQueue.pop() returns None when empty."""
        queue = DeadlineQueue(get_deadline=lambda item: item["deadline"])

        assert queue.pop() is None

    def test_peek_returns_earliest_deadline(self):
        """DeadlineQueue.peek() returns item with earliest deadline."""
        queue = DeadlineQueue(get_deadline=lambda item: item["deadline"])

        queue.push({"name": "late", "deadline": Instant.from_seconds(2.0)})
        queue.push({"name": "early", "deadline": Instant.from_seconds(1.0)})

        assert queue.peek()["name"] == "early"
        assert len(queue) == 2  # Not removed

    def test_is_empty(self):
        """DeadlineQueue.is_empty() returns correct state."""
        queue = DeadlineQueue(get_deadline=lambda item: item["deadline"])

        assert queue.is_empty() is True

        queue.push({"deadline": Instant.from_seconds(1.0)})
        assert queue.is_empty() is False

        queue.pop()
        assert queue.is_empty() is True


class TestDeadlineQueueExpiration:
    """Tests for DeadlineQueue expiration handling."""

    def test_drops_expired_items_on_pop(self):
        """DeadlineQueue drops expired items when popping."""
        current_time = Instant.from_seconds(5.0)
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
            clock_func=lambda: current_time,
        )

        # Add items with past and future deadlines
        queue.push({"name": "expired1", "deadline": Instant.from_seconds(1.0)})
        queue.push({"name": "expired2", "deadline": Instant.from_seconds(2.0)})
        queue.push({"name": "valid", "deadline": Instant.from_seconds(10.0)})

        # Should skip expired and return valid
        result = queue.pop()
        assert result["name"] == "valid"
        assert queue.stats.expired == 2

    def test_pop_returns_none_all_expired(self):
        """DeadlineQueue.pop() returns None if all items expired."""
        current_time = Instant.from_seconds(10.0)
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
            clock_func=lambda: current_time,
        )

        queue.push({"deadline": Instant.from_seconds(1.0)})
        queue.push({"deadline": Instant.from_seconds(2.0)})

        assert queue.pop() is None
        assert queue.stats.expired == 2

    def test_purge_expired(self):
        """DeadlineQueue.purge_expired() removes expired items."""
        current_time = Instant.from_seconds(5.0)
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
            clock_func=lambda: current_time,
        )

        queue.push({"deadline": Instant.from_seconds(1.0)})
        queue.push({"deadline": Instant.from_seconds(2.0)})
        queue.push({"deadline": Instant.from_seconds(10.0)})

        removed = queue.purge_expired()

        assert removed == 2
        assert len(queue) == 1

    def test_count_expired(self):
        """DeadlineQueue.count_expired() counts expired items."""
        current_time = Instant.from_seconds(5.0)
        queue = DeadlineQueue(
            get_deadline=lambda item: item["deadline"],
            clock_func=lambda: current_time,
        )

        queue.push({"deadline": Instant.from_seconds(1.0)})
        queue.push({"deadline": Instant.from_seconds(2.0)})
        queue.push({"deadline": Instant.from_seconds(10.0)})

        assert queue.count_expired() == 2
        assert queue.count_valid() == 1

    def test_no_expiration_without_clock(self):
        """DeadlineQueue doesn't expire items without clock function."""
        queue = DeadlineQueue(get_deadline=lambda item: item["deadline"])

        queue.push({"name": "old", "deadline": Instant.from_seconds(1.0)})

        # Without clock, should return item regardless of deadline
        result = queue.pop()
        assert result["name"] == "old"
        assert queue.stats.expired == 0
