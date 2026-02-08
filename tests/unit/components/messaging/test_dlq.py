"""Tests for DeadLetterQueue."""

import pytest

from happysimulator.components.messaging import (
    DeadLetterQueue,
    MessageQueue,
    Message,
    MessageState,
)
from happysimulator.core.callback_entity import NullEntity
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

_null = NullEntity()


def create_test_message(msg_id: str, delivery_count: int = 1) -> Message:
    """Create a test message."""
    return Message(
        id=msg_id,
        payload=Event(
            time=Instant.Epoch,
            event_type="test",
            target=_null,
        ),
        created_at=Instant.Epoch,
        state=MessageState.REJECTED,
        delivery_count=delivery_count,
    )


class TestDeadLetterQueueCreation:
    """Tests for DeadLetterQueue creation."""

    def test_creates_with_defaults(self):
        """DeadLetterQueue is created with default values."""
        dlq = DeadLetterQueue(name="orders_dlq")

        assert dlq.name == "orders_dlq"
        assert dlq.message_count == 0
        assert dlq.capacity is None

    def test_creates_with_capacity(self):
        """DeadLetterQueue is created with capacity."""
        dlq = DeadLetterQueue(name="dlq", capacity=100)

        assert dlq.capacity == 100
        assert not dlq.is_full


class TestDeadLetterQueueAddMessage:
    """Tests for adding messages."""

    def test_add_message(self):
        """Can add a message."""
        dlq = DeadLetterQueue(name="dlq")
        msg = create_test_message("msg1")

        result = dlq.add_message(msg)

        assert result is True
        assert dlq.message_count == 1
        assert dlq.stats.messages_received == 1

    def test_add_multiple_messages(self):
        """Can add multiple messages."""
        dlq = DeadLetterQueue(name="dlq")

        for i in range(5):
            msg = create_test_message(f"msg{i}")
            dlq.add_message(msg)

        assert dlq.message_count == 5

    def test_add_at_capacity_removes_oldest(self):
        """Adding at capacity removes oldest message."""
        dlq = DeadLetterQueue(name="dlq", capacity=3)

        for i in range(4):
            msg = create_test_message(f"msg{i}")
            dlq.add_message(msg)

        assert dlq.message_count == 3
        # Oldest (msg0) should be gone
        assert dlq.messages[0].id == "msg1"
        assert dlq.stats.messages_discarded == 1


class TestDeadLetterQueueAccess:
    """Tests for accessing messages."""

    def test_get_message_by_index(self):
        """Can get message by index."""
        dlq = DeadLetterQueue(name="dlq")
        msg = create_test_message("msg1")
        dlq.add_message(msg)

        result = dlq.get_message(0)

        assert result is not None
        assert result.id == "msg1"

    def test_get_message_invalid_index(self):
        """Returns None for invalid index."""
        dlq = DeadLetterQueue(name="dlq")

        result = dlq.get_message(0)

        assert result is None

    def test_peek(self):
        """Can peek at oldest message."""
        dlq = DeadLetterQueue(name="dlq")
        for i in range(3):
            dlq.add_message(create_test_message(f"msg{i}"))

        result = dlq.peek()

        assert result is not None
        assert result.id == "msg0"
        assert dlq.message_count == 3  # Not removed

    def test_pop(self):
        """Can pop oldest message."""
        dlq = DeadLetterQueue(name="dlq")
        for i in range(3):
            dlq.add_message(create_test_message(f"msg{i}"))

        result = dlq.pop()

        assert result is not None
        assert result.id == "msg0"
        assert dlq.message_count == 2

    def test_pop_empty(self):
        """Pop from empty DLQ returns None."""
        dlq = DeadLetterQueue(name="dlq")

        result = dlq.pop()

        assert result is None

    def test_messages_property(self):
        """Can get all messages."""
        dlq = DeadLetterQueue(name="dlq")
        for i in range(3):
            dlq.add_message(create_test_message(f"msg{i}"))

        messages = dlq.messages

        assert len(messages) == 3
        assert messages[0].id == "msg0"
        assert messages[2].id == "msg2"


class TestDeadLetterQueueClear:
    """Tests for clearing messages."""

    def test_clear(self):
        """Can clear all messages."""
        dlq = DeadLetterQueue(name="dlq")
        for i in range(5):
            dlq.add_message(create_test_message(f"msg{i}"))

        count = dlq.clear()

        assert count == 5
        assert dlq.message_count == 0
        assert dlq.stats.messages_discarded == 5


class TestDeadLetterQueueFiltering:
    """Tests for filtering messages."""

    def test_get_by_delivery_count(self):
        """Can filter by delivery count."""
        dlq = DeadLetterQueue(name="dlq")
        dlq.add_message(create_test_message("msg1", delivery_count=1))
        dlq.add_message(create_test_message("msg2", delivery_count=3))
        dlq.add_message(create_test_message("msg3", delivery_count=5))

        result = dlq.get_messages_by_delivery_count(3)

        assert len(result) == 2
        assert result[0].id == "msg2"
        assert result[1].id == "msg3"


class TestDeadLetterQueueReprocess:
    """Tests for reprocessing messages."""

    def test_reprocess_creates_event(self):
        """Reprocessing creates republish event."""
        dlq = DeadLetterQueue(name="dlq")
        msg = create_test_message("msg1", delivery_count=3)
        dlq.add_message(msg)

        # Create a simple queue for reprocessing
        class MockQueue(Entity):
            def handle_event(self, event):
                return []

        queue = MockQueue("queue")
        event = dlq.reprocess(msg, queue)

        assert event is not None
        assert event.event_type == "republish"
        assert event.target == queue
        assert event.context['original_message_id'] == "msg1"
        assert dlq.message_count == 0
        assert dlq.stats.messages_reprocessed == 1

    def test_reprocess_all(self):
        """Can reprocess all messages."""
        dlq = DeadLetterQueue(name="dlq")
        for i in range(3):
            dlq.add_message(create_test_message(f"msg{i}"))

        class MockQueue(Entity):
            def handle_event(self, event):
                return []

        queue = MockQueue("queue")
        events = dlq.reprocess_all(queue)

        assert len(events) == 3
        assert dlq.message_count == 0
        assert dlq.stats.messages_reprocessed == 3

    def test_reprocess_unknown_message(self):
        """Reprocessing unknown message returns None."""
        dlq = DeadLetterQueue(name="dlq")
        msg = create_test_message("unknown")

        class MockQueue(Entity):
            def handle_event(self, event):
                return []

        queue = MockQueue("queue")
        event = dlq.reprocess(msg, queue)

        assert event is None


class TestDeadLetterQueueStatistics:
    """Tests for DeadLetterQueue statistics."""

    def test_tracks_received(self):
        """Statistics track received messages."""
        dlq = DeadLetterQueue(name="dlq")

        for i in range(5):
            dlq.add_message(create_test_message(f"msg{i}"))

        assert dlq.stats.messages_received == 5

    def test_tracks_discarded(self):
        """Statistics track discarded messages."""
        dlq = DeadLetterQueue(name="dlq", capacity=2)

        for i in range(5):
            dlq.add_message(create_test_message(f"msg{i}"))

        # 3 oldest should be discarded
        assert dlq.stats.messages_discarded == 3

    def test_tracks_reprocessed(self):
        """Statistics track reprocessed messages."""
        dlq = DeadLetterQueue(name="dlq")
        for i in range(3):
            dlq.add_message(create_test_message(f"msg{i}"))

        class MockQueue(Entity):
            def handle_event(self, event):
                return []

        queue = MockQueue("queue")
        dlq.reprocess_all(queue)

        assert dlq.stats.messages_reprocessed == 3
