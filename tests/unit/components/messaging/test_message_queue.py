"""Tests for MessageQueue."""

import pytest

from happysimulator.components.messaging import (
    MessageQueue,
    MessageState,
    DeadLetterQueue,
)
from happysimulator.core.callback_entity import NullEntity
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

_null = NullEntity()


class DummyConsumer(Entity):
    """Simple consumer for testing."""

    def __init__(self, name: str = "consumer"):
        super().__init__(name)
        self.messages_received = []

    def handle_event(self, event: Event) -> list[Event]:
        self.messages_received.append(event)
        return []


class TestMessageQueueCreation:
    """Tests for MessageQueue creation."""

    def test_creates_with_defaults(self):
        """MessageQueue is created with default values."""
        queue = MessageQueue(name="orders")

        assert queue.name == "orders"
        assert queue.pending_count == 0
        assert queue.in_flight_count == 0
        assert queue.consumer_count == 0

    def test_creates_with_custom_settings(self):
        """MessageQueue is created with custom settings."""
        queue = MessageQueue(
            name="events",
            delivery_latency=0.005,
            redelivery_delay=60.0,
            max_redeliveries=5,
            capacity=1000,
        )

        assert queue.name == "events"
        assert queue.capacity == 1000

    def test_rejects_invalid_redelivery_delay(self):
        """Rejects non-positive redelivery_delay."""
        with pytest.raises(ValueError):
            MessageQueue(name="q", redelivery_delay=0)

        with pytest.raises(ValueError):
            MessageQueue(name="q", redelivery_delay=-1)

    def test_rejects_negative_max_redeliveries(self):
        """Rejects negative max_redeliveries."""
        with pytest.raises(ValueError):
            MessageQueue(name="q", max_redeliveries=-1)


class TestMessageQueueSubscription:
    """Tests for subscription management."""

    def test_subscribe_consumer(self):
        """Can subscribe a consumer."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()

        queue.subscribe(consumer)

        assert queue.consumer_count == 1

    def test_subscribe_multiple_consumers(self):
        """Can subscribe multiple consumers."""
        queue = MessageQueue(name="orders")
        consumers = [DummyConsumer(f"consumer{i}") for i in range(3)]

        for c in consumers:
            queue.subscribe(c)

        assert queue.consumer_count == 3

    def test_unsubscribe_consumer(self):
        """Can unsubscribe a consumer."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()

        queue.subscribe(consumer)
        queue.unsubscribe(consumer)

        assert queue.consumer_count == 0

    def test_duplicate_subscribe_ignored(self):
        """Subscribing same consumer twice is ignored."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()

        queue.subscribe(consumer)
        queue.subscribe(consumer)

        assert queue.consumer_count == 1


class TestMessageQueuePublish:
    """Tests for message publishing."""

    def test_publish_message(self):
        """Can publish a message."""
        queue = MessageQueue(name="orders")
        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        gen = queue.publish(message)
        message_id = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            message_id = e.value

        assert message_id is not None
        assert queue.pending_count == 1
        assert queue.stats.messages_published == 1

    def test_publish_multiple_messages(self):
        """Can publish multiple messages."""
        queue = MessageQueue(name="orders")

        for i in range(5):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert queue.pending_count == 5
        assert queue.stats.messages_published == 5

    def test_publish_respects_capacity(self):
        """Publishing respects queue capacity."""
        queue = MessageQueue(name="orders", capacity=2)

        for i in range(2):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert queue.is_full

        # Third publish should fail
        message = Event(
            time=Instant.Epoch,
            event_type="order3",
            target=_null,
        )
        with pytest.raises(RuntimeError):
            gen = queue.publish(message)
            next(gen)


class TestMessageQueueDelivery:
    """Tests for message delivery."""

    def test_poll_delivers_message(self):
        """Polling delivers message to consumer."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        # Publish
        gen = queue.publish(message)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Poll
        gen = queue.poll()
        delivery_event = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            delivery_event = e.value

        assert delivery_event is not None
        assert delivery_event.target == consumer
        assert queue.pending_count == 0
        assert queue.in_flight_count == 1

    def test_poll_empty_queue(self):
        """Polling empty queue returns None."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        gen = queue.poll()
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result is None

    def test_poll_no_consumers(self):
        """Polling without consumers returns None."""
        queue = MessageQueue(name="orders")

        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        gen = queue.publish(message)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        gen = queue.poll()
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result is None
        assert queue.pending_count == 1  # Still pending

    def test_round_robin_delivery(self):
        """Messages are delivered round-robin to consumers."""
        queue = MessageQueue(name="orders")
        consumers = [DummyConsumer(f"consumer{i}") for i in range(3)]
        for c in consumers:
            queue.subscribe(c)

        # Publish 6 messages
        for i in range(6):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # Deliver all
        delivered_to = []
        for _ in range(6):
            gen = queue.poll()
            delivery = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                delivery = e.value
            if delivery:
                delivered_to.append(delivery.target.name)

        # Each consumer should get 2 messages
        assert delivered_to.count("consumer0") == 2
        assert delivered_to.count("consumer1") == 2
        assert delivered_to.count("consumer2") == 2


class TestMessageQueueAcknowledgment:
    """Tests for message acknowledgment."""

    def test_acknowledge_removes_message(self):
        """Acknowledging removes message from queue."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        # Publish and deliver
        gen = queue.publish(message)
        message_id = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            message_id = e.value

        gen = queue.poll()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Acknowledge
        queue.acknowledge(message_id)

        assert queue.in_flight_count == 0
        assert queue.get_message(message_id) is None
        assert queue.stats.messages_acknowledged == 1

    def test_reject_with_requeue(self):
        """Rejecting with requeue adds message back to pending."""
        queue = MessageQueue(name="orders", max_redeliveries=3)
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        # Publish and deliver
        gen = queue.publish(message)
        message_id = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            message_id = e.value

        gen = queue.poll()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Reject with requeue
        queue.reject(message_id, requeue=True)

        assert queue.in_flight_count == 0
        assert queue.pending_count == 1
        assert queue.stats.messages_rejected == 1

    def test_reject_exceeds_max_redeliveries(self):
        """Rejecting past max redeliveries discards message."""
        queue = MessageQueue(name="orders", max_redeliveries=2)
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        # Publish
        gen = queue.publish(message)
        message_id = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            message_id = e.value

        # Deliver and reject 3 times
        for _ in range(3):
            gen = queue.poll()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

            if queue.in_flight_count > 0:
                queue.reject(message_id, requeue=True)

        # Message should be gone
        assert queue.pending_count == 0
        assert queue.in_flight_count == 0
        assert queue.get_message(message_id) is None


class TestMessageQueueDeadLetter:
    """Tests for dead letter queue integration."""

    def test_messages_go_to_dlq(self):
        """Messages that exceed max redeliveries go to DLQ."""
        dlq = DeadLetterQueue(name="orders_dlq")
        queue = MessageQueue(
            name="orders",
            max_redeliveries=1,
            dead_letter_queue=dlq,
        )
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        message = Event(
            time=Instant.Epoch,
            event_type="order",
            target=_null,
        )

        # Publish
        gen = queue.publish(message)
        message_id = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            message_id = e.value

        # Deliver and reject twice
        for _ in range(2):
            gen = queue.poll()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

            if queue.in_flight_count > 0:
                queue.reject(message_id, requeue=True)

        # Message should be in DLQ
        assert dlq.message_count == 1
        assert queue.stats.messages_dead_lettered == 1


class TestMessageQueueStatistics:
    """Tests for MessageQueue statistics."""

    def test_tracks_delivery_stats(self):
        """Statistics track deliveries."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        # Publish and deliver 5 messages
        for i in range(5):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

            gen = queue.poll()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert queue.stats.messages_published == 5
        assert queue.stats.messages_delivered == 5

    def test_ack_rate_calculation(self):
        """Ack rate is calculated correctly."""
        queue = MessageQueue(name="orders")
        consumer = DummyConsumer()
        queue.subscribe(consumer)

        message_ids = []
        for i in range(4):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            message_id = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                message_id = e.value
            message_ids.append(message_id)

            gen = queue.poll()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # Ack 3, reject 1
        for i, msg_id in enumerate(message_ids):
            if i < 3:
                queue.acknowledge(msg_id)
            else:
                queue.reject(msg_id, requeue=False)

        assert queue.stats.ack_rate == 0.75
