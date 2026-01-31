"""Tests for Topic."""

import pytest

from happysimulator.components.messaging import Topic
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class DummySubscriber(Entity):
    """Simple subscriber for testing."""

    def __init__(self, name: str = "subscriber"):
        super().__init__(name)
        self.messages_received = []

    def handle_event(self, event: Event) -> list[Event]:
        self.messages_received.append(event)
        return []


class TestTopicCreation:
    """Tests for Topic creation."""

    def test_creates_with_defaults(self):
        """Topic is created with default values."""
        topic = Topic(name="notifications")

        assert topic.name == "notifications"
        assert topic.subscriber_count == 0

    def test_creates_with_custom_settings(self):
        """Topic is created with custom settings."""
        topic = Topic(
            name="events",
            delivery_latency=0.005,
            max_subscribers=100,
        )

        assert topic.name == "events"
        assert topic.max_subscribers == 100

    def test_rejects_negative_latency(self):
        """Rejects negative delivery_latency."""
        with pytest.raises(ValueError):
            Topic(name="t", delivery_latency=-1)


class TestTopicSubscription:
    """Tests for subscription management."""

    def test_subscribe_entity(self):
        """Can subscribe an entity."""
        topic = Topic(name="notifications")
        subscriber = DummySubscriber()

        topic.subscribe(subscriber)

        assert topic.subscriber_count == 1
        assert subscriber in topic.subscribers

    def test_subscribe_multiple(self):
        """Can subscribe multiple entities."""
        topic = Topic(name="notifications")
        subscribers = [DummySubscriber(f"sub{i}") for i in range(5)]

        for s in subscribers:
            topic.subscribe(s)

        assert topic.subscriber_count == 5

    def test_unsubscribe_entity(self):
        """Can unsubscribe an entity."""
        topic = Topic(name="notifications")
        subscriber = DummySubscriber()

        topic.subscribe(subscriber)
        topic.unsubscribe(subscriber)

        assert topic.subscriber_count == 0
        assert subscriber not in topic.subscribers

    def test_max_subscribers_enforced(self):
        """Max subscribers limit is enforced."""
        topic = Topic(name="notifications", max_subscribers=2)

        topic.subscribe(DummySubscriber("sub1"))
        topic.subscribe(DummySubscriber("sub2"))

        with pytest.raises(RuntimeError):
            topic.subscribe(DummySubscriber("sub3"))

    def test_resubscribe_after_unsubscribe(self):
        """Can resubscribe after unsubscribing."""
        topic = Topic(name="notifications")
        subscriber = DummySubscriber()

        topic.subscribe(subscriber)
        topic.unsubscribe(subscriber)
        topic.subscribe(subscriber)

        assert topic.subscriber_count == 1


class TestTopicPublish:
    """Tests for message publishing."""

    def test_publish_to_all_subscribers(self):
        """Publishing delivers to all subscribers."""
        topic = Topic(name="notifications")
        subscribers = [DummySubscriber(f"sub{i}") for i in range(3)]
        for s in subscribers:
            topic.subscribe(s)

        message = Event(
            time=Instant.Epoch,
            event_type="notification",
            callback=lambda e: None,
        )

        gen = topic.publish(message)
        events = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            events = e.value

        assert len(events) == 3
        assert topic.stats.messages_published == 1
        assert topic.stats.messages_delivered == 3

    def test_publish_with_no_subscribers(self):
        """Publishing with no subscribers delivers nothing."""
        topic = Topic(name="notifications")

        message = Event(
            time=Instant.Epoch,
            event_type="notification",
            callback=lambda e: None,
        )

        gen = topic.publish(message)
        events = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            events = e.value

        assert len(events) == 0
        assert topic.stats.messages_published == 1
        assert topic.stats.messages_delivered == 0

    def test_publish_sync(self):
        """Synchronous publish works correctly."""
        topic = Topic(name="notifications")
        subscribers = [DummySubscriber(f"sub{i}") for i in range(2)]
        for s in subscribers:
            topic.subscribe(s)

        message = Event(
            time=Instant.Epoch,
            event_type="notification",
            callback=lambda e: None,
        )

        events = topic.publish_sync(message)

        assert len(events) == 2
        assert topic.stats.messages_published == 1

    def test_unsubscribed_entity_not_delivered(self):
        """Unsubscribed entities don't receive messages."""
        topic = Topic(name="notifications")
        sub1 = DummySubscriber("sub1")
        sub2 = DummySubscriber("sub2")

        topic.subscribe(sub1)
        topic.subscribe(sub2)
        topic.unsubscribe(sub1)

        message = Event(
            time=Instant.Epoch,
            event_type="notification",
            callback=lambda e: None,
        )

        gen = topic.publish(message)
        events = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            events = e.value

        # Only sub2 should receive
        assert len(events) == 1
        assert events[0].target == sub2


class TestTopicMessageRetention:
    """Tests for message retention."""

    def test_retain_messages(self):
        """Can retain messages for late subscribers."""
        topic = Topic(name="notifications")
        topic.set_retain_messages(True, max_history=10)

        # Publish some messages
        for i in range(5):
            message = Event(
                time=Instant.Epoch,
                event_type=f"notification{i}",
                callback=lambda e: None,
            )
            topic.publish_sync(message)

        # Late subscriber with replay
        late_sub = DummySubscriber("late")
        events = topic.subscribe(late_sub, replay_history=True)

        assert len(events) == 5

    def test_no_replay_when_disabled(self):
        """No replay when retain_messages is off."""
        topic = Topic(name="notifications")
        # retain_messages is False by default

        # Publish some messages
        for i in range(5):
            message = Event(
                time=Instant.Epoch,
                event_type=f"notification{i}",
                callback=lambda e: None,
            )
            topic.publish_sync(message)

        # Late subscriber with replay requested but nothing retained
        late_sub = DummySubscriber("late")
        events = topic.subscribe(late_sub, replay_history=True)

        assert len(events) == 0


class TestTopicSubscriptionDetails:
    """Tests for subscription details."""

    def test_get_subscription(self):
        """Can get subscription details."""
        topic = Topic(name="notifications")
        subscriber = DummySubscriber()

        topic.subscribe(subscriber)

        subscription = topic.get_subscription(subscriber)

        assert subscription is not None
        assert subscription.subscriber == subscriber
        assert subscription.active is True
        assert subscription.messages_received == 0

    def test_subscription_tracks_messages(self):
        """Subscription tracks messages received."""
        topic = Topic(name="notifications")
        subscriber = DummySubscriber()
        topic.subscribe(subscriber)

        for _ in range(3):
            message = Event(
                time=Instant.Epoch,
                event_type="notification",
                callback=lambda e: None,
            )
            topic.publish_sync(message)

        subscription = topic.get_subscription(subscriber)

        assert subscription.messages_received == 3

    def test_get_subscription_not_found(self):
        """Returns None for non-subscriber."""
        topic = Topic(name="notifications")
        stranger = DummySubscriber()

        subscription = topic.get_subscription(stranger)

        assert subscription is None


class TestTopicStatistics:
    """Tests for Topic statistics."""

    def test_tracks_subscriber_stats(self):
        """Statistics track subscriber additions/removals."""
        topic = Topic(name="notifications")
        sub1 = DummySubscriber("sub1")
        sub2 = DummySubscriber("sub2")

        topic.subscribe(sub1)
        topic.subscribe(sub2)
        topic.unsubscribe(sub1)

        assert topic.stats.subscribers_added == 2
        assert topic.stats.subscribers_removed == 1

    def test_avg_delivery_latency(self):
        """Can calculate average delivery latency."""
        topic = Topic(name="notifications", delivery_latency=0.001)
        subscribers = [DummySubscriber(f"sub{i}") for i in range(2)]
        for s in subscribers:
            topic.subscribe(s)

        for _ in range(5):
            message = Event(
                time=Instant.Epoch,
                event_type="notification",
                callback=lambda e: None,
            )
            gen = topic.publish(message)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # 5 messages * 2 subscribers = 10 deliveries
        assert len(topic.stats.delivery_latencies) == 10
        assert topic.stats.avg_delivery_latency == pytest.approx(0.001)
