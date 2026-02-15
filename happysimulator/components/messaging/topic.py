"""Topic implementation for pub/sub messaging.

Provides a publish-subscribe topic that broadcasts messages to all subscribers.

Example:
    from happysimulator.components.messaging import Topic

    # Create topic
    notifications = Topic(name="user_notifications")

    # Subscribe consumers
    notifications.subscribe(email_service)
    notifications.subscribe(push_service)
    notifications.subscribe(sms_service)

    # Publish message (goes to all subscribers)
    def handle_event(self, event):
        yield from notifications.publish(notification_event)
"""

import logging
from dataclasses import dataclass
from typing import Generator
from collections import deque

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass
class Subscription:
    """Represents a subscription to a topic."""

    subscriber: Entity
    subscribed_at: Instant
    messages_received: int = 0
    active: bool = True


@dataclass(frozen=True)
class TopicStats:
    """Statistics tracked by Topic."""

    messages_published: int = 0
    messages_delivered: int = 0
    subscribers_added: int = 0
    subscribers_removed: int = 0
    delivery_latencies: tuple[float, ...] = ()

    @property
    def avg_delivery_latency(self) -> float:
        """Average delivery latency."""
        if not self.delivery_latencies:
            return 0.0
        return sum(self.delivery_latencies) / len(self.delivery_latencies)


class Topic(Entity):
    """Pub/sub topic with multiple subscribers.

    Broadcasts published messages to all active subscribers.
    Each subscriber receives a copy of every message.

    Attributes:
        name: Entity name for identification.
        subscriber_count: Number of active subscribers.
    """

    def __init__(
        self,
        name: str,
        delivery_latency: float = 0.001,
        max_subscribers: int | None = None,
    ):
        """Initialize the topic.

        Args:
            name: Name for this topic entity.
            delivery_latency: Latency to deliver message to each subscriber.
            max_subscribers: Maximum number of subscribers (None for unlimited).

        Raises:
            ValueError: If parameters are invalid.
        """
        if delivery_latency < 0:
            raise ValueError(f"delivery_latency must be >= 0, got {delivery_latency}")

        super().__init__(name)
        self._delivery_latency = delivery_latency
        self._max_subscribers = max_subscribers

        # Subscriptions
        self._subscriptions: dict[Entity, Subscription] = {}

        # Message history (optional, for late subscribers)
        self._message_history: deque[Event] = deque(maxlen=100)
        self._retain_messages = False

        # Statistics
        self._messages_published = 0
        self._messages_delivered = 0
        self._subscribers_added = 0
        self._subscribers_removed = 0
        self._delivery_latencies: list[float] = []

    @property
    def stats(self) -> TopicStats:
        """Return a frozen snapshot of current statistics."""
        return TopicStats(
            messages_published=self._messages_published,
            messages_delivered=self._messages_delivered,
            subscribers_added=self._subscribers_added,
            subscribers_removed=self._subscribers_removed,
            delivery_latencies=tuple(self._delivery_latencies),
        )

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return sum(1 for sub in self._subscriptions.values() if sub.active)

    @property
    def subscribers(self) -> list[Entity]:
        """List of active subscribers."""
        return [
            sub.subscriber
            for sub in self._subscriptions.values()
            if sub.active
        ]

    @property
    def max_subscribers(self) -> int | None:
        """Maximum number of subscribers."""
        return self._max_subscribers

    def subscribe(
        self,
        subscriber: Entity,
        replay_history: bool = False,
    ) -> list[Event]:
        """Subscribe an entity to the topic.

        Args:
            subscriber: The entity to subscribe.
            replay_history: If True, deliver historical messages.

        Returns:
            List of historical message events (if replay_history=True).

        Raises:
            RuntimeError: If max subscribers reached.
        """
        if self._max_subscribers is not None:
            if self.subscriber_count >= self._max_subscribers:
                raise RuntimeError(f"Topic {self.name} at max subscribers")

        now = self._clock.now if self._clock else Instant.Epoch

        if subscriber in self._subscriptions:
            # Reactivate existing subscription
            self._subscriptions[subscriber].active = True
        else:
            self._subscriptions[subscriber] = Subscription(
                subscriber=subscriber,
                subscribed_at=now,
            )
            self._subscribers_added += 1

        # Optionally replay history
        events = []
        if replay_history and self._retain_messages:
            for msg in self._message_history:
                delivery_event = Event(
                    time=now,
                    event_type="topic_message",
                    target=subscriber,
                    context={
                        'topic': self.name,
                        'payload': msg,
                        'is_replay': True,
                    },
                )
                events.append(delivery_event)

        return events

    def unsubscribe(self, subscriber: Entity) -> None:
        """Unsubscribe an entity from the topic.

        Args:
            subscriber: The entity to unsubscribe.
        """
        if subscriber in self._subscriptions:
            self._subscriptions[subscriber].active = False
            self._subscribers_removed += 1

    def publish(self, message: Event) -> Generator[float, None, list[Event]]:
        """Publish a message to all subscribers.

        Args:
            message: The event to publish.

        Yields:
            Publishing latency (per subscriber delivery).

        Returns:
            List of delivery events for each subscriber.
        """
        now = self._clock.now if self._clock else Instant.Epoch
        self._messages_published += 1

        # Store in history if retaining
        if self._retain_messages:
            self._message_history.append(message)

        # Deliver to all active subscribers
        delivery_events = []
        active_subscribers = [
            sub for sub in self._subscriptions.values()
            if sub.active
        ]

        for subscription in active_subscribers:
            # Delivery latency
            yield self._delivery_latency

            subscription.messages_received += 1
            self._messages_delivered += 1
            self._delivery_latencies.append(self._delivery_latency)

            delivery_event = Event(
                time=now,
                event_type="topic_message",
                target=subscription.subscriber,
                context={
                    'topic': self.name,
                    'payload': message,
                    'is_replay': False,
                },
            )
            delivery_events.append(delivery_event)

        return delivery_events

    def publish_sync(self, message: Event) -> list[Event]:
        """Publish a message synchronously (no delay simulation).

        Args:
            message: The event to publish.

        Returns:
            List of delivery events for each subscriber.
        """
        now = self._clock.now if self._clock else Instant.Epoch
        self._messages_published += 1

        if self._retain_messages:
            self._message_history.append(message)

        delivery_events = []
        for subscription in self._subscriptions.values():
            if subscription.active:
                subscription.messages_received += 1
                self._messages_delivered += 1

                delivery_event = Event(
                    time=now,
                    event_type="topic_message",
                    target=subscription.subscriber,
                    context={
                        'topic': self.name,
                        'payload': message,
                        'is_replay': False,
                    },
                )
                delivery_events.append(delivery_event)

        return delivery_events

    def set_retain_messages(self, retain: bool, max_history: int = 100) -> None:
        """Configure message retention.

        Args:
            retain: Whether to retain messages.
            max_history: Maximum messages to retain.
        """
        self._retain_messages = retain
        self._message_history = deque(maxlen=max_history)

    def get_subscription(self, subscriber: Entity) -> Subscription | None:
        """Get subscription details for a subscriber.

        Args:
            subscriber: The subscriber entity.

        Returns:
            Subscription details, or None if not subscribed.
        """
        return self._subscriptions.get(subscriber)

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]] | list[Event]:
        """Handle topic events."""
        event_type = event.event_type

        if event_type == "publish":
            payload = event.context.get('payload')
            if payload:
                return (yield from self.publish(payload))
            return []

        return []
