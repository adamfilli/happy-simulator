"""Message Queue implementation.

Provides a persistent message queue with acknowledgment, redelivery,
and dead letter queue support.

Example:
    from happysimulator.components.messaging import MessageQueue

    queue = MessageQueue(
        name="orders",
        redelivery_delay=30.0,
        max_redeliveries=3,
    )

    # Publisher
    def handle_event(self, event):
        yield from queue.publish(order_event)

    # Consumer
    queue.subscribe(order_processor)

    # In consumer's handle_event:
    def handle_event(self, event):
        try:
            # Process message
            yield from self.process_order(event)
            queue.acknowledge(event.context['message_id'])
        except Exception:
            queue.reject(event.context['message_id'], requeue=True)
"""

from dataclasses import dataclass, field
from typing import Any, Generator, Callable
from collections import deque
from enum import Enum
import uuid

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class MessageState(Enum):
    """State of a message in the queue."""

    PENDING = "pending"  # Waiting to be delivered
    DELIVERED = "delivered"  # Sent to consumer, awaiting ack
    ACKNOWLEDGED = "acknowledged"  # Successfully processed
    REJECTED = "rejected"  # Failed processing


@dataclass
class Message:
    """Represents a message in the queue."""

    id: str
    payload: Event
    created_at: Instant
    state: MessageState = MessageState.PENDING
    delivery_count: int = 0
    last_delivered_at: Instant | None = None
    consumer: Entity | None = None


@dataclass
class MessageQueueStats:
    """Statistics tracked by MessageQueue."""

    messages_published: int = 0
    messages_delivered: int = 0
    messages_acknowledged: int = 0
    messages_rejected: int = 0
    messages_redelivered: int = 0
    messages_dead_lettered: int = 0
    delivery_latencies: list[float] = field(default_factory=list)

    @property
    def avg_delivery_latency(self) -> float:
        """Average delivery latency."""
        if not self.delivery_latencies:
            return 0.0
        return sum(self.delivery_latencies) / len(self.delivery_latencies)

    @property
    def ack_rate(self) -> float:
        """Acknowledgment rate."""
        total = self.messages_acknowledged + self.messages_rejected
        if total == 0:
            return 0.0
        return self.messages_acknowledged / total


class MessageQueue(Entity):
    """Persistent message queue with acknowledgment.

    Provides reliable message delivery with at-least-once semantics.
    Messages are redelivered if not acknowledged within timeout.

    Attributes:
        name: Entity name for identification.
        pending_count: Number of messages waiting to be delivered.
        in_flight_count: Number of messages delivered but not yet acknowledged.
    """

    def __init__(
        self,
        name: str,
        delivery_latency: float = 0.001,
        redelivery_delay: float = 30.0,
        max_redeliveries: int = 3,
        capacity: int | None = None,
        dead_letter_queue: 'DeadLetterQueue | None' = None,
    ):
        """Initialize the message queue.

        Args:
            name: Name for this queue entity.
            delivery_latency: Latency to deliver message to consumer.
            redelivery_delay: Delay before redelivering unacknowledged message.
            max_redeliveries: Maximum redelivery attempts before dead-lettering.
            capacity: Maximum queue size (None for unlimited).
            dead_letter_queue: Optional DLQ for failed messages.

        Raises:
            ValueError: If parameters are invalid.
        """
        if redelivery_delay <= 0:
            raise ValueError(f"redelivery_delay must be > 0, got {redelivery_delay}")
        if max_redeliveries < 0:
            raise ValueError(f"max_redeliveries must be >= 0, got {max_redeliveries}")

        super().__init__(name)
        self._delivery_latency = delivery_latency
        self._redelivery_delay = redelivery_delay
        self._max_redeliveries = max_redeliveries
        self._capacity = capacity
        self._dead_letter_queue = dead_letter_queue

        # Message storage
        self._messages: dict[str, Message] = {}
        self._pending_queue: deque[str] = deque()  # Message IDs in delivery order
        self._in_flight: dict[str, Message] = {}  # Message ID -> Message

        # Consumers
        self._consumers: list[Entity] = []
        self._consumer_index = 0  # For round-robin delivery

        # Pending redeliveries
        self._redelivery_scheduled: set[str] = set()

        # Statistics
        self.stats = MessageQueueStats()

    @property
    def pending_count(self) -> int:
        """Number of messages waiting to be delivered."""
        return len(self._pending_queue)

    @property
    def in_flight_count(self) -> int:
        """Number of messages delivered but not acknowledged."""
        return len(self._in_flight)

    @property
    def consumer_count(self) -> int:
        """Number of subscribed consumers."""
        return len(self._consumers)

    @property
    def capacity(self) -> int | None:
        """Maximum queue capacity."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Whether the queue is at capacity."""
        if self._capacity is None:
            return False
        return len(self._messages) >= self._capacity

    def subscribe(self, consumer: Entity) -> None:
        """Subscribe a consumer to receive messages.

        Args:
            consumer: The entity to receive messages.
        """
        if consumer not in self._consumers:
            self._consumers.append(consumer)

    def unsubscribe(self, consumer: Entity) -> None:
        """Unsubscribe a consumer.

        Args:
            consumer: The entity to remove.
        """
        if consumer in self._consumers:
            self._consumers.remove(consumer)

    def publish(self, message: Event) -> Generator[float, None, str]:
        """Publish a message to the queue.

        Args:
            message: The event to publish.

        Yields:
            Publishing latency.

        Returns:
            The message ID.

        Raises:
            RuntimeError: If queue is at capacity.
        """
        if self.is_full:
            raise RuntimeError(f"Queue {self.name} is at capacity")

        # Create message wrapper
        message_id = str(uuid.uuid4())
        now = self._clock.now if self._clock else Instant.Epoch

        msg = Message(
            id=message_id,
            payload=message,
            created_at=now,
        )

        self._messages[message_id] = msg
        self._pending_queue.append(message_id)
        self.stats.messages_published += 1

        # Small publish latency
        yield 0.0001

        return message_id

    def _get_next_consumer(self) -> Entity | None:
        """Get next consumer using round-robin."""
        if not self._consumers:
            return None

        consumer = self._consumers[self._consumer_index % len(self._consumers)]
        self._consumer_index += 1
        return consumer

    def _deliver_message(self, message_id: str) -> Generator[float, None, Event | None]:
        """Deliver a message to a consumer.

        Args:
            message_id: ID of message to deliver.

        Yields:
            Delivery latency.

        Returns:
            The delivery event, or None if no consumer available.
        """
        if message_id not in self._messages:
            return None

        consumer = self._get_next_consumer()
        if consumer is None:
            return None

        msg = self._messages[message_id]
        now = self._clock.now if self._clock else Instant.Epoch

        # Update message state
        msg.state = MessageState.DELIVERED
        msg.delivery_count += 1
        msg.last_delivered_at = now
        msg.consumer = consumer

        # Move to in-flight
        if message_id in self._pending_queue:
            self._pending_queue.remove(message_id)
        self._in_flight[message_id] = msg

        # Track delivery latency
        created_time = msg.created_at.to_seconds() if msg.created_at else 0
        now_time = now.to_seconds() if now else 0
        self.stats.delivery_latencies.append(now_time - created_time)

        if msg.delivery_count > 1:
            self.stats.messages_redelivered += 1
        else:
            self.stats.messages_delivered += 1

        yield self._delivery_latency

        # Create delivery event
        delivery_event = Event(
            time=now,
            event_type="message_delivery",
            target=consumer,
            context={
                'message_id': message_id,
                'payload': msg.payload,
                'delivery_count': msg.delivery_count,
                'queue': self.name,
            },
        )

        return delivery_event

    def acknowledge(self, message_id: str) -> None:
        """Acknowledge successful processing of a message.

        Args:
            message_id: ID of the message to acknowledge.
        """
        if message_id not in self._messages:
            return

        msg = self._messages[message_id]
        msg.state = MessageState.ACKNOWLEDGED

        # Remove from in-flight and messages
        self._in_flight.pop(message_id, None)
        self._messages.pop(message_id, None)
        self._redelivery_scheduled.discard(message_id)

        self.stats.messages_acknowledged += 1

    def reject(self, message_id: str, requeue: bool = True) -> None:
        """Reject a message.

        Args:
            message_id: ID of the message to reject.
            requeue: If True, requeue for redelivery. If False, discard or DLQ.
        """
        if message_id not in self._messages:
            return

        msg = self._messages[message_id]
        msg.state = MessageState.REJECTED
        self.stats.messages_rejected += 1

        # Remove from in-flight
        self._in_flight.pop(message_id, None)

        if requeue and msg.delivery_count < self._max_redeliveries:
            # Requeue for redelivery
            msg.state = MessageState.PENDING
            self._pending_queue.append(message_id)
        else:
            # Dead letter or discard
            if self._dead_letter_queue is not None:
                self._dead_letter_queue.add_message(msg)
                self.stats.messages_dead_lettered += 1
            self._messages.pop(message_id, None)
            self._redelivery_scheduled.discard(message_id)

    def poll(self) -> Generator[float, None, Event | None]:
        """Poll for the next message to deliver.

        Yields:
            Polling/delivery latency.

        Returns:
            Delivery event or None if queue empty or no consumers.
        """
        if not self._pending_queue or not self._consumers:
            yield 0.0
            return None

        message_id = self._pending_queue[0]
        event = yield from self._deliver_message(message_id)
        return event

    def schedule_redelivery(self, message_id: str) -> Event | None:
        """Schedule redelivery of an unacknowledged message.

        Args:
            message_id: ID of message to redeliver.

        Returns:
            Redelivery event, or None if message not found.
        """
        if message_id not in self._in_flight:
            return None

        if message_id in self._redelivery_scheduled:
            return None

        msg = self._in_flight[message_id]

        if msg.delivery_count >= self._max_redeliveries:
            # Dead letter
            self.reject(message_id, requeue=False)
            return None

        self._redelivery_scheduled.add(message_id)

        # Move back to pending
        msg.state = MessageState.PENDING
        self._in_flight.pop(message_id, None)
        self._pending_queue.appendleft(message_id)

        now = self._clock.now if self._clock else Instant.Epoch
        redelivery_time = Instant.from_seconds(now.to_seconds() + self._redelivery_delay)

        return Event(
            time=redelivery_time,
            event_type="message_redelivery",
            target=self,
            context={'message_id': message_id},
        )

    def get_message(self, message_id: str) -> Message | None:
        """Get a message by ID.

        Args:
            message_id: The message ID.

        Returns:
            The message, or None if not found.
        """
        return self._messages.get(message_id)

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]] | list[Event]:
        """Handle queue events."""
        event_type = event.event_type

        if event_type == "message_redelivery":
            message_id = event.context.get('message_id')
            if message_id:
                self._redelivery_scheduled.discard(message_id)
                delivery_event = yield from self._deliver_message(message_id)
                if delivery_event:
                    return [delivery_event]
            return []

        if event_type == "poll":
            delivery_event = yield from self.poll()
            if delivery_event:
                return [delivery_event]
            return []

        return []
