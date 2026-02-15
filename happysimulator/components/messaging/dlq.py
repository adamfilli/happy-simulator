"""Dead Letter Queue implementation.

Provides storage for messages that failed processing after all retry attempts.

Example:
    from happysimulator.components.messaging import MessageQueue, DeadLetterQueue

    # Create DLQ
    dlq = DeadLetterQueue(name="orders_dlq")

    # Create queue with DLQ
    queue = MessageQueue(
        name="orders",
        max_redeliveries=3,
        dead_letter_queue=dlq,
    )

    # Messages that fail 3 times go to DLQ
    # Later, inspect and reprocess:
    for msg in dlq.messages:
        print(f"Failed message: {msg.id}, attempts: {msg.delivery_count}")

    # Reprocess all messages
    events = dlq.reprocess_all(queue)
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from collections import deque

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from happysimulator.components.messaging.message_queue import Message, MessageQueue


@dataclass(frozen=True)
class DeadLetterStats:
    """Statistics tracked by DeadLetterQueue."""

    messages_received: int = 0
    messages_reprocessed: int = 0
    messages_discarded: int = 0


class DeadLetterQueue(Entity):
    """Stores messages that failed processing.

    Dead letter queues collect messages that could not be processed
    after exhausting all retry attempts. They can be inspected for
    debugging or reprocessed later.

    Attributes:
        name: Entity name for identification.
        message_count: Number of dead-lettered messages.
    """

    def __init__(
        self,
        name: str,
        capacity: int | None = None,
        retention_period: float | None = None,
    ):
        """Initialize the dead letter queue.

        Args:
            name: Name for this DLQ entity.
            capacity: Maximum messages to store (None for unlimited).
            retention_period: Time in seconds to retain messages (None for forever).
        """
        super().__init__(name)
        self._capacity = capacity
        self._retention_period = retention_period

        # Message storage
        self._messages: deque['Message'] = deque()
        self._message_times: deque[Instant] = deque()

        # Statistics
        self._messages_received = 0
        self._messages_reprocessed = 0
        self._messages_discarded = 0

    @property
    def stats(self) -> DeadLetterStats:
        """Return a frozen snapshot of current statistics."""
        return DeadLetterStats(
            messages_received=self._messages_received,
            messages_reprocessed=self._messages_reprocessed,
            messages_discarded=self._messages_discarded,
        )

    @property
    def message_count(self) -> int:
        """Number of messages in the DLQ."""
        return len(self._messages)

    @property
    def messages(self) -> list['Message']:
        """List of all dead-lettered messages."""
        return list(self._messages)

    @property
    def capacity(self) -> int | None:
        """Maximum capacity."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """Whether the DLQ is at capacity."""
        if self._capacity is None:
            return False
        return len(self._messages) >= self._capacity

    def add_message(self, message: 'Message') -> bool:
        """Add a message to the dead letter queue.

        Args:
            message: The failed message to store.

        Returns:
            True if added, False if at capacity.
        """
        # Cleanup expired messages first
        self._cleanup_expired()

        if self.is_full:
            # Remove oldest if at capacity
            if self._messages:
                self._messages.popleft()
                self._message_times.popleft()
                self._messages_discarded += 1

        now = self._clock.now if self._clock else Instant.Epoch
        self._messages.append(message)
        self._message_times.append(now)
        self._messages_received += 1

        return True

    def _cleanup_expired(self) -> None:
        """Remove messages past retention period."""
        if self._retention_period is None:
            return

        now = self._clock.now if self._clock else Instant.Epoch
        now_seconds = now.to_seconds()

        while self._messages and self._message_times:
            msg_time = self._message_times[0]
            age = now_seconds - msg_time.to_seconds()
            if age > self._retention_period:
                self._messages.popleft()
                self._message_times.popleft()
                self._messages_discarded += 1
            else:
                break

    def get_message(self, index: int) -> 'Message | None':
        """Get a message by index.

        Args:
            index: Index of the message (0 = oldest).

        Returns:
            The message, or None if index out of range.
        """
        if 0 <= index < len(self._messages):
            return self._messages[index]
        return None

    def peek(self) -> 'Message | None':
        """Peek at the oldest message.

        Returns:
            The oldest message, or None if empty.
        """
        if self._messages:
            return self._messages[0]
        return None

    def pop(self) -> 'Message | None':
        """Remove and return the oldest message.

        Returns:
            The oldest message, or None if empty.
        """
        if self._messages:
            self._message_times.popleft()
            return self._messages.popleft()
        return None

    def clear(self) -> int:
        """Clear all messages.

        Returns:
            Number of messages cleared.
        """
        count = len(self._messages)
        self._messages.clear()
        self._message_times.clear()
        self._messages_discarded += count
        return count

    def reprocess(self, message: 'Message', target_queue: 'MessageQueue') -> Event | None:
        """Reprocess a single message by publishing to a queue.

        Args:
            message: The message to reprocess.
            target_queue: The queue to publish to.

        Returns:
            The reprocessed message event, or None if failed.
        """
        # Remove from DLQ
        try:
            idx = list(self._messages).index(message)
            del self._messages[idx]
            del self._message_times[idx]
        except (ValueError, IndexError):
            return None

        self._messages_reprocessed += 1

        # Create republish event
        now = self._clock.now if self._clock else Instant.Epoch
        return Event(
            time=now,
            event_type="republish",
            target=target_queue,
            context={
                'payload': message.payload,
                'original_message_id': message.id,
                'delivery_count': message.delivery_count,
            },
        )

    def reprocess_all(self, target_queue: 'MessageQueue') -> list[Event]:
        """Reprocess all messages in the DLQ.

        Args:
            target_queue: The queue to publish to.

        Returns:
            List of republish events.
        """
        events = []
        while self._messages:
            message = self._messages.popleft()
            self._message_times.popleft()
            self._messages_reprocessed += 1

            now = self._clock.now if self._clock else Instant.Epoch
            event = Event(
                time=now,
                event_type="republish",
                target=target_queue,
                context={
                    'payload': message.payload,
                    'original_message_id': message.id,
                    'delivery_count': message.delivery_count,
                },
            )
            events.append(event)

        return events

    def get_messages_by_age(self, max_age: float) -> list['Message']:
        """Get messages younger than max_age seconds.

        Args:
            max_age: Maximum age in seconds.

        Returns:
            List of messages within age limit.
        """
        now = self._clock.now if self._clock else Instant.Epoch
        now_seconds = now.to_seconds()

        result = []
        for msg, msg_time in zip(self._messages, self._message_times):
            age = now_seconds - msg_time.to_seconds()
            if age <= max_age:
                result.append(msg)

        return result

    def get_messages_by_delivery_count(self, min_count: int) -> list['Message']:
        """Get messages with at least min_count delivery attempts.

        Args:
            min_count: Minimum delivery count.

        Returns:
            List of messages meeting criteria.
        """
        return [msg for msg in self._messages if msg.delivery_count >= min_count]

    def handle_event(self, event: Event) -> list[Event]:
        """Handle DLQ events."""
        # DLQ is mostly passive, but can handle admin commands
        event_type = event.event_type

        if event_type == "clear":
            self.clear()
            return []

        if event_type == "cleanup":
            self._cleanup_expired()
            return []

        return []
