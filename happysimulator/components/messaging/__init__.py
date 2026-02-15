"""Messaging components for pub/sub and message queue patterns.

This module provides:
- MessageQueue: Persistent queue with acknowledgment and redelivery
- Topic: Pub/sub topic for broadcasting to multiple subscribers
- DeadLetterQueue: Storage for messages that failed processing

Example:
    from happysimulator.components.messaging import (
        MessageQueue,
        Topic,
        DeadLetterQueue,
    )

    # Create a message queue with dead letter support
    dlq = DeadLetterQueue(name="orders_dlq")
    orders = MessageQueue(
        name="orders",
        max_redeliveries=3,
        dead_letter_queue=dlq,
    )

    # Create a pub/sub topic
    notifications = Topic(name="notifications")
    notifications.subscribe(email_service)
    notifications.subscribe(sms_service)
"""

from happysimulator.components.messaging.dlq import (
    DeadLetterQueue,
    DeadLetterStats,
)
from happysimulator.components.messaging.message_queue import (
    Message,
    MessageQueue,
    MessageQueueStats,
    MessageState,
)
from happysimulator.components.messaging.topic import (
    Subscription,
    Topic,
    TopicStats,
)

__all__ = [
    # Dead Letter Queue
    "DeadLetterQueue",
    "DeadLetterStats",
    "Message",
    # Message Queue
    "MessageQueue",
    "MessageQueueStats",
    "MessageState",
    "Subscription",
    # Topic
    "Topic",
    "TopicStats",
]
