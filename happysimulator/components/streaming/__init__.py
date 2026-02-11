"""Streaming infrastructure components for event-driven simulation.

Provides Kafka-inspired streaming primitives:

- **EventLog**: Append-only partitioned log with retention policies
- **ConsumerGroup**: Coordinated consumer management with partition assignment
- **StreamProcessor**: Stateful windowed stream processing
"""

from happysimulator.components.streaming.event_log import (
    EventLog,
    EventLogStats,
    Partition,
    Record,
    RetentionPolicy,
    SizeRetention,
    TimeRetention,
)
from happysimulator.components.streaming.consumer_group import (
    ConsumerGroup,
    ConsumerGroupStats,
    ConsumerState,
    PartitionAssignment,
    RangeAssignment,
    RoundRobinAssignment,
    StickyAssignment,
)
from happysimulator.components.streaming.stream_processor import (
    LateEventPolicy,
    SessionWindow,
    SlidingWindow,
    StreamProcessor,
    StreamProcessorStats,
    TumblingWindow,
    WindowState,
    WindowType,
)

__all__ = [
    # Event log
    "EventLog",
    "EventLogStats",
    "Partition",
    "Record",
    "RetentionPolicy",
    "SizeRetention",
    "TimeRetention",
    # Consumer group
    "ConsumerGroup",
    "ConsumerGroupStats",
    "ConsumerState",
    "PartitionAssignment",
    "RangeAssignment",
    "RoundRobinAssignment",
    "StickyAssignment",
    # Stream processor
    "LateEventPolicy",
    "SessionWindow",
    "SlidingWindow",
    "StreamProcessor",
    "StreamProcessorStats",
    "TumblingWindow",
    "WindowState",
    "WindowType",
]
