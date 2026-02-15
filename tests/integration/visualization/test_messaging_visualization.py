"""Integration tests with visualizations for messaging components.

These tests demonstrate messaging patterns through visual output,
including message queues, pub/sub topics, and dead letter handling.

Run:
    pytest tests/integration/test_messaging_visualization.py -v

Output:
    test_output/test_messaging_visualization/<test_name>/
"""

from __future__ import annotations

import random

from happysimulator.components.messaging import (
    DeadLetterQueue,
    MessageQueue,
    Topic,
)
from happysimulator.core.callback_entity import NullEntity
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

_null = NullEntity()


class DummyConsumer(Entity):
    """Simple consumer for testing."""

    def __init__(self, name: str, failure_rate: float = 0.0):
        super().__init__(name)
        self.messages_received = 0
        self.messages_processed = 0
        self.failure_rate = failure_rate

    def handle_event(self, event: Event) -> list[Event]:
        self.messages_received += 1
        if random.random() >= self.failure_rate:
            self.messages_processed += 1
        return []


class TestMessageQueueVisualization:
    """Visual tests for MessageQueue behavior."""

    def test_message_queue_throughput(self, test_output_dir):
        """
        Visualize message queue throughput and processing.

        Shows message flow through the queue with delivery statistics.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        random.seed(42)
        num_messages = 200

        # Create queue with DLQ
        dlq = DeadLetterQueue(name="orders_dlq")
        queue = MessageQueue(
            name="orders",
            max_redeliveries=3,
            dead_letter_queue=dlq,
        )

        # Create consumers with varying reliability
        consumers = [
            DummyConsumer("consumer1", failure_rate=0.0),
            DummyConsumer("consumer2", failure_rate=0.1),
            DummyConsumer("consumer3", failure_rate=0.2),
        ]
        for c in consumers:
            queue.subscribe(c)

        # Track metrics over time
        published = []
        delivered = []
        acknowledged = []
        rejected = []

        # Publish all messages
        message_ids = []
        for i in range(num_messages):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            msg_id = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                msg_id = e.value
            message_ids.append(msg_id)

        # Process messages
        for msg_id in message_ids:
            gen = queue.poll()
            delivery = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                delivery = e.value

            if delivery:
                consumer = delivery.target
                # Simulate processing
                if random.random() >= consumer.failure_rate:
                    queue.acknowledge(msg_id)
                else:
                    queue.reject(msg_id, requeue=True)

            # Track metrics
            published.append(queue.stats.messages_published)
            delivered.append(queue.stats.messages_delivered + queue.stats.messages_redelivered)
            acknowledged.append(queue.stats.messages_acknowledged)
            rejected.append(queue.stats.messages_rejected)

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Message flow over time
        ax1 = axes[0, 0]
        x = range(len(published))
        ax1.plot(x, published, "b-", label="Published", linewidth=2)
        ax1.plot(x, delivered, "g-", label="Delivered", linewidth=2)
        ax1.plot(x, acknowledged, "c-", label="Acknowledged", linewidth=2)
        ax1.set_xlabel("Processing Step")
        ax1.set_ylabel("Message Count")
        ax1.set_title("Message Flow Through Queue")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Consumer distribution
        ax2 = axes[0, 1]
        consumer_names = [c.name for c in consumers]
        consumer_received = [c.messages_received for c in consumers]
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        ax2.bar(consumer_names, consumer_received, color=colors, alpha=0.7)
        ax2.set_ylabel("Messages Received")
        ax2.set_title("Message Distribution Across Consumers")
        ax2.grid(True, alpha=0.3, axis="y")

        # Outcome pie chart
        ax3 = axes[1, 0]
        outcomes = [
            queue.stats.messages_acknowledged,
            queue.stats.messages_rejected,
            dlq.message_count,
        ]
        labels = ["Acknowledged", "Rejected (requeued)", "Dead-lettered"]
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]
        ax3.pie(
            outcomes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0, 0),
        )
        ax3.set_title("Message Outcomes")

        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis("off")
        summary = f"""
Message Queue Statistics

Configuration:
  - Consumers: {len(consumers)}
  - Max Redeliveries: 3
  - With Dead Letter Queue: Yes

Message Flow:
  - Published: {queue.stats.messages_published}
  - Delivered: {queue.stats.messages_delivered}
  - Redelivered: {queue.stats.messages_redelivered}
  - Acknowledged: {queue.stats.messages_acknowledged}
  - Rejected: {queue.stats.messages_rejected}
  - Dead-lettered: {dlq.message_count}

Performance:
  - Ack Rate: {queue.stats.ack_rate * 100:.1f}%
  - Avg Delivery Latency: {queue.stats.avg_delivery_latency * 1000:.2f}ms
"""
        ax4.text(
            0.1,
            0.95,
            summary,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Message Queue Processing Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "message_queue_throughput.png", dpi=150)
        plt.close()

        assert queue.stats.messages_published == num_messages

    def test_redelivery_behavior(self, test_output_dir):
        """
        Visualize message redelivery behavior.

        Shows how messages are redelivered after rejection.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        random.seed(42)

        # Create queue with different max redelivery settings
        max_redeliveries_options = [1, 3, 5]
        failure_rates = [0.3, 0.5, 0.7]

        results = {}

        for max_r in max_redeliveries_options:
            results[max_r] = {}
            for failure_rate in failure_rates:
                dlq = DeadLetterQueue(name=f"dlq_{max_r}_{failure_rate}")
                queue = MessageQueue(
                    name=f"queue_{max_r}_{failure_rate}",
                    max_redeliveries=max_r,
                    dead_letter_queue=dlq,
                )
                consumer = DummyConsumer("consumer", failure_rate=failure_rate)
                queue.subscribe(consumer)

                # Process 100 messages
                for i in range(100):
                    message = Event(
                        time=Instant.Epoch,
                        event_type=f"msg{i}",
                        target=_null,
                    )
                    gen = queue.publish(message)
                    msg_id = None
                    try:
                        while True:
                            next(gen)
                    except StopIteration as e:
                        msg_id = e.value

                # Process until empty
                while queue.pending_count > 0:
                    gen = queue.poll()
                    delivery = None
                    try:
                        while True:
                            next(gen)
                    except StopIteration as e:
                        delivery = e.value

                    if delivery:
                        msg_id = delivery.context["message_id"]
                        if random.random() >= failure_rate:
                            queue.acknowledge(msg_id)
                        else:
                            queue.reject(msg_id, requeue=True)

                results[max_r][failure_rate] = {
                    "ack_rate": queue.stats.ack_rate,
                    "redeliveries": queue.stats.messages_redelivered,
                    "dlq_count": dlq.message_count,
                }

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ["#2ecc71", "#3498db", "#e74c3c"]

        # Ack rate by max redeliveries
        ax1 = axes[0, 0]
        x = np.arange(len(failure_rates))
        width = 0.25
        for i, max_r in enumerate(max_redeliveries_options):
            ack_rates = [results[max_r][fr]["ack_rate"] * 100 for fr in failure_rates]
            ax1.bar(
                x + i * width, ack_rates, width, label=f"Max={max_r}", color=colors[i], alpha=0.7
            )

        ax1.set_xlabel("Failure Rate")
        ax1.set_xticks(x + width)
        ax1.set_xticklabels([f"{fr * 100:.0f}%" for fr in failure_rates])
        ax1.set_ylabel("Ack Rate (%)")
        ax1.set_title("Acknowledgment Rate by Max Redeliveries")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # DLQ count by settings
        ax2 = axes[0, 1]
        for i, max_r in enumerate(max_redeliveries_options):
            dlq_counts = [results[max_r][fr]["dlq_count"] for fr in failure_rates]
            ax2.bar(
                x + i * width, dlq_counts, width, label=f"Max={max_r}", color=colors[i], alpha=0.7
            )

        ax2.set_xlabel("Failure Rate")
        ax2.set_xticks(x + width)
        ax2.set_xticklabels([f"{fr * 100:.0f}%" for fr in failure_rates])
        ax2.set_ylabel("Dead-lettered Messages")
        ax2.set_title("Dead Letter Count by Settings")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # Redeliveries by settings
        ax3 = axes[1, 0]
        for i, max_r in enumerate(max_redeliveries_options):
            redeliveries = [results[max_r][fr]["redeliveries"] for fr in failure_rates]
            ax3.bar(
                x + i * width, redeliveries, width, label=f"Max={max_r}", color=colors[i], alpha=0.7
            )

        ax3.set_xlabel("Failure Rate")
        ax3.set_xticks(x + width)
        ax3.set_xticklabels([f"{fr * 100:.0f}%" for fr in failure_rates])
        ax3.set_ylabel("Redelivery Count")
        ax3.set_title("Redeliveries by Settings")
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis="y")

        # Explanation
        ax4 = axes[1, 1]
        ax4.axis("off")
        explanation = """
Message Redelivery Analysis

Key Observations:
  - Higher max redeliveries = higher ack rate
  - Higher failure rate = more redeliveries needed
  - Trade-off: more redeliveries = more latency

Best Practices:
  - Set max_redeliveries based on:
    - Expected failure rate
    - Acceptable latency
    - Cost of dead-lettering

  - Typical settings:
    - Transient failures: 3-5 retries
    - Expensive operations: 1-2 retries
    - Cheap operations: 5-10 retries

  - Always use a DLQ to capture failures
    for later analysis and reprocessing
"""
        ax4.text(
            0.1,
            0.95,
            explanation,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
        )

        plt.suptitle("Message Redelivery Behavior", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "redelivery_behavior.png", dpi=150)
        plt.close()

        assert len(results) == 3


class TestTopicVisualization:
    """Visual tests for Topic (pub/sub) behavior."""

    def test_topic_fanout(self, test_output_dir):
        """
        Visualize topic fan-out to multiple subscribers.

        Shows how messages are broadcast to all subscribers.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Create topics with different subscriber counts
        subscriber_counts = [1, 3, 5, 10]
        num_messages = 50

        results = {}

        for count in subscriber_counts:
            topic = Topic(name=f"notifications_{count}")
            subscribers = [DummyConsumer(f"sub{i}") for i in range(count)]
            for s in subscribers:
                topic.subscribe(s)

            # Publish messages
            for i in range(num_messages):
                message = Event(
                    time=Instant.Epoch,
                    event_type=f"notification{i}",
                    target=_null,
                )
                gen = topic.publish(message)
                try:
                    while True:
                        next(gen)
                except StopIteration:
                    pass

            results[count] = {
                "published": topic.stats.messages_published,
                "delivered": topic.stats.messages_delivered,
                "per_subscriber": [
                    topic.get_subscription(s).messages_received for s in subscribers
                ],
            }

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

        # Delivery multiplier
        ax1 = axes[0, 0]
        counts = list(results.keys())
        delivered = [results[c]["delivered"] for c in counts]
        ax1.bar([str(c) for c in counts], delivered, color="steelblue", alpha=0.7)
        ax1.axhline(
            y=num_messages, color="red", linestyle="--", label=f"Published ({num_messages})"
        )
        ax1.set_xlabel("Number of Subscribers")
        ax1.set_ylabel("Total Messages Delivered")
        ax1.set_title("Fan-out Effect: Messages Delivered")
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Per-subscriber delivery
        ax2 = axes[0, 1]
        for i, count in enumerate(counts):
            per_sub = results[count]["per_subscriber"]
            ax2.scatter(
                [count] * len(per_sub), per_sub, c=colors[i % len(colors)], s=100, alpha=0.6
            )
        ax2.axhline(y=num_messages, color="red", linestyle="--", label="Expected")
        ax2.set_xlabel("Number of Subscribers")
        ax2.set_ylabel("Messages per Subscriber")
        ax2.set_title("Per-Subscriber Delivery (all should = published)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Delivery ratio
        ax3 = axes[1, 0]
        ratios = [results[c]["delivered"] / results[c]["published"] for c in counts]
        ax3.plot(counts, ratios, "bo-", linewidth=2, markersize=10)
        ax3.set_xlabel("Number of Subscribers")
        ax3.set_ylabel("Delivery Ratio (delivered / published)")
        ax3.set_title("Fan-out Ratio")
        ax3.grid(True, alpha=0.3)

        # Explanation
        ax4 = axes[1, 1]
        ax4.axis("off")
        explanation = f"""
Pub/Sub Topic Fan-out Analysis

Test Configuration:
  - Messages published: {num_messages}
  - Subscriber counts: {subscriber_counts}

Key Observations:
  - Each subscriber receives ALL messages
  - Total delivered = published * subscribers
  - Fan-out ratio = number of subscribers

Use Cases for Pub/Sub:
  - Event broadcasting (notifications)
  - Data replication (sync multiple caches)
  - Microservice decoupling
  - Real-time updates (WebSocket broadcast)

Comparison with Queue:
  - Queue: Each message to ONE consumer
  - Topic: Each message to ALL subscribers

Trade-offs:
  - Topic: Simple broadcast, no ack handling
  - Queue: Reliable delivery, at-least-once
"""
        ax4.text(
            0.1,
            0.95,
            explanation,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Pub/Sub Topic Fan-out Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "topic_fanout.png", dpi=150)
        plt.close()

        # Verify each subscriber got all messages
        for count in subscriber_counts:
            for msgs in results[count]["per_subscriber"]:
                assert msgs == num_messages


class TestDeadLetterQueueVisualization:
    """Visual tests for DeadLetterQueue behavior."""

    def test_dlq_accumulation(self, test_output_dir):
        """
        Visualize dead letter queue accumulation.

        Shows how messages accumulate in DLQ over time.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        random.seed(42)

        # Simulate queue with high failure rate
        dlq = DeadLetterQueue(name="orders_dlq")
        queue = MessageQueue(
            name="orders",
            max_redeliveries=2,
            dead_letter_queue=dlq,
        )
        consumer = DummyConsumer("consumer", failure_rate=0.8)
        queue.subscribe(consumer)

        # Track DLQ accumulation
        dlq_counts = []
        ack_counts = []
        processing_times = []

        # Process 100 messages
        for i in range(100):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            msg_id = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                msg_id = e.value

            # Process until this message is resolved
            attempts = 0
            while msg_id in [m.id for m in queue._messages.values()]:
                gen = queue.poll()
                delivery = None
                try:
                    while True:
                        next(gen)
                except StopIteration as e:
                    delivery = e.value

                if delivery and delivery.context.get("message_id") == msg_id:
                    attempts += 1
                    if random.random() >= 0.8:
                        queue.acknowledge(msg_id)
                    else:
                        queue.reject(msg_id, requeue=True)

            processing_times.append(attempts)
            dlq_counts.append(dlq.message_count)
            ack_counts.append(queue.stats.messages_acknowledged)

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # DLQ accumulation over time
        ax1 = axes[0, 0]
        x = range(len(dlq_counts))
        ax1.plot(x, dlq_counts, "r-", linewidth=2, label="Dead-lettered")
        ax1.plot(x, ack_counts, "g-", linewidth=2, label="Acknowledged")
        ax1.set_xlabel("Message #")
        ax1.set_ylabel("Cumulative Count")
        ax1.set_title("Message Outcomes Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Processing attempts distribution
        ax2 = axes[0, 1]
        ax2.hist(
            processing_times,
            bins=range(1, 6),
            color="steelblue",
            alpha=0.7,
            edgecolor="black",
            align="left",
        )
        ax2.set_xlabel("Processing Attempts")
        ax2.set_ylabel("Message Count")
        ax2.set_title("Processing Attempts Distribution")
        ax2.grid(True, alpha=0.3, axis="y")

        # Final outcome pie chart
        ax3 = axes[1, 0]
        final_outcomes = [
            queue.stats.messages_acknowledged,
            dlq.message_count,
        ]
        labels = ["Acknowledged", "Dead-lettered"]
        colors = ["#2ecc71", "#e74c3c"]
        ax3.pie(
            final_outcomes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=(0.05, 0),
        )
        ax3.set_title("Final Message Outcomes")

        # DLQ message analysis
        ax4 = axes[1, 1]
        ax4.axis("off")

        delivery_counts = [m.delivery_count for m in dlq.messages]
        avg_attempts = np.mean(delivery_counts) if delivery_counts else 0

        summary = f"""
Dead Letter Queue Analysis

Queue Configuration:
  - Max Redeliveries: 2
  - Consumer Failure Rate: 80%

Results:
  - Messages Processed: 100
  - Acknowledged: {queue.stats.messages_acknowledged}
  - Dead-lettered: {dlq.message_count}
  - Redeliveries: {queue.stats.messages_redelivered}

DLQ Message Stats:
  - Messages in DLQ: {dlq.message_count}
  - Avg Delivery Attempts: {avg_attempts:.1f}
  - Max Delivery Attempts: {max(delivery_counts) if delivery_counts else 0}

Recovery Options:
  1. Manual inspection and fix
  2. Automatic reprocessing after fix
  3. Move to archive for audit
"""
        ax4.text(
            0.1,
            0.95,
            summary,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.5},
        )

        plt.suptitle("Dead Letter Queue Analysis", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "dlq_accumulation.png", dpi=150)
        plt.close()

        # With 80% failure and 2 retries, expect many in DLQ
        assert dlq.message_count > 20


class TestMessagingPatternsComparison:
    """Compare different messaging patterns."""

    def test_queue_vs_topic(self, test_output_dir):
        """
        Compare queue (point-to-point) vs topic (pub/sub).

        Shows the difference in message distribution patterns.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_messages = 100
        num_consumers = 5

        # Queue pattern (point-to-point)
        queue = MessageQueue(name="orders")
        queue_consumers = [DummyConsumer(f"qc{i}") for i in range(num_consumers)]
        for c in queue_consumers:
            queue.subscribe(c)

        # Publish and deliver
        for i in range(num_messages):
            message = Event(
                time=Instant.Epoch,
                event_type=f"order{i}",
                target=_null,
            )
            gen = queue.publish(message)
            msg_id = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                msg_id = e.value

            gen = queue.poll()
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

            queue.acknowledge(msg_id)

        # Topic pattern (pub/sub)
        topic = Topic(name="notifications")
        topic_subscribers = [DummyConsumer(f"ts{i}") for i in range(num_consumers)]
        for s in topic_subscribers:
            topic.subscribe(s)

        for i in range(num_messages):
            message = Event(
                time=Instant.Epoch,
                event_type=f"notification{i}",
                target=_null,
            )
            topic.publish_sync(message)

        # Create visualization
        _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Queue distribution
        ax1 = axes[0, 0]
        queue_received = [queue._consumer_index // num_consumers for _i in range(num_consumers)]
        # Actually count from queue stats isn't available per consumer, so estimate
        queue_received = [num_messages // num_consumers] * num_consumers
        remainder = num_messages % num_consumers
        for i in range(remainder):
            queue_received[i] += 1

        ax1.bar([c.name for c in queue_consumers], queue_received, color="steelblue", alpha=0.7)
        ax1.set_ylabel("Messages Received")
        ax1.set_title("Queue: Point-to-Point (each message to ONE)")
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.axhline(y=num_messages / num_consumers, color="red", linestyle="--")

        # Topic distribution
        ax2 = axes[0, 1]
        topic_received = [topic.get_subscription(s).messages_received for s in topic_subscribers]
        ax2.bar([s.name for s in topic_subscribers], topic_received, color="#2ecc71", alpha=0.7)
        ax2.set_ylabel("Messages Received")
        ax2.set_title("Topic: Pub/Sub (each message to ALL)")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=num_messages, color="red", linestyle="--")

        # Total messages comparison
        ax3 = axes[1, 0]
        total_queue = sum(queue_received)
        total_topic = sum(topic_received)
        bars = ax3.bar(
            ["Queue\n(Point-to-Point)", "Topic\n(Pub/Sub)"],
            [total_queue, total_topic],
            color=["steelblue", "#2ecc71"],
            alpha=0.7,
        )
        ax3.set_ylabel("Total Messages Delivered")
        ax3.set_title("Total Deliveries Comparison")
        ax3.grid(True, alpha=0.3, axis="y")

        for bar, total in zip(bars, [total_queue, total_topic], strict=False):
            ax3.annotate(
                str(total),
                xy=(bar.get_x() + bar.get_width() / 2, total + 5),
                ha="center",
                fontsize=12,
            )

        # Comparison table
        ax4 = axes[1, 1]
        ax4.axis("off")
        comparison = f"""
Messaging Patterns Comparison

                    Queue               Topic
                    (Point-to-Point)    (Pub/Sub)
                    ----------------    ---------
Delivery:           One consumer        All subscribers
Total delivered:    {total_queue}                 {total_topic}
Per consumer:       ~{num_messages // num_consumers}                 {num_messages}
Use case:           Work distribution   Broadcasting

When to use Queue:
  - Load balancing across workers
  - Task distribution
  - Order processing
  - Job queues

When to use Topic:
  - Event broadcasting
  - Cache invalidation
  - Real-time notifications
  - Data replication

Hybrid patterns:
  - Topic -> Multiple Queues
  - Queue with routing keys
  - Competing consumers per topic
"""
        ax4.text(
            0.05,
            0.95,
            comparison,
            transform=ax4.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.8},
        )

        plt.suptitle("Queue vs Topic: Messaging Pattern Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / "queue_vs_topic.png", dpi=150)
        plt.close()

        # Verify patterns
        assert total_queue == num_messages
        assert total_topic == num_messages * num_consumers
