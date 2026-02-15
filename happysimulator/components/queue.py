"""Bounded buffer queue with pluggable ordering policy.

Implements a queue entity that buffers incoming events and delivers them
to a downstream driver on demand. Uses QueuePolicy implementations to
control ordering (FIFO, LIFO, Priority).

The Queue/QueueDriver separation exists to decouple queue management from
the target entity. The target entity does not need to know it is being fed
by a queue.
"""

import logging
from dataclasses import dataclass, field

from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


class QueuePollEvent(Event):
    """Request from driver to queue for the next work item.

    Sent when the driver's target has capacity and is ready for work.
    The queue responds with QueueDeliverEvent if items are available.
    """

    __slots__ = ("requestor",)

    def __init__(self, *, time: Instant, target, requestor: Entity = None, **kwargs):
        super().__init__(time=time, event_type="QUEUE_POLL", target=target, **kwargs)
        self.requestor = requestor


class QueueNotifyEvent(Event):
    """Notification from queue to driver that work is available.

    Sent when an item is enqueued into a previously empty queue.
    The driver should check target capacity and poll if ready.
    """

    __slots__ = ("queue_entity",)

    def __init__(self, *, time: Instant, target, queue_entity: Entity = None, **kwargs):
        super().__init__(time=time, event_type="QUEUE_NOTIFY", target=target, **kwargs)
        self.queue_entity = queue_entity


class QueueDeliverEvent(Event):
    """Delivery from queue to driver containing one work item.

    The payload event is passed unmodified. The driver is responsible
    for retargeting it to the downstream entity before scheduling.
    """

    __slots__ = ("payload", "queue_entity")

    def __init__(
        self,
        *,
        time: Instant,
        target,
        payload: Event | None = None,
        queue_entity: Entity | None = None,
        **kwargs,
    ):
        super().__init__(time=time, event_type="QUEUE_DELIVER", target=target, **kwargs)
        self.payload = payload
        self.queue_entity = queue_entity


@dataclass
class Queue(Entity):
    """Bounded buffer that stores events and delivers them on demand.

    Accepts incoming events and buffers them according to its QueuePolicy.
    Notifies the egress (typically a QueueDriver) when items are available.
    Responds to poll requests by delivering the next item.

    The queue tracks acceptance and drop statistics for capacity analysis.

    Attributes:
        name: Identifier for logging.
        egress: Downstream entity to notify (usually QueueDriver).
        policy: Ordering strategy (FIFO, LIFO, Priority). Defaults to FIFO.
        stats_dropped: Count of items rejected due to capacity.
        stats_accepted: Count of items successfully enqueued.
    """

    name: str = "Queue"
    egress: Entity = None  # The driver that will process items
    policy: QueuePolicy = None  # Queue policy (FIFO, LIFO, Priority, etc.)

    # Statistics
    stats_dropped: int = field(default=0, init=False)
    stats_accepted: int = field(default=0, init=False)

    def __post_init__(self):
        # Default to unbounded FIFO if no policy provided
        if self.policy is None:
            self.policy = FIFOQueue()

    def has_capacity(self) -> bool:
        """Return True if the queue can accept more items."""
        # Delegate capacity check to policy
        return len(self.policy) < self.policy.capacity

    def handle_event(self, event: Event) -> list[Event]:
        if isinstance(event, QueuePollEvent):
            return self._handle_poll(event)

        # Any other event is work to be queued
        return self._handle_enqueue(event)

    def _handle_enqueue(self, event: Event) -> list[Event]:
        """Buffer incoming work and notify driver if queue was empty."""
        was_empty = self.policy.is_empty()

        accepted = self.policy.push(event)
        if not accepted:
            self.stats_dropped += 1
            logger.debug(
                "[%s] Dropped event (capacity full): type=%s depth=%d capacity=%s",
                self.name,
                event.event_type,
                len(self.policy),
                self.policy.capacity,
            )
            return []

        self.stats_accepted += 1
        logger.debug(
            "[%s] Enqueued event: type=%s depth=%d", self.name, event.event_type, len(self.policy)
        )

        # If queue was empty, the driver might be idle—wake it up
        if was_empty:
            logger.debug("[%s] Queue was empty, notifying driver", self.name)
            return [QueueNotifyEvent(time=self.now, target=self.egress, queue_entity=self)]
        return []

    def _handle_poll(self, event: QueuePollEvent) -> list[Event]:
        """Driver is asking for work."""
        next_item = self.policy.pop()
        if next_item is None:
            logger.debug("[%s] Poll received but queue is empty", self.name)
            return []

        logger.debug(
            "[%s] Delivering event to driver: type=%s depth=%d",
            self.name,
            next_item.event_type,
            len(self.policy),
        )
        return [
            QueueDeliverEvent(
                time=self.now, target=event.requestor, payload=next_item, queue_entity=self
            )
        ]

    @property
    def depth(self) -> int:
        return len(self.policy)
