from dataclasses import dataclass, field
from collections import deque

from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event


@dataclass
class QueuePollEvent(Event):
    """
    Sent by the Driver to the Queue.
    Meaning: "I am free. Give me the next item."
    """
    event_type: str = field(default="QUEUE_POLL", init=False)
    requestor: Entity = None


@dataclass
class QueueNotifyEvent(Event):
    """
    Sent by the Queue to the Driver.
    Meaning: "I have data available. You should poll me."
    """
    event_type: str = field(default="QUEUE_NOTIFY", init=False)
    queue_entity: Entity = None


@dataclass
class Queue(Entity):
    """
    A bounded buffer that stores events and notifies a downstream driver.
    
    The queue doesn't interact with the server directly—it only knows
    about its egress (typically a QueueDriver that wraps the server).
    """
    name: str = "Queue"
    egress: Entity = None  # The driver that will process items
    capacity: int = 0  # 0 = unbounded
    
    # Internal storage
    _queue: deque = field(default_factory=deque, init=False)
    
    # Statistics
    stats_dropped: int = field(default=0, init=False)
    stats_accepted: int = field(default=0, init=False)

    def has_capacity(self) -> bool:
        """Return True if the queue can accept more items."""
        if self.capacity == 0:
            return True
        return len(self._queue) < self.capacity

    def handle_event(self, event: Event) -> list[Event]:
        if isinstance(event, QueuePollEvent):
            return self._handle_poll(event)
        
        # Any other event is work to be queued
        return self._handle_enqueue(event)

    def _handle_enqueue(self, event: Event) -> list[Event]:
        """Buffer incoming work and notify driver if queue was empty."""
        if not self.has_capacity():
            self.stats_dropped += 1
            return []

        was_empty = len(self._queue) == 0
        
        self._queue.append(event)
        self.stats_accepted += 1
        
        # If queue was empty, the driver might be idle—wake it up
        if was_empty:
            return [QueueNotifyEvent(
                time=event.time,
                target=self.egress,
                queue_entity=self
            )]
        return []

    def _handle_poll(self, event: QueuePollEvent) -> list[Event]:
        """Driver is asking for work."""
        if not self._queue:
            # Nothing available; driver will wait for QueueNotifyEvent
            return []
        
        next_item = self._queue.popleft()
        next_item.target = event.requestor
        next_item.time = event.time
        
        return [next_item]

    @property
    def depth(self) -> int:
        return len(self._queue)