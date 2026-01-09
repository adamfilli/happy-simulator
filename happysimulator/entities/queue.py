from dataclasses import dataclass, field

from happysimulator.entities.entity import Entity
from happysimulator.entities.queue_policy import QueuePolicy, FIFOQueue
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
class QueueDeliverEvent(Event):
    """
    Sent by the Queue to the Driver.
    Meaning: "Here is one payload event you asked for."

    Note:
        The payload is not mutated by the queue. The driver is responsible for
        cloning/retargeting as needed before re-emitting to the simulation.
    """
    event_type: str = field(default="QUEUE_DELIVER", init=False)
    payload: Event | None = None
    queue_entity: Entity | None = None

@dataclass
class Queue(Entity):
    """
    A bounded buffer that stores events and notifies a downstream driver.
    
    The queue doesn't interact with the server directly—it only knows
    about its egress (typically a QueueDriver that wraps the server).
    
    Args:
        name: Entity name for identification.
        egress: The downstream entity (typically a QueueDriver) to notify.
        policy: Queue policy controlling item ordering (FIFO, LIFO, Priority).
                Defaults to FIFOQueue with unlimited capacity.
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
            return []
        
        self.stats_accepted += 1
        
        # If queue was empty, the driver might be idle—wake it up
        if was_empty:
            return [QueueNotifyEvent(
                time=self.now,
                target=self.egress,
                queue_entity=self
            )]
        return []

    def _handle_poll(self, event: QueuePollEvent) -> list[Event]:
        """Driver is asking for work."""
        next_item = self.policy.pop()
        if next_item is None:
            # Nothing available; driver will wait for QueueNotifyEvent
            return []
        
        # Do not mutate `next_item` (events are treated as immutable payloads).
        # Wrap it in a delivery event timestamped to the global simulation clock.
        return [QueueDeliverEvent(
            time=self.now,
            target=event.requestor,
            payload=next_item,
            queue_entity=self
        )]

    @property
    def depth(self) -> int:
        return len(self.policy)