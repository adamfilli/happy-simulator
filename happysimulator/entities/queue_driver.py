from dataclasses import dataclass
from typing import Generator

from happysimulator.entities.entity import Entity
from happysimulator.entities.queue import QueueDeliverEvent, QueueNotifyEvent, QueuePollEvent
from happysimulator.events.event import Event
from happysimulator.utils.instant import Instant


@dataclass
class QueueDriver(Entity):
    """
    Bridges a Queue and a downstream target without the target knowing about queues.
    
    The driver polls for work whenever the target has capacity. It delegates
    all capacity decisions to the target via `target.has_capacity()`.
    
    Flow:
        Queue --QueueNotifyEvent--> Driver --poll (if target has capacity)--> Queue
        Queue --WorkEvent--------> Driver --delegate--> Target
        Target --response--------> Driver --poll (if target has capacity)--> Queue
    """
    name: str = "QueueDriver"
    queue: Entity = None
    target: Entity = None

    def handle_event(self, event: Event) -> Generator[Instant, None, list[Event]] | list[Event]:
        if isinstance(event, QueueNotifyEvent):
            return self._handle_notify(event)
        
        if isinstance(event, QueueDeliverEvent):
            return self._handle_delivery(event)
        
        return None
        
    def _handle_delivery(self, event: QueueDeliverEvent) -> list[Event]:
        """Queue delivered one payload event; clone/retarget and re-emit."""
        if event.payload is None:
            return []
        return self._handle_work_payload(event.payload)
    
    def _handle_work_payload(self, payload: Event) -> list[Event]:
        def schedule_poll(time: Instant):
            # Always emit using the current simulation time (global clock).
            if self.target.has_capacity():
                return QueuePollEvent(time=time, target=self.queue, requestor=self)
            return None
        
        target_event = payload
        target_event.time = self.now
        target_event.target = self.target
        target_event.add_completion_hook(schedule_poll)
        return [target_event]

    def _handle_notify(self, _: QueueNotifyEvent) -> list[Event]:
        """Queue has work available—poll if target has capacity."""
        if not self.target.has_capacity():
            return []
        
        return [QueuePollEvent(time=self.now, target=self.queue, requestor=self)]

    def _handle_work(self, event: Event) -> Generator[Instant, None, list[Event]]:
        # 1. Re-target to downstream target
        event.target = self.target
        
        # 2. Define the Hook
        # "When you finish (at time 't'), please check the queue again."
        def schedule_poll(finish_time: Instant):
            # Check capacity again NOW (at finish time)
            if self.target.has_capacity():
                return QueuePollEvent(
                    time=finish_time,
                    target=self.queue,
                    requestor=self
                )
            return []

        # 3. Attach it
        event.add_completion_hook(schedule_poll)
        
        # 4. Re-emit
        return [event]