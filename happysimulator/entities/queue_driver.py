from dataclasses import dataclass
from typing import Generator

from happysimulator.entities.entity import Entity
from happysimulator.entities.queue import QueueNotifyEvent, QueuePollEvent
from happysimulator.events.event import Event
from happysimulator.utils.instant import Instant


@dataclass
class QueueDriver(Entity):
    """
    Bridges a Queue and a Server without the server knowing about queues.
    
    The driver polls for work whenever the server has capacity. It delegates
    all capacity decisions to the server via `server.has_capacity()`.
    
    Flow:
        Queue --QueueNotifyEvent--> Driver --poll (if server has capacity)--> Queue
        Queue --WorkEvent--------> Driver --delegate--> Server
        Server --response--------> Driver --poll (if server has capacity)--> Queue
    """
    name: str = "QueueDriver"
    queue: Entity = None
    server: Entity = None

    def handle_event(self, event: Event) -> Generator[Instant, None, list[Event]] | list[Event]:
        if isinstance(event, QueueNotifyEvent):
            return self._handle_notify(event)
        
        return self._handle_work(event)

    def _handle_notify(self, _: QueueNotifyEvent) -> list[Event]:
        """Queue has work available—poll if server has capacity."""
        if not self.server.has_capacity():
            return []
        
        return [QueuePollEvent(time=self.now, target=self.queue, requestor=self)]

    def _handle_work(self, event: Event) -> Generator[Instant, None, list[Event]]:
        # 1. Re-target to server
        event.target = self.server
        
        # 2. Define the Hook
        # "When you finish (at time 't'), please check the queue again."
        def schedule_poll(finish_time: Instant):
            # Check capacity again NOW (at finish time)
            if self.server.has_capacity():
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