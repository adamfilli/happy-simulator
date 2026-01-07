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

    def _handle_notify(self, event: QueueNotifyEvent) -> list[Event]:
        """Queue has work available—poll if server has capacity."""
        if not self.server.has_capacity():
            return []
        
        return [
            QueuePollEvent(time=event.time, target=self.queue, requestor=self)]

    def _handle_work(self, event: Event) -> Generator[Instant, None, list[Event]]:
        """Process a work item by delegating to the server."""
        result = self.server.handle_event(event)
        
        if isinstance(result, Generator):
            server_output = yield from result
        else:
            server_output = result if result else []
        
        # After server completes, poll for more work if server still has capacity
        if self.server.has_capacity():
            poll = QueuePollEvent(
                time=event.time,
                target=self.queue,
                requestor=self
            )
            return server_output + [poll]
        
        return server_output