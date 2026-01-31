"""Queue driver that mediates between a queue and its target entity.

The driver decouples the target from queue awareness. The target entity
receives work events without knowing they came from a queue. The driver
manages polling based on target capacity and re-polls after work completion.
"""

import logging
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.components.queue import QueueDeliverEvent, QueueNotifyEvent, QueuePollEvent

logger = logging.getLogger(__name__)


@dataclass
class QueueDriver(Entity):
    """Mediator between Queue and target entity.

    Polls the queue for work when the target has capacity. Retargets
    delivered events to the target and attaches completion hooks to
    re-poll after processing finishes.

    Event flow:
    1. Queue sends QueueNotifyEvent when items are available
    2. Driver checks target.has_capacity() and polls if ready
    3. Queue sends QueueDeliverEvent with payload
    4. Driver retargets payload to target and schedules it
    5. On completion, driver re-polls if target has capacity

    Attributes:
        name: Identifier for logging.
        queue: The upstream Queue entity.
        target: The downstream entity that processes work.
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
            logger.debug("[%s] Received empty delivery", self.name)
            return []
        logger.debug(
            "[%s] Received delivery: type=%s, forwarding to target",
            self.name, event.payload.event_type
        )
        return self._handle_work_payload(event.payload)

    def _handle_work_payload(self, payload: Event) -> list[Event]:
        def schedule_poll(time: Instant):
            if self.target.has_capacity():
                logger.debug("[%s] Target has capacity, scheduling poll", self.name)
                return QueuePollEvent(time=time, target=self.queue, requestor=self)
            logger.debug("[%s] Target at capacity, deferring poll", self.name)
            return None

        target_event = payload
        target_event.time = self.now
        target_event.target = self.target
        target_event.add_completion_hook(schedule_poll)
        return [target_event]

    def _handle_notify(self, _: QueueNotifyEvent) -> list[Event]:
        """Queue has work available—poll if target has capacity."""
        if not self.target.has_capacity():
            logger.debug("[%s] Notify received but target at capacity", self.name)
            return []

        logger.debug("[%s] Notify received, polling queue", self.name)
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