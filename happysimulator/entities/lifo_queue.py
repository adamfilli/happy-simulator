import logging

import queue

from happysimulator.data import Data
from happysimulator.entity import Entity
from happysimulator.event import Event
from happysimulator.events.measurement_event import MeasurementEvent
from happysimulator.events.queue_event import QueueEvent

logger = logging.getLogger(__name__)

class LifoQueue(Entity):
    def __init__(self, name: str, size: int = 0):
        super().__init__(name)
        self._size = size
        self._queue = queue.LifoQueue(0) # always make it unbounded, we handle the size ourselves, now LIFO
        self._puts = Data()
        self._pops = Data()
        self._depth = Data()
        self._queue_time = Data()

    def put(self, event: QueueEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}][{event.queued_event.name}] Queue put called. Depth before put is {self._queue.qsize()}")
        self._depth.add_stat(self._queue.qsize(), event.time)
        self._puts.add_stat(1, event.time)

        self._queue.put(event.queued_event)

        if event.immediate_pop:
            logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}][{event.queued_event.name}] Server has capacity when submitting to queue, secheduling a pop immediately.")
            return [Event(name="QueuePop", time=event.time, callback=self.pop)]

        return [] # adding to a queue does not trigger anything under normal circumstances

    def pop(self, event: Event) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Queue pop called. Depth before pop is {self._queue.qsize()}")


        if self._queue.empty():
            logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Queue is empty, will not pop anything.")
            return []

        popped_event = self._queue.get(block=False)

        self._queue_time.add_stat((event.time - popped_event.time).to_seconds(), event.time) # how long was this event sitting in the queue?
        popped_event.time = event.time # move forward the event time, we don't have any additional queue delays

        self._pops.add_stat(1, event.time)

        return [popped_event]

    def has_capacity(self) -> bool:
        return self._size == 0 or self._queue.qsize() < self._size

    def depth(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for queue depth. Current depth is {self._queue.qsize()}")
        self.sink_data(self._depth, event)

    def queue_time(self, event: MeasurementEvent) -> list[Event]:
        logger.info(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for queue time.")
        self.sink_data(self._queue_time, event)
