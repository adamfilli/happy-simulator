import heapq
from typing import Union

from happysimulator.events.event import Event

class EventHeap:
    def __init__(self, events: list[Event] | None = None):
        self._heap = list(events) if events else []
        heapq.heapify(self._heap)

    def push(self, events: Union[Event, list[Event]]):
        """Push an Event or an iterable of Events/continuations onto the heap."""
        if isinstance(events, list):
            for event in events:
                heapq.heappush(self._heap, event)
        else:
            heapq.heappush(self._heap, events)

    def pop(self) -> Event:
        return heapq.heappop(self._heap)

    def peek(self) -> Event:
        return self._heap[0]

    def has_events(self) -> bool:
        return bool(self._heap)

    def size(self) -> int:
        return len(self._heap)