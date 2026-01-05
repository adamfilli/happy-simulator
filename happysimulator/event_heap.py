import heapq
from typing import Union

from happysimulator.events.event import Event

class EventHeap:
    def __init__(self, events: list[Event] | None = None):
        self._primary_event_count = 0
        self._heap = list(events) if events else []
        heapq.heapify(self._heap)

    def push(self, events: Union[Event, list[Event]]):
        if isinstance(events, list):
            for event in events:
                self._push_single(event)
        else:
            self._push_single(events)

    def pop(self) -> Event:
        popped = heapq.heappop(self._heap)
        self._primary_event_count -= 1
        return popped

    def peek(self) -> Event:
        return self._heap[0]

    def has_events(self) -> bool:
        return bool(self._heap)
    
    def has_primary_events(self) -> bool:
        """Returns True if there is at least one non-daemon event pending."""
        return self._primary_event_count > 0

    def size(self) -> int:
        return len(self._heap)
    
    def _push_single(self, event: Event):
        heapq.heappush(self._heap, event)
        self._primary_event_count += 1