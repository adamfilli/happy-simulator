import heapq
from typing import Union

from happysimulator.events.event import Event
from happysimulator.tracing.recorder import TraceRecorder, NullTraceRecorder
from happysimulator.utils.instant import Instant


class EventHeap:
    def __init__(
        self,
        events: list[Event] | None = None,
        trace_recorder: TraceRecorder | None = None,
    ):
        self._primary_event_count = 0
        self._current_time = Instant.Epoch
        self._heap = list(events) if events else []
        heapq.heapify(self._heap)
        self._trace = trace_recorder or NullTraceRecorder()

    def set_current_time(self, time: Instant) -> None:
        """Update the current simulation time for accurate trace timestamps."""
        self._current_time = time

    def push(self, events: Union[Event, list[Event]]):
        if isinstance(events, list):
            for event in events:
                self._push_single(event)
        else:
            self._push_single(events)

    def pop(self) -> Event:
        popped = heapq.heappop(self._heap)
        if not popped.daemon:
            self._primary_event_count -= 1
        self._current_time = popped.time
        self._trace.record(
            time=self._current_time,
            kind="heap.pop",
            event_id=popped.context.get("id"),
            event_type=popped.event_type,
            heap_size=len(self._heap),
        )
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
        if not event.daemon:
            self._primary_event_count += 1
        self._trace.record(
            time=self._current_time,
            kind="heap.push",
            event_id=event.context.get("id"),
            event_type=event.event_type,
            scheduled_for=event.time,
            heap_size=len(self._heap),
        )