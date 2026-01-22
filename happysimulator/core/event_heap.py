"""Priority queue for scheduling and dispatching simulation events.

The heap ensures events are processed in chronological order, breaking ties
by insertion order for deterministic FIFO behavior among simultaneous events.
"""

import heapq
import logging
from typing import Union

from happysimulator.core.event import Event
from happysimulator.core.instant import Instant
from happysimulator.instrumentation.recorder import TraceRecorder, NullTraceRecorder

logger = logging.getLogger(__name__)


class EventHeap:
    """Min-heap priority queue for event scheduling.

    Events are ordered by (time, insertion_order) to guarantee deterministic
    execution order. The heap distinguishes between primary and daemon events
    to support auto-termination: when only daemon events remain, the simulation
    can choose to stop rather than run forever.

    The heap tracks a separate count of primary (non-daemon) events to enable
    O(1) checks for auto-termination without scanning the entire heap.

    Args:
        events: Optional initial events to populate the heap.
        trace_recorder: Optional recorder for debugging heap operations.
    """
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

    def push(self, events: Union[Event, list[Event]]) -> None:
        """Schedule one or more events for future processing.

        Events are inserted in O(log n) time per event. Each push is traced
        for debugging and visualization purposes.
        """
        if isinstance(events, list):
            for event in events:
                self._push_single(event)
        else:
            self._push_single(events)

    def pop(self) -> Event:
        """Remove and return the earliest scheduled event.

        Also updates the internal primary event counter for auto-termination
        tracking and advances the heap's time reference.
        """
        popped = heapq.heappop(self._heap)
        if not popped.daemon:
            self._primary_event_count -= 1
        self._current_time = popped.time
        logger.debug(
            "Popped event: type=%s time=%r heap_size=%d primary_remaining=%d",
            popped.event_type, popped.time, len(self._heap), self._primary_event_count
        )
        self._trace.record(
            time=self._current_time,
            kind="heap.pop",
            event_id=popped.context.get("id"),
            event_type=popped.event_type,
            heap_size=len(self._heap),
        )
        return popped

    def peek(self) -> Event:
        """Return the next event without removing it from the heap."""
        return self._heap[0]

    def has_events(self) -> bool:
        """Check if any events remain to be processed."""
        return bool(self._heap)

    def has_primary_events(self) -> bool:
        """Check if non-daemon events remain for auto-termination decisions."""
        return self._primary_event_count > 0

    def size(self) -> int:
        """Return the total number of pending events."""
        return len(self._heap)

    def _push_single(self, event: Event) -> None:
        heapq.heappush(self._heap, event)
        if not event.daemon:
            self._primary_event_count += 1
        logger.debug(
            "Pushed event: type=%s scheduled_for=%r daemon=%s heap_size=%d",
            event.event_type, event.time, event.daemon, len(self._heap)
        )
        self._trace.record(
            time=self._current_time,
            kind="heap.push",
            event_id=event.context.get("id"),
            event_type=event.event_type,
            scheduled_for=event.time,
            heap_size=len(self._heap),
        )