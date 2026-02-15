"""Trace analysis utilities for reconstructing event lifecycles.

Works with the existing InMemoryTraceRecorder to provide structured
views of event processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from happysimulator.core.temporal import Duration, Instant
    from happysimulator.instrumentation.recorder import InMemoryTraceRecorder


@dataclass
class EventLifecycle:
    """Reconstructed lifecycle of a single event through the simulation."""

    event_id: str
    event_type: str | None = None
    scheduled_at: Instant | None = None
    dequeued_at: Instant | None = None
    completed_at: Instant | None = None
    child_event_ids: list[str] = field(default_factory=list)

    @property
    def wait_time(self) -> Duration | None:
        """Time between scheduling and dequeue."""
        if self.scheduled_at is not None and self.dequeued_at is not None:
            return self.dequeued_at - self.scheduled_at
        return None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"event_id": self.event_id}
        if self.event_type:
            result["event_type"] = self.event_type
        if self.scheduled_at is not None:
            result["scheduled_at_s"] = self.scheduled_at.to_seconds()
        if self.dequeued_at is not None:
            result["dequeued_at_s"] = self.dequeued_at.to_seconds()
        if self.completed_at is not None:
            result["completed_at_s"] = self.completed_at.to_seconds()
        if self.wait_time is not None:
            result["wait_time_s"] = self.wait_time.to_seconds()
        if self.child_event_ids:
            result["children"] = list(self.child_event_ids)
        return result

    def __str__(self) -> str:
        parts = [f"Event {self.event_id}"]
        if self.event_type:
            parts[0] += f" ({self.event_type})"
        if self.scheduled_at is not None:
            parts.append(f"  scheduled: {self.scheduled_at}")
        if self.dequeued_at is not None:
            parts.append(f"  dequeued:  {self.dequeued_at}")
        if self.wait_time is not None:
            parts.append(f"  wait:      {self.wait_time}")
        if self.child_event_ids:
            parts.append(f"  children:  {len(self.child_event_ids)}")
        return "\n".join(parts)


def trace_event_lifecycle(
    recorder: InMemoryTraceRecorder,
    event_id: str,
) -> EventLifecycle | None:
    """Reconstruct the full lifecycle of a single event from trace spans.

    Args:
        recorder: The trace recorder that captured the simulation.
        event_id: The event ID to look up.

    Returns:
        EventLifecycle with timing data, or None if event_id not found.
    """
    spans = recorder.filter_by_event(event_id)
    if not spans:
        return None

    lifecycle = EventLifecycle(event_id=event_id)

    for span in spans:
        kind = span.get("kind", "")
        time = span.get("time")

        if kind == "simulation.schedule":
            lifecycle.scheduled_at = time
            lifecycle.event_type = span.get("event_type")
        elif kind == "simulation.dequeue":
            lifecycle.dequeued_at = time
            if lifecycle.event_type is None:
                lifecycle.event_type = span.get("event_type")

    # Look for child events spawned by this event
    # Find schedule spans where this event's dequeue time matches
    if lifecycle.dequeued_at is not None:
        for span in recorder.spans:
            if (
                span.get("kind") == "simulation.schedule"
                and span.get("time") == lifecycle.dequeued_at
            ):
                child_id = span.get("event_id")
                if child_id and child_id != event_id:
                    lifecycle.child_event_ids.append(child_id)

    return lifecycle


def list_event_lifecycles(
    recorder: InMemoryTraceRecorder,
    *,
    event_type: str | None = None,
) -> list[EventLifecycle]:
    """Reconstruct lifecycles for all events (or filtered by type).

    Args:
        recorder: The trace recorder that captured the simulation.
        event_type: If provided, only return events of this type.

    Returns:
        List of EventLifecycle objects.
    """
    # Collect unique event IDs
    seen: set[str] = set()
    for span in recorder.spans:
        eid = span.get("event_id")
        if eid and eid not in seen:
            if event_type is not None:
                et = span.get("event_type")
                if et != event_type:
                    continue
            seen.add(eid)

    lifecycles = []
    for eid in seen:
        lc = trace_event_lifecycle(recorder, eid)
        if lc is not None:
            lifecycles.append(lc)

    return lifecycles
