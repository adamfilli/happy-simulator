"""Trace recorders for engine-level simulation instrumentation.

Engine traces capture scheduling decisions (heap push/pop, simulation loop events)
and are kept separate from application-level traces on Event.context["trace"].
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from happysimulator.utils.instant import Instant


class TraceRecorder(Protocol):
    """Protocol for recording engine-level trace spans.
    
    Implementations can store traces in memory, write to files,
    send to external systems, or simply discard them.
    """
    
    def record(
        self,
        *,
        time: Instant,
        kind: str,
        event_id: str | None = None,
        event_type: str | None = None,
        **data: Any,
    ) -> None:
        """Record an engine-level trace span.
        
        Args:
            time: Simulation time when the span occurred.
            kind: Category of span (e.g., "heap.push", "heap.pop", "simulation.dequeue").
            event_id: ID of the associated event (from event.context["id"]).
            event_type: Type of the associated event.
            **data: Additional structured data for the span.
        """

@dataclass
class InMemoryTraceRecorder:
    """Stores engine traces in memory for later inspection.
    
    Useful for testing and debugging simulation behavior.
    """
    
    spans: list[dict[str, Any]] = field(default_factory=list)
    
    def record(
        self,
        *,
        time: Instant,
        kind: str,
        event_id: str | None = None,
        event_type: str | None = None,
        **data: Any,
    ) -> None:
        span: dict[str, Any] = {
            "time": time,
            "kind": kind,
        }
        if event_id is not None:
            span["event_id"] = event_id
        if event_type is not None:
            span["event_type"] = event_type
        if data:
            span["data"] = data
        self.spans.append(span)
    
    def clear(self) -> None:
        """Clear all recorded spans."""
        self.spans.clear()
    
    def filter_by_kind(self, kind: str) -> list[dict[str, Any]]:
        """Return spans matching the given kind."""
        return [s for s in self.spans if s["kind"] == kind]
    
    def filter_by_event(self, event_id: str) -> list[dict[str, Any]]:
        """Return spans for a specific event ID."""
        return [s for s in self.spans if s.get("event_id") == event_id]


@dataclass
class NullTraceRecorder:
    """No-op recorder that discards all traces.
    
    Use when tracing is disabled for performance.
    """
    
    def record(
        self,
        *,
        time: Instant,
        kind: str,
        event_id: str | None = None,
        event_type: str | None = None,
        **data: Any,
    ) -> None:
        pass
