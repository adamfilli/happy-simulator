"""Trace recorders for engine-level simulation instrumentation.

Engine traces capture low-level scheduling decisions (heap push/pop, simulation
loop events) separate from application-level traces stored in Event.context["trace"].

Use InMemoryTraceRecorder for debugging and testing. Use NullTraceRecorder (the
default) when tracing is not needed, avoiding any overhead.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from happysimulator.core.temporal import Instant


class TraceRecorder(Protocol):
    """Protocol defining the trace recording interface.

    Implementations can store traces in memory, write to files, send to
    external monitoring systems, or discard them entirely.
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
            kind: Category (e.g., "heap.push", "heap.pop", "simulation.dequeue").
            event_id: ID of the associated event (from event.context["id"]).
            event_type: Type of the associated event.
            **data: Additional structured data for the span.
        """


@dataclass
class InMemoryTraceRecorder:
    """Trace recorder that stores spans in memory.

    Useful for debugging, testing, and post-simulation analysis.
    Provides filtering methods to query specific span types or events.

    Attributes:
        spans: List of recorded spans as dictionaries.
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
