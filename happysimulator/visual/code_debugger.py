"""Code-level debugging for generator-based entity handlers.

Provides line-by-line tracing of entity generators using Python's frame
tracing (gen.gi_frame.f_trace). Two modes:

1. **Recording** (default): Each line appends to a buffer. Zero blocking.
   After gen.send() completes, the buffer is sent to the frontend for
   animated replay.

2. **Breakpoint**: When a line matches a code breakpoint, the trace
   function blocks the sim thread via threading.Event. The frontend
   sends a continue/step command to resume.
"""

from __future__ import annotations

import inspect
import logging
import sys
import threading
import uuid
from dataclasses import dataclass, field
from types import FrameType
from typing import Any

logger = logging.getLogger(__name__)

# Deadman timeout — if the frontend never sends continue, unblock after 30s
DEADMAN_TIMEOUT_S = 30.0


@dataclass
class CodeBreakpoint:
    """A breakpoint on a specific source line of an entity class."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_name: str = ""
    line_number: int = 0  # 1-indexed, relative to the source file

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_name": self.entity_name,
            "line_number": self.line_number,
        }


@dataclass
class CodeLocation:
    """Describes the source location of an entity's handler method."""

    entity_name: str
    class_name: str
    method_name: str
    source_lines: list[str]
    start_line: int  # 1-indexed line number in the source file

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "source_lines": self.source_lines,
            "start_line": self.start_line,
        }


@dataclass
class LineRecord:
    """A single recorded line execution within a generator trace."""

    line_number: int  # 1-indexed, absolute file line number
    locals_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"line_number": self.line_number}
        if self.locals_snapshot is not None:
            d["locals"] = self.locals_snapshot
        return d


@dataclass
class ExecutionTrace:
    """Complete trace of a generator invocation (one gen.send() call)."""

    entity_name: str
    method_name: str
    start_line: int
    lines: list[LineRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "method_name": self.method_name,
            "start_line": self.start_line,
            "lines": [lr.to_dict() for lr in self.lines],
        }


def _safe_serialize_locals(frame_locals: dict[str, Any], max_depth: int = 2) -> dict[str, Any]:
    """Serialize frame locals to a JSON-safe dict with depth limits."""
    result: dict[str, Any] = {}
    for key, val in frame_locals.items():
        if key.startswith("_") or key == "self":
            continue
        result[key] = _safe_serialize_value(val, depth=0, max_depth=max_depth)
    return result


def _safe_serialize_value(val: Any, depth: int, max_depth: int) -> Any:
    """Recursively serialize a value to JSON-safe form."""
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    if depth >= max_depth:
        return repr(val)
    if isinstance(val, dict):
        return {
            str(k): _safe_serialize_value(v, depth + 1, max_depth)
            for k, v in list(val.items())[:20]
        }
    if isinstance(val, (list, tuple)):
        items = [_safe_serialize_value(v, depth + 1, max_depth) for v in val[:20]]
        return items
    return repr(val)


@dataclass
class _ActiveTrace:
    """Mutable state for an in-progress trace on a specific generator."""

    entity_name: str
    method_name: str
    start_line: int
    buffer: list[LineRecord] = field(default_factory=list)
    capture_locals: bool = False


class CodeDebugger:
    """Manages code-level debugging for entity generators.

    Tracks which entities have active code panels, installs trace functions
    on their generators, records line execution, and manages code breakpoints.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # entity_name -> CodeLocation (cached source info)
        self._source_cache: dict[str, CodeLocation] = {}
        # entity_name -> True (entities with open code panels)
        self._active_entities: set[str] = set()
        # breakpoint_id -> CodeBreakpoint
        self._breakpoints: dict[str, CodeBreakpoint] = {}
        # Completed traces waiting for the bridge to drain
        self._completed_traces: list[ExecutionTrace] = []
        # Per-generator active trace (keyed by id(generator))
        self._active_traces: dict[int, _ActiveTrace] = {}
        # Blocking state for breakpoint pauses
        self._pause_event = threading.Event()
        self._pause_event.set()  # Start unblocked
        self._paused_entity: str | None = None
        self._paused_line: int | None = None
        self._paused_locals: dict[str, Any] | None = None
        # Step mode: None = continue freely, "step" = stop at next line,
        # "step_over" = stop at next line in same frame,
        # "step_out" = stop when frame returns
        self._step_mode: str | None = None
        self._step_frame_depth: int | None = None

    def activate_entity(self, entity_name: str) -> None:
        """Mark an entity for code-level debugging."""
        with self._lock:
            self._active_entities.add(entity_name)
        logger.debug("Code debug activated for entity: %s", entity_name)

    def deactivate_entity(self, entity_name: str) -> None:
        """Remove an entity from code-level debugging."""
        with self._lock:
            self._active_entities.discard(entity_name)
            # Remove breakpoints for this entity
            to_remove = [
                bp_id for bp_id, bp in self._breakpoints.items()
                if bp.entity_name == entity_name
            ]
            for bp_id in to_remove:
                del self._breakpoints[bp_id]
        logger.debug("Code debug deactivated for entity: %s", entity_name)

    def is_active(self, entity_name: str) -> bool:
        """Check if an entity has code debugging active."""
        return entity_name in self._active_entities

    def get_source(self, entity: Any) -> CodeLocation | None:
        """Retrieve source code for an entity's handler method.

        Tries handle_queued_event first (for QueuedResource subclasses),
        then falls back to handle_event. Caches results per class.
        """
        entity_name = getattr(entity, "name", type(entity).__name__)

        with self._lock:
            if entity_name in self._source_cache:
                return self._source_cache[entity_name]

        cls = type(entity)
        class_name = cls.__name__

        # Try handle_queued_event first (QueuedResource subclasses)
        for method_name in ("handle_queued_event", "handle_event"):
            method = getattr(cls, method_name, None)
            if method is None:
                continue

            # Skip if it's the abstract/base method (not overridden)
            if getattr(method, "__isabstractmethod__", False):
                continue

            try:
                source_lines, start_line = inspect.getsourcelines(method)
                location = CodeLocation(
                    entity_name=entity_name,
                    class_name=class_name,
                    method_name=method_name,
                    source_lines=[line.rstrip("\n") for line in source_lines],
                    start_line=start_line,
                )
                with self._lock:
                    self._source_cache[entity_name] = location
                return location
            except (OSError, TypeError):
                continue

        return None

    def add_breakpoint(self, bp: CodeBreakpoint) -> str:
        """Add a code breakpoint. Returns the breakpoint ID."""
        with self._lock:
            self._breakpoints[bp.id] = bp
        logger.debug("Code breakpoint added: entity=%s line=%d id=%s",
                      bp.entity_name, bp.line_number, bp.id)
        return bp.id

    def remove_breakpoint(self, bp_id: str) -> bool:
        """Remove a code breakpoint by ID. Returns True if found."""
        with self._lock:
            return self._breakpoints.pop(bp_id, None) is not None

    def list_breakpoints(self) -> list[CodeBreakpoint]:
        """Return all active code breakpoints."""
        with self._lock:
            return list(self._breakpoints.values())

    def install_trace(self, gen: Any, entity_name: str) -> None:
        """Install a trace function on a generator's frame.

        Called before gen.send() in ProcessContinuation.invoke().
        Sets sys.settrace to enable the tracing infrastructure (required
        for frame.f_trace to fire), then sets the generator frame's f_trace.
        """
        frame = getattr(gen, "gi_frame", None)
        if frame is None:
            return

        location = self._source_cache.get(entity_name)
        method_name = location.method_name if location else "handle_event"
        start_line = location.start_line if location else 0

        trace = _ActiveTrace(
            entity_name=entity_name,
            method_name=method_name,
            start_line=start_line,
        )

        gen_id = id(gen)
        with self._lock:
            self._active_traces[gen_id] = trace
            # Save the previous sys.settrace so we can restore it
            self._prev_settrace = sys.gettrace()

        def trace_fn(frame: FrameType, event: str, arg: Any) -> Any:
            if event == "line":
                line_no = frame.f_lineno
                record = LineRecord(line_number=line_no)

                # Check breakpoints
                should_break = False
                with self._lock:
                    for bp in self._breakpoints.values():
                        if bp.entity_name == entity_name and bp.line_number == line_no:
                            should_break = True
                            break

                # Check step mode
                if self._step_mode == "step":
                    should_break = True
                elif self._step_mode == "step_over":
                    should_break = True
                elif self._step_mode == "step_out":
                    pass  # Only break on return

                if should_break:
                    # Capture locals for the paused state
                    locals_snapshot = _safe_serialize_locals(dict(frame.f_locals))
                    record.locals_snapshot = locals_snapshot
                    trace.buffer.append(record)

                    self._step_mode = None

                    # Block the sim thread
                    self._paused_entity = entity_name
                    self._paused_line = line_no
                    self._paused_locals = locals_snapshot
                    self._pause_event.clear()
                    logger.debug("Code breakpoint hit: entity=%s line=%d", entity_name, line_no)

                    # Wait for continue/step command (with deadman timeout)
                    if not self._pause_event.wait(timeout=DEADMAN_TIMEOUT_S):
                        logger.warning("Code breakpoint deadman timeout after %ds", DEADMAN_TIMEOUT_S)

                    self._paused_entity = None
                    self._paused_line = None
                    self._paused_locals = None
                else:
                    trace.buffer.append(record)

            elif event == "return":
                if self._step_mode == "step_out":
                    self._step_mode = None

            return trace_fn

        # Enable thread-level tracing (required for f_trace to fire)
        # Use a minimal function that returns None for all non-generator frames
        def _global_trace(frame: FrameType, event: str, arg: Any) -> Any:
            return None

        sys.settrace(_global_trace)
        frame.f_trace = trace_fn

    def remove_trace(self, gen: Any) -> None:
        """Remove trace from a generator and flush its buffer to completed traces.

        Called after gen.send() in ProcessContinuation.invoke() (in finally).
        Restores the previous sys.settrace.
        """
        gen_id = id(gen)
        frame = getattr(gen, "gi_frame", None)
        if frame is not None:
            frame.f_trace = None

        with self._lock:
            trace = self._active_traces.pop(gen_id, None)
            if trace is not None and trace.buffer:
                execution_trace = ExecutionTrace(
                    entity_name=trace.entity_name,
                    method_name=trace.method_name,
                    start_line=trace.start_line,
                    lines=list(trace.buffer),
                )
                self._completed_traces.append(execution_trace)
            # Restore previous sys.settrace
            prev = getattr(self, '_prev_settrace', None)
            sys.settrace(prev)

    def drain_completed_traces(self) -> list[ExecutionTrace]:
        """Return and clear all completed execution traces."""
        with self._lock:
            traces = self._completed_traces
            self._completed_traces = []
            return traces

    def is_code_paused(self) -> bool:
        """Check if the debugger is paused at a code breakpoint."""
        return self._paused_entity is not None

    def get_paused_state(self) -> dict[str, Any] | None:
        """Return the current paused state, or None if not paused."""
        if self._paused_entity is None:
            return None
        return {
            "entity_name": self._paused_entity,
            "line_number": self._paused_line,
            "locals": self._paused_locals,
        }

    def code_continue(self) -> None:
        """Resume from a code breakpoint pause."""
        self._step_mode = None
        self._pause_event.set()

    def code_step(self) -> None:
        """Step to the next line (step into)."""
        self._step_mode = "step"
        self._pause_event.set()

    def code_step_over(self) -> None:
        """Step over — next line in the same frame."""
        self._step_mode = "step_over"
        self._pause_event.set()

    def code_step_out(self) -> None:
        """Step out — continue until the current frame returns."""
        self._step_mode = "step_out"
        self._pause_event.set()

    def get_state(self) -> dict[str, Any]:
        """Return the full code debug state for the frontend."""
        with self._lock:
            return {
                "active_entities": list(self._active_entities),
                "breakpoints": [bp.to_dict() for bp in self._breakpoints.values()],
                "is_paused": self.is_code_paused(),
                "paused_state": self.get_paused_state(),
            }

    def reset(self) -> None:
        """Reset all code debug state."""
        # If paused, unblock first
        if self.is_code_paused():
            self.code_continue()

        with self._lock:
            self._source_cache.clear()
            self._active_entities.clear()
            self._breakpoints.clear()
            self._completed_traces.clear()
            self._active_traces.clear()
            self._step_mode = None
