import uuid
import logging
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from happysimulator.utils.instant import Instant

logger = logging.getLogger(__name__)

_global_event_counter = count()

EventCallback = Callable[['Event'], Any]

@dataclass
class Event:
    time: Instant
    event_type: str
    daemon: bool = field(default=False, repr=False)
    
    # Option A: The "Model" way (Send to Entity)
    target: Optional['Entity'] = None
    
    # Option B: The "Scripting" way (Call specific function)
    callback: Optional[EventCallback] = field(default=None, repr=False)

    # Context & Tracing
    context: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    # Sorting Internals
    _sort_index: int = field(default_factory=_global_event_counter.__next__, init=False, repr=False)
    _id: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)

    def __post_init__(self):
        # Validation: Ensure mutually exclusive but at least one exists
        if self.target is None and self.callback is None:
            raise ValueError(f"Event {self} must have EITHER a 'target' OR a 'callback'.")
        
        if self.target is not None and self.callback is not None:
            raise ValueError(f"Event {self} cannot have BOTH 'target' and 'callback'.")
        
        # Always ensure trace context exists (even if caller passed partial context)
        self.context.setdefault("id", str(self._id))
        self.context.setdefault("created_at", self.time)
        self.context.setdefault("stack", [])
        self.context.setdefault("metadata", {})
        self.context.setdefault("trace", {"spans": []})

    def __repr__(self) -> str:
        """Return a concise representation showing time, type, and target/callback."""
        if self.target is not None:
            target_name = getattr(self.target, "name", None) or type(self.target).__name__
            return f"Event({self.time!r}, {self.event_type!r}, target={target_name})"
        else:
            callback_name = getattr(self.callback, "__qualname__", None) or repr(self.callback)
            return f"Event({self.time!r}, {self.event_type!r}, callback={callback_name})"

    def trace(self, action: str, **data: Any) -> None:
        """Append a structured span to this event's application-level trace.

        Args:
            action: Short action name (e.g., "handle.start", "process.yield").
            **data: Extra structured fields for debugging.
        """
        entry: Dict[str, Any] = {
            "time": self.time,
            "action": action,
            "event_id": self.context["id"],
            "event_type": self.event_type,
        }
        if data:
            entry["data"] = data
        self.context["trace"]["spans"].append(entry)

    def invoke(self) -> List['Event']:
        handler_kind = "callback" if self.callback else "entity"
        handler_label = (
            getattr(self.callback, "__qualname__", repr(self.callback))
            if self.callback
            else getattr(self.target, "name", type(self.target).__name__)
        )
        self.trace("handle.start", handler=handler_kind, handler_label=handler_label)

        try:
            raw_result = None

            # Path 1: Callback (High Priority / Explicit)
            if self.callback:
                raw_result = self.callback(self)

            # Path 2: Target Entity (Standard Model Flow)
            elif self.target:
                raw_result = self.target.handle_event(self)
            else:
                raise ValueError(f"Event {self} must have EITHER a 'target' OR a 'callback'.")
            
            # Normalize Result: Did the handler return a Generator? (Start of a Process)
            if isinstance(raw_result, Generator):
                self.trace("handle.end", result_kind="process")
                return self._start_process(raw_result)
            
            normalized = self._normalize_return(raw_result)
            self.trace("handle.end", result_kind="immediate", produced=len(normalized))
            return normalized

        except Exception as exc:
            self.trace("handle.error", error=type(exc).__name__, message=str(exc))
            raise
    
    def _start_process(self, gen: Generator) -> List["Event"]:
        continuation = ProcessContinuation(
                time=self.time,
                event_type=self.event_type,
                daemon=self.daemon,
                target=self.target,
                callback=self.callback,
                process=gen,   # Pass the SAME generator forward
                context=self.context)
        
        # Execute it immediately to get to the first 'yield'
        return continuation.invoke()
    
    def _normalize_return(self, value: Any) -> List['Event']:
        """Standardizes return values into List[Event]"""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, Event):
            return [value]
        return []

    def __lt__(self, other: "Event") -> bool:
        """
        1. Time (Primary)
        2. Insert Order (Secondary - guarantees FIFO for simultaneous events)
        """
        if self.time != other.time:
            return self.time < other.time
        return self._sort_index < other._sort_index

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self._id == other._id

    def add_context(self, key: str, value: Any):
        self.context.setdefault("metadata", {})[key] = value

    def get_context(self, key: str) -> Any:
        return self.context.get("metadata", {}).get(key)
    
@dataclass
class ProcessContinuation(Event):
    """
    Internal Event Type: Represents a paused Python generator waiting to resume.
    """
    process: Generator = field(default=None, repr=False)

    def invoke(self) -> List["Event"]:
        """Resumes the generator, handles the yield, and schedules the NEXT resume."""
        self.trace("process.resume.start")

        try:
            # 1. Wake up the process
            yielded_val = next(self.process)
            
            # 2. Parse the yield
            delay, side_effects = self._normalize_yield(yielded_val)
            self.trace("process.yield", delay_s=delay)

            # 3. Schedule the next Resume (Recursive Continuation)
            resume_time = self.time + delay
            next_continuation = ProcessContinuation(
                time=resume_time,
                event_type=self.event_type,
                daemon=self.daemon,
                target=self.target,     # Keep targeting the same entity
                callback=self.callback,
                process=self.process,   # Pass the SAME generator forward
                context=self.context    # Preserve trace context
            )
            
            if side_effects is None:
                side_effects = []
            elif isinstance(side_effects, Event):
                side_effects = [side_effects]
                
            result = list(side_effects)
            result.append(next_continuation)

            self.trace("process.resume.end", produced=len(result))
            return result

        except StopIteration as e:
            # Process Finished. Return the final value (if any).
            finished = self._normalize_return(e.value)
            self.trace("process.stop", produced=len(finished))
            return finished

        except Exception as exc:
            self.trace("process.error", error=type(exc).__name__, message=str(exc))
            raise
        
    def _normalize_yield(self, value: Any) -> Tuple[float, List["Event"]]:
        """Unpacks `yield 0.1` vs `yield 0.1, [events]`"""
        if isinstance(value, tuple):
            # (delay, [side_effects])
            delay = value[0]
            effects = value[1]
            # Normalize None -> [], single Event -> [Event], keep lists as-is
            if effects is None:
                effects = []
            elif isinstance(effects, Event):
                effects = [effects]
            return float(delay), effects
        elif isinstance(value, (int, float)):
            return float(value), []
        else:
            logger.warning("Generator yielded unknown type %s; assuming 0 delay.", type(value))
            return 0.0, []