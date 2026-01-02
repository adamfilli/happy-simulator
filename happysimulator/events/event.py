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
        
        if not self.context:
            self.context = {
                "id": str(self._id),
                "created_at": self.time,
                "stack": [],
                "metadata": {}
            }

    def invoke(self) -> List['Event']:
        raw_result = None

        # Path 1: Callback (High Priority / Explicit)
        if self.callback:
            # We pass 'self' so the callback has access to event data/time
            raw_result = self.callback(self)

        # Path 2: Target Entity (Standard Model Flow)
        elif self.target:
            raw_result = self.target.handle_event(self)
        else:
            raise ValueError(f"Event {self} must have EITHER a 'target' OR a 'callback'.")
        
        # 2. Normalize Result
        # Did the handler return a Generator? (Start of a Process)
        if isinstance(raw_result, Generator):
            return self._start_process(raw_result)
        
        return self._normalize_return(raw_result)
    
    def _start_process(self, gen: Generator) -> List["Event"]:
        continuation = ProcessContinuation(
                time=self.time,
                event_type=self.event_type,
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
        """
        Resumes the generator, handles the yield, and schedules the NEXT resume.
        """
        try:
            # 1. Wake up the process
            yielded_val = next(self.process)
            
            # 2. Parse the yield
            delay, side_effects = self._normalize_yield(yielded_val)

            # 3. Schedule the next Resume (Recursive Continuation)
            resume_time = self.time + delay
            next_continuation = ProcessContinuation(
                time=resume_time,
                event_type=self.event_type,
                target=self.target,     # Keep targeting the same entity
                callback=self.callback,
                process=self.process,   # Pass the SAME generator forward
                context=self.context    # Preserve Trace ID
            )
            
            if side_effects is None:
                side_effects = []
            elif isinstance(side_effects, Event):
                side_effects = [side_effects]
                
            result = list(side_effects)
            if next_continuation is not None:
                result.append(next_continuation)
            return result

        except StopIteration as e:
            # Process Finished. Return the final value (if any).
            return self._normalize_return(e.value)
        
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
             logger.warning(f"Generator yielded unknown type {type(value)}; assuming 0 delay.")
             return 0.0, []