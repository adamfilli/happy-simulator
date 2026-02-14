"""Event types that form the fundamental units of simulation work.

Events drive the simulation forward. Each event represents something that happens
at a specific point in simulation time. When invoked, an event calls its target
entity's handle_event() method. For function-based dispatch, use Event.once()
which wraps a function in a CallbackEntity.

This module also provides ProcessContinuation for generator-based multi-step
processes, enabling entities to yield delays and resume execution later.
"""

import uuid
import logging
from dataclasses import dataclass, field
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from happysimulator.core.protocols import Simulatable

logger = logging.getLogger(__name__)

_global_event_counter = count()

CompletionHook = Callable[[Instant], Union[List['Event'], 'Event', None]]
"""Signature for hooks that run when an event or process finishes."""


@dataclass
class Event:
    """The fundamental unit of simulation work.

    Events are scheduled onto the EventHeap and processed in chronological order.
    Each event targets an Entity whose handle_event() method processes it.

    For function-based dispatch without a full Entity, use the ``Event.once()``
    static constructor which wraps a function in a CallbackEntity.

    Events support two additional mechanisms:

    1. **Generators**: When handle_event() returns a generator, the simulation
       wraps it as a ProcessContinuation, enabling multi-step processes that
       yield delays between steps.

    2. **Completion Hooks**: Functions registered via on_complete run when the
       event finishes (including after generator exhaustion). Used for chaining
       actions or notifying dependent entities.

    Sorting uses (time, insertion_order) to ensure deterministic FIFO ordering
    for events scheduled at the same instant.

    Attributes:
        time: When this event should be processed.
        event_type: Human-readable label for debugging and tracing.
        daemon: If True, this event won't block auto-termination.
        target: Entity to receive this event.
        on_complete: Hooks to run when processing finishes.
        context: Arbitrary metadata for tracing and debugging.
    """
    time: Instant
    event_type: str
    daemon: bool = field(default=False, repr=False)
    target: Optional['Simulatable'] = None
    on_complete: List[CompletionHook] = field(default_factory=list, repr=False, compare=False)
    
    # Context & Tracing
    context: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    # Sorting Internals
    _sort_index: int = field(default_factory=_global_event_counter.__next__, init=False, repr=False)
    _id: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)
    _cancelled: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def cancelled(self) -> bool:
        """Whether this event has been cancelled."""
        return self._cancelled

    def cancel(self) -> None:
        """Mark this event as cancelled. The simulation loop will skip it on pop.

        Cancelling an already-cancelled or already-processed event is a no-op.
        """
        self._cancelled = True

    def __post_init__(self):
        if self.target is None:
            raise ValueError(f"Event '{self.event_type}' must have a 'target'.")

        # Always ensure trace context exists (even if caller passed partial context)
        self.context.setdefault("id", str(self._id))
        self.context.setdefault("created_at", self.time)
        self.context.setdefault("stack", [])
        self.context.setdefault("metadata", {})
        self.context.setdefault("trace", {"spans": []})

    def __repr__(self) -> str:
        """Return a concise representation showing time, type, and target."""
        target_name = getattr(self.target, "name", None) or type(self.target).__name__
        return f"Event({self.time!r}, {self.event_type!r}, target={target_name})"

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
        
    def add_completion_hook(self, hook: CompletionHook) -> None:
        """Attach a function to run when this event finishes processing.

        Completion hooks enable dependency chains and notification patterns.
        For example, a QueueDriver uses hooks to know when its target entity
        has finished processing work and is ready for more.

        Args:
            hook: Function called with the finish time when processing completes.
        """
        self.on_complete.append(hook)

    def invoke(self) -> List['Event']:
        """Execute this event and return any resulting events.

        Dispatches to the target entity's handle_event() method. If the handler
        returns a generator, it's automatically wrapped as a ProcessContinuation
        for multi-step execution.

        Returns:
            New events to schedule, including any from completion hooks.
        """
        if getattr(self.target, '_crashed', False):
            return []

        handler_label = getattr(self.target, "name", type(self.target).__name__)
        self.context["stack"].append(handler_label)
        self.trace("handle.start", handler="entity", handler_label=handler_label)

        try:
            raw_result = self.target.handle_event(self)

            if isinstance(raw_result, Generator):
                self.trace("handle.end", result_kind="process")
                return self._start_process(raw_result)

            normalized = self._normalize_return(raw_result)
            self.trace("handle.end", result_kind="immediate", produced=len(normalized))

            completion_events = self._run_completion_hooks(self.time)

            return normalized + completion_events

        except Exception as exc:
            self.trace("handle.error", error=type(exc).__name__, message=str(exc))
            raise
        
    def _run_completion_hooks(self, time: Instant) -> List['Event']:
        """Helper to execute all hooks and flatten results.

        Hooks are expected to be one-shot: they run once when the event (or
        generator-based process) finishes. After running, the hook list is
        cleared to prevent accidental double execution.
        """
        hooks = list(self.on_complete)
        self.on_complete.clear()

        results: List["Event"] = []
        for hook in hooks:
            hook_result = hook(time)

            if not hook_result:
                continue
            if isinstance(hook_result, list):
                results.extend(hook_result)
            else:
                results.append(hook_result)

        return results
    
    def _start_process(self, gen: Generator) -> List["Event"]:
        continuation = ProcessContinuation(
                time=self.time,
                event_type=self.event_type,
                daemon=self.daemon,
                target=self.target,
                process=gen,
                on_complete=self.on_complete,
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

    @staticmethod
    def once(
        time: Instant,
        event_type: str,
        fn: Callable[['Event'], Any],
        *,
        daemon: bool = False,
        context: Dict[str, Any] | None = None,
    ) -> 'Event':
        """Create a one-shot event that invokes a function.

        Wraps the function in a CallbackEntity so that all events
        use target-based dispatch uniformly.

        Args:
            time: When this event should fire.
            event_type: Human-readable label for debugging.
            fn: Function called with the event.
            daemon: If True, won't block auto-termination.
            context: Optional metadata dict.
        """
        from happysimulator.core.callback_entity import CallbackEntity

        entity = CallbackEntity(name=f"once:{event_type}", fn=fn)
        return Event(
            time=time,
            event_type=event_type,
            target=entity,
            daemon=daemon,
            context=context or {},
        )

@dataclass
class ProcessContinuation(Event):
    """Internal event that resumes a paused generator-based process.

    When an entity's handle_event() returns a generator, the simulation wraps
    it in a ProcessContinuation. Each invocation advances the generator to its
    next yield point, schedules another continuation for the yielded delay,
    and collects any side-effect events.

    This enables entities to express multi-step, time-consuming operations
    naturally using Python's generator syntax:

        def handle_event(self, event):
            yield 0.05  # Wait 50ms for network latency
            yield self.compute_time  # Wait for processing
            return self.create_response(event)

    Yields are interpreted as:
    - ``yield delay`` - Wait for delay seconds before resuming
    - ``yield (delay, events)`` - Wait and also schedule side-effect events
    - ``yield SimFuture()`` - Park until the future is resolved

    Attributes:
        process: The Python generator being executed incrementally.
    """
    process: Generator = field(default=None, repr=False)

    # Set by SimFuture._resume() when resuming from a future
    _send_value: Any = field(default=None, init=False, repr=False)

    def invoke(self) -> List["Event"]:
        """Advance the generator to its next yield and schedule the continuation."""
        from happysimulator.core.sim_future import SimFuture

        self.trace("process.resume.start")

        try:
            # 1. Wake up the process
            yielded_val = self.process.send(self._send_value)

            # 2. Check for SimFuture yield (park instead of scheduling)
            if isinstance(yielded_val, SimFuture):
                yielded_val._park(self)
                self.trace("process.park", future="SimFuture")
                return []

            # 3. Parse the yield (delay or delay+side_effects)
            delay, side_effects = self._normalize_yield(yielded_val)
            self.trace("process.yield", delay_s=delay)

            # 4. Schedule the next Resume (Recursive Continuation)
            resume_time = self.time + delay
            next_continuation = ProcessContinuation(
                time=resume_time,
                event_type=self.event_type,
                daemon=self.daemon,
                target=self.target,     # Keep targeting the same entity
                on_complete=self.on_complete,
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
            # Process finished. Return the final value (if any) PLUS completion hooks.
            finished = self._normalize_return(e.value)
            completion_events = self._run_completion_hooks(self.time)
            self.trace(
                "process.stop",
                produced=len(finished) + len(completion_events),
                finished_produced=len(finished),
                completion_produced=len(completion_events),
            )
            return finished + completion_events

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