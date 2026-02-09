"""Breakpoint definitions for pausing simulations on conditions.

Breakpoints are evaluated after each event is processed. When a breakpoint's
``should_break()`` returns True, the simulation pauses. One-shot breakpoints
are automatically removed after triggering.

Five concrete implementations cover common debugging scenarios:

- TimeBreakpoint: pause at a specific simulation time
- EventCountBreakpoint: pause after N events processed
- ConditionBreakpoint: pause when a custom predicate is satisfied
- MetricBreakpoint: pause when an entity attribute crosses a threshold
- EventTypeBreakpoint: pause when a specific event type is processed
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from happysimulator.core.control.state import BreakpointContext
from happysimulator.core.temporal import Instant


@runtime_checkable
class Breakpoint(Protocol):
    """Protocol for breakpoint implementations.

    Breakpoints are evaluated after each event. Implement ``should_break``
    to define the trigger condition. Set ``one_shot`` to True for breakpoints
    that should auto-remove after triggering once.
    """

    @property
    def one_shot(self) -> bool: ...

    def should_break(self, context: BreakpointContext) -> bool: ...


_OPERATORS: dict[str, Callable[[object, object], bool]] = {
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
}


@dataclass(frozen=True)
class TimeBreakpoint:
    """Pause when simulation time reaches or exceeds a threshold.

    Args:
        time: The simulation time at which to break.
        one_shot: If True (default), removed after first trigger.
    """
    time: Instant
    one_shot: bool = True

    def should_break(self, context: BreakpointContext) -> bool:
        return context.current_time >= self.time

    def __str__(self) -> str:
        return f"TimeBreakpoint(time={self.time!r}, one_shot={self.one_shot})"


@dataclass(frozen=True)
class EventCountBreakpoint:
    """Pause after a given number of events have been processed.

    Args:
        count: The event count threshold.
        one_shot: If True (default), removed after first trigger.
    """
    count: int
    one_shot: bool = True

    def should_break(self, context: BreakpointContext) -> bool:
        return context.events_processed >= self.count

    def __str__(self) -> str:
        return f"EventCountBreakpoint(count={self.count}, one_shot={self.one_shot})"


@dataclass(frozen=True)
class ConditionBreakpoint:
    """Pause when a user-defined predicate returns True.

    Args:
        fn: Callable receiving BreakpointContext, returns bool.
        description: Human-readable description for listing breakpoints.
        one_shot: If True, removed after first trigger. Defaults to False.
    """
    fn: Callable[[BreakpointContext], bool] = field(repr=False)
    description: str = "custom condition"
    one_shot: bool = False

    def should_break(self, context: BreakpointContext) -> bool:
        return self.fn(context)

    def __str__(self) -> str:
        return f"ConditionBreakpoint({self.description!r}, one_shot={self.one_shot})"


@dataclass(frozen=True)
class MetricBreakpoint:
    """Pause when an entity attribute crosses a threshold.

    Looks up the named entity in the simulation, reads the given attribute,
    and applies the comparison operator against the threshold.

    Args:
        entity_name: Name of the entity to inspect.
        attribute: Attribute name to read from the entity.
        operator: Comparison operator: "gt", "ge", "lt", "le", "eq", "ne".
        threshold: Value to compare against.
        one_shot: If True, removed after first trigger. Defaults to False.

    Raises:
        ValueError: If the operator string is not recognized.
    """
    entity_name: str
    attribute: str
    operator: str
    threshold: float
    one_shot: bool = False

    def __post_init__(self) -> None:
        if self.operator not in _OPERATORS:
            raise ValueError(
                f"Unknown operator {self.operator!r}. "
                f"Expected one of: {', '.join(sorted(_OPERATORS))}"
            )

    def should_break(self, context: BreakpointContext) -> bool:
        entity = self._find_entity(context)
        if entity is None:
            return False
        value = getattr(entity, self.attribute, None)
        if value is None:
            return False
        op_fn = _OPERATORS[self.operator]
        return op_fn(value, self.threshold)

    def _find_entity(self, context: BreakpointContext) -> object | None:
        for component in context.simulation._entities:
            if getattr(component, "name", None) == self.entity_name:
                return component
        return None

    def __str__(self) -> str:
        return (
            f"MetricBreakpoint({self.entity_name}.{self.attribute} "
            f"{self.operator} {self.threshold}, one_shot={self.one_shot})"
        )


@dataclass(frozen=True)
class EventTypeBreakpoint:
    """Pause when the last processed event matches a specific type.

    Args:
        event_type: The event_type string to match.
        one_shot: If True, removed after first trigger. Defaults to False.
    """
    event_type: str
    one_shot: bool = False

    def should_break(self, context: BreakpointContext) -> bool:
        return context.last_event.event_type == self.event_type

    def __str__(self) -> str:
        return f"EventTypeBreakpoint({self.event_type!r}, one_shot={self.one_shot})"
