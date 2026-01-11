"""Protocol definitions for duck-typed simulation components.

This module defines structural types (Protocols) that allow any class with
the right methods to participate in a simulation, without requiring inheritance
from Entity.

Two approaches are supported:
1. Inherit from Entity (traditional, explicit contract)
2. Implement the Simulatable protocol (duck typing, no inheritance required)

Example using duck typing:
    class MyClient:
        def __init__(self, name: str):
            self.name = name
            self._clock: Clock | None = None

        def set_clock(self, clock: Clock) -> None:
            self._clock = clock

        @property
        def now(self) -> Instant:
            return self._clock.now

        def handle_event(self, event: Event) -> list[Event] | None:
            # process event
            return []

Or use the @simulatable decorator for automatic clock injection:
    from happysimulator.core.decorators import simulatable

    @simulatable
    class MyClient:
        def __init__(self, name: str):
            self.name = name

        def handle_event(self, event: Event) -> list[Event] | None:
            print(f"Processing at {self.now}")  # self.now is injected
            return []
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generator, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from happysimulator.core.clock import Clock
    from happysimulator.core.event import Event
    from happysimulator.core.instant import Instant


# Type alias matching Entity's return type
SimYield = Union[float, tuple[float, list["Event"], "Event"]]
SimReturn = Union[list["Event"], "Event", None]
HandleEventReturn = Union[Generator[SimYield, None, SimReturn], list["Event"], "Event", None]


@runtime_checkable
class Simulatable(Protocol):
    """Protocol for objects that can participate in a simulation.

    Any class that implements these methods can be used as an entity in the
    simulation, without inheriting from Entity. The @runtime_checkable decorator
    allows isinstance() checks at runtime.

    Required attributes:
        name: Identifier for logging and debugging.

    Required methods:
        set_clock: Called by Simulation to inject the shared clock.
        handle_event: Process incoming events and return reactions.

    Optional methods:
        has_capacity: Override to model resource constraints (defaults to True).
    """

    name: str

    def set_clock(self, clock: "Clock") -> None:
        """Receive the simulation clock for time-aware operations.

        Called automatically by Simulation during initialization.
        Store the clock to access simulation time via clock.now.
        """
        ...

    def handle_event(self, event: "Event") -> HandleEventReturn:
        """Process an incoming event and return any resulting events.

        Returns:
            Generator: For multi-step processes. Yield delays; optionally return
                events on completion.
            list[Event] | Event | None: For immediate, single-step responses.
        """
        ...


@runtime_checkable
class HasCapacity(Protocol):
    """Optional protocol for entities with resource constraints.

    Entities that implement has_capacity() can signal when they're unable
    to accept additional work (e.g., server at max connections).
    """

    def has_capacity(self) -> bool:
        """Check if this entity can accept additional work."""
        ...
