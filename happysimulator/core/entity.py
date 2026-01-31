"""Base class for simulation actors that respond to events.

Entities are the building blocks of a simulation model. Each entity receives
events via handle_event() and returns reactions (new events or generators).
"""

import logging
from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, Union

from happysimulator.core.temporal import Instant
from happysimulator.core.clock import Clock
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)

SimYield = Union[float, Tuple[float, list[Event], Event]]
"""Type alias for generator yield values: delay or (delay, side_effects)."""

SimReturn = Optional[Union[list[Event], Event]]
"""Type alias for generator return values: events to schedule on completion."""


class Entity(ABC):
    """Abstract base class for all simulation actors.

    Entities receive events through handle_event() and produce reactions.
    They maintain a reference to the simulation clock for time-aware logic.

    The simulation injects the clock during initialization, so entities
    should not be used outside a simulation context.

    Subclasses must implement handle_event() to define their behavior.
    Optionally override has_capacity() to model resource constraints.

    Attributes:
        name: Identifier for logging and debugging.
    """

    def __init__(self, name: str):
        self.name = name
        self._clock: Optional[Clock] = None

    def set_clock(self, clock: Clock) -> None:
        """Inject the simulation clock. Called automatically during setup."""
        self._clock = clock
        logger.debug("[%s] Clock injected", self.name)

    @property
    def now(self) -> Instant:
        """Current simulation time from the injected clock.

        Raises:
            RuntimeError: If accessed before clock injection.
        """
        if self._clock is None:
            logger.error("[%s] Attempted to access time before clock injection", self.name)
            raise RuntimeError(f"Entity {self.name} is not attached to a simulation (Clock is None).")
        return self._clock.now

    @abstractmethod
    def handle_event(self, event: Event) -> Union[Generator[SimYield, None, SimReturn], list[Event], Event, None]:
        """Process an incoming event and return any resulting events.

        Returns:
            Generator: For multi-step processes. Yield delays; optionally return
                events on completion.
            list[Event] | Event | None: For immediate, single-step responses.
        """
        raise NotImplementedError

    def has_capacity(self) -> bool:
        """Check if this entity can accept additional work.

        Override in subclasses with concurrency limits, rate limits, or
        other resource constraints. Returns True by default.
        """
        return True

