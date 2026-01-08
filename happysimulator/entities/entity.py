from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, Union

from archive.experiments.arrival_distributions import Instant
from happysimulator.utils.clock import Clock

from ..events.event import Event

# Alias for what the Generator yields: 
# Either a delay (float) OR a tuple of (delay, side_effect_events)
SimYield = Union[float, Tuple[float, list[Event], Event]]

# Alias for what the Generator returns when it finishes (via return statement)
SimReturn = Optional[Union[list[Event], Event]]

class Entity(ABC):
    def __init__(self, name):
        self.name = name
        self._clock: Optional[Clock] = None # Injected later

    def set_clock(self, clock: Clock):
        """Called by Simulation during initialization."""
        self._clock = clock

    @property
    def now(self) -> Instant:
        """Convenience accessor for current simulation time."""
        if self._clock is None:
            raise RuntimeError(f"Entity {self.name} is not attached to a simulation (Clock is None).")
        return self._clock.now
        
    @abstractmethod
    def handle_event(self, event: Event) -> Union[Generator[SimYield, None, SimReturn], list[Event], Event, None]:
        """Handle an event and return the reaction.

        Returns:
            Generator: For sequential processes that yield delays/control (e.g., yield 0.1).
                The generator may also return a final list[Event] upon completion.
            list[Event] | Event | None: For immediate, atomic event scheduling.
        """
        raise NotImplementedError

    def has_capacity(self) -> bool:
        """
        Return True if this entity can accept more work.
        
        Override in subclasses that have concurrency limits, rate limits,
        or other resource constraints. Default implementation always returns True.
        """
        return True

