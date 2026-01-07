from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, Union

from ..events.event import Event

# Alias for what the Generator yields: 
# Either a delay (float) OR a tuple of (delay, side_effect_events)
SimYield = Union[float, Tuple[float, list[Event], Event]]

# Alias for what the Generator returns when it finishes (via return statement)
SimReturn = Optional[Union[list[Event], Event]]

class Entity(ABC):
    def __init__(self, name):
        self.name = name
        
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

