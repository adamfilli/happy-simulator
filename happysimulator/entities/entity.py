from abc import ABC
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
        
    def handle_event(self, event: Event) -> Union[Generator[SimYield, None, SimReturn], list[Event], Event, None]:
            """
            Handles an event and returns the reaction.
            
            Returns:
                Generator: For sequential processes that yield delays/control (e.g., yield 0.1).
                        The generator may also return a final list[Event] upon completion.
                list[Event]: For immediate, atomic event scheduling (legacy style).
                None: If the event requires no reaction.
            """
            pass

