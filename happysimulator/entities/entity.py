from abc import ABC
from typing import Generator, Optional, Tuple, Union

from ..events.event import Event
from ..data.data import Data

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

    def sink_data(self, data: Union[Data, float, int], event):
        for sink in event.sink:
            if isinstance(data, Data):
                if isinstance(event.stat, list):  # Handling multiple stats
                    stats = {}
                    for stat in event.stat:
                        stats[stat.name] = data.get_stats(begin=event.time - event.interval, end=event.time,
                                                    aggregator=stat)
                    sink.add_stat(event.time, stats)
                else:  # Handling a single stat
                    stats = data.get_stats(begin=event.time - event.interval, end=event.time,
                                           aggregator=event.stat)
                    sink.add_stat(event.time, stats)
            elif isinstance(data, (float, int)):
                if isinstance(event.stat, list) and len(event.stat) > 1:
                    raise RuntimeError("A single data value provided but multiple stat names are defined in the event.")
                else:
                    sink.add_stat(event.time, float(data))
            else:
                raise RuntimeError("Unsupported data type submitted to data_sink.")

