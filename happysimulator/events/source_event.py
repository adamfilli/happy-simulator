from dataclasses import dataclass
from happysimulator.events.event import Event
from happysimulator.utils.instant import Instant
from happysimulator.entities.entity import Entity 

@dataclass
class SourceEvent(Event):
    """
    A self-scheduled 'tick' that tells the Source to produce data.
    """
    def __init__(self, time: Instant, source_entity: Entity, daemon: bool = False):
        super().__init__(
            time=time,
            event_type="source_event",
            daemon=daemon,
            target=source_entity,
            callback=None)