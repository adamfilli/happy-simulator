from happysimulator.events.event import Event
from happysimulator.utils.instant import Instant


class GenerateEvent(Event):
    def __init__(self, time: Instant, callback, name: str):
        super().__init__(time, name, callback)
