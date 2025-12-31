from typing import Union, List

from happysimulator.data.datasink import DataSink
from happysimulator.events.event import Event
from happysimulator.data.stat import Stat
from happysimulator.utils.instant import Instant


class MeasurementEvent(Event):
    def __init__(self, time: Instant, callback, name: str, stat: Union[Stat, List[Stat]], interval: Instant, sink: list[DataSink]):
        super().__init__(time, name, callback)
        self.stat = stat
        self.interval = interval
        self.sink = sink