from typing import List
from happysimulator.data.data import Data
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.utils.instant import Instant


class _ProbeProfile(Profile):
    """
    Internal: Fixed rate profile for the probe's polling interval.
    If the probe interval is 0.5s, the rate is 2.0Hz.
    """
    def __init__(self, interval_seconds: float):
        if interval_seconds <= 0:
            raise ValueError("Probe interval must be positive.")
        self.rate = 1.0 / interval_seconds
        self._interval = interval_seconds
        
    def get_rate(self, time: Instant) -> float:
        return self.rate


class _ProbeEventProvider(EventProvider):
    """Creates probe events that measure a metric on a target entity via callback."""
    
    def __init__(self, target: Entity, metric: str, data_sink: Data):
        super().__init__()
        self.target = target
        self.metric = metric
        self.data_sink = data_sink
    
    def _create_measurement_callback(self):
        """Create a callback that pulls the measurement from the target and stores it."""
        target = self.target
        metric = self.metric
        data_sink = self.data_sink
        
        def measure_callback(event: Event) -> list[Event]:
            val = 0.0
            if hasattr(target, metric):
                raw_val = getattr(target, metric)
                if callable(raw_val):
                    val = raw_val()
                else:
                    val = raw_val
            data_sink.add_stat(val, event.time)
            return []
        
        return measure_callback
    
    def get_events(self, time: Instant) -> List[Event]:
        callback = self._create_measurement_callback()
        return [
            Event(
                time=time,
                daemon=True,
                event_type="probe_event",
                target=None,
                callback=callback)
            ]


class Probe(Source):
    """
    A Probe periodically measures a metric on a target entity and stores the data.
    
    Uses callback-style events to collect measurements at a fixed interval.
    The callback pulls the metric value from the target via reflection.
    """
    
    def __init__(self, target: Entity, metric: str, data: Data, interval: float = 1.0, start_time: Instant | None = None):
        self.target = target
        self.metric = metric
        self.data_sink = data
        
        if start_time is None:
            start_time = Instant.Epoch
        
        profile = _ProbeProfile(interval)
        provider = _ProbeEventProvider(target, metric, data)
        arrival_time_provider = ConstantArrivalTimeProvider(profile, start_time=start_time)
        
        super().__init__(
            name=f"Probe_{target.name}_{metric}",
            event_provider=provider,
            arrival_time_provider=arrival_time_provider
        )
