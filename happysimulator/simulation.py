from types import GeneratorType
from archive.arrival_distribution import ArrivalDistribution
from happysimulator.entities.entity import Entity
from happysimulator.event_heap import EventHeap
from archive.generate_event import GenerateEvent
from archive.measurement_event import MeasurementEvent
from happysimulator.load.source import Source
from happysimulator.data.measurement import Measurement
from happysimulator.events.process_continuation import ProcessContinuation
from happysimulator.load import ConstantProfile
from happysimulator.data.simulation_result import SimulationResult
from happysimulator.utils.instant import Instant
from .events.event import Event


def create_sources(measurements: list[Measurement], end_time):
    sources = []

    for measurement in measurements:

        # the default arg fixes outerscope issue: https://stackoverflow.com/questions/50298582/why-does-python-asyncio-loop-call-soon-overwrite-data
        def create_measurement_event_func(time: Instant, measurement=measurement):
            return [MeasurementEvent(name=measurement.name, callback=measurement.func, time=time, interval=measurement.interval, stat=measurement.stat, sink=measurement.sinks)]

        sources.append(Source(name=f"{measurement.name}_generator", func=create_measurement_event_func, distribution=ArrivalDistribution.CONSTANT, arrival_rate_profile=ConstantProfile.from_period(measurement.interval)))

    return sources

class Simulation:
    def __init__(self, end_time: Instant, sources: list[Source], measurements: list[Measurement], entities: list[Entity] = None):
        self._end_time = end_time
        self._entities = entities
        self._sources = sources
        self._measurements = measurements

        self._measurement_sources = create_sources(measurements, end_time)

        self._event_heap = EventHeap(
            [item for source in self._sources for item in source.next(event=GenerateEvent(time=Instant.from_seconds(0), callback=None, name="BootstrapEvent"))] +
            [item for source in self._measurement_sources for item in source.generate(event=GenerateEvent(time=Instant.from_seconds(0), callback=None, name="SecondBootstrappedEvent"))]
        )

    def run(self) -> SimulationResult:
        current_time = Instant.from_seconds(0)
        
        while self._event_heap.has_events() and self._end_time > current_time:
            # 1. Pop unified event
            event = self._event_heap.pop()
            current_time = event.time

            # 2. Check: Is this a Resume (Generator) or a New Event?
            if event.is_continuation():
                # It's a paused process
                gen = event.payload
                entity = event.entity
                
                try:
                    # WAKE UP the generator
                    delay, side_effects = next(gen)
                    
                    # SCHEDULE the side effect events
                    for side_effect_event in side_effects:
                        self._event_heap.push(side_effect_event)
                    
                    # SCHEDULE the next resume event
                    resume_time = current_time + delay
                    
                    # Create continuation event, inheriting context (Trace ID)
                    next_event = Event.create_continuation(
                        time=resume_time,
                        entity=entity,
                        generator=gen,
                        parent_context=event.context
                    )
                    self._event_heap.push(next_event)

                except StopIteration as e:
                    # Generator Finished
                    if e.value is not None:
                        self._event_heap.push(e.value)

            else:
                # It's a standard event (start of a flow)
                result = event.entity.handle_event(event)
                
                # Helper to handle different return types from handle_event
                if isinstance(result, Source):
                    # Handler started a new process immediately
                    try:
                        delay = next(result)
                        next_event = Event.create_continuation(
                            time=current_time + delay,
                            entity=event.entity,
                            generator=result,
                            parent_context=event.context
                        )
                        self._event_heap.push(next_event)
                    except StopIteration as e:
                        if e.value is not None:
                            self._event_heap.push(e.value)
                
                elif isinstance(result, list[Event]):
                    for event in result:
                        self._event_heap.push(event)
                elif isinstance(result, Event):
                    self._event_heap.push(event)

        return SimulationResult(sinks=[sink for measurement in self._measurements for sink in measurement.sinks])

