from happysimulator.entities.entity import Entity
from happysimulator.event_heap import EventHeap
from happysimulator.events.event import Event
from happysimulator.load.source import Source
from happysimulator.utils.instant import Instant

class Simulation:
    def __init__(self, start_time: Instant, end_time: Instant, sources: list[Source], entities: list[Entity] = None):
        self._start_time = start_time
        self._end_time = end_time
        self._entities = entities
        self._sources = sources
        
        # TODO: add measurements back in, see notepad.

        self._event_heap = EventHeap()
        
        for source in self._sources:
            # The source calculates its first event and returns it
            initial_events = source.start(start_time)
            
            # We push it to the heap to prime the simulation
            for event in initial_events:
                self._event_heap.push(event)

    def run(self) -> None:
        current_time = Instant.from_seconds(0)
        
        while self._event_heap.has_events() and self._end_time > current_time:
            # 1. Pop event
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
                if event.entity is not None:
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
                    
                    elif isinstance(result, list):
                        for event in result:
                            self._event_heap.push(event)
                    elif isinstance(result, Event):
                        self._event_heap.push(event)

        return

