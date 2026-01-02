from happysimulator.entities.entity import Entity
from happysimulator.event_heap import EventHeap
from happysimulator.load.source import Source
from happysimulator.utils.instant import Instant

class Simulation:
    def __init__(self, start_time: Instant = None, end_time: Instant = None, sources: list[Source] = None, entities: list[Entity] = None):
        self._start_time = start_time
        if self._start_time is None:
            self._start_time = Instant.Epoch
        
        self._end_time = end_time
        if self._end_time is None:
            self._end_time = Instant.Infinity
        
        self._entities = entities
        self._sources = sources
        
        # TODO: add measurements back in, see notepad.

        self._event_heap = EventHeap()
        
        for source in self._sources:
            # The source calculates its first event and returns it
            initial_events = source.start(self._start_time)
            
            # We push it to the heap to prime the simulation
            for event in initial_events:
                self._event_heap.push(event)

    def run(self) -> None:
        current_time = Instant.Epoch
        
        while self._event_heap.has_events() and self._end_time >= current_time:
            # 1. Pop
            event = self._event_heap.pop()
            current_time = event.time # Advance clock
            
            # 2. Invoke
            # The event itself knows how to run and what to return
            new_events = event.invoke()
            
            # 3. Push
            if new_events:
                self._event_heap.push(new_events)

        return

