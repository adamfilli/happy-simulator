from happysimulator.entities.entity import Entity
from happysimulator.event_heap import EventHeap
from happysimulator.load.source import Source
from happysimulator.tracing.recorder import TraceRecorder, NullTraceRecorder
from happysimulator.utils.instant import Instant


class Simulation:
    def __init__(
        self,
        start_time: Instant = None,
        end_time: Instant = None,
        sources: list[Source] = None,
        entities: list[Entity] = None,
        probes: list[Source] = None,
        trace_recorder: TraceRecorder | None = None,
    ):
        self._start_time = start_time
        if self._start_time is None:
            self._start_time = Instant.Epoch
        
        self._end_time = end_time
        if self._end_time is None:
            self._end_time = Instant.Infinity
        
        self._entities = entities or []
        self._sources = sources or []
        self._probes = probes or []
        
        self._trace = trace_recorder or NullTraceRecorder()
        self._event_heap = EventHeap(trace_recorder=self._trace)
        
        self._trace.record(
            time=self._start_time,
            kind="simulation.init",
            num_sources=len(self._sources),
            num_entities=len(self._entities),
            num_probes=len(self._probes),
        )
        
        for source in self._sources:
            # The source calculates its first event and returns it
            initial_events = source.start(self._start_time)
            
            # We push it to the heap to prime the simulation
            for event in initial_events:
                self._event_heap.push(event)
        
        for probe in self._probes:
            initial_events = probe.start(self._start_time)
            for event in initial_events:
                self._event_heap.push(event)

    @property
    def trace_recorder(self) -> TraceRecorder:
        """Access the trace recorder for inspection after simulation."""
        return self._trace

    def run(self) -> None:
        current_time = self._start_time
        self._event_heap.set_current_time(current_time)
        
        self._trace.record(
            time=current_time,
            kind="simulation.start",
            heap_size=self._event_heap.size(),
        )
        
        while self._event_heap.has_events() and self._end_time >= current_time:
            
            # TERMINATION CHECK:
            # If we rely on auto-termination (end_time is Infinity),
            # and we have no primary events left (only probes), STOP.
            if self._end_time == Instant.Infinity and not self._event_heap.has_primary_events():
                self._trace.record(
                    time=current_time,
                    kind="simulation.auto_terminate",
                    reason="no_primary_events",
                )
                break
            
            # 1. Pop
            event = self._event_heap.pop()
            current_time = event.time  # Advance clock
            self._event_heap.set_current_time(current_time)
            
            self._trace.record(
                time=current_time,
                kind="simulation.dequeue",
                event_id=event.context.get("id"),
                event_type=event.event_type,
            )
            
            # 2. Invoke
            # The event itself knows how to run and what to return
            new_events = event.invoke()
            
            # 3. Push
            if new_events:
                for new_event in new_events:
                    self._trace.record(
                        time=current_time,
                        kind="simulation.schedule",
                        event_id=new_event.context.get("id"),
                        event_type=new_event.event_type,
                        scheduled_time=new_event.time,
                    )
                self._event_heap.push(new_events)
        
        self._trace.record(
            time=current_time,
            kind="simulation.end",
            final_heap_size=self._event_heap.size(),
        )
        
        return

