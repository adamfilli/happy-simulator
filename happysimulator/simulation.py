import logging

from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from happysimulator.event_heap import EventHeap
from happysimulator.load.source import Source
from happysimulator.tracing.recorder import TraceRecorder, NullTraceRecorder
from happysimulator.utils.clock import Clock
from happysimulator.utils.instant import Instant

logger = logging.getLogger(__name__)


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
            
        self._clock = Clock(self._start_time)
        
        self._entities = entities or []
        self._sources = sources or []
        self._probes = probes or []
        
        all_components = self._entities + self._sources + self._probes
        for component in all_components:
            if isinstance(component, Entity):
                component.set_clock(self._clock)
        
        self._trace = trace_recorder or NullTraceRecorder()
        self._event_heap = EventHeap(trace_recorder=self._trace)
        
        logger.info(
            "Simulation initialized: start=%r, end=%r, sources=%d, entities=%d, probes=%d",
            self._start_time, self._end_time,
            len(self._sources), len(self._entities), len(self._probes),
        )
        
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
            logger.debug("Source '%s' produced %d initial event(s)", source.name, len(initial_events))
            
            # We push it to the heap to prime the simulation
            for event in initial_events:
                self._event_heap.push(event)
        
        for probe in self._probes:
            initial_events = probe.start(self._start_time)
            logger.debug("Probe '%s' produced %d initial event(s)", probe.name, len(initial_events))
            for event in initial_events:
                self._event_heap.push(event)
        
        logger.debug("Initialization complete, heap size: %d", self._event_heap.size())

    @property
    def trace_recorder(self) -> TraceRecorder:
        """Access the trace recorder for inspection after simulation."""
        return self._trace

    def schedule(self, events: Event | list[Event]) -> None:
        """Schedule one or more events into the simulation heap."""
        self._event_heap.push(events)

    def run(self) -> None:
        current_time = self._start_time
        self._event_heap.set_current_time(current_time)
        
        logger.info("Simulation starting at %r with %d event(s) in heap", current_time, self._event_heap.size())
        
        if not self._event_heap.has_events():
            logger.warning("Simulation started with empty event heap")
        
        self._trace.record(
            time=current_time,
            kind="simulation.start",
            heap_size=self._event_heap.size(),
        )
        
        events_processed = 0
        
        while self._event_heap.has_events() and self._end_time >= current_time:
            
            # TERMINATION CHECK:
            # If we rely on auto-termination (end_time is Infinity),
            # and we have no primary events left (only probes), STOP.
            if self._end_time == Instant.Infinity and not self._event_heap.has_primary_events():
                logger.info(
                    "Auto-terminating at %r: no primary events remaining (only daemon/probe events)",
                    current_time,
                )
                self._trace.record(
                    time=current_time,
                    kind="simulation.auto_terminate",
                    reason="no_primary_events",
                )
                break
            
            # 1. Pop
            event = self._event_heap.pop()

            if event.time < current_time:
                logger.warning(
                    "Time travel detected: next event scheduled at %r, but current simulation time is %r. "
                    "event_type=%s event_id=%s",
                    event.time,
                    current_time,
                    event.event_type,
                    event.context.get("id"),
                )
            current_time = event.time  # Advance clock
            self._event_heap.set_current_time(current_time)
            events_processed += 1
            
            logger.debug(
                "Processing event #%d: %r",
                events_processed, event,
            )
            
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
                logger.debug(
                    "Event %r produced %d new event(s)",
                    event.event_type, len(new_events),
                )
                for new_event in new_events:
                    logger.debug(
                        "  Scheduling %r for %r",
                        new_event.event_type, new_event.time,
                    )
                    self._trace.record(
                        time=current_time,
                        kind="simulation.schedule",
                        event_id=new_event.context.get("id"),
                        event_type=new_event.event_type,
                        scheduled_time=new_event.time,
                    )
                self._event_heap.push(new_events)
        
        # Determine why loop ended
        if not self._event_heap.has_events():
            logger.info("Simulation ended at %r: event heap exhausted", current_time)
        elif self._end_time < current_time:
            logger.info(
                "Simulation ended: current time %r exceeded end_time %r",
                current_time, self._end_time,
            )
        
        logger.info(
            "Simulation complete: processed %d events, final time %r, %d event(s) remaining in heap",
            events_processed, current_time, self._event_heap.size(),
        )
        
        if self._event_heap.size() > 0:
            logger.debug("Unprocessed events remain in heap (scheduled past end_time)")
        
        self._trace.record(
            time=current_time,
            kind="simulation.end",
            final_heap_size=self._event_heap.size(),
        )
        
        return

