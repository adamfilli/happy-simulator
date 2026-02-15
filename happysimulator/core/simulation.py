"""Core simulation orchestrator for the discrete-event simulation engine.

This module implements the main simulation loop using a pop-invoke-push pattern:
events are popped from a priority heap, invoked to perform their work, and any
resulting events are pushed back for future processing.
"""

import logging
import time as _time
from typing import TYPE_CHECKING

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event, _clear_active_code_debugger, _set_active_code_debugger
from happysimulator.core.event_heap import EventHeap
from happysimulator.core.protocols import Simulatable
from happysimulator.core.sim_future import _clear_active_context, _set_active_context
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.recorder import NullTraceRecorder, TraceRecorder
from happysimulator.instrumentation.summary import (
    EntitySummary,
    QueueStats,
    SimulationSummary,
)
from happysimulator.load.source import Source

if TYPE_CHECKING:
    from happysimulator.faults.schedule import FaultSchedule

logger = logging.getLogger(__name__)


class Simulation:
    """Orchestrates discrete-event simulation execution.

    The simulation maintains a central event heap and advances time by processing
    events in chronological order. It supports two termination modes:

    - Explicit: Run until `end_time` is reached
    - Auto: Run until no primary (non-daemon) events remain, allowing probes
      to keep running without blocking termination

    All entities share a common Clock instance, ensuring consistent time views
    across the simulation. Sources and probes are bootstrapped during initialization,
    priming the heap with their first events.

    Interactive control (pause/resume, stepping, breakpoints) is available via
    the ``control`` property, which lazily creates a ``SimulationControl`` instance.
    When ``control`` is never accessed, the run loop incurs zero overhead.

    Args:
        start_time: When the simulation begins. Defaults to Instant.Epoch (t=0).
        end_time: When to stop processing. Defaults to Instant.Infinity (auto-terminate).
        sources: Load generators that produce events at specified intervals.
        entities: Simulation actors that respond to events.
        probes: Measurement sources that run as daemons (won't block termination).
        trace_recorder: Optional recorder for debugging/visualization.
        fault_schedule: Optional fault injection schedule.
    """

    def __init__(
        self,
        start_time: Instant = None,
        end_time: Instant = None,
        sources: list[Source] | None = None,
        entities: list[Simulatable] | None = None,
        probes: list[Source] | None = None,
        trace_recorder: TraceRecorder | None = None,
        fault_schedule: "FaultSchedule | None" = None,
        duration: float | None = None,
    ):
        if duration is not None and end_time is not None:
            raise ValueError("Cannot specify both 'duration' and 'end_time'")

        self._start_time = start_time
        if self._start_time is None:
            self._start_time = Instant.Epoch

        if duration is not None:
            self._end_time = self._start_time + duration
        elif end_time is not None:
            self._end_time = end_time
        else:
            self._end_time = Instant.Infinity

        self._clock = Clock(self._start_time)

        self._entities = entities or []
        self._sources = sources or []
        self._probes = probes or []

        all_components = self._entities + self._sources + self._probes
        for component in all_components:
            if isinstance(component, Simulatable):
                component.set_clock(self._clock)

        self._trace = trace_recorder or NullTraceRecorder()
        self._event_heap = EventHeap(trace_recorder=self._trace)
        self._summary: SimulationSummary | None = None

        # Run state — promoted from run() locals for re-entrancy
        self._current_time: Instant = self._start_time
        self._events_processed: int = 0
        self._is_running: bool = False
        self._is_paused: bool = False
        self._wall_start: float | None = None
        self._last_event: Event | None = None
        self._events_cancelled: int = 0

        # Control surface — lazy-created on first access
        self._control = None

        logger.info(
            "Simulation initialized: start=%r, end=%r, sources=%d, entities=%d, probes=%d",
            self._start_time,
            self._end_time,
            len(self._sources),
            len(self._entities),
            len(self._probes),
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
            logger.debug(
                "Source '%s' produced %d initial event(s)", source.name, len(initial_events)
            )

            # We push it to the heap to prime the simulation
            for event in initial_events:
                self._event_heap.push(event)

        for probe in self._probes:
            initial_events = probe.start(self._start_time)
            logger.debug("Probe '%s' produced %d initial event(s)", probe.name, len(initial_events))
            for event in initial_events:
                self._event_heap.push(event)

        # Bootstrap fault schedule (if provided)
        self._fault_schedule = fault_schedule
        if self._fault_schedule is not None:
            self._fault_schedule.set_clock(self._clock)
            fault_events = self._fault_schedule.start(self._start_time, self)
            logger.debug("FaultSchedule produced %d event(s)", len(fault_events))
            for event in fault_events:
                self._event_heap.push(event)

        logger.debug("Initialization complete, heap size: %d", self._event_heap.size())

    @property
    def control(self):
        """Access the simulation control surface for interactive debugging.

        Lazily creates a SimulationControl instance on first access.
        """
        if self._control is None:
            from happysimulator.core.control import SimulationControl

            self._control = SimulationControl(self)
        return self._control

    @property
    def trace_recorder(self) -> TraceRecorder:
        """Access the trace recorder for inspection after simulation."""
        return self._trace

    @property
    def summary(self) -> SimulationSummary | None:
        """Summary from the most recent run(), or None if not yet run."""
        return self._summary

    def schedule(self, events: Event | list[Event]) -> None:
        """Inject events into the simulation from outside the event loop.

        Useful for adding events programmatically after initialization but before
        or during simulation execution.
        """
        self._event_heap.push(events)

    def run(self) -> SimulationSummary:
        """Execute the simulation until termination or pause.

        Implements the core pop-invoke-push loop:
        1. Pop the earliest event from the heap
        2. Advance simulation time to that event's timestamp
        3. Invoke the event (calling its target entity or callback)
        4. Push any resulting events back onto the heap
        5. Repeat until termination condition is met

        The method is re-entrant: calling run() on a paused simulation
        resumes from where it left off. When the simulation is paused
        (via control.pause(), control.step(), or a breakpoint), a partial
        SimulationSummary is returned.

        Termination occurs when:
        - The event heap is exhausted, or
        - Current time exceeds end_time, or
        - Auto-terminate mode with no primary events remaining, or
        - The simulation is paused (returns partial summary)

        Returns:
            SimulationSummary with run statistics (partial if paused).
        """
        # First call: initialize wall clock and mark as running
        if not self._is_running:
            self._wall_start = _time.monotonic()
            self._current_time = self._start_time
            self._events_processed = 0
            self._is_running = True
            self._event_heap.set_current_time(self._current_time)

            logger.info(
                "Simulation starting at %r with %d event(s) in heap",
                self._current_time,
                self._event_heap.size(),
            )

            if not self._event_heap.has_events():
                logger.warning("Simulation started with empty event heap")

            self._trace.record(
                time=self._current_time,
                kind="simulation.start",
                heap_size=self._event_heap.size(),
            )
        else:
            # Resuming from pause
            logger.info(
                "Simulation resuming at %r with %d event(s) in heap",
                self._current_time,
                self._event_heap.size(),
            )

        # Set active context so SimFuture.resolve()/fail() can schedule events
        _set_active_context(self._event_heap, self._clock)

        # Set code debugger context if one has been injected by the visual bridge
        _set_active_code_debugger(getattr(self, "_code_debugger", None))

        try:
            return self._run_loop()
        finally:
            _clear_active_code_debugger()
            _clear_active_context()

    def _run_loop(self) -> SimulationSummary:
        """Inner loop extracted for clean active-context scoping."""
        while self._event_heap.has_events() and self._end_time >= self._current_time:
            # CONTROL CHECK: should we pause before popping?
            if self._control is not None and self._control._should_pause():
                self._is_paused = True
                logger.info(
                    "Simulation paused at %r after %d events",
                    self._current_time,
                    self._events_processed,
                )
                return self._build_summary()

            # TERMINATION CHECK:
            # If we rely on auto-termination (end_time is Infinity),
            # and we have no primary events left (only probes), STOP.
            if self._end_time == Instant.Infinity and not self._event_heap.has_primary_events():
                logger.info(
                    "Auto-terminating at %r: no primary events remaining (only daemon/probe events)",
                    self._current_time,
                )
                self._trace.record(
                    time=self._current_time,
                    kind="simulation.auto_terminate",
                    reason="no_primary_events",
                )
                break

            # 1. Pop
            event = self._event_heap.pop()

            # Skip cancelled events (lazy deletion)
            if event.cancelled:
                self._events_cancelled += 1
                continue

            if event.time < self._current_time:
                logger.warning(
                    "Time travel detected: next event scheduled at %r, but current simulation time is %r. "
                    "Skipping event. event_type=%s event_id=%s",
                    event.time,
                    self._current_time,
                    event.event_type,
                    event.context.get("id"),
                )
                continue

            prev_time = self._current_time
            self._current_time = event.time  # Advance clock
            self._clock.update(self._current_time)
            self._event_heap.set_current_time(self._current_time)
            self._events_processed += 1
            self._last_event = event

            # CONTROL CHECK: notify time advance
            if self._control is not None and self._current_time != prev_time:
                self._control._notify_time_advance(self._current_time)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Processing event #%d: %r",
                    self._events_processed,
                    event,
                )

            self._trace.record(
                time=self._current_time,
                kind="simulation.dequeue",
                event_id=event.context.get("id"),
                event_type=event.event_type,
            )

            # 2. Invoke
            # The event itself knows how to run and what to return
            new_events = event.invoke()

            # 3. Push
            if new_events:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Event %r produced %d new event(s)",
                        event.event_type,
                        len(new_events),
                    )
                    for new_event in new_events:
                        logger.debug(
                            "  Scheduling %r for %r",
                            new_event.event_type,
                            new_event.time,
                        )
                for new_event in new_events:
                    self._trace.record(
                        time=self._current_time,
                        kind="simulation.schedule",
                        event_id=new_event.context.get("id"),
                        event_type=new_event.event_type,
                        scheduled_time=new_event.time,
                    )
                self._event_heap.push(new_events)

            # CONTROL CHECK: notify event processed + check breakpoints
            if self._control is not None:
                self._control._notify_event_processed(event)
                if self._control._check_breakpoints():
                    self._is_paused = True
                    logger.info(
                        "Simulation paused by breakpoint at %r after %d events",
                        self._current_time,
                        self._events_processed,
                    )
                    return self._build_summary()

        # Simulation complete
        self._is_running = False
        self._is_paused = False

        # Determine why loop ended
        if not self._event_heap.has_events():
            logger.info("Simulation ended at %r: event heap exhausted", self._current_time)
        elif self._end_time < self._current_time:
            logger.info(
                "Simulation ended: current time %r exceeded end_time %r",
                self._current_time,
                self._end_time,
            )

        logger.info(
            "Simulation complete: processed %d events, final time %r, %d event(s) remaining in heap",
            self._events_processed,
            self._current_time,
            self._event_heap.size(),
        )

        if self._event_heap.size() > 0:
            logger.debug("Unprocessed events remain in heap (scheduled past end_time)")

        self._trace.record(
            time=self._current_time,
            kind="simulation.end",
            final_heap_size=self._event_heap.size(),
        )

        self._summary = self._build_summary()
        return self._summary

    def _build_summary(self) -> SimulationSummary:
        """Build a SimulationSummary from current state."""
        wall_elapsed = _time.monotonic() - self._wall_start if self._wall_start else 0.0
        duration_s = (self._current_time - self._start_time).to_seconds()
        eps = self._events_processed / duration_s if duration_s > 0 else 0.0

        entity_summaries = self._build_entity_summaries()

        return SimulationSummary(
            duration_s=duration_s,
            total_events_processed=self._events_processed,
            events_cancelled=self._events_cancelled,
            events_per_second=eps,
            wall_clock_seconds=wall_elapsed,
            entities=entity_summaries,
        )

    def _build_entity_summaries(self) -> dict[str, EntitySummary]:
        """Build per-entity summaries from registered entities."""
        from happysimulator.components.queued_resource import QueuedResource

        summaries: dict[str, EntitySummary] = {}
        for component in self._entities:
            if not isinstance(component, Entity):
                continue

            queue_stats = None
            if isinstance(component, QueuedResource):
                queue_stats = QueueStats(
                    peak_depth=0,  # Not tracked yet; would need probe data
                    total_accepted=component.stats_accepted,
                    total_dropped=component.stats_dropped,
                )

            # Count events handled if entity tracks it
            events_handled = 0
            for attr in ("count", "events_received", "stats_processed"):
                val = getattr(component, attr, None)
                if isinstance(val, int):
                    events_handled = val
                    break

            summaries[component.name] = EntitySummary(
                name=component.name,
                entity_type=type(component).__name__,
                events_handled=events_handled,
                queue_stats=queue_stats,
            )
        return summaries
