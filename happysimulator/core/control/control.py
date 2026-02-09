"""Interactive simulation control for debugging and inspection.

SimulationControl provides pause/resume, single-stepping, breakpoints,
event hooks, and heap introspection. It is accessed via the
``Simulation.control`` property and adds zero overhead when not used.
"""

from __future__ import annotations

import heapq
import logging
import uuid
from typing import TYPE_CHECKING, Callable

from happysimulator.core.control.breakpoints import Breakpoint
from happysimulator.core.control.state import BreakpointContext, SimulationState
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from happysimulator.core.simulation import Simulation

logger = logging.getLogger(__name__)


class SimulationControl:
    """Interactive control surface for a running simulation.

    Created lazily via ``sim.control``. Provides:

    - **Execution control**: pause, resume, step, reset
    - **Breakpoints**: condition-based pausing (time, count, metric, etc.)
    - **Event hooks**: callbacks on event processing and time advance
    - **Heap introspection**: peek at upcoming events without consuming them

    Args:
        simulation: The simulation instance to control.
    """

    def __init__(self, simulation: Simulation) -> None:
        self._sim = simulation
        self._pause_requested = False
        self._steps_remaining: int | None = None
        self._breakpoints: dict[str, Breakpoint] = {}
        self._event_hooks: dict[str, Callable[[Event], None]] = {}
        self._time_hooks: dict[str, Callable[[Instant], None]] = {}

    # ------------------------------------------------------------------
    # Execution control
    # ------------------------------------------------------------------

    @property
    def is_paused(self) -> bool:
        """True if the simulation is currently paused."""
        return self._sim._is_paused

    @property
    def is_running(self) -> bool:
        """True if the simulation has started and has not completed."""
        return self._sim._is_running

    def pause(self) -> None:
        """Request the simulation to pause before processing the next event.

        Takes effect at the next control check point in the event loop.
        """
        self._pause_requested = True
        logger.info("Pause requested")

    def resume(self):
        """Resume a paused simulation.

        Returns:
            SimulationSummary: Partial summary if paused again, or
            final summary if the simulation completes.
        """
        if not self._sim._is_paused:
            raise RuntimeError("Cannot resume: simulation is not paused")
        self._pause_requested = False
        self._steps_remaining = None
        self._sim._is_paused = False
        logger.info("Resuming simulation")
        return self._sim.run()

    def step(self, n: int = 1):
        """Process exactly n events then pause.

        Args:
            n: Number of events to process. Must be >= 1.

        Returns:
            SimulationSummary: Partial summary after stepping.
        """
        if n < 1:
            raise ValueError("step count must be >= 1")
        if not self._sim._is_running:
            raise RuntimeError("Cannot step: simulation is not running")
        self._pause_requested = False
        self._steps_remaining = n
        self._sim._is_paused = False
        logger.info("Stepping %d event(s)", n)
        return self._sim.run()

    def get_state(self) -> SimulationState:
        """Return a snapshot of the simulation's current state."""
        import time as _time

        wall_elapsed = 0.0
        if self._sim._wall_start is not None:
            wall_elapsed = _time.monotonic() - self._sim._wall_start

        return SimulationState(
            current_time=self._sim._current_time,
            events_processed=self._sim._events_processed,
            heap_size=self._sim._event_heap.size(),
            primary_events_remaining=self._sim._event_heap._primary_event_count,
            is_paused=self._sim._is_paused,
            is_running=self._sim._is_running,
            is_complete=not self._sim._is_running and self._sim._events_processed > 0,
            last_event=self._sim._last_event,
            wall_clock_elapsed=wall_elapsed,
        )

    def reset(self) -> None:
        """Reset the simulation to its initial state.

        Clears the event heap, resets the clock and counters, and re-primes
        sources and probes. Does NOT reset entity internal state.
        """
        if self._sim._is_running and not self._sim._is_paused:
            raise RuntimeError("Cannot reset while simulation is actively running")

        logger.info("Resetting simulation")

        # Clear heap
        self._sim._event_heap = type(self._sim._event_heap)(
            trace_recorder=self._sim._trace,
        )

        # Reset clock
        self._sim._clock.update(self._sim._start_time)

        # Reset run state
        self._sim._current_time = self._sim._start_time
        self._sim._events_processed = 0
        self._sim._is_running = False
        self._sim._is_paused = False
        self._sim._last_event = None
        self._sim._wall_start = None
        self._sim._summary = None
        self._pause_requested = False
        self._steps_remaining = None

        # Re-prime sources and probes
        for source in self._sim._sources:
            initial_events = source.start(self._sim._start_time)
            for event in initial_events:
                self._sim._event_heap.push(event)

        for probe in self._sim._probes:
            initial_events = probe.start(self._sim._start_time)
            for event in initial_events:
                self._sim._event_heap.push(event)

        logger.info("Reset complete, heap size: %d", self._sim._event_heap.size())

    # ------------------------------------------------------------------
    # Breakpoints
    # ------------------------------------------------------------------

    def add_breakpoint(self, bp: Breakpoint) -> str:
        """Register a breakpoint. Returns a unique ID for removal."""
        bp_id = str(uuid.uuid4())[:8]
        self._breakpoints[bp_id] = bp
        logger.info("Added breakpoint %s: %s", bp_id, bp)
        return bp_id

    def remove_breakpoint(self, bp_id: str) -> None:
        """Remove a breakpoint by its ID.

        Raises:
            KeyError: If the ID is not found.
        """
        del self._breakpoints[bp_id]
        logger.info("Removed breakpoint %s", bp_id)

    def list_breakpoints(self) -> list[tuple[str, Breakpoint]]:
        """Return all registered breakpoints as (id, breakpoint) pairs."""
        return list(self._breakpoints.items())

    def clear_breakpoints(self) -> None:
        """Remove all breakpoints."""
        self._breakpoints.clear()
        logger.info("Cleared all breakpoints")

    # ------------------------------------------------------------------
    # Event hooks
    # ------------------------------------------------------------------

    def on_event(self, callback: Callable[[Event], None]) -> str:
        """Register a hook called after each event is processed.

        Args:
            callback: Function receiving the processed Event.

        Returns:
            Hook ID for later removal.
        """
        hook_id = str(uuid.uuid4())[:8]
        self._event_hooks[hook_id] = callback
        return hook_id

    def on_time_advance(self, callback: Callable[[Instant], None]) -> str:
        """Register a hook called when simulation time advances.

        Args:
            callback: Function receiving the new simulation time.

        Returns:
            Hook ID for later removal.
        """
        hook_id = str(uuid.uuid4())[:8]
        self._time_hooks[hook_id] = callback
        return hook_id

    def remove_hook(self, hook_id: str) -> None:
        """Remove an event or time hook by its ID.

        Raises:
            KeyError: If the ID is not found in either hook registry.
        """
        if hook_id in self._event_hooks:
            del self._event_hooks[hook_id]
            return
        if hook_id in self._time_hooks:
            del self._time_hooks[hook_id]
            return
        raise KeyError(f"Hook {hook_id!r} not found")

    # ------------------------------------------------------------------
    # Heap introspection
    # ------------------------------------------------------------------

    def peek_next(self, n: int = 1) -> list[Event]:
        """Preview the next n events without removing them from the heap.

        Only available when the simulation is paused.

        Args:
            n: Number of events to preview.

        Returns:
            List of up to n events in scheduled order.
        """
        if not self._sim._is_paused:
            raise RuntimeError("peek_next() is only available when paused")
        heap_list = self._sim._event_heap._heap
        return heapq.nsmallest(n, heap_list)

    def find_events(self, predicate: Callable[[Event], bool]) -> list[Event]:
        """Find all heap events matching a predicate.

        Only available when the simulation is paused. Performs a linear scan.

        Args:
            predicate: Function returning True for matching events.

        Returns:
            List of matching events (unsorted).
        """
        if not self._sim._is_paused:
            raise RuntimeError("find_events() is only available when paused")
        return [e for e in self._sim._event_heap._heap if predicate(e)]

    # ------------------------------------------------------------------
    # Internal methods (called by Simulation.run())
    # ------------------------------------------------------------------

    def _should_pause(self) -> bool:
        """Check if the simulation should pause before the next pop.

        Called by the run loop before each event is popped.
        """
        if self._pause_requested:
            return True
        if self._steps_remaining is not None and self._steps_remaining <= 0:
            return True
        return False

    def _notify_time_advance(self, new_time: Instant) -> None:
        """Fire time-advance hooks."""
        for callback in self._time_hooks.values():
            callback(new_time)

    def _notify_event_processed(self, event: Event) -> None:
        """Fire event-processed hooks and decrement step counter."""
        if self._steps_remaining is not None:
            self._steps_remaining -= 1
        for callback in self._event_hooks.values():
            callback(event)

    def _check_breakpoints(self) -> bool:
        """Evaluate all breakpoints. Return True if any triggered."""
        if not self._breakpoints:
            return False

        context = BreakpointContext(
            current_time=self._sim._current_time,
            events_processed=self._sim._events_processed,
            last_event=self._sim._last_event,
            simulation=self._sim,
        )

        triggered = False
        to_remove: list[str] = []

        for bp_id, bp in self._breakpoints.items():
            if bp.should_break(context):
                logger.info("Breakpoint %s triggered: %s", bp_id, bp)
                triggered = True
                if bp.one_shot:
                    to_remove.append(bp_id)

        for bp_id in to_remove:
            del self._breakpoints[bp_id]

        return triggered
