"""State and context dataclasses for simulation control.

SimulationState provides a snapshot of the simulation's current status.
BreakpointContext is passed to breakpoint predicates for evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from happysimulator.core.event import Event
    from happysimulator.core.simulation import Simulation


@dataclass(frozen=True)
class SimulationState:
    """Immutable snapshot of the simulation's current status.

    Returned by ``SimulationControl.get_state()`` for inspection
    without modifying simulation internals.

    Attributes:
        current_time: The simulation clock's current time.
        events_processed: Total events processed so far.
        heap_size: Total events remaining in the event heap.
        primary_events_remaining: Non-daemon events remaining.
        is_paused: True if execution was explicitly paused.
        is_running: True if the simulation has started and not completed.
        is_complete: True if the simulation has finished.
        last_event: The most recently processed event, or None.
        wall_clock_elapsed: Real-time seconds since the simulation started.
    """
    current_time: Instant
    events_processed: int
    heap_size: int
    primary_events_remaining: int
    is_paused: bool
    is_running: bool
    is_complete: bool
    last_event: Event | None
    wall_clock_elapsed: float


@dataclass(frozen=True)
class BreakpointContext:
    """Context passed to breakpoint predicates for evaluation.

    Provides read-only access to simulation state so breakpoints can
    make decisions based on current time, event count, the last event
    processed, and entity attributes.

    Attributes:
        current_time: Current simulation time.
        events_processed: Total events processed so far.
        last_event: The event that was just processed.
        simulation: Reference to the simulation for entity inspection.
    """
    current_time: Instant
    events_processed: int
    last_event: Event
    simulation: Simulation
