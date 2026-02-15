"""Simulation clock for tracking current simulation time.

The Clock provides a shared time reference for all entities in a simulation.
The Simulation advances the clock as it processes events. Entities access
the current time via their injected clock reference.
"""

from happysimulator.core.temporal import Instant


class Clock:
    """Tracks the current simulation time.

    The simulation creates one Clock instance and shares it with all entities.
    As events are processed, the simulation calls update() to advance time.
    Entities query now to get the current time for their logic.

    This indirection allows entities to access simulation time without direct
    coupling to the Simulation class.

    Attributes:
        now: The current simulation time.
    """

    def __init__(self, start_time: Instant):
        """Initialize clock at the given start time."""
        self._current_time = start_time

    @property
    def now(self) -> Instant:
        """The current simulation time."""
        return self._current_time

    def update(self, time: Instant) -> None:
        """Advance the clock to a new time. Called by the simulation loop."""
        self._current_time = time
