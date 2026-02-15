"""Appointment-based arrival scheduling.

AppointmentScheduler generates arrivals at fixed appointment times with a
configurable no-show rate. Can be combined with a separate Poisson source
for walk-in traffic.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)

_APPOINTMENT_TICK = "_AppointmentTick"


@dataclass(frozen=True)
class AppointmentStats:
    """Snapshot of appointment scheduler statistics."""

    total_scheduled: int = 0
    arrivals: int = 0
    no_shows: int = 0


class AppointmentScheduler(Entity):
    """Source-like entity that generates arrivals at fixed appointment times.

    Appointments are defined as a list of times (in seconds). At each
    appointment time, the entity generates an arrival event with
    probability ``(1 - no_show_rate)``.

    Args:
        name: Identifier for logging.
        target: Entity to receive appointment arrivals.
        appointments: List of appointment times in seconds.
        no_show_rate: Probability of a no-show (0.0-1.0).
        event_type: Event type string for generated events.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        appointments: list[float],
        no_show_rate: float = 0.0,
        event_type: str = "Appointment",
    ):
        if not (0.0 <= no_show_rate <= 1.0):
            raise ValueError(f"no_show_rate must be in [0.0, 1.0], got {no_show_rate}")
        super().__init__(name)
        self.target = target
        self.appointments = sorted(appointments)
        self.no_show_rate = no_show_rate
        self.event_type = event_type

        self._total_scheduled = len(appointments)
        self._arrivals = 0
        self._no_shows = 0

    @property
    def stats(self) -> AppointmentStats:
        return AppointmentStats(
            total_scheduled=self._total_scheduled,
            arrivals=self._arrivals,
            no_shows=self._no_shows,
        )

    def start_events(self) -> list[Event]:
        """Create events for all appointments.

        Schedule these into the simulation::

            for e in scheduler.start_events():
                sim.schedule(e)
        """
        events: list[Event] = [
            Event(
                time=Instant.from_seconds(t),
                event_type=_APPOINTMENT_TICK,
                target=self,
                context={"appointment_time": t},
            )
            for t in self.appointments
        ]
        return events

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type != _APPOINTMENT_TICK:
            return []

        if random.random() < self.no_show_rate:
            self._no_shows += 1
            logger.debug("[%s] No-show at t=%.2f", self.name, self.now.to_seconds())
            return []

        self._arrivals += 1
        return [
            Event(
                time=self.now,
                event_type=self.event_type,
                target=self.target,
                context={
                    "created_at": self.now,
                    "appointment_time": event.context.get("appointment_time"),
                },
            )
        ]
