"""Random machine breakdowns for target entities.

BreakdownScheduler alternates a target entity between UP and DOWN states.
Time-to-failure and repair time are drawn from configurable distributions.
During DOWN state, a ``_broken`` flag is set on the target so that
``has_capacity()`` checks can respect it.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)

_BREAKDOWN = "_Breakdown"
_REPAIR_COMPLETE = "_RepairComplete"


@dataclass(frozen=True)
class BreakdownStats:
    """Snapshot of breakdown statistics."""

    breakdown_count: int = 0
    total_downtime_s: float = 0.0
    total_uptime_s: float = 0.0

    @property
    def availability(self) -> float:
        """Fraction of time the machine was operational (0.0-1.0)."""
        total = self.total_uptime_s + self.total_downtime_s
        if total == 0:
            return 1.0
        return self.total_uptime_s / total


@runtime_checkable
class Breakable(Protocol):
    """Protocol for entities that can be broken down."""

    _broken: bool


class BreakdownScheduler(Entity):
    """Entity that schedules random breakdowns for a target.

    Alternates between UP (exponential time-to-failure) and DOWN (repair
    time) states. During DOWN, sets ``target._broken = True`` so that
    ``has_capacity()`` implementations can check it.

    The scheduler must be registered as an entity in the simulation and
    needs to be started by scheduling an initial event to it (or by calling
    ``start_event()``).

    Args:
        name: Identifier for logging.
        target: The entity subject to breakdowns.
        mean_time_to_failure: Mean time between breakdowns (seconds).
        mean_repair_time: Mean time to repair (seconds).
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        mean_time_to_failure: float = 100.0,
        mean_repair_time: float = 5.0,
    ):
        super().__init__(name)
        if not hasattr(target, "_broken"):
            target._broken = False  # type: ignore[attr-defined]
        self.target: Breakable = target  # type: ignore[assignment]
        self.mean_time_to_failure = mean_time_to_failure
        self.mean_repair_time = mean_repair_time

        self._breakdown_count = 0
        self._total_downtime_s = 0.0
        self._total_uptime_s = 0.0
        self._last_state_change_s = 0.0
        self._is_down = False

    @property
    def is_down(self) -> bool:
        return self._is_down

    @property
    def stats(self) -> BreakdownStats:
        return BreakdownStats(
            breakdown_count=self._breakdown_count,
            total_downtime_s=self._total_downtime_s,
            total_uptime_s=self._total_uptime_s,
        )

    def start_event(self) -> Event:
        """Create the initial event that starts the breakdown cycle.

        Schedule this event in the simulation to activate breakdowns::

            sim.schedule(breakdown_scheduler.start_event())
        """
        from happysimulator.core.temporal import Instant

        ttf = random.expovariate(1.0 / self.mean_time_to_failure)
        return Event(
            time=Instant.from_seconds(ttf),
            event_type=_BREAKDOWN,
            target=self,
            daemon=True,
        )

    def handle_event(self, event: Event) -> list[Event]:
        from happysimulator.core.temporal import Instant

        now_s = self.now.to_seconds()

        if event.event_type == _BREAKDOWN:
            # Record uptime
            self._total_uptime_s += now_s - self._last_state_change_s
            self._last_state_change_s = now_s

            # Break the target
            self._is_down = True
            self.target._broken = True
            self._breakdown_count += 1

            # Schedule repair
            repair_time = random.expovariate(1.0 / self.mean_repair_time)
            logger.debug(
                "[%s] Breakdown #%d at t=%.2f, repair in %.2fs",
                self.name,
                self._breakdown_count,
                now_s,
                repair_time,
            )

            return [
                Event(
                    time=Instant.from_seconds(now_s + repair_time),
                    event_type=_REPAIR_COMPLETE,
                    target=self,
                    daemon=True,
                )
            ]

        if event.event_type == _REPAIR_COMPLETE:
            # Record downtime
            self._total_downtime_s += now_s - self._last_state_change_s
            self._last_state_change_s = now_s

            # Restore the target
            self._is_down = False
            self.target._broken = False

            # Schedule next failure
            ttf = random.expovariate(1.0 / self.mean_time_to_failure)
            logger.debug(
                "[%s] Repair complete at t=%.2f, next failure in %.2fs",
                self.name,
                now_s,
                ttf,
            )

            return [
                Event(
                    time=Instant.from_seconds(now_s + ttf),
                    event_type=_BREAKDOWN,
                    target=self,
                    daemon=True,
                )
            ]

        return []
