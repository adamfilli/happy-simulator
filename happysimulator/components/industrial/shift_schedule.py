"""Shift-based capacity scheduling for server entities.

ShiftSchedule defines time-varying capacity via a list of Shift periods.
ShiftedServer extends QueuedResource, adjusting its concurrency at shift
boundaries by scheduling self-check events.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.core.entity import Entity

logger = logging.getLogger(__name__)

_SHIFT_CHANGE = "_ShiftChange"


@dataclass(frozen=True)
class Shift:
    """A single shift defining capacity over a time window.

    Args:
        start_s: Start time in seconds.
        end_s: End time in seconds.
        capacity: Number of concurrent workers during this shift.
    """

    start_s: float
    end_s: float
    capacity: int


class ShiftSchedule:
    """Collection of shifts that defines time-varying capacity.

    Shifts should be non-overlapping and sorted by start time. Gaps
    between shifts default to ``default_capacity``.

    Args:
        shifts: List of Shift definitions.
        default_capacity: Capacity outside of any defined shift.
    """

    def __init__(
        self,
        shifts: list[Shift],
        default_capacity: int = 0,
    ):
        self.shifts = sorted(shifts, key=lambda s: s.start_s)
        self.default_capacity = default_capacity

    def capacity_at(self, time_s: float) -> int:
        """Return the capacity at the given time."""
        for shift in self.shifts:
            if shift.start_s <= time_s < shift.end_s:
                return shift.capacity
        return self.default_capacity

    def next_transition_after(self, time_s: float) -> float | None:
        """Return the next transition time strictly after ``time_s``, or None."""
        times = set()
        for shift in self.shifts:
            times.add(shift.start_s)
            times.add(shift.end_s)
        future = sorted(t for t in times if t > time_s)
        return future[0] if future else None

    def transition_times(self) -> list[float]:
        """Return all shift boundary times (sorted, deduplicated)."""
        times: set[float] = set()
        for shift in self.shifts:
            times.add(shift.start_s)
            times.add(shift.end_s)
        return sorted(times)


class ShiftedServer(QueuedResource):
    """QueuedResource whose concurrency varies according to a ShiftSchedule.

    At each shift boundary, the server adjusts its concurrency to match
    the schedule. Uses a self-perpetuating pattern: each shift change
    event schedules the next one.

    Args:
        name: Identifier for logging.
        schedule: ShiftSchedule defining capacity over time.
        service_time: Seconds per item processed.
        downstream: Entity to forward processed items to (or None).
        policy: Queue ordering policy (default FIFO).
    """

    def __init__(
        self,
        name: str,
        schedule: ShiftSchedule,
        service_time: float = 0.1,
        downstream: Entity | None = None,
        policy: QueuePolicy | None = None,
    ):
        super().__init__(name, policy=policy or FIFOQueue())
        self.schedule = schedule
        self.service_time = service_time
        self.downstream = downstream
        self._current_capacity = schedule.capacity_at(0.0)
        self._active = 0
        self._processed = 0
        self._initialized = False

    @property
    def current_capacity(self) -> int:
        return self._current_capacity

    @property
    def processed(self) -> int:
        return self._processed

    def has_capacity(self) -> bool:
        return self._active < self._current_capacity

    def handle_event(self, event: Event):
        if event.event_type == _SHIFT_CHANGE:
            return self._handle_shift_change()

        # On first real event, schedule the first shift change
        if not self._initialized:
            self._initialized = True
            next_event = self._schedule_next_shift()
            result = super().handle_event(event)
            if next_event and isinstance(result, list):
                return [*result, next_event]
            return result

        return super().handle_event(event)

    def _handle_shift_change(self) -> list[Event]:
        time_s = self.now.to_seconds()
        new_capacity = self.schedule.capacity_at(time_s)
        old_capacity = self._current_capacity
        self._current_capacity = new_capacity

        logger.debug(
            "[%s] Shift change at t=%.1f: capacity %d -> %d",
            self.name,
            time_s,
            old_capacity,
            new_capacity,
        )

        # Schedule the next shift change (self-perpetuating)
        next_event = self._schedule_next_shift()
        return [next_event] if next_event else []

    def _schedule_next_shift(self) -> Event | None:
        """Schedule only the next transition event."""
        from happysimulator.core.temporal import Instant

        current_s = self.now.to_seconds()
        next_t = self.schedule.next_transition_after(current_s)
        if next_t is None:
            return None

        return Event(
            time=Instant.from_seconds(next_t),
            event_type=_SHIFT_CHANGE,
            target=self,
            daemon=True,
        )

    def handle_queued_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None]:
        self._active += 1
        try:
            yield self.service_time
        finally:
            self._active -= 1

        self._processed += 1

        if self.downstream is not None:
            return [
                Event(
                    time=self.now,
                    event_type=event.event_type,
                    target=self.downstream,
                    context=event.context,
                )
            ]
        return None
