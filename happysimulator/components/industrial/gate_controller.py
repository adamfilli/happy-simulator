"""Gate that opens/closes on schedule or programmatically.

GateController manages access to a downstream entity. When closed,
arrivals are queued. When opened, the queue is flushed. Supports
both schedule-based and programmatic open/close transitions.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)

_GATE_OPEN = "_GateOpen"
_GATE_CLOSE = "_GateClose"


@dataclass(frozen=True)
class GateStats:
    """Snapshot of gate controller statistics."""

    passed_through: int = 0
    queued_while_closed: int = 0
    rejected: int = 0
    open_cycles: int = 0
    is_open: bool = True


class GateController(Entity):
    """Entity that opens/closes on schedule or programmatically.

    When open, events pass through immediately. When closed, events
    are queued. On opening, the queue is flushed to downstream.

    Args:
        name: Identifier for logging.
        downstream: Entity to forward events to.
        schedule: List of ``(open_at_s, close_at_s)`` intervals.
        initially_open: Whether the gate starts open.
        queue_capacity: Maximum queue size (0 = unlimited).
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        schedule: list[tuple[float, float]] | None = None,
        initially_open: bool = True,
        queue_capacity: int = 0,
    ):
        super().__init__(name)
        self.downstream = downstream
        self.schedule = schedule or []
        self._is_open = initially_open
        self._queue_capacity = queue_capacity

        self._queue: deque[Event] = deque()
        self._passed_through = 0
        self._queued_while_closed = 0
        self._rejected = 0
        self._open_cycles = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> GateStats:
        return GateStats(
            passed_through=self._passed_through,
            queued_while_closed=self._queued_while_closed,
            rejected=self._rejected,
            open_cycles=self._open_cycles,
            is_open=self._is_open,
        )

    def start_events(self) -> list[Event]:
        """Create schedule-based open/close events."""
        from happysimulator.core.temporal import Instant

        events: list[Event] = []
        for open_at, close_at in self.schedule:
            events.append(
                Event(
                    time=Instant.from_seconds(open_at),
                    event_type=_GATE_OPEN,
                    target=self,
                    daemon=True,
                )
            )
            events.append(
                Event(
                    time=Instant.from_seconds(close_at),
                    event_type=_GATE_CLOSE,
                    target=self,
                    daemon=True,
                )
            )
        return events

    def open(self) -> list[Event]:
        """Programmatically open the gate and flush queued events."""
        return self._do_open()

    def close(self) -> list[Event]:
        """Programmatically close the gate."""
        return self._do_close()

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type == _GATE_OPEN:
            return self._do_open()
        if event.event_type == _GATE_CLOSE:
            return self._do_close()

        # Regular event
        if self._is_open:
            self._passed_through += 1
            return [
                Event(
                    time=self.now,
                    event_type=event.event_type,
                    target=self.downstream,
                    context=event.context,
                )
            ]

        # Gate is closed — queue or reject
        if self._queue_capacity > 0 and len(self._queue) >= self._queue_capacity:
            self._rejected += 1
            logger.debug(
                "[%s] Rejected (gate closed, queue full: %d/%d)",
                self.name, len(self._queue), self._queue_capacity,
            )
            return []

        self._queue.append(event)
        self._queued_while_closed += 1
        return []

    def _do_open(self) -> list[Event]:
        if self._is_open:
            return []

        self._is_open = True
        self._open_cycles += 1
        logger.debug(
            "[%s] Gate opened (flushing %d queued items)",
            self.name, len(self._queue),
        )

        # Flush queue
        results: list[Event] = []
        while self._queue:
            queued = self._queue.popleft()
            self._passed_through += 1
            results.append(
                Event(
                    time=self.now,
                    event_type=queued.event_type,
                    target=self.downstream,
                    context=queued.context,
                )
            )
        return results

    def _do_close(self) -> list[Event]:
        if not self._is_open:
            return []
        self._is_open = False
        logger.debug("[%s] Gate closed", self.name)
        return []
