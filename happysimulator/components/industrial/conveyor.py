"""Fixed transit-time transport between stations.

ConveyorBelt models a simple delay element: events arrive, spend a fixed
``transit_time`` in transit, then are forwarded to the downstream entity.
An optional capacity limit models physical conveyor length.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConveyorStats:
    """Snapshot of conveyor belt statistics."""

    items_transported: int = 0
    items_in_transit: int = 0
    items_rejected: int = 0


class ConveyorBelt(Entity):
    """Entity that models fixed transit time between stations.

    Receives events, holds them for ``transit_time`` seconds, then
    forwards to ``downstream``. An optional ``capacity`` limits
    how many items can be in transit simultaneously.

    Args:
        name: Identifier for logging.
        downstream: Entity to forward events to after transit.
        transit_time: Seconds each item spends on the conveyor.
        capacity: Maximum items in transit at once (default unlimited).
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        transit_time: float,
        capacity: int = 0,
    ):
        if transit_time < 0:
            raise ValueError(f"transit_time must be >= 0, got {transit_time}")
        super().__init__(name)
        self.downstream = downstream
        self.transit_time = transit_time
        self._capacity = capacity  # 0 means unlimited
        self._items_in_transit = 0
        self._items_transported = 0
        self._items_rejected = 0

    @property
    def items_in_transit(self) -> int:
        return self._items_in_transit

    @property
    def items_transported(self) -> int:
        return self._items_transported

    @property
    def items_rejected(self) -> int:
        return self._items_rejected

    @property
    def stats(self) -> ConveyorStats:
        return ConveyorStats(
            items_transported=self._items_transported,
            items_in_transit=self._items_in_transit,
            items_rejected=self._items_rejected,
        )

    def has_capacity(self) -> bool:
        if self._capacity <= 0:
            return True
        return self._items_in_transit < self._capacity

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]] | list[Event]:
        if self._capacity > 0 and self._items_in_transit >= self._capacity:
            self._items_rejected += 1
            logger.debug(
                "[%s] Rejected item (at capacity %d)", self.name, self._capacity
            )
            return []

        self._items_in_transit += 1
        return self._transport(event)

    def _transport(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.transit_time

        self._items_in_transit -= 1
        self._items_transported += 1

        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.downstream,
                context=event.context,
            )
        ]
