"""QueuedResource subclass where items expire based on patience.

RenegingQueuedResource checks dequeued items against their patience
time. If an item has waited longer than its patience, it is routed to
``reneged_target`` instead of being processed.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generator

from happysimulator.components.queued_resource import QueuedResource
from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenegingStats:
    """Snapshot of reneging statistics."""

    served: int
    reneged: int


class RenegingQueuedResource(QueuedResource):
    """Abstract QueuedResource where items can renege (leave) if they wait too long.

    When an item is dequeued, the resource checks whether
    ``(now - created_at) > patience``. If so, the item is routed to
    ``reneged_target`` instead of being processed.

    Patience is determined from ``event.context["patience_s"]`` if
    present, otherwise ``default_patience_s`` is used.

    Subclasses implement ``_handle_served_event()`` for items that
    are still within their patience window.

    Args:
        name: Identifier for logging.
        reneged_target: Entity to receive reneged items (or None to discard).
        default_patience_s: Default patience in seconds.
        policy: Queue ordering policy (default FIFO).
    """

    def __init__(
        self,
        name: str,
        reneged_target: Entity | None = None,
        default_patience_s: float = float("inf"),
        policy: QueuePolicy | None = None,
    ):
        super().__init__(name, policy=policy or FIFOQueue())
        self.reneged_target = reneged_target
        self.default_patience_s = default_patience_s
        self._served = 0
        self._reneged = 0

    @property
    def served(self) -> int:
        return self._served

    @property
    def reneged(self) -> int:
        return self._reneged

    @property
    def reneging_stats(self) -> RenegingStats:
        return RenegingStats(served=self._served, reneged=self._reneged)

    def handle_queued_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        created_at = event.context.get("created_at", self.now)
        patience_s = event.context.get("patience_s", self.default_patience_s)

        wait_time = (self.now - created_at).to_seconds()

        if wait_time > patience_s:
            self._reneged += 1
            logger.debug(
                "[%s] Item reneged (waited %.3fs > patience %.3fs)",
                self.name, wait_time, patience_s,
            )
            if self.reneged_target is not None:
                return [
                    Event(
                        time=self.now,
                        event_type="Reneged",
                        target=self.reneged_target,
                        context=event.context,
                    )
                ]
            return []

        self._served += 1
        return self._handle_served_event(event)

    @abstractmethod
    def _handle_served_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle an item that is within its patience window.

        Subclasses implement their processing logic here.
        """
        raise NotImplementedError
