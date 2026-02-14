"""Probabilistic pass/fail inspection station.

InspectionStation extends QueuedResource: each item is inspected with a
configurable ``inspection_time``, then routed to ``pass_target`` with
probability ``pass_rate`` or to ``fail_target`` otherwise.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Generator

from happysimulator.components.queued_resource import QueuedResource
from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InspectionStats:
    """Snapshot of inspection station statistics."""

    inspected: int
    passed: int
    failed: int


class InspectionStation(QueuedResource):
    """QueuedResource that inspects items and routes by pass/fail outcome.

    Args:
        name: Identifier for logging.
        pass_target: Entity to receive items that pass inspection.
        fail_target: Entity to receive items that fail inspection.
        inspection_time: Seconds per inspection.
        pass_rate: Probability of passing (0.0-1.0).
        policy: Queue ordering policy (default FIFO).
    """

    def __init__(
        self,
        name: str,
        pass_target: Entity,
        fail_target: Entity,
        inspection_time: float = 0.1,
        pass_rate: float = 0.95,
        policy: QueuePolicy | None = None,
    ):
        super().__init__(name, policy=policy or FIFOQueue())
        self.pass_target = pass_target
        self.fail_target = fail_target
        self.inspection_time = inspection_time
        self.pass_rate = pass_rate
        self._inspected = 0
        self._passed = 0
        self._failed = 0

    @property
    def inspected(self) -> int:
        return self._inspected

    @property
    def passed(self) -> int:
        return self._passed

    @property
    def failed(self) -> int:
        return self._failed

    @property
    def stats(self) -> InspectionStats:
        return InspectionStats(
            inspected=self._inspected,
            passed=self._passed,
            failed=self._failed,
        )

    def handle_queued_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]]:
        yield self.inspection_time

        self._inspected += 1
        if random.random() < self.pass_rate:
            self._passed += 1
            target = self.pass_target
        else:
            self._failed += 1
            target = self.fail_target

        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=target,
                context=event.context,
            )
        ]
