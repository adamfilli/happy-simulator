"""Fan-out / fan-in pattern for parallel sub-task processing.

SplitMerge fans out one event to N parallel targets, waits for all
sub-tasks to complete via ``all_of``, then forwards the merged result
downstream. Targets must resolve ``event.context["reply_future"]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture, all_of

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitMergeStats:
    """Snapshot of split-merge statistics."""

    splits_initiated: int
    merges_completed: int
    fan_out: int


class SplitMerge(Entity):
    """Entity that fans out events to parallel targets and merges results.

    For each incoming event, creates N sub-events (one per target) with
    a ``reply_future`` in context. Uses ``all_of`` to wait for all
    targets to resolve their futures, then forwards the merged result
    downstream with ``context["sub_results"]``.

    Targets must call ``event.context["reply_future"].resolve(value)``
    when their work is complete.

    Args:
        name: Identifier for logging.
        targets: List of entities to fan out to.
        downstream: Entity to forward merged results to.
        split_event_type: Event type for sub-task events.
        merge_event_type: Event type for the merged result event.
    """

    def __init__(
        self,
        name: str,
        targets: list[Entity],
        downstream: Entity,
        split_event_type: str = "SubTask",
        merge_event_type: str = "Merged",
    ):
        super().__init__(name)
        self.targets = targets
        self.downstream = downstream
        self.split_event_type = split_event_type
        self.merge_event_type = merge_event_type

        self._splits_initiated = 0
        self._merges_completed = 0

    @property
    def fan_out(self) -> int:
        return len(self.targets)

    @property
    def stats(self) -> SplitMergeStats:
        return SplitMergeStats(
            splits_initiated=self._splits_initiated,
            merges_completed=self._merges_completed,
            fan_out=len(self.targets),
        )

    def handle_event(
        self, event: Event
    ) -> Generator[float | SimFuture, None, list[Event]]:
        self._splits_initiated += 1

        # Create futures and sub-events
        futures: list[SimFuture] = []
        sub_events: list[Event] = []
        for target in self.targets:
            future = SimFuture()
            futures.append(future)
            sub_events.append(
                Event(
                    time=self.now,
                    event_type=self.split_event_type,
                    target=target,
                    context={
                        **event.context,
                        "reply_future": future,
                    },
                )
            )

        # Schedule all sub-events immediately
        yield 0.0, sub_events

        # Wait for all sub-tasks to complete
        results = yield all_of(*futures)

        self._merges_completed += 1

        return [
            Event(
                time=self.now,
                event_type=self.merge_event_type,
                target=self.downstream,
                context={
                    **event.context,
                    "sub_results": results,
                },
            )
        ]
