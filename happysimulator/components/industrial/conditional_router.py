"""Declarative event routing based on context predicates.

ConditionalRouter evaluates an ordered list of ``(predicate, target)``
routes against incoming events. The first predicate that returns True
wins, and the event is forwarded to the corresponding target. If no
predicate matches, the event goes to ``default`` (or is dropped).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouterStats:
    """Snapshot of router statistics."""

    total_routed: int = 0
    dropped: int = 0
    by_target: dict[str, int] = field(default_factory=dict)


class ConditionalRouter(Entity):
    """Entity that routes events based on ordered predicate matching.

    Each incoming event is tested against ``routes`` in order. The first
    ``(predicate, target)`` pair whose predicate returns True forwards
    the event to that target. If no predicate matches, the event goes
    to ``default`` (if set) or is dropped.

    Args:
        name: Identifier for logging.
        routes: Ordered list of ``(predicate, target)`` pairs.
        default: Fallback target when no predicate matches.
        drop_unmatched: If True (and no default), silently drop unmatched
            events. If False, log a warning on drop.
    """

    def __init__(
        self,
        name: str,
        routes: list[tuple[Callable[[Event], bool], Entity]],
        default: Entity | None = None,
        drop_unmatched: bool = False,
    ):
        super().__init__(name)
        self.routes = routes
        self.default = default
        self.drop_unmatched = drop_unmatched

        self._routed_counts: dict[str, int] = defaultdict(int)
        self._total_routed = 0
        self._dropped = 0

    @property
    def routed_counts(self) -> dict[str, int]:
        return dict(self._routed_counts)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def total_routed(self) -> int:
        return self._total_routed

    @property
    def stats(self) -> RouterStats:
        return RouterStats(
            total_routed=self._total_routed,
            dropped=self._dropped,
            by_target=dict(self._routed_counts),
        )

    @classmethod
    def by_context_field(
        cls,
        name: str,
        field: str,
        mapping: dict[str, Entity],
        default: Entity | None = None,
    ) -> ConditionalRouter:
        """Create a router that dispatches based on a context field value.

        Args:
            name: Identifier for logging.
            field: The context key to inspect.
            mapping: Dict of ``{field_value: target_entity}``.
            default: Fallback target for unmapped values.
        """
        routes: list[tuple[Callable[[Event], bool], Entity]] = []
        for value, target in mapping.items():
            routes.append((lambda e, v=value, f=field: e.context.get(f) == v, target))
        return cls(name, routes=routes, default=default)

    def handle_event(self, event: Event) -> list[Event]:
        for predicate, target in self.routes:
            if predicate(event):
                self._total_routed += 1
                self._routed_counts[target.name] += 1
                return [
                    Event(
                        time=self.now,
                        event_type=event.event_type,
                        target=target,
                        context=event.context,
                    )
                ]

        # No predicate matched
        if self.default is not None:
            self._total_routed += 1
            self._routed_counts[self.default.name] += 1
            return [
                Event(
                    time=self.now,
                    event_type=event.event_type,
                    target=self.default,
                    context=event.context,
                )
            ]

        self._dropped += 1
        if not self.drop_unmatched:
            logger.warning(
                "[%s] No matching route for event type=%s, dropped",
                self.name,
                event.event_type,
            )
        return []
