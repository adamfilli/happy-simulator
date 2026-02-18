"""Base class for simulation actors that respond to events.

Entities are the building blocks of a simulation model. Each entity receives
events via handle_event() and returns reactions (new events or generators).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

from happysimulator.core.event import Event

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.core.clock import Clock
    from happysimulator.core.sim_future import SimFuture
    from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)

SimYield = Union[float, tuple[float, list[Event] | Event | None], "SimFuture"]
"""Type alias for generator yield values: delay, (delay, side_effects), or SimFuture."""

SimReturn = list[Event] | Event | None
"""Type alias for generator return values: events to schedule on completion."""


class Entity(ABC):
    """Abstract base class for all simulation actors.

    Entities receive events through handle_event() and produce reactions.
    They maintain a reference to the simulation clock for time-aware logic.

    The simulation injects the clock during initialization, so entities
    should not be used outside a simulation context.

    Subclasses must implement handle_event() to define their behavior.
    Optionally override has_capacity() to model resource constraints.

    Attributes:
        name: Identifier for logging and debugging.
    """

    def __init__(self, name: str):
        self.name = name
        self._clock: Clock | None = None

    def set_clock(self, clock: Clock) -> None:
        """Inject the simulation clock. Called automatically during setup."""
        self._clock = clock
        logger.debug("[%s] Clock injected", self.name)

    @property
    def now(self) -> Instant:
        """Current simulation time from the injected clock.

        Raises:
            RuntimeError: If accessed before clock injection.
        """
        if self._clock is None:
            logger.error("[%s] Attempted to access time before clock injection", self.name)
            raise RuntimeError(
                f"Entity {self.name} is not attached to a simulation (Clock is None)."
            )
        return self._clock.now

    @abstractmethod
    def handle_event(
        self, event: Event
    ) -> Union[Generator[SimYield, None, SimReturn], list[Event], Event, None]:
        """Process an incoming event and return any resulting events.

        Returns:
            Generator: For multi-step processes. Yield delays; optionally return
                events on completion.
            list[Event] | Event | None: For immediate, single-step responses.
        """
        raise NotImplementedError

    def forward(self, event: Event, target: Entity, event_type: str | None = None) -> Event:
        """Create a forwarding event that preserves the original event's context.

        This method only *creates* the event — it does not schedule it.
        The caller must include the returned event in their return value
        (e.g., ``return [self.forward(event, downstream)]``) for the
        simulation loop to schedule it.

        Args:
            event: The original event whose context to preserve.
            target: The downstream entity to receive the new event.
            event_type: Override the event type (default: keep original).

        Returns:
            A new Event at the current time with the same context, targeted
            at the given downstream entity.
        """
        return Event(
            time=self.now,
            event_type=event_type or event.event_type,
            target=target,
            context=event.context,
        )

    def has_capacity(self) -> bool:
        """Check if this entity can accept additional work.

        Override in subclasses with concurrency limits, rate limits, or
        other resource constraints. Returns True by default.
        """
        return True

    def downstream_entities(self) -> list[Entity]:
        """Return downstream entities for topology discovery.

        Override in subclasses that store downstream targets under
        non-standard attribute names. The default implementation
        returns an empty list; topology discovery falls back to
        scanning common attribute names (``downstream``, ``targets``,
        ``_downstream``, etc.) when this returns empty.
        """
        return []
