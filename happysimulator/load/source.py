"""Self-perpetuating event generator for load generation.

A Source periodically generates payload events (e.g., requests) using an
EventProvider. The timing between events is determined by an ArrivalTimeProvider,
which can be constant (deterministic) or follow a distribution (e.g., Poisson).

Sources bootstrap themselves by scheduling a SourceEvent, which triggers
payload generation and schedules the next SourceEvent.
"""

from __future__ import annotations

import logging
from typing import List

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.load.source_event import SourceEvent
from happysimulator.load.arrival_time_provider import ArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import ConstantRateProfile, Profile


logger = logging.getLogger(__name__)


class _SimpleEventProvider(EventProvider):
    """Internal event provider that creates targeted events with metadata.

    Generates events with auto-incrementing request IDs and ``created_at``
    timestamps in context. Used by Source factory methods to eliminate the
    need for custom EventProvider subclasses in common cases.
    """

    def __init__(
        self,
        target: Entity,
        event_type: str = "Request",
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._event_type = event_type
        self._stop_after = stop_after
        self._generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self._generated += 1
        return [
            Event(
                time=time,
                event_type=self._event_type,
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self._generated,
                },
            )
        ]


class Source(Entity):
    """Self-scheduling entity that generates load events at specified intervals.

    Combines an EventProvider (what to generate) with an ArrivalTimeProvider
    (when to generate) to produce a stream of events. The source maintains
    its own schedule by creating SourceEvents that trigger the next generation.

    Attributes:
        name: Identifier for logging.

    Args:
        name: Source identifier.
        event_provider: Creates the payload events.
        arrival_time_provider: Determines timing between events.
    """

    def __init__(
        self,
        name: str,
        event_provider: EventProvider,
        arrival_time_provider: ArrivalTimeProvider,
    ):
        super().__init__(name)
        self._event_provider = event_provider
        self._time_provider = arrival_time_provider
        self._nmb_generated = 0

    def start(self, start_time: Instant) -> List[Event]:
        """Bootstrap the source by scheduling its first tick.

        Called by Simulation during initialization. Synchronizes the
        arrival time provider to the simulation start time.
        """
        # Sync the provider to the simulation start time
        self._time_provider.current_time = start_time
        
        try:
            # Calculate when the first event should happen
            first_time = self._time_provider.next_arrival_time()
            
            logger.info(f"[{self.name}] Source starting. First event at {first_time}")
            
            # Return the first 'Tick'
            return [SourceEvent(time=first_time, source_entity=self)]
            
        except RuntimeError:
            logger.warning(f"[{self.name}] Rate is zero indefinitely. Source will not start.")
            return []

    def handle_event(self, event: Event) -> List[Event]:
        """Generate payload events and schedule the next tick.

        This implements the source's self-perpetuating loop:
        1. Create payload events via the EventProvider
        2. Calculate next arrival time
        3. Schedule the next SourceEvent
        4. Return both payload and next tick for scheduling
        """
        if not isinstance(event, SourceEvent):
            # If for some reason a Source receives a non-generate event, ignore it
            return []

        current_time = event.time

        # --- A. Generate Payload (The "Real" Events) ---
        payload_events = self._event_provider.get_events(current_time)
        self._nmb_generated += 1

        logger.debug(
            "[%s] Generated %d payload event(s) (#%d total)",
            self.name, len(payload_events), self._nmb_generated
        )

        # --- B. Schedule Next Tick (Self-Perpetuation) ---
        try:
            next_time = self._time_provider.next_arrival_time()
            next_tick = SourceEvent(time=next_time, source_entity=self)

            logger.debug("[%s] Next tick scheduled for %r", self.name, next_time)
            return payload_events + [next_tick]

        except RuntimeError:
            logger.info("[%s] Source exhausted after %d events. Stopping.", self.name, self._nmb_generated)
            return payload_events
            
    @classmethod
    def constant(
        cls,
        rate: float,
        target: Entity,
        event_type: str = "Request",
        *,
        name: str = "Source",
        stop_after: float | Instant | None = None,
    ) -> Source:
        """Create a Source with constant (deterministic) arrival rate.

        Args:
            rate: Events per second.
            target: Entity to receive generated events.
            event_type: Type string for generated events.
            name: Source identifier for logging.
            stop_after: Stop generating events after this time.
                        Accepts seconds (float) or Instant.

        Returns:
            A fully-wired Source ready for use in a Simulation.
        """
        from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider

        stop_instant = cls._resolve_stop_after(stop_after)
        return cls(
            name=name,
            event_provider=_SimpleEventProvider(target, event_type, stop_instant),
            arrival_time_provider=ConstantArrivalTimeProvider(
                ConstantRateProfile(rate=rate),
                start_time=Instant.Epoch,
            ),
        )

    @classmethod
    def poisson(
        cls,
        rate: float,
        target: Entity,
        event_type: str = "Request",
        *,
        name: str = "Source",
        stop_after: float | Instant | None = None,
    ) -> Source:
        """Create a Source with Poisson (stochastic) arrival rate.

        Args:
            rate: Mean events per second.
            target: Entity to receive generated events.
            event_type: Type string for generated events.
            name: Source identifier for logging.
            stop_after: Stop generating events after this time.
                        Accepts seconds (float) or Instant.

        Returns:
            A fully-wired Source ready for use in a Simulation.
        """
        from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider

        stop_instant = cls._resolve_stop_after(stop_after)
        return cls(
            name=name,
            event_provider=_SimpleEventProvider(target, event_type, stop_instant),
            arrival_time_provider=PoissonArrivalTimeProvider(
                ConstantRateProfile(rate=rate),
                start_time=Instant.Epoch,
            ),
        )

    @classmethod
    def with_profile(
        cls,
        profile: Profile,
        target: Entity,
        event_type: str = "Request",
        *,
        poisson: bool = True,
        name: str = "Source",
        stop_after: float | Instant | None = None,
    ) -> Source:
        """Create a Source with a custom rate profile.

        Args:
            profile: Rate profile defining how arrival rate varies over time.
            target: Entity to receive generated events.
            event_type: Type string for generated events.
            poisson: If True (default), use stochastic Poisson arrivals.
                     If False, use deterministic constant arrivals.
            name: Source identifier for logging.
            stop_after: Stop generating events after this time.
                        Accepts seconds (float) or Instant.

        Returns:
            A fully-wired Source ready for use in a Simulation.
        """
        from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
        from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider

        stop_instant = cls._resolve_stop_after(stop_after)
        provider_cls = PoissonArrivalTimeProvider if poisson else ConstantArrivalTimeProvider
        return cls(
            name=name,
            event_provider=_SimpleEventProvider(target, event_type, stop_instant),
            arrival_time_provider=provider_cls(
                profile,
                start_time=Instant.Epoch,
            ),
        )

    @staticmethod
    def _resolve_stop_after(stop_after: float | Instant | None) -> Instant | None:
        """Convert a stop_after value to an Instant."""
        if stop_after is None:
            return None
        if isinstance(stop_after, Instant):
            return stop_after
        return Instant.from_seconds(stop_after)

    def __repr__(self):
        return f"<Source {self.name}>"