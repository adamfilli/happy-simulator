"""Self-perpetuating event generator for load generation.

A Source periodically generates payload events (e.g., requests) using an
EventProvider. The timing between events is determined by an ArrivalTimeProvider,
which can be constant (deterministic) or follow a distribution (e.g., Poisson).

Sources bootstrap themselves by scheduling a SourceEvent, which triggers
payload generation and schedules the next SourceEvent.
"""

import logging
from typing import List

from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from happysimulator.events.source_event import SourceEvent
from happysimulator.load.arrival_time_provider import ArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.utils.instant import Instant


logger = logging.getLogger(__name__)


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
        # Delegate to the EventProvider (e.g., create an HttpRequest)
        payload_events = self._event_provider.get_events(current_time)
        self._nmb_generated += 1
        
        # --- B. Schedule Next Tick (Self-Perpetuation) ---
        try:
            next_time = self._time_provider.next_arrival_time()     
            next_tick = SourceEvent(time=next_time, source_entity=self)
            
            # Return both the payload AND the next tick to be pushed to the heap
            return payload_events + [next_tick]
            
        except RuntimeError:
            # Rate has dropped to zero forever (or profile ended)
            logger.info(f"[{self.name}] Source exhausted. Stopping.")
            return payload_events
            
    def __repr__(self):
        return f"<Source {self.name}>"