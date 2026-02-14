"""Self-scheduling tick event for load generation sources.

Sources use SourceEvents to drive their periodic event generation. Each tick
causes the source to produce payload events and schedule its next tick,
creating a self-perpetuating loop until the simulation ends.
"""

from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.core.entity import Entity


class SourceEvent(Event):
    """Tick event that triggers a Source to generate its next payload.

    When processed, the Source creates payload events via its EventProvider
    and schedules the next SourceEvent based on its ArrivalTimeProvider.
    This bootstrapping pattern keeps the source running without external
    intervention.

    Args:
        time: When this tick should fire.
        source_entity: The Source that will handle this event.
        daemon: If True, this tick won't prevent auto-termination.
    """

    def __init__(self, time: Instant, source_entity: Entity, daemon: bool = False):
        super().__init__(
            time=time,
            event_type="source_event",
            daemon=daemon,
            target=source_entity)