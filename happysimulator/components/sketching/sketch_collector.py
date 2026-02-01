"""Generic sketch collector entity wrapper.

Provides a generic entity wrapper for any sketch algorithm. Events flow
through the collector, values are extracted, and the sketch is updated.
"""

from __future__ import annotations

from typing import Callable, Generic, TypeVar

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.sketching.base import Sketch

S = TypeVar('S', bound=Sketch)
T = TypeVar('T')


class SketchCollector(Entity, Generic[S, T]):
    """Generic entity wrapper for any sketch algorithm.

    Extracts values from events and updates the underlying sketch.
    Can optionally extract weights for weighted updates.

    Args:
        name: Entity name for logging.
        sketch: The sketch instance to update.
        value_extractor: Function to extract value from event.
        weight_extractor: Optional function to extract weight from event.
            If provided, add(value, count=weight) is called.

    Example:
        # Track customer frequencies with Count-Min Sketch
        cms = CountMinSketch.from_error_rate(epsilon=0.01, delta=0.01)
        collector = SketchCollector(
            name="customer_freq",
            sketch=cms,
            value_extractor=lambda e: e.context.get("customer_id"),
        )

        # In simulation, route events to collector
        router = RandomRouter([collector])
    """

    def __init__(
        self,
        name: str,
        sketch: S,
        value_extractor: Callable[[Event], T],
        weight_extractor: Callable[[Event], int] | None = None,
    ):
        """Initialize SketchCollector.

        Args:
            name: Entity name.
            sketch: Sketch instance to update.
            value_extractor: Function to extract value from Event.
            weight_extractor: Optional function to extract weight from Event.
        """
        super().__init__(name)
        self._sketch = sketch
        self._value_extractor = value_extractor
        self._weight_extractor = weight_extractor
        self._events_processed = 0

    @property
    def sketch(self) -> S:
        """Access the underlying sketch."""
        return self._sketch

    @property
    def events_processed(self) -> int:
        """Number of events processed."""
        return self._events_processed

    def handle_event(self, event: Event) -> list[Event]:
        """Process an event by updating the sketch.

        Args:
            event: Incoming event.

        Returns:
            Empty list (sketch collector is a sink).
        """
        value = self._value_extractor(event)

        if value is not None:
            if self._weight_extractor is not None:
                weight = self._weight_extractor(event)
                self._sketch.add(value, count=weight)
            else:
                self._sketch.add(value)

        self._events_processed += 1
        return []

    def clear(self) -> None:
        """Clear the sketch and reset counters."""
        self._sketch.clear()
        self._events_processed = 0
