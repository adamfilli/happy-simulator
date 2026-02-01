"""TopK collector entity for heavy hitter tracking.

Specialized entity wrapper for TopK (Space-Saving) algorithm that provides
convenient access to top-k queries and statistics.
"""

from __future__ import annotations

from typing import Callable, TypeVar, Hashable

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.sketching.topk import TopK
from happysimulator.sketching.base import FrequencyEstimate

T = TypeVar('T', bound=Hashable)


class TopKCollector(Entity):
    """Entity wrapper for TopK heavy hitter tracking.

    Tracks the most frequent values extracted from events.
    Provides convenient query methods for top-k analysis.

    Args:
        name: Entity name for logging.
        k: Number of items to track.
        value_extractor: Function to extract trackable value from event.
        count_extractor: Optional function to extract count from event.
            Defaults to 1 per event.
        seed: Random seed for reproducibility.

    Example:
        # Track top 100 customers
        collector = TopKCollector(
            name="top_customers",
            k=100,
            value_extractor=lambda e: e.context.get("customer_id"),
        )

        # After simulation
        for estimate in collector.top(10):
            print(f"Customer {estimate.item}: ~{estimate.count} requests")
    """

    def __init__(
        self,
        name: str,
        k: int,
        value_extractor: Callable[[Event], T],
        count_extractor: Callable[[Event], int] | None = None,
        seed: int | None = None,
    ):
        """Initialize TopKCollector.

        Args:
            name: Entity name.
            k: Number of items to track.
            value_extractor: Function to extract value from Event.
            count_extractor: Optional function to extract count from Event.
            seed: Random seed (passed to TopK for API consistency).
        """
        super().__init__(name)
        self._topk: TopK[T] = TopK(k=k, seed=seed)
        self._value_extractor = value_extractor
        self._count_extractor = count_extractor
        self._events_processed = 0

    @property
    def k(self) -> int:
        """Number of items being tracked."""
        return self._topk.k

    @property
    def events_processed(self) -> int:
        """Number of events processed."""
        return self._events_processed

    @property
    def total_count(self) -> int:
        """Total count of all items added."""
        return self._topk.item_count

    @property
    def tracked_count(self) -> int:
        """Number of distinct items currently tracked."""
        return self._topk.tracked_count

    def handle_event(self, event: Event) -> list[Event]:
        """Process an event by updating the TopK sketch.

        Args:
            event: Incoming event.

        Returns:
            Empty list (collector is a sink).
        """
        value = self._value_extractor(event)

        if value is not None:
            if self._count_extractor is not None:
                count = self._count_extractor(event)
                self._topk.add(value, count=count)
            else:
                self._topk.add(value)

        self._events_processed += 1
        return []

    def top(self, n: int | None = None) -> list[FrequencyEstimate]:
        """Get the top-n most frequent items.

        Args:
            n: Number of items to return. If None, returns all tracked items.

        Returns:
            List of FrequencyEstimate objects sorted by count (descending).
        """
        return self._topk.top(n)

    def estimate(self, item: T) -> int:
        """Estimate the frequency of a specific item.

        Args:
            item: Item to estimate.

        Returns:
            Estimated count for the item.
        """
        return self._topk.estimate(item)

    def __contains__(self, item: T) -> bool:
        """Check if an item is currently being tracked."""
        return item in self._topk

    def max_error(self) -> int:
        """Maximum possible error for any estimate."""
        return self._topk.max_error()

    def guaranteed_threshold(self) -> int:
        """Frequency threshold above which items are guaranteed tracked."""
        return self._topk.guaranteed_threshold()

    def clear(self) -> None:
        """Clear the sketch and reset counters."""
        self._topk.clear()
        self._events_processed = 0
