"""Top-K heavy hitters detection using the Space-Saving algorithm.

The Space-Saving algorithm efficiently finds the most frequent items in a data
stream using bounded memory. It maintains a fixed number of counters (k) and
guarantees that:
- Any item with true frequency > N/k will be tracked
- Frequency estimates have error bounded by N/k where N is total count

This is ideal for:
- Finding hot keys in caches
- Identifying top customers by request volume
- Detecting popular API endpoints
- Any scenario where a small set of items dominates traffic

Reference:
    Metwally, Agrawal, El Abbadi. "Efficient Computation of Frequent and Top-k
    Elements in Data Streams" (2005)
"""

from __future__ import annotations

import sys
from collections.abc import Hashable
from dataclasses import dataclass
from typing import TypeVar

from happysimulator.sketching.base import FrequencyEstimate, FrequencySketch

T = TypeVar("T", bound=Hashable)


@dataclass(slots=True)
class _Counter[T: Hashable]:
    """Internal counter for Space-Saving algorithm.

    Tracks an item, its count, and the error (overestimation) introduced
    when the counter was reassigned from another item.
    """

    item: T
    count: int
    error: int  # Overestimation bound from counter reuse


class TopK(FrequencySketch[T]):
    """Top-K heavy hitters using the Space-Saving algorithm.

    Maintains k counters to track the approximately k most frequent items.
    When a new item arrives that isn't tracked, it replaces the item with
    the minimum count, inheriting that count as an error bound.

    Args:
        k: Number of items to track. Larger k = more accurate but more memory.
        seed: Not used (algorithm is deterministic) but accepted for API consistency.

    Example:
        # Track top 100 customers by request count
        topk = TopK[int](k=100)

        for customer_id in request_stream:
            topk.add(customer_id)

        # Get the 10 most frequent customers
        for estimate in topk.top(10):
            print(f"Customer {estimate.item}: ~{estimate.count} requests "
                  f"(error <= {estimate.error})")
    """

    def __init__(self, k: int, seed: int | None = None):
        """Initialize TopK sketch.

        Args:
            k: Number of items to track. Must be positive.
            seed: Unused (algorithm is deterministic). Accepted for API consistency.

        Raises:
            ValueError: If k <= 0.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        self._k = k
        self._counters: dict[T, _Counter[T]] = {}
        self._total_count = 0

    @property
    def k(self) -> int:
        """Number of items being tracked."""
        return self._k

    def add(self, item: T, count: int = 1) -> None:
        """Add an item occurrence to the sketch.

        Args:
            item: The item to add. Must be hashable.
            count: Number of occurrences to add (default 1).

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")
        if count == 0:
            return

        self._total_count += count

        if item in self._counters:
            # Item already tracked - increment its count
            self._counters[item].count += count
        elif len(self._counters) < self._k:
            # Space available - add new counter
            self._counters[item] = _Counter(item=item, count=count, error=0)
        else:
            # No space - replace minimum counter
            min_counter = min(self._counters.values(), key=lambda c: c.count)
            min_count = min_counter.count

            # Remove old item
            del self._counters[min_counter.item]

            # Add new item, inheriting min_count as error
            self._counters[item] = _Counter(
                item=item,
                count=min_count + count,
                error=min_count,
            )

    def estimate(self, item: T) -> int:
        """Estimate the frequency of an item.

        Args:
            item: The item to estimate.

        Returns:
            Estimated count. If item is tracked, returns its counter value.
            If not tracked, returns 0 (it may have been evicted, but its
            true count is bounded by max_error()).
        """
        if item in self._counters:
            return self._counters[item].count
        return 0

    def estimate_with_error(self, item: T) -> FrequencyEstimate[T]:
        """Get frequency estimate with error bounds.

        Args:
            item: The item to estimate.

        Returns:
            FrequencyEstimate with count and error bound.
        """
        if item in self._counters:
            counter = self._counters[item]
            return FrequencyEstimate(
                item=item,
                count=counter.count,
                error=counter.error,
            )
        return FrequencyEstimate(
            item=item,
            count=0,
            error=self.max_error(),
        )

    def top(self, n: int | None = None) -> list[FrequencyEstimate[T]]:
        """Return the top-n most frequent items.

        Args:
            n: Number of top items to return. If None, returns all tracked items.

        Returns:
            List of FrequencyEstimate objects sorted by count (descending).
        """
        if n is None:
            n = len(self._counters)

        sorted_counters = sorted(
            self._counters.values(),
            key=lambda c: c.count,
            reverse=True,
        )

        return [
            FrequencyEstimate(item=c.item, count=c.count, error=c.error)
            for c in sorted_counters[:n]
        ]

    def __contains__(self, item: T) -> bool:
        """Check if an item is currently being tracked."""
        return item in self._counters

    def max_error(self) -> int:
        """Maximum possible error for any estimate.

        Returns:
            Upper bound on estimation error. For any item, its true count
            differs from estimate() by at most this amount.
        """
        if not self._counters:
            return 0
        return min(c.count for c in self._counters.values())

    def guaranteed_threshold(self) -> int:
        """Frequency threshold above which items are guaranteed to be tracked.

        Returns:
            Threshold T such that any item with true frequency > T is
            guaranteed to be in the top-k. Equal to N/k where N is total count.
        """
        if self._k == 0:
            return 0
        return self._total_count // self._k

    def merge(self, other: TopK[T]) -> None:
        """Merge another TopK sketch into this one.

        After merging, this sketch approximates the top-k of the combined stream.
        Note that merging TopK sketches can lose accuracy compared to a single
        sketch processing both streams - this is a fundamental limitation.

        Args:
            other: Another TopK sketch with the same k value.

        Raises:
            TypeError: If other is not a TopK.
            ValueError: If other has different k value.
        """
        if not isinstance(other, TopK):
            raise TypeError(f"Can only merge with TopK, got {type(other).__name__}")
        if other._k != self._k:
            raise ValueError(f"Cannot merge TopK with k={other._k} into k={self._k}")

        # Capture items already tracked BEFORE merging (for correct total calculation)
        items_already_tracked = set(self._counters.keys())

        # Simple merge: add all items from other
        # This isn't optimal but provides correctness
        for counter in other._counters.values():
            # For tracked items, we merge counts
            if counter.item in self._counters:
                self._counters[counter.item].count += counter.count
                # Combine errors conservatively
                self._counters[counter.item].error += counter.error
            else:
                # Add as new, counting it as 'count' occurrences
                # This is approximate - we use the count from other
                self.add(counter.item, counter.count)
                # Track merged error
                if counter.item in self._counters:
                    self._counters[counter.item].error += counter.error

        # Update total: add other's total, subtract what add() already added
        # (add() increments _total_count for items that weren't already tracked)
        self._total_count += other._total_count - sum(
            c.count for c in other._counters.values() if c.item not in items_already_tracked
        )

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Rough estimate: dict overhead + counter objects
        # Each counter: item reference (8) + count (8) + error (8) = 24 bytes
        # Plus dict overhead per entry (~50 bytes on 64-bit Python)
        base_size = sys.getsizeof(self._counters)
        per_item = 24 + 50  # Counter fields + dict entry overhead
        return base_size + len(self._counters) * per_item

    @property
    def item_count(self) -> int:
        """Total count of items added."""
        return self._total_count

    @property
    def tracked_count(self) -> int:
        """Number of distinct items currently tracked."""
        return len(self._counters)

    def clear(self) -> None:
        """Reset the sketch to empty state."""
        self._counters.clear()
        self._total_count = 0

    def __repr__(self) -> str:
        return f"TopK(k={self._k}, tracked={len(self._counters)}, total={self._total_count})"
