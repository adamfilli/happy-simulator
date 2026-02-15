"""Reservoir Sampling for uniform random sampling from a stream.

Reservoir Sampling maintains a uniform random sample of a fixed size from
a stream of unknown (possibly infinite) length. Each item in the stream has
an equal probability of being in the final sample.

Key properties:
- Space: O(k) where k is sample size
- Update: O(1)
- Query: O(k)
- Guarantee: Exact uniform sampling (no approximation error)

This is ideal for:
- Selecting random samples for analysis
- Load testing with representative data
- A/B testing traffic sampling
- Any case requiring unbiased samples from streams

Reference:
    Vitter. "Random Sampling with a Reservoir" (1985)
"""

from __future__ import annotations

import random
import sys
from typing import TYPE_CHECKING, TypeVar

from happysimulator.sketching.base import SamplingSketch

if TYPE_CHECKING:
    from collections.abc import Iterator

T = TypeVar("T")


class ReservoirSampler(SamplingSketch[T]):
    """Reservoir Sampling for uniform stream sampling.

    Maintains a uniform random sample of fixed size from a stream.
    Uses Algorithm R from Vitter's paper for O(1) update time.

    Args:
        size: Maximum number of items to keep in the sample.
        seed: Random seed for reproducibility.

    Example:
        # Sample 1000 requests for analysis
        sampler = ReservoirSampler[Request](size=1000, seed=42)

        for request in request_stream:
            sampler.add(request)

        # Get uniform sample
        sample = sampler.sample()
        analyze(sample)
    """

    def __init__(self, size: int, seed: int | None = None):
        """Initialize Reservoir Sampler.

        Args:
            size: Maximum sample size. Must be > 0.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If size <= 0.
        """
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")

        self._size = size
        self._reservoir: list[T] = []
        self._total_count = 0
        self._rng = random.Random(seed)

    @property
    def capacity(self) -> int:
        """Maximum number of items in the sample."""
        return self._size

    def add(self, item: T, count: int = 1) -> None:
        """Add an item to the stream.

        The item may or may not be kept in the reservoir, depending on
        random selection to maintain uniform sampling.

        Args:
            item: The item to add.
            count: Number of times to add this item. Each occurrence is
                   considered independently for sampling.

        Raises:
            ValueError: If count is negative.
        """
        if count < 0:
            raise ValueError(f"count must be non-negative, got {count}")

        for _ in range(count):
            self._add_one(item)

    def _add_one(self, item: T) -> None:
        """Add a single item to the stream."""
        self._total_count += 1

        if len(self._reservoir) < self._size:
            # Reservoir not full - add directly
            self._reservoir.append(item)
        else:
            # Reservoir full - randomly replace with probability size/count
            j = self._rng.randint(0, self._total_count - 1)
            if j < self._size:
                self._reservoir[j] = item

    def sample(self) -> list[T]:
        """Return the current sample.

        Returns:
            List of sampled items. Length is min(capacity, items_seen).
        """
        return list(self._reservoir)

    def __iter__(self) -> Iterator[T]:
        """Iterate over sampled items."""
        return iter(self._reservoir)

    def __len__(self) -> int:
        """Number of items currently in the reservoir."""
        return len(self._reservoir)

    def __getitem__(self, index: int) -> T:
        """Get item at index in the reservoir."""
        return self._reservoir[index]

    @property
    def is_full(self) -> bool:
        """Whether the reservoir is at capacity."""
        return len(self._reservoir) >= self._size

    def merge(self, other: ReservoirSampler[T]) -> None:
        """Merge another reservoir into this one.

        Maintains uniform sampling over the combined streams.
        Uses weighted random selection based on total counts.

        Args:
            other: Another ReservoirSampler with same capacity.

        Raises:
            TypeError: If other is not a ReservoirSampler.
            ValueError: If other has different capacity.
        """
        if not isinstance(other, ReservoirSampler):
            raise TypeError(f"Can only merge with ReservoirSampler, got {type(other).__name__}")
        if other._size != self._size:
            raise ValueError(f"Cannot merge: capacity differs ({self._size} vs {other._size})")

        # Merge using weighted selection
        # Each item in self has probability self._total_count / combined_total
        # Each item in other has probability other._total_count / combined_total
        combined_total = self._total_count + other._total_count

        if combined_total == 0:
            return

        # Create new reservoir by weighted sampling
        new_reservoir: list[T] = []

        for _i in range(min(self._size, combined_total)):
            # Decide which reservoir to sample from
            if self._rng.random() < self._total_count / combined_total:
                # Sample from self
                if self._reservoir:
                    idx = self._rng.randint(0, len(self._reservoir) - 1)
                    new_reservoir.append(self._reservoir[idx])
            else:
                # Sample from other
                if other._reservoir:
                    idx = self._rng.randint(0, len(other._reservoir) - 1)
                    new_reservoir.append(other._reservoir[idx])

        self._reservoir = new_reservoir[: self._size]
        self._total_count = combined_total

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # List + item references (8 bytes each on 64-bit)
        reservoir_bytes = len(self._reservoir) * 8
        return reservoir_bytes + sys.getsizeof(self._reservoir) + sys.getsizeof(self)

    @property
    def item_count(self) -> int:
        """Total count of items seen (not sample size)."""
        return self._total_count

    @property
    def sample_size(self) -> int:
        """Current number of items in the sample."""
        return len(self._reservoir)

    def clear(self) -> None:
        """Reset the sampler to empty state."""
        self._reservoir.clear()
        self._total_count = 0

    def __repr__(self) -> str:
        return (
            f"ReservoirSampler(capacity={self._size}, "
            f"sampled={len(self._reservoir)}, "
            f"seen={self._total_count})"
        )
