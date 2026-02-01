"""Count-Min Sketch for frequency estimation.

The Count-Min Sketch is a probabilistic data structure that estimates the
frequency of items in a data stream. It uses multiple hash functions and
a 2D array of counters to provide frequency estimates with controlled error.

Key properties:
- Space: O(width * depth)
- Update: O(depth) time
- Query: O(depth) time
- Error: At most εN with probability ≥ 1-δ where ε=e/width and δ=e^(-depth)
- Always overestimates (never underestimates)

This is ideal for:
- Frequency estimation when TopK guarantees aren't needed
- Situations requiring frequency queries for arbitrary items
- Applications where always-overestimate is acceptable

Reference:
    Cormode, Muthukrishnan. "An Improved Data Stream Summary: The Count-Min
    Sketch and its Applications" (2004)
"""

from __future__ import annotations

import hashlib
import math
import struct
import sys
from typing import Generic, TypeVar, Hashable

from happysimulator.sketching.base import FrequencySketch, FrequencyEstimate

T = TypeVar('T', bound=Hashable)


def _optimal_width(epsilon: float) -> int:
    """Calculate optimal width for target error rate epsilon."""
    return int(math.ceil(math.e / epsilon))


def _optimal_depth(delta: float) -> int:
    """Calculate optimal depth for target failure probability delta."""
    return int(math.ceil(math.log(1.0 / delta)))


class CountMinSketch(FrequencySketch[T]):
    """Count-Min Sketch for frequency estimation.

    A probabilistic data structure that estimates item frequencies using
    multiple hash functions and a 2D counter array. Provides strong
    guarantees on estimation error.

    Args:
        width: Number of counters per row. Larger = more accurate.
        depth: Number of hash functions/rows. Larger = higher confidence.
        seed: Random seed for hash function initialization.

    Alternatively, create from error bounds:
        CountMinSketch.from_error_rate(epsilon=0.01, delta=0.01)

    Error guarantees:
        - Query always returns count >= true count (never underestimates)
        - With probability ≥ 1-δ: estimate ≤ true_count + εN
        - Where ε = e/width, δ = e^(-depth), N = total item count

    Example:
        # Create sketch for 1% error with 99% confidence
        cms = CountMinSketch.from_error_rate(epsilon=0.01, delta=0.01)

        for item in data_stream:
            cms.add(item)

        # Query frequency (may overestimate)
        freq = cms.estimate("hot_key")
    """

    def __init__(self, width: int, depth: int, seed: int | None = None):
        """Initialize Count-Min Sketch.

        Args:
            width: Number of counters per row. Must be positive.
            depth: Number of hash functions/rows. Must be positive.
            seed: Random seed for reproducible hash functions.

        Raises:
            ValueError: If width or depth <= 0.
        """
        if width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self._width = width
        self._depth = depth
        self._seed = seed if seed is not None else 0

        # 2D array of counters [depth][width]
        self._counters: list[list[int]] = [
            [0] * width for _ in range(depth)
        ]
        self._total_count = 0

        # Precompute hash seeds for each row
        self._hash_seeds = self._generate_hash_seeds()

    @classmethod
    def from_error_rate(
        cls,
        epsilon: float,
        delta: float,
        seed: int | None = None,
    ) -> "CountMinSketch[T]":
        """Create a sketch with specified error guarantees.

        Args:
            epsilon: Maximum relative error (e.g., 0.01 for 1%).
            delta: Failure probability (e.g., 0.01 for 99% confidence).
            seed: Random seed for reproducibility.

        Returns:
            CountMinSketch configured for the specified guarantees.

        Raises:
            ValueError: If epsilon or delta not in (0, 1).
        """
        if not 0 < epsilon < 1:
            raise ValueError(f"epsilon must be in (0, 1), got {epsilon}")
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")

        width = _optimal_width(epsilon)
        depth = _optimal_depth(delta)
        return cls(width=width, depth=depth, seed=seed)

    def _generate_hash_seeds(self) -> list[int]:
        """Generate deterministic hash seeds for each row."""
        seeds = []
        for i in range(self._depth):
            # Use seed to generate per-row seeds deterministically
            h = hashlib.sha256()
            h.update(struct.pack(">QQ", self._seed, i))
            seeds.append(struct.unpack(">Q", h.digest()[:8])[0])
        return seeds

    def _hash(self, item: T, row: int) -> int:
        """Hash an item to a column index for a specific row."""
        # Combine item hash with row-specific seed
        item_hash = hash(item)
        combined = item_hash ^ self._hash_seeds[row]
        # Mask to 64 bits to avoid overflow in struct.pack
        combined = combined & 0xFFFFFFFFFFFFFFFF
        # Use another round of hashing for better distribution
        h = hashlib.sha256()
        h.update(struct.pack(">Q", combined))
        return struct.unpack(">Q", h.digest()[:8])[0] % self._width

    @property
    def width(self) -> int:
        """Number of counters per row."""
        return self._width

    @property
    def depth(self) -> int:
        """Number of hash functions/rows."""
        return self._depth

    @property
    def epsilon(self) -> float:
        """Error rate parameter (e/width)."""
        return math.e / self._width

    @property
    def delta(self) -> float:
        """Failure probability parameter (e^-depth)."""
        return math.exp(-self._depth)

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

        for row in range(self._depth):
            col = self._hash(item, row)
            self._counters[row][col] += count

    def estimate(self, item: T) -> int:
        """Estimate the frequency of an item.

        Args:
            item: The item to estimate frequency for.

        Returns:
            Estimated count. This is guaranteed to be >= true count,
            and with high probability <= true_count + epsilon * total_count.
        """
        min_count = float('inf')
        for row in range(self._depth):
            col = self._hash(item, row)
            min_count = min(min_count, self._counters[row][col])
        return int(min_count)

    def estimate_with_error(self, item: T) -> FrequencyEstimate[T]:
        """Get frequency estimate with error bounds.

        Args:
            item: The item to estimate.

        Returns:
            FrequencyEstimate with count and error bound.
        """
        count = self.estimate(item)
        error = int(math.ceil(self.epsilon * self._total_count))
        return FrequencyEstimate(item=item, count=count, error=error)

    def top(self, k: int) -> list[FrequencyEstimate[T]]:
        """Return top-k items.

        Note: Count-Min Sketch doesn't track items, only counts. This method
        cannot efficiently find top-k. For top-k queries, use TopK instead.

        Args:
            k: Number of top items to return.

        Returns:
            Empty list (Count-Min Sketch doesn't track item identities).

        Raises:
            NotImplementedError: Always, as CMS cannot enumerate items.
        """
        raise NotImplementedError(
            "Count-Min Sketch cannot enumerate top-k items. "
            "Use TopK (Space-Saving) for heavy hitters queries."
        )

    def inner_product(self, other: "CountMinSketch[T]") -> int:
        """Estimate the inner product of two streams.

        The inner product is sum(count_A[x] * count_B[x]) over all items x.
        Useful for comparing similarity of two streams.

        Args:
            other: Another Count-Min Sketch with same dimensions.

        Returns:
            Estimated inner product.

        Raises:
            ValueError: If sketches have different dimensions.
        """
        if self._width != other._width or self._depth != other._depth:
            raise ValueError(
                f"Cannot compute inner product: dimensions differ "
                f"({self._width}x{self._depth} vs {other._width}x{other._depth})"
            )

        min_product = float('inf')
        for row in range(self._depth):
            product = sum(
                self._counters[row][col] * other._counters[row][col]
                for col in range(self._width)
            )
            min_product = min(min_product, product)
        return int(min_product)

    def merge(self, other: "CountMinSketch[T]") -> None:
        """Merge another Count-Min Sketch into this one.

        After merging, this sketch estimates frequencies for the combined stream.

        Args:
            other: Another Count-Min Sketch with same dimensions and seed.

        Raises:
            TypeError: If other is not a CountMinSketch.
            ValueError: If other has different dimensions or seed.
        """
        if not isinstance(other, CountMinSketch):
            raise TypeError(f"Can only merge with CountMinSketch, got {type(other).__name__}")
        if self._width != other._width or self._depth != other._depth:
            raise ValueError(
                f"Cannot merge: dimensions differ "
                f"({self._width}x{self._depth} vs {other._width}x{other._depth})"
            )
        if self._seed != other._seed:
            raise ValueError(
                f"Cannot merge: seeds differ ({self._seed} vs {other._seed})"
            )

        for row in range(self._depth):
            for col in range(self._width):
                self._counters[row][col] += other._counters[row][col]
        self._total_count += other._total_count

    @property
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes."""
        # Counter array: depth * width * 8 bytes per int
        # Plus list overhead
        counter_bytes = self._depth * self._width * 8
        list_overhead = sys.getsizeof([]) * (self._depth + 1)
        return counter_bytes + list_overhead + sys.getsizeof(self)

    @property
    def item_count(self) -> int:
        """Total count of items added."""
        return self._total_count

    def clear(self) -> None:
        """Reset the sketch to empty state."""
        for row in range(self._depth):
            for col in range(self._width):
                self._counters[row][col] = 0
        self._total_count = 0

    def __repr__(self) -> str:
        return (
            f"CountMinSketch(width={self._width}, depth={self._depth}, "
            f"ε={self.epsilon:.4f}, δ={self.delta:.4f}, total={self._total_count})"
        )
