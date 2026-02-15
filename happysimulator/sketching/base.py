"""Base protocols for streaming/sketching algorithms.

Sketching algorithms provide approximate statistics over data streams using
bounded memory. They trade exact accuracy for space efficiency, making them
ideal for high-throughput systems where storing all data is impractical.

This module defines the core protocols that all sketch implementations follow:
- Sketch: Base protocol with common operations (add, merge, clear)
- FrequencySketch: For frequency estimation (TopK, Count-Min Sketch)
- QuantileSketch: For quantile estimation (T-Digest)
- CardinalitySketch: For cardinality estimation (HyperLogLog)
- MembershipSketch: For membership testing (Bloom Filter)
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class Sketch(ABC):
    """Base protocol for all streaming/sketching algorithms.

    Sketches process a stream of items and provide approximate answers to
    queries about the stream. They support:
    - Adding items (with optional counts)
    - Merging two sketches of the same type
    - Estimating memory usage
    - Clearing state for reuse

    All sketches should accept a `seed` parameter for reproducibility when
    the algorithm uses randomization.
    """

    @abstractmethod
    def add(self, item: T, count: int = 1) -> None:
        """Add an item to the sketch.

        Args:
            item: The item to add.
            count: Number of occurrences to add (default 1).
        """

    @abstractmethod
    def merge(self, other: "Sketch") -> None:
        """Merge another sketch of the same type into this one.

        After merging, this sketch contains the combined information from
        both sketches, as if all items from both had been added to one sketch.

        Args:
            other: Another sketch of the same type and configuration.

        Raises:
            TypeError: If other is not the same sketch type.
            ValueError: If other has incompatible configuration.
        """

    @property
    @abstractmethod
    def memory_bytes(self) -> int:
        """Estimated memory usage in bytes.

        Returns:
            Approximate memory footprint of the sketch data structures.
        """

    @property
    @abstractmethod
    def item_count(self) -> int:
        """Total count of items added to the sketch.

        Returns:
            Sum of all counts added via add().
        """

    @abstractmethod
    def clear(self) -> None:
        """Reset the sketch to its initial empty state."""


@dataclass(frozen=True, slots=True)
class FrequencyEstimate[T]:
    """A frequency estimate for an item.

    Attributes:
        item: The item being estimated.
        count: Estimated frequency count.
        error: Upper bound on the estimation error.
    """

    item: T
    count: int
    error: int


class FrequencySketch[T](Sketch):
    """Protocol for sketches that estimate item frequencies.

    Used for finding heavy hitters (most frequent items) and estimating
    how many times specific items appeared in the stream.

    Implementations: TopK (Space-Saving), Count-Min Sketch
    """

    @abstractmethod
    def estimate(self, item: T) -> int:
        """Estimate the frequency of an item.

        Args:
            item: The item to estimate frequency for.

        Returns:
            Estimated count. May be an overestimate (never an underestimate
            for Count-Min) or approximate (for Space-Saving).
        """

    @abstractmethod
    def top(self, k: int) -> list[FrequencyEstimate[T]]:
        """Return the top-k most frequent items.

        Args:
            k: Number of top items to return.

        Returns:
            List of FrequencyEstimate objects sorted by count (descending).
            May return fewer than k items if fewer distinct items were seen.
        """


class QuantileSketch(Sketch):
    """Protocol for sketches that estimate quantiles/percentiles.

    Used for estimating latency percentiles (p50, p95, p99) without storing
    all values.

    Implementations: T-Digest
    """

    @abstractmethod
    def quantile(self, q: float) -> float:
        """Estimate the value at a given quantile.

        Args:
            q: Quantile to estimate (0.0 to 1.0).
               - 0.5 = median (p50)
               - 0.95 = 95th percentile (p95)
               - 0.99 = 99th percentile (p99)

        Returns:
            Estimated value at the quantile.

        Raises:
            ValueError: If q is not in [0, 1].
        """

    @abstractmethod
    def cdf(self, value: float) -> float:
        """Estimate the cumulative distribution function at a value.

        Args:
            value: The value to get CDF for.

        Returns:
            Estimated probability that a random sample <= value (0.0 to 1.0).
        """

    def percentile(self, p: float) -> float:
        """Convenience method for percentile estimation.

        Args:
            p: Percentile (0 to 100).

        Returns:
            Estimated value at the percentile.

        Raises:
            ValueError: If p is not in [0, 100].
        """
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile must be in [0, 100], got {p}")
        return self.quantile(p / 100.0)


class CardinalitySketch(Sketch):
    """Protocol for sketches that estimate cardinality (distinct count).

    Used for estimating the number of unique items in a stream without
    storing all items.

    Implementations: HyperLogLog
    """

    @abstractmethod
    def cardinality(self) -> int:
        """Estimate the number of distinct items.

        Returns:
            Estimated count of unique items added.
        """


class MembershipSketch[T](Sketch):
    """Protocol for sketches that test set membership.

    Used for efficiently testing if an item was (probably) seen before.
    False positives are possible; false negatives are not.

    Implementations: Bloom Filter
    """

    @abstractmethod
    def contains(self, item: T) -> bool:
        """Test if an item is (probably) in the set.

        Args:
            item: The item to test.

        Returns:
            True if the item might be in the set (possible false positive).
            False if the item is definitely not in the set (no false negatives).
        """

    @property
    @abstractmethod
    def false_positive_rate(self) -> float:
        """Estimated false positive rate given current fill level.

        Returns:
            Probability that contains() returns True for an item not in the set.
        """


class SamplingSketch[T](Sketch):
    """Protocol for sketches that maintain a sample of stream items.

    Used for maintaining a representative sample of a stream for later analysis.

    Implementations: Reservoir Sampling
    """

    @abstractmethod
    def sample(self) -> list[T]:
        """Return the current sample.

        Returns:
            List of sampled items (may be smaller than capacity if fewer items seen).
        """

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over sampled items."""

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Maximum number of items in the sample.

        Returns:
            Sample size limit.
        """
