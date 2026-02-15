"""Zipf (power-law) distribution for discrete value sampling.

Zipf's law describes the frequency distribution commonly observed in natural
and artificial systems where a small number of items account for most of the
activity. The probability of selecting an item with rank k is proportional
to 1/k^s, where s is the Zipf exponent.

Common applications in distributed systems:
- Web caching: Popular URLs receive most traffic
- Database access: Hot keys dominate read/write patterns
- API usage: Power users generate disproportionate requests
- Content delivery: Viral content vs long tail

The Zipf exponent (s) controls the degree of skew:
- s=0: Uniform distribution (no skew)
- s=1: Classic Zipf's law (rank k appears 1/k as often as rank 1)
- s>1: More extreme concentration on top-ranked items
"""

import bisect
import random
from collections.abc import Sequence
from typing import TypeVar

from happysimulator.distributions.value_distribution import ValueDistribution

T = TypeVar("T")


class ZipfDistribution(ValueDistribution[T]):
    """Samples values following Zipf's law (power-law distribution).

    Zipf's law states that the frequency of an item is inversely proportional
    to its rank raised to a power: P(rank=k) proportional to 1/k^s

    The distribution is implemented using inverse transform sampling with
    precomputed cumulative probabilities for O(1) sampling after initialization.

    Args:
        values: The population of values to sample from. The first value is
                rank 1 (most frequent), second is rank 2, etc.
        s: Zipf exponent (default 1.0).
           - s=0: uniform distribution
           - s=1: classic Zipf (item at rank k appears 1/k as often as rank 1)
           - s>1: more extreme skew toward popular items
        seed: Optional random seed for reproducibility.

    Example:
        # Customer IDs 0-999 with classic Zipf distribution
        dist = ZipfDistribution(range(1000), s=1.0)
        customer_id = dist.sample()  # Most likely returns low IDs

        # String values with extreme skew
        dist = ZipfDistribution(["hot", "warm", "cool", "cold"], s=1.5, seed=42)
        category = dist.sample()  # "hot" appears most frequently
    """

    def __init__(
        self,
        values: Sequence[T],
        s: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize with population and Zipf parameters.

        Args:
            values: Sequence of values to sample from (rank ordered).
            s: Zipf exponent controlling skew (default 1.0).
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If values is empty or s is negative.
        """
        if len(values) == 0:
            raise ValueError("values must not be empty")
        if s < 0:
            raise ValueError(f"s must be non-negative, got {s}")

        self._values = list(values)
        self._s = s
        self._rng = random.Random(seed)

        # Precompute cumulative probabilities for O(1) sampling
        self._probabilities = self._compute_probabilities()
        self._cum_probs = self._compute_cumulative_probs()

    def _compute_probabilities(self) -> list[float]:
        """Compute probability for each rank."""
        n = len(self._values)

        if self._s == 0:
            # Uniform distribution
            return [1.0 / n] * n

        # Zipf: P(k) = (1/k^s) / H_n,s where H_n,s is generalized harmonic number
        weights = [1.0 / ((k + 1) ** self._s) for k in range(n)]
        total = sum(weights)
        return [w / total for w in weights]

    def _compute_cumulative_probs(self) -> list[float]:
        """Compute cumulative probability distribution."""
        cum = []
        running = 0.0
        for p in self._probabilities:
            running += p
            cum.append(running)
        # Ensure last value is exactly 1.0 to avoid floating point issues
        if cum:
            cum[-1] = 1.0
        return cum

    def sample(self) -> T:
        """Sample a value using inverse transform sampling.

        Returns:
            A value from the population according to Zipf distribution.
        """
        u = self._rng.random()
        # Binary search for the appropriate bucket
        idx = bisect.bisect_left(self._cum_probs, u)
        return self._values[min(idx, len(self._values) - 1)]

    @property
    def population(self) -> Sequence[T]:
        """Return the complete population of possible values."""
        return list(self._values)

    @property
    def size(self) -> int:
        """Return the number of distinct values in the population."""
        return len(self._values)

    @property
    def s(self) -> float:
        """Return the Zipf exponent."""
        return self._s

    def probability(self, rank: int) -> float:
        """Return the probability for a given rank (1-indexed).

        Args:
            rank: The rank of the item (1 = most popular).

        Returns:
            Probability of sampling an item with this rank.

        Raises:
            ValueError: If rank is out of range.
        """
        if rank < 1 or rank > len(self._values):
            raise ValueError(f"Rank must be 1-{len(self._values)}, got {rank}")
        return self._probabilities[rank - 1]

    def probability_for_value(self, value: T) -> float:
        """Return the probability for a specific value.

        Args:
            value: The value to get probability for.

        Returns:
            Probability of sampling this value.

        Raises:
            ValueError: If value is not in the population.
        """
        try:
            idx = self._values.index(value)
            return self._probabilities[idx]
        except ValueError as err:
            raise ValueError(f"Value {value!r} not in population") from err

    def expected_frequency(self, rank: int, n_samples: int) -> float:
        """Return the expected count for a rank given n samples.

        Args:
            rank: The rank of the item (1 = most popular).
            n_samples: Total number of samples.

        Returns:
            Expected count for items with this rank.
        """
        return self.probability(rank) * n_samples

    def top_n_probability(self, n: int) -> float:
        """Return the combined probability of the top n ranked items.

        Args:
            n: Number of top items to include.

        Returns:
            Combined probability (0.0 to 1.0).

        Raises:
            ValueError: If n is out of range.
        """
        if n < 0 or n > len(self._values):
            raise ValueError(f"n must be 0-{len(self._values)}, got {n}")
        if n == 0:
            return 0.0
        return self._cum_probs[n - 1]

    def __repr__(self) -> str:
        return f"ZipfDistribution(size={len(self._values)}, s={self._s})"
