"""Uniform distribution for discrete value sampling.

Provides uniform random sampling from a finite population where each
value has equal probability of being selected. Useful as a baseline
comparison against skewed distributions like Zipf.
"""

import random
from typing import Sequence, TypeVar

from happysimulator.distributions.value_distribution import ValueDistribution

T = TypeVar('T')


class UniformDistribution(ValueDistribution[T]):
    """Uniform random sampling from a population.

    Each value in the population has an equal probability of being selected.
    Useful as a baseline comparison against skewed distributions.

    Args:
        values: The population of values to sample from.
        seed: Optional random seed for reproducibility.

    Example:
        # Uniform distribution over regions
        dist = UniformDistribution(["us-east", "us-west", "eu"], seed=42)
        region = dist.sample()  # Each region equally likely

        # Uniform distribution over integer IDs
        dist = UniformDistribution(range(1000))
        id = dist.sample()  # Each ID has 1/1000 probability
    """

    def __init__(
        self,
        values: Sequence[T],
        seed: int | None = None,
    ):
        """Initialize with population.

        Args:
            values: Sequence of values to sample from.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If values is empty.
        """
        if len(values) == 0:
            raise ValueError("values must not be empty")

        self._values = list(values)
        self._rng = random.Random(seed)

    def sample(self) -> T:
        """Sample a value uniformly at random.

        Returns:
            A randomly selected value from the population.
        """
        return self._rng.choice(self._values)

    @property
    def population(self) -> Sequence[T]:
        """Return the complete population of possible values."""
        return list(self._values)

    @property
    def size(self) -> int:
        """Return the number of distinct values in the population."""
        return len(self._values)

    def probability(self) -> float:
        """Return the probability of selecting any single value.

        Returns:
            1/n where n is the population size.
        """
        return 1.0 / len(self._values)

    def __repr__(self) -> str:
        return f"UniformDistribution(size={len(self._values)})"
