"""Abstract base class for sampling discrete values from distributions.

ValueDistribution provides an interface for sampling from a finite set of
discrete values. Unlike LatencyDistribution (which samples continuous positive
floats), ValueDistribution is designed for categorical data like customer IDs,
region codes, or product SKUs.

Common use cases:
- Zipf-distributed customer IDs (power-law access patterns)
- Uniform random selection from a pool of values
- Weighted categorical distributions
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Sequence

T = TypeVar('T')


class ValueDistribution(ABC, Generic[T]):
    """Abstract base for sampling discrete values from a distribution.

    Subclasses implement sample() to return values according to their
    specific probability distribution. The population of possible values
    is finite and defined at construction time.

    Type Parameters:
        T: The type of values in the distribution (int, str, custom objects, etc.)

    Example:
        class MyDistribution(ValueDistribution[str]):
            def sample(self) -> str:
                return "value"
    """

    @abstractmethod
    def sample(self) -> T:
        """Sample a single value from the distribution.

        Returns:
            A value from the population according to the distribution.
        """
        pass

    def sample_n(self, n: int) -> list[T]:
        """Sample n values from the distribution.

        Args:
            n: Number of samples to generate.

        Returns:
            List of n sampled values.
        """
        return [self.sample() for _ in range(n)]

    @property
    @abstractmethod
    def population(self) -> Sequence[T]:
        """Return the complete population of possible values.

        Returns:
            Sequence of all values that can be sampled.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of distinct values in the population.

        Returns:
            Population size.
        """
        pass
