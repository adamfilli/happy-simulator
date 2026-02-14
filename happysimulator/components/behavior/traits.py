"""Personality traits for behavioral agents.

Provides a TraitSet protocol for arbitrary trait dimensions, a concrete
PersonalityTraits implementation, and distributions for sampling traits
across a population.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class TraitSet(Protocol):
    """Protocol for accessing named personality dimensions."""

    def get(self, name: str) -> float:
        """Return the value of the named trait (0.0 to 1.0)."""
        ...

    def names(self) -> Sequence[str]:
        """Return the names of all trait dimensions."""
        ...


@dataclass(frozen=True)
class PersonalityTraits:
    """Immutable float-vector personality model.

    Each dimension is a value in [0.0, 1.0]. Dimensions are stored in a
    dict keyed by dimension name.

    Attributes:
        dimensions: Mapping from trait name to value (0.0-1.0).
    """

    dimensions: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        """Return the trait value, defaulting to 0.5 if absent."""
        return self.dimensions.get(name, 0.5)

    def names(self) -> Sequence[str]:
        """Return all trait dimension names."""
        return list(self.dimensions.keys())

    @staticmethod
    def big_five(
        openness: float = 0.5,
        conscientiousness: float = 0.5,
        extraversion: float = 0.5,
        agreeableness: float = 0.5,
        neuroticism: float = 0.5,
    ) -> PersonalityTraits:
        """Construct traits using the Big Five personality model."""
        return PersonalityTraits(dimensions={
            "openness": _clamp(openness),
            "conscientiousness": _clamp(conscientiousness),
            "extraversion": _clamp(extraversion),
            "agreeableness": _clamp(agreeableness),
            "neuroticism": _clamp(neuroticism),
        })


@runtime_checkable
class TraitDistribution(Protocol):
    """Protocol for sampling TraitSet instances from a distribution."""

    def sample(self, rng: random.Random) -> TraitSet:
        """Return a new TraitSet sampled from this distribution."""
        ...


class NormalTraitDistribution:
    """Per-dimension normal distribution, clamped to [0, 1].

    Args:
        means: Mean value per dimension.
        stds: Standard deviation per dimension.
    """

    def __init__(self, means: dict[str, float], stds: dict[str, float] | None = None):
        self._means = means
        self._stds = stds or {k: 0.15 for k in means}

    def sample(self, rng: random.Random) -> PersonalityTraits:
        dims = {}
        for name, mean in self._means.items():
            std = self._stds.get(name, 0.15)
            dims[name] = _clamp(rng.gauss(mean, std))
        return PersonalityTraits(dimensions=dims)


class UniformTraitDistribution:
    """Uniform distribution across all dimensions.

    Args:
        dimension_names: Names of the trait dimensions to sample.
    """

    def __init__(self, dimension_names: Sequence[str]):
        self._names = list(dimension_names)

    def sample(self, rng: random.Random) -> PersonalityTraits:
        dims = {name: rng.random() for name in self._names}
        return PersonalityTraits(dimensions=dims)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))
