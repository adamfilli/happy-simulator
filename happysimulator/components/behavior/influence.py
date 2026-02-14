"""Influence models for opinion dynamics.

Provides an InfluenceModel protocol and three implementations:
DeGroot (weighted average), Bounded Confidence (Hegselmann-Krause),
and Voter Model (random adoption).
"""

from __future__ import annotations

import random
from typing import Protocol, runtime_checkable


@runtime_checkable
class InfluenceModel(Protocol):
    """Protocol for computing how opinions change under social influence."""

    def compute_influence(
        self,
        current: float,
        influencer_opinions: list[float],
        weights: list[float],
        rng: random.Random,
    ) -> float:
        """Compute the updated opinion value.

        Args:
            current: The agent's current opinion on a topic (-1 to 1).
            influencer_opinions: Opinions of influencing agents.
            weights: Corresponding influence weights (same length).
            rng: Random number generator.

        Returns:
            Updated opinion value.
        """
        ...


class DeGrootModel:
    """Weighted average convergence (consensus model).

    At each round, the agent's opinion becomes the weighted average
    of its own opinion and those of its influencers.

    Args:
        self_weight: Weight the agent places on its own opinion.
    """

    def __init__(self, self_weight: float = 0.5):
        self.self_weight = self_weight

    def compute_influence(
        self,
        current: float,
        influencer_opinions: list[float],
        weights: list[float],
        rng: random.Random,
    ) -> float:
        if not influencer_opinions:
            return current

        total_w = sum(weights)
        if total_w <= 0:
            return current

        # Normalize influencer weights to sum to (1 - self_weight)
        other_weight = 1.0 - self.self_weight
        weighted_sum = sum(o * w for o, w in zip(influencer_opinions, weights))
        neighbor_avg = weighted_sum / total_w

        return self.self_weight * current + other_weight * neighbor_avg


class BoundedConfidenceModel:
    """Hegselmann-Krause bounded confidence model.

    Only considers influencers whose opinions are within *epsilon*
    of the agent's current opinion, then averages them.

    Args:
        epsilon: Maximum opinion distance to consider.
        self_weight: Weight the agent places on its own opinion.
    """

    def __init__(self, epsilon: float = 0.3, self_weight: float = 0.5):
        self.epsilon = epsilon
        self.self_weight = self_weight

    def compute_influence(
        self,
        current: float,
        influencer_opinions: list[float],
        weights: list[float],
        rng: random.Random,
    ) -> float:
        if not influencer_opinions:
            return current

        # Filter to opinions within epsilon
        close_opinions: list[float] = []
        close_weights: list[float] = []
        for opinion, w in zip(influencer_opinions, weights):
            if abs(opinion - current) <= self.epsilon:
                close_opinions.append(opinion)
                close_weights.append(w)

        if not close_opinions:
            return current

        total_w = sum(close_weights)
        if total_w <= 0:
            return current

        other_weight = 1.0 - self.self_weight
        weighted_sum = sum(o * w for o, w in zip(close_opinions, close_weights))
        neighbor_avg = weighted_sum / total_w

        return self.self_weight * current + other_weight * neighbor_avg


class VoterModel:
    """Random adoption from one neighbor.

    At each round, the agent randomly selects one influencer (weighted
    by influence weight) and adopts their opinion entirely.
    """

    def compute_influence(
        self,
        current: float,
        influencer_opinions: list[float],
        weights: list[float],
        rng: random.Random,
    ) -> float:
        if not influencer_opinions:
            return current

        total_w = sum(weights)
        if total_w <= 0:
            return current

        chosen = rng.choices(influencer_opinions, weights=weights, k=1)[0]
        return chosen
