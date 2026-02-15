"""Population factory for creating groups of behavioral agents.

Provides convenience constructors for creating populations with
uniform traits or demographic segments, along with auto-generated
social graphs.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from typing import Callable

from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.traits import (
    PersonalityTraits,
    TraitDistribution,
    UniformTraitDistribution,
)
from happysimulator.components.behavior.state import AgentState
from happysimulator.components.behavior.decision import DecisionModel
from happysimulator.components.behavior.social_network import SocialGraph
from happysimulator.components.behavior.stats import PopulationStats


@dataclass
class DemographicSegment:
    """Description of a sub-population segment.

    Attributes:
        name: Segment label (e.g. "innovators", "majority").
        fraction: Proportion of total population (0 to 1).
        trait_distribution: Distribution for sampling personality traits.
        decision_model_factory: Callable that creates a DecisionModel per agent.
        initial_state_factory: Optional callable that creates initial AgentState.
        seed: Optional seed for the trait distribution RNG.
    """

    name: str
    fraction: float
    trait_distribution: TraitDistribution | None = None
    decision_model_factory: Callable[[], DecisionModel] | None = None
    initial_state_factory: Callable[[], AgentState] | None = None
    seed: int | None = None


class Population:
    """A collection of agents with an associated social graph.

    Use the class methods ``uniform()`` and ``from_segments()`` to
    construct populations conveniently.

    Attributes:
        agents: The list of Agent instances.
        social_graph: The social graph connecting agents.
    """

    def __init__(
        self,
        agents: list[Agent],
        social_graph: SocialGraph,
    ):
        self.agents = agents
        self.social_graph = social_graph

    @property
    def size(self) -> int:
        return len(self.agents)

    @property
    def stats(self) -> PopulationStats:
        """Frozen snapshot of aggregate statistics across all agents."""
        total_events = 0
        total_decisions = 0
        total_actions: dict[str, int] = {}
        for agent in self.agents:
            agent_stats = agent.stats
            total_events += agent_stats.events_received
            total_decisions += agent_stats.decisions_made
            for action, count in agent_stats.actions_by_type.items():
                total_actions[action] = total_actions.get(action, 0) + count
        return PopulationStats(
            size=self.size,
            total_events=total_events,
            total_decisions=total_decisions,
            total_actions=total_actions,
        )

    @classmethod
    def uniform(
        cls,
        size: int,
        decision_model: DecisionModel | None = None,
        graph_type: str = "small_world",
        seed: int | None = None,
        name_prefix: str = "agent",
    ) -> Population:
        """Create a population with uniform trait distribution.

        Args:
            size: Number of agents.
            decision_model: Shared decision model for all agents.
            graph_type: One of "small_world", "complete", "random".
            seed: Random seed for reproducibility.
            name_prefix: Prefix for auto-generated agent names.
        """
        rng = random.Random(seed)
        big_five_names = ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]
        dist = UniformTraitDistribution(big_five_names)

        agents: list[Agent] = []
        for i in range(size):
            traits = dist.sample(rng)
            agents.append(Agent(
                name=f"{name_prefix}_{i}",
                traits=traits,
                decision_model=decision_model,
                seed=rng.randint(0, 2**31),
            ))

        names = [a.name for a in agents]
        graph = _build_graph(names, graph_type, rng)
        return cls(agents, graph)

    @classmethod
    def from_segments(
        cls,
        total_size: int,
        segments: list[DemographicSegment],
        graph_type: str = "small_world",
        seed: int | None = None,
        name_prefix: str = "agent",
    ) -> Population:
        """Create a population from demographic segments.

        Each segment contributes ``floor(fraction * total_size)`` agents.
        Remaining agents are assigned to the largest segment.

        Args:
            total_size: Total number of agents.
            segments: Segment definitions with fractions summing to ~1.0.
            graph_type: Social graph topology.
            seed: Random seed for reproducibility.
            name_prefix: Prefix for agent names.
        """
        rng = random.Random(seed)
        agents: list[Agent] = []
        big_five_names = ["openness", "conscientiousness", "extraversion",
                          "agreeableness", "neuroticism"]

        # Calculate segment sizes
        sizes: list[int] = []
        for seg in segments:
            sizes.append(int(seg.fraction * total_size))

        # Distribute remainder to largest segment
        remainder = total_size - sum(sizes)
        if remainder > 0 and sizes:
            max_idx = sizes.index(max(sizes))
            sizes[max_idx] += remainder

        agent_idx = 0
        for seg, seg_size in zip(segments, sizes):
            seg_rng = random.Random(seg.seed if seg.seed is not None else rng.randint(0, 2**31))
            dist = seg.trait_distribution or UniformTraitDistribution(big_five_names)

            for _ in range(seg_size):
                traits = dist.sample(seg_rng)
                state = seg.initial_state_factory() if seg.initial_state_factory else AgentState()
                model = seg.decision_model_factory() if seg.decision_model_factory else None

                agents.append(Agent(
                    name=f"{name_prefix}_{agent_idx}",
                    traits=traits,
                    decision_model=model,
                    state=state,
                    seed=seg_rng.randint(0, 2**31),
                ))
                agent_idx += 1

        names = [a.name for a in agents]
        graph = _build_graph(names, graph_type, rng)
        return cls(agents, graph)


def _build_graph(
    names: list[str], graph_type: str, rng: random.Random
) -> SocialGraph:
    """Build a social graph of the specified type."""
    if graph_type == "complete":
        return SocialGraph.complete(names, rng=rng)
    elif graph_type == "random":
        return SocialGraph.random_erdos_renyi(names, p=0.1, rng=rng)
    else:  # "small_world" default
        k = min(4, len(names) - 1) if len(names) > 1 else 0
        if k < 2:
            return SocialGraph.complete(names, rng=rng)
        return SocialGraph.small_world(names, k=k, p_rewire=0.1, rng=rng)
