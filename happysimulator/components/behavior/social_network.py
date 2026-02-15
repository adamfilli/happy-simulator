"""Social graph data structure for behavioral agents.

A directed weighted graph of relationships between agents. This is a
pure data structure (like FIFOQueue), not an Entity. It provides O(1)
neighbor lookups and graph generator class methods.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)
from dataclasses import dataclass


@dataclass
class Relationship:
    """A directed edge in the social graph.

    Attributes:
        source: Name of the source agent.
        target: Name of the target agent.
        weight: General relationship strength (0-1).
        trust: How much the source trusts the target (0-1).
        interaction_count: Number of interactions recorded.
    """

    source: str
    target: str
    weight: float = 0.5
    trust: float = 0.5
    interaction_count: int = 0


class SocialGraph:
    """Directed weighted graph of agent relationships.

    Edges are stored as adjacency lists keyed by source agent name.
    Supports O(1) neighbor queries and several common graph generation
    algorithms.
    """

    def __init__(self) -> None:
        self._adjacency: dict[str, dict[str, Relationship]] = {}
        self._nodes: set[str] = set()

    @property
    def nodes(self) -> set[str]:
        """All node names in the graph."""
        return set(self._nodes)

    @property
    def edge_count(self) -> int:
        """Total number of directed edges."""
        return sum(len(targets) for targets in self._adjacency.values())

    def add_node(self, name: str) -> None:
        """Add a node (agent name) to the graph."""
        self._nodes.add(name)
        self._adjacency.setdefault(name, {})

    def add_edge(
        self,
        source: str,
        target: str,
        weight: float = 0.5,
        trust: float = 0.5,
    ) -> Relationship:
        """Add a directed edge from source to target."""
        self.add_node(source)
        self.add_node(target)
        rel = Relationship(source=source, target=target, weight=weight, trust=trust)
        self._adjacency[source][target] = rel
        return rel

    def add_bidirectional_edge(
        self,
        a: str,
        b: str,
        weight: float = 0.5,
        trust: float = 0.5,
    ) -> tuple[Relationship, Relationship]:
        """Add edges in both directions."""
        r1 = self.add_edge(a, b, weight, trust)
        r2 = self.add_edge(b, a, weight, trust)
        return r1, r2

    def get_edge(self, source: str, target: str) -> Relationship | None:
        """Return the relationship from source to target, or None."""
        return self._adjacency.get(source, {}).get(target)

    def neighbors(self, name: str) -> list[str]:
        """Return names of agents that *name* has edges to."""
        return list(self._adjacency.get(name, {}).keys())

    def influencers(self, name: str) -> list[str]:
        """Return names of agents that have edges *to* name."""
        result: list[str] = []
        for source, targets in self._adjacency.items():
            if name in targets:
                result.append(source)
        return result

    def influence_weights(self, name: str) -> dict[str, float]:
        """Return {influencer_name: weight} for edges pointing at *name*."""
        result: dict[str, float] = {}
        for source, targets in self._adjacency.items():
            if name in targets:
                result[source] = targets[name].weight
        return result

    def record_interaction(self, source: str, target: str) -> None:
        """Increment the interaction count on an existing edge."""
        rel = self.get_edge(source, target)
        if rel is not None:
            rel.interaction_count += 1

    # -----------------------------------------------------------------
    # Graph generators
    # -----------------------------------------------------------------

    @classmethod
    def complete(
        cls,
        names: list[str],
        weight: float = 0.5,
        trust: float = 0.5,
        rng: random.Random | None = None,
    ) -> SocialGraph:
        """Fully connected graph (every pair has bidirectional edges)."""
        g = cls()
        for name in names:
            g.add_node(name)
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                g.add_bidirectional_edge(a, b, weight, trust)
        return g

    @classmethod
    def random_erdos_renyi(
        cls,
        names: list[str],
        p: float = 0.1,
        weight: float = 0.5,
        trust: float = 0.5,
        rng: random.Random | None = None,
    ) -> SocialGraph:
        """Erdos-Renyi random graph: each directed edge exists with probability p."""
        rng = rng or random.Random()
        g = cls()
        for name in names:
            g.add_node(name)
        for a in names:
            for b in names:
                if a != b and rng.random() < p:
                    g.add_edge(a, b, weight, trust)
        return g

    @classmethod
    def small_world(
        cls,
        names: list[str],
        k: int = 4,
        p_rewire: float = 0.1,
        weight: float = 0.5,
        trust: float = 0.5,
        rng: random.Random | None = None,
    ) -> SocialGraph:
        """Watts-Strogatz small-world graph.

        Starts with a ring lattice where each node connects to its k
        nearest neighbors, then rewires each edge with probability p.

        Args:
            names: Node names.
            k: Number of nearest neighbors in the ring (must be even).
            p_rewire: Probability of rewiring each edge.
            weight: Default edge weight.
            trust: Default edge trust.
            rng: Random number generator for determinism.
        """
        rng = rng or random.Random()
        n = len(names)
        if n < 3:
            return cls.complete(names, weight, trust)

        k = min(k, n - 1)
        half_k = k // 2

        g = cls()
        for name in names:
            g.add_node(name)

        # Build ring lattice
        for i in range(n):
            for j in range(1, half_k + 1):
                target_idx = (i + j) % n
                g.add_bidirectional_edge(names[i], names[target_idx], weight, trust)

        # Rewire
        for i in range(n):
            for j in range(1, half_k + 1):
                if rng.random() < p_rewire:
                    target_idx = (i + j) % n
                    # Remove old edge
                    old_target = names[target_idx]
                    if old_target in g._adjacency.get(names[i], {}):
                        del g._adjacency[names[i]][old_target]

                    # Pick new target (avoid self and existing neighbors)
                    candidates = [
                        names[c]
                        for c in range(n)
                        if c != i and names[c] not in g._adjacency.get(names[i], {})
                    ]
                    if candidates:
                        new_target = rng.choice(candidates)
                        g.add_edge(names[i], new_target, weight, trust)

        return g
