"""Environment entity for behavioral simulations.

The Environment mediates between external stimuli and agents. It
receives broadcast/targeted stimulus events and influence propagation
triggers, routing them to the appropriate agents with enriched context.

Follows the Network pattern: receives events and routes them to
registered entities, propagating clocks.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.clock import Clock
from happysimulator.core.temporal import Instant

from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.social_network import SocialGraph
from happysimulator.components.behavior.influence import InfluenceModel, DeGrootModel
from happysimulator.components.behavior.stats import EnvironmentStats

logger = logging.getLogger(__name__)


class Environment(Entity):
    """Mediator entity that routes stimuli to behavioral agents.

    Handles three event types:
    - ``BroadcastStimulus``: Forward to all registered agents.
    - ``TargetedStimulus``: Forward to named agents only.
    - ``InfluencePropagation``: Iterate social graph and create
      SocialMessage events between connected agents.
    - ``StateChange``: Update a shared state variable.

    Args:
        name: Identifier for this environment.
        agents: List of Agent entities to manage.
        social_graph: Optional social graph for influence propagation.
        shared_state: Initial shared state (prices, policies, etc.).
        influence_model: Model for computing opinion updates.
        seed: Random seed for deterministic behavior.
    """

    def __init__(
        self,
        name: str,
        agents: list[Agent] | None = None,
        social_graph: SocialGraph | None = None,
        shared_state: dict[str, Any] | None = None,
        influence_model: InfluenceModel | None = None,
        seed: int | None = None,
    ):
        super().__init__(name)
        self._agents: dict[str, Agent] = {}
        self.social_graph = social_graph or SocialGraph()
        self.shared_state: dict[str, Any] = shared_state or {}
        self.influence_model = influence_model or DeGrootModel()
        self.stats = EnvironmentStats()
        self._rng = random.Random(seed)

        for agent in (agents or []):
            self.register_agent(agent)

    def register_agent(self, agent: Agent) -> None:
        """Add an agent to this environment."""
        self._agents[agent.name] = agent
        self.social_graph.add_node(agent.name)

    @property
    def agents(self) -> list[Agent]:
        """All registered agents."""
        return list(self._agents.values())

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to all registered agents."""
        super().set_clock(clock)
        for agent in self._agents.values():
            agent.set_clock(clock)

    def handle_event(self, event: Event) -> list[Event] | None:
        if event.event_type == "BroadcastStimulus":
            return self._handle_broadcast(event)
        elif event.event_type == "TargetedStimulus":
            return self._handle_targeted(event)
        elif event.event_type == "InfluencePropagation":
            return self._handle_influence(event)
        elif event.event_type == "StateChange":
            return self._handle_state_change(event)
        return None

    def _handle_broadcast(self, event: Event) -> list[Event]:
        """Forward stimulus to all registered agents."""
        self.stats.broadcasts_sent += 1
        metadata = event.context.get("metadata", {})
        stimulus_type = metadata.get("stimulus_type", "Stimulus")

        events: list[Event] = []
        for agent in self._agents.values():
            enriched = self._enrich_metadata(metadata, agent.name)
            events.append(Event(
                time=self.now,
                event_type=stimulus_type,
                target=agent,
                context={"metadata": enriched},
            ))
        return events

    def _handle_targeted(self, event: Event) -> list[Event]:
        """Forward stimulus to named agents only."""
        self.stats.targeted_sends += 1
        metadata = event.context.get("metadata", {})
        stimulus_type = metadata.get("stimulus_type", "Stimulus")
        targets: list[str] = metadata.get("targets", [])

        events: list[Event] = []
        for name in targets:
            agent = self._agents.get(name)
            if agent is not None:
                enriched = self._enrich_metadata(metadata, agent.name)
                events.append(Event(
                    time=self.now,
                    event_type=stimulus_type,
                    target=agent,
                    context={"metadata": enriched},
                ))
        return events

    def _handle_influence(self, event: Event) -> list[Event]:
        """Execute one round of social influence propagation."""
        self.stats.influence_rounds += 1
        metadata = event.context.get("metadata", {})
        topic = metadata.get("topic", "")

        events: list[Event] = []
        for agent_name, agent in self._agents.items():
            influencer_names = self.social_graph.influencers(agent_name)
            if not influencer_names:
                continue

            # Gather influencer opinions and weights
            opinions: list[float] = []
            weights: list[float] = []
            for inf_name in influencer_names:
                inf_agent = self._agents.get(inf_name)
                if inf_agent is None:
                    continue
                opinion = inf_agent.state.beliefs.get(topic, 0.0)
                rel = self.social_graph.get_edge(inf_name, agent_name)
                w = rel.weight if rel else 0.5
                opinions.append(opinion)
                weights.append(w)

            if not opinions:
                continue

            # Compute new opinion
            current = agent.state.beliefs.get(topic, 0.0)
            new_opinion = self.influence_model.compute_influence(
                current, opinions, weights, self._rng
            )

            # Send social message with the aggregated influence
            avg_trust = sum(
                (self.social_graph.get_edge(n, agent_name) or _default_rel).trust
                for n in influencer_names
                if self._agents.get(n) is not None
            ) / max(1, len(influencer_names))

            events.append(Event(
                time=self.now,
                event_type="SocialMessage",
                target=agent,
                context={"metadata": {
                    "topic": topic,
                    "opinion": new_opinion,
                    "credibility": avg_trust,
                }},
            ))

        return events

    def _handle_state_change(self, event: Event) -> list[Event] | None:
        """Update shared state from event metadata."""
        self.stats.state_changes += 1
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        if key is not None:
            self.shared_state[key] = value
        return None

    def _enrich_metadata(self, metadata: dict[str, Any], agent_name: str) -> dict[str, Any]:
        """Add environment and social context to stimulus metadata."""
        enriched = dict(metadata)
        enriched["environment"] = dict(self.shared_state)

        # Add social context: peer actions from neighbors
        neighbors = self.social_graph.neighbors(agent_name)
        peer_actions: dict[str, int] = {}
        for n_name in neighbors:
            n_agent = self._agents.get(n_name)
            if n_agent is not None:
                for action, count in n_agent.stats.actions_by_type.items():
                    peer_actions[action] = peer_actions.get(action, 0) + count

        enriched["social_context"] = {"peer_actions": peer_actions}
        return enriched


class _DefaultRel:
    """Fallback relationship for trust computation."""
    trust = 0.5

_default_rel = _DefaultRel()
