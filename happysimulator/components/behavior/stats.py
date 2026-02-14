"""Statistics dataclasses for behavioral simulation components."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentStats:
    """Per-agent statistics tracked during simulation.

    Attributes:
        events_received: Total events handled by this agent.
        decisions_made: Number of times the decision model was invoked.
        actions_by_type: Count of each action type chosen.
        social_messages_received: Number of social influence messages processed.
    """

    events_received: int = 0
    decisions_made: int = 0
    actions_by_type: dict[str, int] = field(default_factory=dict)
    social_messages_received: int = 0

    def record_action(self, action: str) -> None:
        """Increment the count for the given action type."""
        self.actions_by_type[action] = self.actions_by_type.get(action, 0) + 1


@dataclass
class PopulationStats:
    """Aggregate statistics across a population of agents.

    Attributes:
        size: Number of agents in the population.
        total_events: Sum of events received by all agents.
        total_decisions: Sum of decisions made by all agents.
        total_actions: Aggregate action counts across all agents.
    """

    size: int = 0
    total_events: int = 0
    total_decisions: int = 0
    total_actions: dict[str, int] = field(default_factory=dict)


@dataclass
class EnvironmentStats:
    """Statistics tracked by the Environment entity.

    Attributes:
        broadcasts_sent: Number of broadcast stimulus events dispatched.
        targeted_sends: Number of targeted stimulus events dispatched.
        influence_rounds: Number of influence propagation rounds executed.
        state_changes: Number of shared-state mutations applied.
    """

    broadcasts_sent: int = 0
    targeted_sends: int = 0
    influence_rounds: int = 0
    state_changes: int = 0
