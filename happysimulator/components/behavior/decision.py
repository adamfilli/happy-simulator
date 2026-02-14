"""Decision models for behavioral agents.

Provides a DecisionModel protocol and five concrete implementations
covering rational choice, rule-based heuristics, satisficing, social
conformity, and composite voting.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from happysimulator.components.behavior.traits import TraitSet
from happysimulator.components.behavior.state import AgentState


@dataclass(frozen=True)
class Choice:
    """A candidate action the agent may select.

    Attributes:
        action: Short identifier for the action (e.g. "buy", "wait").
        context: Arbitrary metadata about this choice.
    """

    action: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionContext:
    """Everything an agent knows when making a decision.

    Attributes:
        traits: The agent's personality traits.
        state: The agent's current internal state.
        choices: Available actions to choose from.
        stimulus: Metadata about the triggering event.
        environment: Shared environment state (prices, policies, etc.).
        social_context: Information about peer behavior.
    """

    traits: TraitSet
    state: AgentState
    choices: list[Choice]
    stimulus: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    social_context: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DecisionModel(Protocol):
    """Protocol for agent decision-making strategies."""

    def decide(self, context: DecisionContext, rng: random.Random) -> Choice | None:
        """Select a choice from the context, or None to abstain."""
        ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

UtilityFunction = Callable[[Choice, DecisionContext], float]


class UtilityModel:
    """Rational choice: maximize a utility function.

    Optionally applies softmax temperature for stochastic selection.

    Args:
        utility_fn: Maps (choice, context) to a scalar utility.
        temperature: Softmax temperature. 0 = deterministic argmax.
    """

    def __init__(self, utility_fn: UtilityFunction, temperature: float = 0.0):
        self._utility_fn = utility_fn
        self.temperature = temperature

    def decide(self, context: DecisionContext, rng: random.Random) -> Choice | None:
        if not context.choices:
            return None

        utilities = [(c, self._utility_fn(c, context)) for c in context.choices]

        if self.temperature <= 0:
            return max(utilities, key=lambda x: x[1])[0]

        # Softmax selection
        max_u = max(u for _, u in utilities)
        weights = [math.exp((u - max_u) / self.temperature) for _, u in utilities]
        total = sum(weights)
        probs = [w / total for w in weights]
        return rng.choices([c for c, _ in utilities], weights=probs, k=1)[0]


RuleCondition = Callable[[DecisionContext], bool]


@dataclass
class Rule:
    """A condition-action pair for rule-based decision making.

    Attributes:
        condition: Predicate that tests whether this rule applies.
        action: The action to take if the condition is met.
        priority: Higher priority rules are evaluated first.
    """

    condition: RuleCondition
    action: str
    priority: int = 0


class RuleBasedModel:
    """Heuristic decision making via priority-ordered if-then rules.

    Rules are evaluated in descending priority order. The first rule
    whose condition is satisfied selects the matching choice.

    Args:
        rules: List of Rule instances.
        default_action: Action to fall back to if no rule fires.
    """

    def __init__(self, rules: list[Rule], default_action: str | None = None):
        self._rules = sorted(rules, key=lambda r: r.priority, reverse=True)
        self._default = default_action

    def decide(self, context: DecisionContext, rng: random.Random) -> Choice | None:
        for rule in self._rules:
            if rule.condition(context):
                for choice in context.choices:
                    if choice.action == rule.action:
                        return choice
                return None

        if self._default is not None:
            for choice in context.choices:
                if choice.action == self._default:
                    return choice
        return None


class BoundedRationalityModel:
    """Satisficing (Simon): accept the first option exceeding an aspiration level.

    Args:
        utility_fn: Maps (choice, context) to a scalar utility.
        aspiration: Minimum acceptable utility threshold.
    """

    def __init__(self, utility_fn: UtilityFunction, aspiration: float = 0.5):
        self._utility_fn = utility_fn
        self.aspiration = aspiration

    def decide(self, context: DecisionContext, rng: random.Random) -> Choice | None:
        if not context.choices:
            return None

        # Shuffle to avoid order bias
        shuffled = list(context.choices)
        rng.shuffle(shuffled)

        for choice in shuffled:
            if self._utility_fn(choice, context) >= self.aspiration:
                return choice

        # If nothing exceeds aspiration, pick the best available
        return max(context.choices, key=lambda c: self._utility_fn(c, context))


class SocialInfluenceModel:
    """Conformity-based decision making weighted by peer behavior.

    Looks at ``context.social_context["peer_actions"]`` (a dict mapping
    action name to count) and selects proportionally, weighted by the
    agent's agreeableness trait.

    Args:
        individual_fn: Fallback utility function for individual preference.
        conformity_weight: Base weight given to peer actions (0-1).
    """

    def __init__(
        self,
        individual_fn: UtilityFunction,
        conformity_weight: float = 0.5,
    ):
        self._individual_fn = individual_fn
        self._conformity_weight = conformity_weight

    def decide(self, context: DecisionContext, rng: random.Random) -> Choice | None:
        if not context.choices:
            return None

        peer_actions: dict[str, int] = context.social_context.get("peer_actions", {})
        agreeableness = context.traits.get("agreeableness")
        effective_conformity = self._conformity_weight * agreeableness

        # Combine individual utility with peer conformity
        scores: list[tuple[Choice, float]] = []
        total_peers = max(1, sum(peer_actions.values()))

        for choice in context.choices:
            individual = self._individual_fn(choice, context)
            peer_fraction = peer_actions.get(choice.action, 0) / total_peers
            combined = (1.0 - effective_conformity) * individual + effective_conformity * peer_fraction
            scores.append((choice, combined))

        # Weighted random selection
        total = sum(s for _, s in scores)
        if total <= 0:
            return rng.choice(context.choices)

        weights = [s / total for _, s in scores]
        return rng.choices([c for c, _ in scores], weights=weights, k=1)[0]


class CompositeModel:
    """Hybrid decision making via weighted voting across sub-models.

    Each sub-model votes for a choice; votes are weighted and tallied.

    Args:
        models: List of (DecisionModel, weight) pairs.
    """

    def __init__(self, models: list[tuple[DecisionModel, float]]):
        self._models = models

    def decide(self, context: DecisionContext, rng: random.Random) -> Choice | None:
        if not context.choices:
            return None

        votes: dict[str, float] = {}
        choice_map: dict[str, Choice] = {c.action: c for c in context.choices}

        for model, weight in self._models:
            pick = model.decide(context, rng)
            if pick is not None and pick.action in choice_map:
                votes[pick.action] = votes.get(pick.action, 0) + weight

        if not votes:
            return None

        winner = max(votes, key=lambda a: votes[a])
        return choice_map[winner]
