"""Behavioral simulation components for modeling human agents.

Provides personality traits, decision models, social networks, and
environment mediation for simulating individual and collective human
behavior such as product adoption, opinion dynamics, and policy impact.
"""

from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.decision import (
    BoundedRationalityModel,
    Choice,
    CompositeModel,
    DecisionContext,
    DecisionModel,
    Rule,
    RuleBasedModel,
    SocialInfluenceModel,
    UtilityModel,
)
from happysimulator.components.behavior.environment import Environment
from happysimulator.components.behavior.social_network import (
    Relationship,
    SocialGraph,
)
from happysimulator.components.behavior.state import (
    AgentState,
    Memory,
)
from happysimulator.components.behavior.traits import (
    NormalTraitDistribution,
    PersonalityTraits,
    TraitDistribution,
    TraitSet,
    UniformTraitDistribution,
)

BehaviorEnvironment = Environment
from happysimulator.components.behavior.influence import (
    BoundedConfidenceModel,
    DeGrootModel,
    InfluenceModel,
    VoterModel,
)
from happysimulator.components.behavior.population import (
    DemographicSegment,
    Population,
)
from happysimulator.components.behavior.stats import (
    AgentStats,
    EnvironmentStats,
    PopulationStats,
)
from happysimulator.components.behavior.stimulus import (
    broadcast_stimulus,
    influence_propagation,
    policy_announcement,
    price_change,
    targeted_stimulus,
)

__all__ = [
    # Agent
    "Agent",
    # State
    "AgentState",
    # Stats
    "AgentStats",
    "BehaviorEnvironment",
    "BoundedConfidenceModel",
    "BoundedRationalityModel",
    # Decision
    "Choice",
    "CompositeModel",
    "DeGrootModel",
    "DecisionContext",
    "DecisionModel",
    # Population
    "DemographicSegment",
    # Environment
    "Environment",
    "EnvironmentStats",
    # Influence
    "InfluenceModel",
    "Memory",
    "NormalTraitDistribution",
    "PersonalityTraits",
    "Population",
    "PopulationStats",
    # Social
    "Relationship",
    "Rule",
    "RuleBasedModel",
    "SocialGraph",
    "SocialInfluenceModel",
    "TraitDistribution",
    # Traits
    "TraitSet",
    "UniformTraitDistribution",
    "UtilityModel",
    "VoterModel",
    # Stimulus factories
    "broadcast_stimulus",
    "influence_propagation",
    "policy_announcement",
    "price_change",
    "targeted_stimulus",
]
