"""Behavioral simulation components for modeling human agents.

Provides personality traits, decision models, social networks, and
environment mediation for simulating individual and collective human
behavior such as product adoption, opinion dynamics, and policy impact.
"""

from happysimulator.components.behavior.traits import (
    TraitSet,
    PersonalityTraits,
    TraitDistribution,
    NormalTraitDistribution,
    UniformTraitDistribution,
)
from happysimulator.components.behavior.state import (
    AgentState,
    Memory,
)
from happysimulator.components.behavior.decision import (
    Choice,
    DecisionContext,
    DecisionModel,
    UtilityModel,
    Rule,
    RuleBasedModel,
    BoundedRationalityModel,
    SocialInfluenceModel,
    CompositeModel,
)
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.social_network import (
    Relationship,
    SocialGraph,
)
from happysimulator.components.behavior.environment import Environment
BehaviorEnvironment = Environment
from happysimulator.components.behavior.influence import (
    InfluenceModel,
    DeGrootModel,
    BoundedConfidenceModel,
    VoterModel,
)
from happysimulator.components.behavior.population import (
    DemographicSegment,
    Population,
)
from happysimulator.components.behavior.stimulus import (
    broadcast_stimulus,
    targeted_stimulus,
    price_change,
    policy_announcement,
    influence_propagation,
)
from happysimulator.components.behavior.stats import (
    AgentStats,
    PopulationStats,
    EnvironmentStats,
)

__all__ = [
    # Traits
    "TraitSet",
    "PersonalityTraits",
    "TraitDistribution",
    "NormalTraitDistribution",
    "UniformTraitDistribution",
    # State
    "AgentState",
    "Memory",
    # Decision
    "Choice",
    "DecisionContext",
    "DecisionModel",
    "UtilityModel",
    "Rule",
    "RuleBasedModel",
    "BoundedRationalityModel",
    "SocialInfluenceModel",
    "CompositeModel",
    # Agent
    "Agent",
    # Social
    "Relationship",
    "SocialGraph",
    # Environment
    "Environment",
    "BehaviorEnvironment",
    # Influence
    "InfluenceModel",
    "DeGrootModel",
    "BoundedConfidenceModel",
    "VoterModel",
    # Population
    "DemographicSegment",
    "Population",
    # Stimulus factories
    "broadcast_stimulus",
    "targeted_stimulus",
    "price_change",
    "policy_announcement",
    "influence_propagation",
    # Stats
    "AgentStats",
    "PopulationStats",
    "EnvironmentStats",
]
