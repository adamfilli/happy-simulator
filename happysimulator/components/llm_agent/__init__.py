"""LLM-powered human agents for behavioral simulation.

Agents use language models as decision engines while maintaining
rich state (physiological, emotional, cognitive, social) that
evolves in pure code. The LLM is only called for decisions.
"""

from happysimulator.components.llm_agent.agent import ActionHandler, HumanAgent
from happysimulator.components.llm_agent.backend import LLMBackend, MockLLMBackend
from happysimulator.components.llm_agent.loops import (
    HeuristicLoop,
    ReactiveLoop,
    ReasoningLoop,
    select_loop,
)
from happysimulator.components.llm_agent.memory import EpisodicMemory, MemoryEntry
from happysimulator.components.llm_agent.prompt import (
    PromptBuilder,
    SalienceThreshold,
)
from happysimulator.components.llm_agent.sanity import (
    NoDoubleEating,
    NoSleepWhileSleeping,
    SanityCheck,
    run_checks,
)
from happysimulator.components.llm_agent.state import (
    CognitiveState,
    EmotionalState,
    HumanState,
    PhysiologicalState,
    SocialRelation,
)
from happysimulator.components.llm_agent.stats import HumanAgentStats
from happysimulator.components.llm_agent.trace import DecisionTrace

__all__ = [
    "ActionHandler",
    "CognitiveState",
    "DecisionTrace",
    "EmotionalState",
    "EpisodicMemory",
    "HeuristicLoop",
    "HumanAgent",
    "HumanAgentStats",
    "HumanState",
    "LLMBackend",
    "MemoryEntry",
    "MockLLMBackend",
    "NoDoubleEating",
    "NoSleepWhileSleeping",
    "PhysiologicalState",
    "PromptBuilder",
    "ReactiveLoop",
    "ReasoningLoop",
    "SalienceThreshold",
    "SanityCheck",
    "SocialRelation",
    "run_checks",
    "select_loop",
]
