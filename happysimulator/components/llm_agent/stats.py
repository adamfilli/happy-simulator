"""Statistics dataclass for LLM-powered human agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HumanAgentStats:
    """Frozen snapshot of agent statistics.

    Attributes:
        events_received: Total events handled.
        decisions_made: Times the LLM was invoked for a decision.
        llm_calls: Total LLM calls (may exceed decisions for multi-call loops).
        actions_by_type: Count of each chosen action.
        sanity_violations: Number of sanity check failures.
        conversations_participated: Distinct conversation IDs seen.
        memory_compressions: Times memory was compressed.
    """

    events_received: int = 0
    decisions_made: int = 0
    llm_calls: int = 0
    actions_by_type: dict[str, int] = field(default_factory=dict)
    sanity_violations: int = 0
    conversations_participated: int = 0
    memory_compressions: int = 0
