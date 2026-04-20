"""Decision audit trail for LLM-powered agents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DecisionTrace:
    """Frozen record of a single decision for debugging and analysis.

    Attributes:
        time_s: Simulation time when the decision was made.
        event_summary: What triggered the decision.
        loop_used: Reasoning loop type (e.g. "reactive", "heuristic").
        state_snapshot: Salient state dimensions at decision time.
        prompt_summary: First ~200 chars of the prompt sent to the LLM.
        raw_response: Full LLM response text.
        decision: The chosen action string.
        reasoning: Extracted reasoning for memory/analysis.
        llm_calls: Number of LLM calls made for this decision.
        model_used: Which model produced the decision.
        escalated: Whether loop escalation occurred.
        sanity_failures: Any sanity check violations encountered.
    """

    time_s: float
    event_summary: str
    loop_used: str
    state_snapshot: dict[str, object] = field(default_factory=dict)
    prompt_summary: str = ""
    raw_response: str = ""
    decision: str = ""
    reasoning: str = ""
    llm_calls: int = 1
    model_used: str = ""
    escalated: bool = False
    sanity_failures: tuple[str, ...] = ()
