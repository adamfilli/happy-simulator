"""State-to-prompt translation with salience thresholds.

Bridges numerical state and LLM input. Each state dimension has a
salience threshold — below it, the dimension is omitted from the
prompt. Above it, it appears with intensity-appropriate language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from happysimulator.components.llm_agent.memory import EpisodicMemory
from happysimulator.components.llm_agent.state import HumanState


@dataclass(frozen=True)
class SalienceThreshold:
    """Salience configuration for a single state dimension.

    Attributes:
        dimension: State dimension key (e.g. "hunger", "anger").
        threshold: Value below which the dimension is omitted.
        descriptions: Level -> text mappings.
            Keys should be "low", "medium", "high" mapping to
            intensity-appropriate natural language.
    """

    dimension: str
    threshold: float
    descriptions: dict[str, str]


# Default salience thresholds from the design doc
_DEFAULT_THRESHOLDS: list[SalienceThreshold] = [
    SalienceThreshold(
        "hunger",
        0.4,
        {
            "low": "You're getting hungry.",
            "medium": "You're quite hungry and finding it hard to concentrate.",
            "high": "You're starving. Finding food is your top priority.",
        },
    ),
    SalienceThreshold(
        "fatigue",
        0.5,
        {
            "low": "You're a bit tired.",
            "medium": "You're quite fatigued and your thinking is slower.",
            "high": "You're exhausted and struggling to stay focused.",
        },
    ),
    SalienceThreshold(
        "arousal",
        0.5,
        {
            "low": "You feel alert and engaged.",
            "medium": "You're quite agitated.",
            "high": "Your heart is pounding and adrenaline is surging.",
        },
    ),
    SalienceThreshold(
        "anger",
        0.3,
        {
            "low": "You feel a bit irritated.",
            "medium": "You're angry and it's affecting your judgment.",
            "high": "You're furious and struggling to think clearly.",
        },
    ),
    SalienceThreshold(
        "anxiety",
        0.3,
        {
            "low": "You feel somewhat uneasy.",
            "medium": "You're anxious and can't stop worrying.",
            "high": "You're overwhelmed with anxiety.",
        },
    ),
    SalienceThreshold(
        "pain",
        0.1,
        {
            "low": "You have a mild ache.",
            "medium": "You're in noticeable pain.",
            "high": "You're in severe pain that dominates your attention.",
        },
    ),
    SalienceThreshold(
        "decision_fatigue",
        0.6,
        {
            "low": "You're getting tired of making decisions.",
            "medium": "You're mentally drained from too many decisions.",
            "high": "You can barely bring yourself to decide anything.",
        },
    ),
]


def _intensity_level(value: float, threshold: float) -> str:
    """Map a value above threshold to low/medium/high intensity."""
    # Normalize remaining range [threshold, 1.0] to [0, 1]
    span = 1.0 - threshold
    if span <= 0:
        return "high"
    normalized = (value - threshold) / span
    if normalized < 0.33:
        return "low"
    if normalized < 0.67:
        return "medium"
    return "high"


class PromptBuilder:
    """Translates agent state into natural language prompts.

    Args:
        persona: Natural language persona description.
        thresholds: Salience thresholds for state dimensions.
            Defaults to the built-in thresholds if not provided.
    """

    def __init__(
        self,
        persona: str,
        thresholds: list[SalienceThreshold] | None = None,
    ):
        self.persona = persona
        self.thresholds = thresholds or list(_DEFAULT_THRESHOLDS)

    def describe_state(self, state: HumanState) -> str:
        """Describe only salient state dimensions in natural language.

        Dimensions below their salience threshold are omitted.
        """
        snapshot = state.snapshot()
        parts: list[str] = []

        for threshold in self.thresholds:
            value = snapshot.get(threshold.dimension)
            if value is None:
                continue
            if value < threshold.threshold:
                continue
            level = _intensity_level(value, threshold.threshold)
            text = threshold.descriptions.get(level)
            if text:
                parts.append(text)

        return " ".join(parts)

    def build_decision_prompt(
        self,
        state: HumanState,
        event_description: str,
        memory: EpisodicMemory,
        available_actions: list[str],
        social_context: dict[str, Any] | None = None,
    ) -> str:
        """Assemble the full decision prompt.

        Structure: persona + salient state + memory + situation + actions.

        Args:
            state: Current human state.
            event_description: What just happened.
            memory: Agent's episodic memory.
            available_actions: Actions the agent can choose from.
            social_context: Optional social relationship info.

        Returns:
            Complete prompt string for the LLM.
        """
        sections: list[str] = []

        # Persona
        sections.append(f"You are {self.persona}.")

        # Salient state
        state_desc = self.describe_state(state)
        if state_desc:
            sections.append(f"Current state: {state_desc}")

        # Memory
        memory_text = memory.format_for_prompt()
        if memory_text:
            sections.append(memory_text)

        # Social context
        if social_context:
            rel_parts: list[str] = []
            for name, info in social_context.items():
                if isinstance(info, dict):
                    trust = info.get("trust", 0.5)
                    liking = info.get("liking", 0.0)
                    desc = f"{name} (trust: {trust:.1f}, liking: {liking:.1f})"
                else:
                    desc = f"{name}: {info}"
                rel_parts.append(desc)
            if rel_parts:
                sections.append("Relationships: " + "; ".join(rel_parts))

        # Situation
        sections.append(f"Situation: {event_description}")

        # Available actions
        actions_str = ", ".join(available_actions)
        sections.append(
            f"Available actions: {actions_str}\n\n"
            f"Choose one action and explain your reasoning briefly. "
            f"Respond with the action name first, then your reasoning."
        )

        return "\n\n".join(sections)
