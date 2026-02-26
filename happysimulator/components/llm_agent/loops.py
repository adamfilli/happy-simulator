"""Reasoning loop protocol and implementations.

Each loop is a strategy for how to use the LLM to make a decision.
They differ in number of LLM calls, context depth, and reasoning depth.
Phase 1 implements ReactiveLoop and HeuristicLoop.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from happysimulator.components.llm_agent.backend import LLMBackend
from happysimulator.components.llm_agent.state import HumanState


@runtime_checkable
class ReasoningLoop(Protocol):
    """Protocol for reasoning loop strategies.

    Each loop takes a prompt and available actions, runs one or more
    LLM calls, and returns the chosen action plus reasoning text.
    """

    def run(
        self,
        backend: LLMBackend,
        prompt: str,
        available_actions: list[str],
        quality: float,
    ) -> tuple[str, str]:
        """Execute the reasoning loop.

        Args:
            backend: LLM backend to call.
            prompt: Full decision prompt.
            available_actions: Valid action strings.
            quality: 0-1 quality parameter affecting depth/token budget.

        Returns:
            Tuple of (chosen_action, reasoning_text).
        """
        ...


def _parse_action(
    response: str,
    available_actions: list[str],
    default: str,
) -> tuple[str, str]:
    """Extract an action and reasoning from LLM response text.

    Tries several parsing strategies:
    1. JSON with "action" key
    2. First line matching an available action
    3. First available action mentioned anywhere in the response

    Args:
        response: Raw LLM response.
        available_actions: Valid action strings.
        default: Fallback action if parsing fails.

    Returns:
        Tuple of (action, reasoning).
    """
    # Try JSON
    try:
        data = json.loads(response)
        if isinstance(data, dict):
            action = data.get("action", "")
            reasoning = data.get("reasoning", "")
            if action in available_actions:
                return action, reasoning
    except (json.JSONDecodeError, TypeError):
        pass

    # Try first line match
    lines = response.strip().splitlines()
    if lines:
        first_line = lines[0].strip().lower()
        for action in available_actions:
            if action.lower() == first_line or first_line.startswith(action.lower()):
                reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
                return action, reasoning

    # Try mention anywhere
    response_lower = response.lower()
    for action in available_actions:
        if action.lower() in response_lower:
            return action, response.strip()

    return default, response.strip()


class ReactiveLoop:
    """Single fast call, minimal context. 1 LLM call.

    Used when cognitive capacity is severely depleted or
    for routine, low-stakes responses.
    """

    def run(
        self,
        backend: LLMBackend,
        prompt: str,
        available_actions: list[str],
        quality: float,
    ) -> tuple[str, str]:
        max_tokens = max(50, int(150 * quality))
        response = backend.complete(prompt, temperature=0.8, max_tokens=max_tokens)
        default = available_actions[0] if available_actions else "wait"
        return _parse_action(response, available_actions, default)


class HeuristicLoop:
    """Single call with full persona + situation. 1 LLM call.

    Used for familiar situations and moderate-stakes decisions
    where the agent has enough capacity for a considered response.
    """

    def run(
        self,
        backend: LLMBackend,
        prompt: str,
        available_actions: list[str],
        quality: float,
    ) -> tuple[str, str]:
        max_tokens = max(100, int(500 * quality))
        response = backend.complete(prompt, temperature=0.7, max_tokens=max_tokens)
        default = available_actions[0] if available_actions else "wait"
        return _parse_action(response, available_actions, default)


def select_loop(
    state: HumanState,
    stakes: float,
    involves_others: bool,
) -> ReasoningLoop:
    """Select reasoning loop based on cognitive capacity and stakes.

    Pure-code loop selection — no LLM calls. Phase 1 only returns
    ReactiveLoop or HeuristicLoop.

    Args:
        state: Current human state.
        stakes: 0-1 importance of the decision.
        involves_others: Whether other agents are involved.

    Returns:
        Appropriate ReasoningLoop instance.
    """
    effective_cap = state.cognition.effective_capacity

    # Arousal multiplier: high arousal can partially compensate fatigue
    arousal = state.emotion.arousal
    if arousal > 0.5:
        arousal_mult = 1.0 + (arousal - 0.5)  # up to 1.5x
    else:
        arousal_mult = 1.0
    effective_cap = min(1.0, effective_cap * arousal_mult)

    if effective_cap < 0.3:
        return ReactiveLoop()

    if effective_cap < 0.5 or state.cognition.decision_fatigue > 0.8:
        return ReactiveLoop()

    # Phase 1: everything above reactive threshold gets heuristic
    return HeuristicLoop()
