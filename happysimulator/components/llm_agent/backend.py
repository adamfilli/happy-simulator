"""LLM backend protocol and mock implementation.

The backend is a clean interface for swapping LLM providers.
All LLM calls go through this protocol — the agent never
imports a specific provider directly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends.

    Implementations must provide text completion and structured
    (JSON-schema-guided) completion, plus a model identifier.
    """

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The full prompt text.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text.
        """
        ...

    def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Generate a structured (JSON) completion.

        Args:
            prompt: The full prompt text.
            schema: JSON schema the response must conform to.
            temperature: Sampling temperature.

        Returns:
            Parsed response matching the schema.
        """
        ...

    @property
    def model_id(self) -> str:
        """Identifier for this model (e.g. 'claude-sonnet-4-6')."""
        ...


class MockLLMBackend:
    """Deterministic backend for testing.

    Matches prompt keywords to canned responses, or returns a default.
    Useful for unit and integration tests that need predictable behavior.

    Args:
        responses: Mapping of keyword -> response text. If a keyword
            appears in the prompt, the corresponding response is returned.
        default_action: Action returned when no keyword matches.
        default_reasoning: Reasoning text returned with the default action.
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        default_action: str = "wait",
        default_reasoning: str = "No strong reason to act.",
    ):
        self._responses = responses or {}
        self._default_action = default_action
        self._default_reasoning = default_reasoning
        self._call_count = 0
        self._prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        self._call_count += 1
        self._prompts.append(prompt)

        # Check keyword matches
        for keyword, response in self._responses.items():
            if keyword.lower() in prompt.lower():
                return response

        return self._default_action

    def complete_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        self._call_count += 1
        self._prompts.append(prompt)

        # Check keyword matches — try to parse as JSON
        for keyword, response in self._responses.items():
            if keyword.lower() in prompt.lower():
                try:
                    return json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    return {"action": response, "reasoning": response}

        return {
            "action": self._default_action,
            "reasoning": self._default_reasoning,
        }

    @property
    def model_id(self) -> str:
        return "mock-llm"

    @property
    def call_count(self) -> int:
        """Total number of LLM calls made."""
        return self._call_count

    @property
    def prompts(self) -> list[str]:
        """All prompts sent to this backend."""
        return list(self._prompts)

    def _parse_action_from_response(self, response: str) -> str:
        """Extract an action name from a response string.

        Tries JSON first, then looks for 'action:' pattern, then
        returns the first word.
        """
        # Try JSON
        try:
            data = json.loads(response)
            if isinstance(data, dict) and "action" in data:
                return data["action"]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try "action: <word>" pattern
        match = re.search(r"action:\s*(\w+)", response, re.IGNORECASE)
        if match:
            return match.group(1)

        # Return first word
        words = response.strip().split()
        return words[0] if words else self._default_action
