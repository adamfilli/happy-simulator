"""Sanity checks — invariant assertions on agent actions.

Checks run after the LLM decision but before execution. They catch
obviously impossible actions (not behavioral realism validation).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from happysimulator.components.llm_agent.state import HumanState


@runtime_checkable
class SanityCheck(Protocol):
    """Protocol for action invariant checks.

    Returns an error message if the invariant is violated, None if OK.
    """

    def check(
        self,
        state: HumanState,
        action: str,
        context: dict[str, Any],
    ) -> str | None: ...


class NoDoubleEating:
    """Can't eat two large meals within 30 simulated minutes.

    Tracks the last eat time via context["last_eat_time"].
    """

    def check(
        self,
        state: HumanState,
        action: str,
        context: dict[str, Any],
    ) -> str | None:
        if action != "eat":
            return None

        current_time = context.get("current_time", 0.0)
        last_eat_time = context.get("last_eat_time")

        if last_eat_time is not None:
            elapsed = current_time - last_eat_time
            if elapsed < 1800.0:  # 30 minutes
                return (
                    f"Cannot eat again so soon "
                    f"(only {elapsed:.0f}s since last meal, "
                    f"minimum 1800s)."
                )
        return None


class NoSleepWhileSleeping:
    """Can't start sleeping if already sleeping.

    Checks context["is_sleeping"] flag.
    """

    def check(
        self,
        state: HumanState,
        action: str,
        context: dict[str, Any],
    ) -> str | None:
        if action != "sleep":
            return None

        if context.get("is_sleeping", False):
            return "Cannot start sleeping while already sleeping."
        return None


def run_checks(
    checks: list[SanityCheck],
    state: HumanState,
    action: str,
    context: dict[str, Any],
) -> list[str]:
    """Run all sanity checks and return list of violation messages.

    Args:
        checks: List of SanityCheck implementations to run.
        state: Current human state.
        action: Proposed action.
        context: Action context (timing info, flags, etc.).

    Returns:
        List of failure messages (empty if all checks pass).
    """
    failures: list[str] = []
    for check in checks:
        result = check.check(state, action, context)
        if result is not None:
            failures.append(result)
    return failures
