"""Stimulus event factory functions.

Convenience constructors for creating events that target an Environment
entity. Follows the pattern of network condition factories
(``local_network()``, ``datacenter_network()``, etc.).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Sequence

logger = logging.getLogger(__name__)

from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.components.behavior.decision import Choice

if TYPE_CHECKING:
    from happysimulator.components.behavior.environment import Environment


def broadcast_stimulus(
    time: Instant | float,
    environment: Environment,
    stimulus_type: str,
    choices: list[Choice | str | dict] | None = None,
    **metadata: Any,
) -> Event:
    """Create a broadcast stimulus event targeting an Environment.

    The Environment will forward this as individual Stimulus events to
    all registered agents.

    Args:
        time: When the stimulus occurs (Instant or float seconds).
        environment: The Environment entity to receive the broadcast.
        stimulus_type: Label for the stimulus (becomes inner event_type).
        choices: Available actions for agents (Choice, str, or dict).
        **metadata: Additional context passed through to agents.
    """
    t = _to_instant(time)
    ctx_meta: dict[str, Any] = {
        "stimulus_type": stimulus_type,
        "choices": _normalize_choices(choices),
        **metadata,
    }
    return Event(
        time=t,
        event_type="BroadcastStimulus",
        target=environment,
        context={"metadata": ctx_meta},
    )


def targeted_stimulus(
    time: Instant | float,
    environment: Environment,
    targets: Sequence[str],
    stimulus_type: str,
    choices: list[Choice | str | dict] | None = None,
    **metadata: Any,
) -> Event:
    """Create a targeted stimulus event for specific agents.

    Args:
        time: When the stimulus occurs.
        environment: The Environment entity.
        targets: Agent names to receive the stimulus.
        stimulus_type: Label for the stimulus.
        choices: Available actions for agents.
        **metadata: Additional context.
    """
    t = _to_instant(time)
    ctx_meta: dict[str, Any] = {
        "stimulus_type": stimulus_type,
        "targets": list(targets),
        "choices": _normalize_choices(choices),
        **metadata,
    }
    return Event(
        time=t,
        event_type="TargetedStimulus",
        target=environment,
        context={"metadata": ctx_meta},
    )


def price_change(
    time: Instant | float,
    environment: Environment,
    product: str,
    old_price: float,
    new_price: float,
) -> Event:
    """Create a price-change broadcast with pre-built buy/wait/switch choices.

    Args:
        time: When the price change takes effect.
        environment: The Environment entity.
        product: Product identifier.
        old_price: Previous price.
        new_price: New price.
    """
    choices = [
        Choice(action="buy", context={"product": product, "price": new_price}),
        Choice(action="wait", context={"product": product}),
        Choice(action="switch", context={"product": product}),
    ]
    valence = 0.3 if new_price < old_price else -0.3
    return broadcast_stimulus(
        time,
        environment,
        stimulus_type="PriceChange",
        choices=choices,
        product=product,
        old_price=old_price,
        new_price=new_price,
        valence=valence,
    )


def policy_announcement(
    time: Instant | float,
    environment: Environment,
    policy: str,
    description: str,
    valence: float = 0.0,
) -> Event:
    """Create a policy announcement with accept/protest/ignore choices.

    Args:
        time: When the announcement occurs.
        environment: The Environment entity.
        policy: Policy identifier.
        description: Human-readable description.
        valence: Positive or negative framing (-1 to 1).
    """
    choices = [
        Choice(action="accept", context={"policy": policy}),
        Choice(action="protest", context={"policy": policy}),
        Choice(action="ignore", context={"policy": policy}),
    ]
    return broadcast_stimulus(
        time,
        environment,
        stimulus_type="PolicyAnnouncement",
        choices=choices,
        policy=policy,
        description=description,
        valence=valence,
    )


def influence_propagation(
    time: Instant | float,
    environment: Environment,
    topic: str,
) -> Event:
    """Trigger one round of social influence propagation.

    Args:
        time: When the influence round occurs.
        environment: The Environment entity.
        topic: The belief topic to propagate.
    """
    t = _to_instant(time)
    return Event(
        time=t,
        event_type="InfluencePropagation",
        target=environment,
        context={"metadata": {"topic": topic}},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_instant(time: Instant | float) -> Instant:
    if isinstance(time, Instant):
        return time
    return Instant.from_seconds(time)


def _normalize_choices(
    choices: list[Choice | str | dict] | None,
) -> list[Choice]:
    if choices is None:
        return []
    result: list[Choice] = []
    for c in choices:
        if isinstance(c, Choice):
            result.append(c)
        elif isinstance(c, str):
            result.append(Choice(action=c))
        elif isinstance(c, dict):
            result.append(Choice(action=c.get("action", "unknown"), context=c.get("context", {})))
    return result
