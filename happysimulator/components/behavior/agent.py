"""Behavioral agent entity.

The Agent is the core actor in behavioral simulations. It receives
stimulus events, applies state decay, records memories, invokes a
decision model, and dispatches the chosen action through registered
action handlers.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

from happysimulator.components.behavior.traits import TraitSet, PersonalityTraits
from happysimulator.components.behavior.state import AgentState, Memory
from happysimulator.components.behavior.decision import (
    Choice,
    DecisionContext,
    DecisionModel,
)
from happysimulator.components.behavior.stats import AgentStats

logger = logging.getLogger(__name__)

ActionHandler = Callable[["Agent", Choice, Event], list[Event] | Event | None]


class Agent(Entity):
    """A behavioral agent that receives stimuli and makes decisions.

    The agent maintains personality traits, mutable internal state, and
    a pluggable decision model. Incoming events trigger the decision
    pipeline: state decay -> memory recording -> decision -> action.

    Action handlers are registered per action name and produce downstream
    events.

    Args:
        name: Unique identifier for this agent.
        traits: Personality trait vector.
        decision_model: Strategy for choosing actions.
        state: Initial internal state (defaults to fresh AgentState).
        seed: Random seed for deterministic behavior.
        heartbeat_interval: If > 0, schedule periodic self-maintenance events (seconds).
        action_delay: Simulated delay (seconds) between decision and action execution.
    """

    def __init__(
        self,
        name: str,
        traits: TraitSet | None = None,
        decision_model: DecisionModel | None = None,
        state: AgentState | None = None,
        seed: int | None = None,
        heartbeat_interval: float = 0.0,
        action_delay: float = 0.0,
    ):
        super().__init__(name)
        self.traits: TraitSet = traits or PersonalityTraits.big_five()
        self.decision_model = decision_model
        self.state = state or AgentState()
        self.action_delay = action_delay
        self.heartbeat_interval = heartbeat_interval
        self._rng = random.Random(seed)

        # Statistics (private counters → frozen snapshot via @property)
        self._events_received = 0
        self._decisions_made = 0
        self._actions_by_type: dict[str, int] = {}
        self._social_messages_received = 0
        self._action_handlers: dict[str, ActionHandler] = {}
        self._last_event_time: float | None = None
        self._heartbeat_scheduled = False

    def on_action(self, action: str, handler: ActionHandler) -> None:
        """Register a handler for the given action name.

        When the decision model selects this action, the handler is
        called with ``(agent, choice, event)`` and should return events
        (or None).
        """
        self._action_handlers[action] = handler

    # -----------------------------------------------------------------
    # Event dispatch
    # -----------------------------------------------------------------

    @property
    def stats(self) -> AgentStats:
        """Frozen snapshot of agent statistics."""
        return AgentStats(
            events_received=self._events_received,
            decisions_made=self._decisions_made,
            actions_by_type=dict(self._actions_by_type),
            social_messages_received=self._social_messages_received,
        )

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        self._events_received += 1
        current_time_s = self.now.to_seconds()

        # Apply passive decay since last event
        if self._last_event_time is not None:
            dt = current_time_s - self._last_event_time
            self.state.decay(dt)
        self._last_event_time = current_time_s

        # Route by event type
        if event.event_type.startswith("heartbeat::"):
            return self._handle_heartbeat(event)

        if event.event_type == "SocialMessage":
            return self._handle_social_message(event)

        # General stimulus -> decision pipeline
        return self._handle_stimulus(event)

    # -----------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------

    def _handle_heartbeat(self, event: Event) -> list[Event]:
        """Periodic self-maintenance: reschedule next heartbeat."""
        events: list[Event] = []
        if self.heartbeat_interval > 0:
            next_time = self.now + self.heartbeat_interval
            events.append(Event(
                time=next_time,
                event_type=f"heartbeat::{self.name}",
                target=self,
                daemon=True,
            ))
        return events

    def schedule_first_heartbeat(self, start_time: Instant) -> Event | None:
        """Create the initial heartbeat event (call before sim.run)."""
        if self.heartbeat_interval <= 0 or self._heartbeat_scheduled:
            return None
        self._heartbeat_scheduled = True
        return Event(
            time=start_time + self.heartbeat_interval,
            event_type=f"heartbeat::{self.name}",
            target=self,
            daemon=True,
        )

    # -----------------------------------------------------------------
    # Social messages
    # -----------------------------------------------------------------

    def _handle_social_message(self, event: Event) -> list[Event] | None:
        """Update beliefs/knowledge based on incoming social influence."""
        self._social_messages_received += 1
        metadata = event.context.get("metadata", {})
        topic = metadata.get("topic", "")
        opinion = metadata.get("opinion", 0.0)
        credibility = metadata.get("credibility", 0.5)
        knowledge_items: list[str] = metadata.get("knowledge", [])

        # Susceptibility = agreeableness * credibility
        agreeableness = self.traits.get("agreeableness")
        susceptibility = agreeableness * credibility

        # Update belief toward influencer opinion
        if topic and topic in self.state.beliefs:
            current = self.state.beliefs[topic]
            self.state.beliefs[topic] = current + susceptibility * (opinion - current)
        elif topic:
            # Adopt with susceptibility as weight from neutral
            self.state.beliefs[topic] = susceptibility * opinion

        # Absorb knowledge
        for item in knowledge_items:
            self.state.knowledge.add(item)

        return None

    # -----------------------------------------------------------------
    # Stimulus -> Decision pipeline
    # -----------------------------------------------------------------

    def _handle_stimulus(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        """Process a stimulus event through the decision pipeline."""
        metadata = event.context.get("metadata", {})

        # Record memory
        valence = metadata.get("valence", 0.0)
        source_name = metadata.get("source", "")
        self.state.add_memory(Memory(
            time=self.now.to_seconds(),
            event_type=event.event_type,
            source=source_name,
            valence=valence,
            details=dict(metadata),
        ))

        # Update mood based on valence
        if valence != 0:
            self.state.mood = max(0.0, min(1.0, self.state.mood + valence * 0.1))

        # Build decision context
        choices = self._extract_choices(event)
        if not choices or self.decision_model is None:
            return None

        dc = DecisionContext(
            traits=self.traits,
            state=self.state,
            choices=choices,
            stimulus=metadata,
            environment=metadata.get("environment", {}),
            social_context=metadata.get("social_context", {}),
        )

        # Decide
        chosen = self.decision_model.decide(dc, self._rng)
        if chosen is None:
            return None

        self._decisions_made += 1
        self._actions_by_type[chosen.action] = self._actions_by_type.get(chosen.action, 0) + 1

        # Execute action
        return self._execute_action(chosen, event)

    def _extract_choices(self, event: Event) -> list[Choice]:
        """Extract choices from event metadata."""
        raw = event.context.get("metadata", {}).get("choices", [])
        choices: list[Choice] = []
        for item in raw:
            if isinstance(item, Choice):
                choices.append(item)
            elif isinstance(item, dict):
                choices.append(Choice(
                    action=item.get("action", "unknown"),
                    context=item.get("context", {}),
                ))
            elif isinstance(item, str):
                choices.append(Choice(action=item))
        return choices

    def _execute_action(
        self, choice: Choice, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        handler = self._action_handlers.get(choice.action)
        if handler is None:
            logger.debug(
                "[%s] No handler for action '%s'", self.name, choice.action
            )
            return None

        if self.action_delay > 0:
            return self._delayed_action(handler, choice, event)

        result = handler(self, choice, event)
        return _normalize_handler_result(result)

    def _delayed_action(
        self, handler: ActionHandler, choice: Choice, event: Event
    ) -> Generator[float, None, list[Event]]:
        yield self.action_delay
        result = handler(self, choice, event)
        return _normalize_handler_result(result) or []


def _normalize_handler_result(
    result: list[Event] | Event | None,
) -> list[Event] | None:
    if result is None:
        return None
    if isinstance(result, Event):
        return [result]
    return result
