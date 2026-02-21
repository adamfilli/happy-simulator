"""LLM-powered human agent entity.

The main class that composes state, memory, reasoning loops, and LLM
backend into a simulation Entity. Follows the Agent pattern from the
behavior module but with richer state and LLM-based decision-making.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

from happysimulator.components.llm_agent.backend import LLMBackend
from happysimulator.components.llm_agent.loops import (
    ReasoningLoop,
    select_loop,
)
from happysimulator.components.llm_agent.memory import EpisodicMemory, MemoryEntry
from happysimulator.components.llm_agent.prompt import PromptBuilder
from happysimulator.components.llm_agent.sanity import SanityCheck, run_checks
from happysimulator.components.llm_agent.state import HumanState
from happysimulator.components.llm_agent.stats import HumanAgentStats
from happysimulator.components.llm_agent.trace import DecisionTrace
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)

ActionHandler = Callable[["HumanAgent", str, Event], list[Event] | Event | None]
"""Signature: (agent, action_string, triggering_event) -> events."""

# Default base stakes for common event types
_DEFAULT_STAKES: dict[str, float] = {
    "Talk": 0.3,
    "Stimulus": 0.5,
    "heartbeat": 0.0,
}

# Maximum sanity re-tries before giving up
_MAX_SANITY_RETRIES = 2


class HumanAgent(Entity):
    """LLM-powered human agent that makes decisions using language models.

    Composes: HumanState (pure-code evolution), EpisodicMemory,
    ReasoningLoop selection, LLMBackend, PromptBuilder, and SanityChecks.

    The LLM is only called for decisions. State ticks, emotional decay,
    and mechanical bookkeeping are pure code.

    Args:
        name: Unique agent identifier.
        persona: Natural language persona description.
        backend: LLM backend for decision-making.
        state: Initial human state (defaults to fresh HumanState).
        memory: Episodic memory (defaults to fresh EpisodicMemory).
        prompt_builder: State-to-prompt translator (auto-created from persona).
        sanity_checks: Invariant checks run after each decision.
        action_handlers: Pre-registered action handlers.
        heartbeat_interval: Physio tick interval in seconds (0 = disabled).
        action_delay: Simulated think time in seconds.
        stakes_config: Event type -> base stakes level.
    """

    def __init__(
        self,
        name: str,
        persona: str,
        backend: LLMBackend,
        *,
        state: HumanState | None = None,
        memory: EpisodicMemory | None = None,
        prompt_builder: PromptBuilder | None = None,
        sanity_checks: list[SanityCheck] | None = None,
        action_handlers: dict[str, ActionHandler] | None = None,
        heartbeat_interval: float = 0.0,
        action_delay: float = 0.1,
        stakes_config: dict[str, float] | None = None,
    ):
        super().__init__(name)
        self.persona = persona
        self.backend = backend
        self.state = state or HumanState()
        self.memory = memory or EpisodicMemory()
        self.prompt_builder = prompt_builder or PromptBuilder(persona)
        self.sanity_checks = sanity_checks or []
        self.heartbeat_interval = heartbeat_interval
        self.action_delay = action_delay
        self._stakes_config = stakes_config or dict(_DEFAULT_STAKES)

        self._action_handlers: dict[str, ActionHandler] = dict(action_handlers or {})
        self._traces: list[DecisionTrace] = []
        self._conversation_ids: set[str] = set()
        self._heartbeat_scheduled = False

        # Mutable counters (snapshot via stats property)
        self._events_received = 0
        self._decisions_made = 0
        self._llm_calls = 0
        self._actions_by_type: dict[str, int] = {}
        self._sanity_violations = 0
        self._memory_compressions = 0
        self._last_event_time: float | None = None

        # Sanity check context (persists across events)
        self._sanity_context: dict[str, Any] = {
            "is_sleeping": False,
            "last_eat_time": None,
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def on_action(self, action: str, handler: ActionHandler) -> None:
        """Register a handler for a specific action string.

        When the LLM chooses this action, the handler is called with
        ``(agent, action, event)`` and should return events (or None).
        """
        self._action_handlers[action] = handler

    @property
    def stats(self) -> HumanAgentStats:
        """Frozen snapshot of agent statistics."""
        return HumanAgentStats(
            events_received=self._events_received,
            decisions_made=self._decisions_made,
            llm_calls=self._llm_calls,
            actions_by_type=dict(self._actions_by_type),
            sanity_violations=self._sanity_violations,
            conversations_participated=len(self._conversation_ids),
            memory_compressions=self._memory_compressions,
        )

    @property
    def traces(self) -> list[DecisionTrace]:
        """List of all decision traces for analysis."""
        return list(self._traces)

    def downstream_entities(self) -> list[Entity]:
        """Return entities referenced by action handlers."""
        entities: list[Entity] = []
        for handler in self._action_handlers.values():
            # Check if handler has a bound entity reference
            target = getattr(handler, "__self__", None)
            if isinstance(target, Entity):
                entities.append(target)
        return entities

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
    # Event dispatch
    # -----------------------------------------------------------------

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        self._events_received += 1
        current_time_s = self.now.to_seconds()

        # 1. Tick state (dt since last event)
        if self._last_event_time is not None:
            dt = current_time_s - self._last_event_time
            time_of_day = (current_time_s / 3600.0) % 24.0
            self.state.tick(dt, time_of_day)
        self._last_event_time = current_time_s

        # Route by event type
        if event.event_type.startswith("heartbeat::"):
            return self._handle_heartbeat(event)

        if event.event_type == "Talk":
            return self._handle_talk(event)

        # Default: stimulus requiring a decision
        return self._handle_stimulus(event)

    # -----------------------------------------------------------------
    # Heartbeat
    # -----------------------------------------------------------------

    def _handle_heartbeat(self, event: Event) -> list[Event]:
        """Periodic self-maintenance: just reschedule."""
        events: list[Event] = []
        if self.heartbeat_interval > 0:
            next_time = self.now + self.heartbeat_interval
            events.append(
                Event(
                    time=next_time,
                    event_type=f"heartbeat::{self.name}",
                    target=self,
                    daemon=True,
                )
            )
        return events

    # -----------------------------------------------------------------
    # Talk (conversation)
    # -----------------------------------------------------------------

    def _handle_talk(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        message = event.context.get("message", "")
        speaker = event.context.get("speaker", "someone")
        conversation_id = event.context.get("conversation_id", "")
        tone = event.context.get("tone", "neutral")

        if conversation_id:
            self._conversation_ids.add(conversation_id)

        # Update social state from conversation
        relation = self.state.get_relation(speaker)
        relation.familiarity = min(1.0, relation.familiarity + 0.05)

        # Emotional impact from tone
        valence = {"positive": 0.1, "negative": -0.1, "neutral": 0.0}.get(tone, 0.0)
        self.state.emotion.apply_event_impact(valence, arousal_delta=0.05)

        # Add to memory
        self.memory.add(
            MemoryEntry(
                time=self.now.to_seconds(),
                summary=f'{speaker} said: "{message}"',
                valence=valence,
                participants=(speaker,),
                tags=("conversation", speaker),
            )
        )

        # Build available actions
        available_actions = ["respond", "stay_silent", "end_conversation"]
        available_actions.extend(
            a for a in self._action_handlers if a not in available_actions
        )

        # Make decision
        event_desc = f'{speaker} says to you: "{message}" (tone: {tone})'
        social_context = {
            speaker: {
                "trust": relation.trust,
                "liking": relation.liking,
                "familiarity": relation.familiarity,
            }
        }

        return self._decide_and_act(
            event, event_desc, available_actions, social_context
        )

    # -----------------------------------------------------------------
    # Generic stimulus
    # -----------------------------------------------------------------

    def _handle_stimulus(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        description = event.context.get("description", event.event_type)
        valence = event.context.get("metadata", {}).get("valence", 0.0)
        arousal_delta = event.context.get("metadata", {}).get("arousal", 0.1)

        # Update emotional state
        self.state.emotion.apply_event_impact(valence, arousal_delta)

        # Add to memory
        self.memory.add(
            MemoryEntry(
                time=self.now.to_seconds(),
                summary=f"Event: {description}",
                valence=valence,
                tags=(event.event_type,),
            )
        )

        # Build available actions
        choices = event.context.get("metadata", {}).get("choices", [])
        if choices:
            available_actions = list(choices)
        else:
            available_actions = list(self._action_handlers.keys()) or ["wait"]

        return self._decide_and_act(event, description, available_actions)

    # -----------------------------------------------------------------
    # Core decision pipeline
    # -----------------------------------------------------------------

    def _decide_and_act(
        self,
        event: Event,
        event_description: str,
        available_actions: list[str],
        social_context: dict[str, Any] | None = None,
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        """Run the full decision pipeline: select loop, prompt, decide, act."""
        current_time_s = self.now.to_seconds()

        # Compute stakes
        base_stakes = self._stakes_config.get(event.event_type, 0.5)
        stakes = min(1.0, base_stakes)

        # Select reasoning loop
        involves_others = social_context is not None and len(social_context) > 0
        loop = select_loop(self.state, stakes, involves_others)
        loop_name = type(loop).__name__

        # Build prompt
        prompt = self.prompt_builder.build_decision_prompt(
            self.state,
            event_description,
            self.memory,
            available_actions,
            social_context,
        )

        # Quality based on effective capacity
        quality = self.state.cognition.effective_capacity

        if self.action_delay > 0:
            return self._delayed_decide(
                event,
                event_description,
                available_actions,
                loop,
                loop_name,
                prompt,
                quality,
                current_time_s,
            )

        return self._immediate_decide(
            event,
            event_description,
            available_actions,
            loop,
            loop_name,
            prompt,
            quality,
            current_time_s,
        )

    def _delayed_decide(
        self,
        event: Event,
        event_description: str,
        available_actions: list[str],
        loop: ReasoningLoop,
        loop_name: str,
        prompt: str,
        quality: float,
        current_time_s: float,
    ) -> Generator[float, None, list[Event]]:
        """Decision with simulated think-time delay."""
        yield self.action_delay

        action, reasoning = self._run_loop_with_sanity(
            loop, loop_name, prompt, available_actions, quality, current_time_s
        )

        self._record_trace(
            current_time_s, event_description, loop_name, prompt,
            reasoning, action, reasoning,
        )

        return self._execute_action(action, event) or []

    def _immediate_decide(
        self,
        event: Event,
        event_description: str,
        available_actions: list[str],
        loop: ReasoningLoop,
        loop_name: str,
        prompt: str,
        quality: float,
        current_time_s: float,
    ) -> list[Event] | None:
        """Decision without delay."""
        action, reasoning = self._run_loop_with_sanity(
            loop, loop_name, prompt, available_actions, quality, current_time_s
        )

        self._record_trace(
            current_time_s, event_description, loop_name, prompt,
            reasoning, action, reasoning,
        )

        return self._execute_action(action, event)

    def _run_loop_with_sanity(
        self,
        loop: ReasoningLoop,
        loop_name: str,
        prompt: str,
        available_actions: list[str],
        quality: float,
        current_time_s: float,
    ) -> tuple[str, str]:
        """Run the reasoning loop with sanity check retries."""
        all_failures: list[str] = []

        for attempt in range(_MAX_SANITY_RETRIES + 1):
            action, reasoning = loop.run(
                self.backend, prompt, available_actions, quality
            )
            self._llm_calls += 1
            self._decisions_made += 1

            # Run sanity checks
            self._sanity_context["current_time"] = current_time_s
            failures = run_checks(
                self.sanity_checks, self.state, action, self._sanity_context
            )

            if not failures:
                # Update post-decision state
                self.state.cognition.record_decision()
                self._actions_by_type[action] = (
                    self._actions_by_type.get(action, 0) + 1
                )

                # Update sanity context for future checks
                if action == "eat":
                    self._sanity_context["last_eat_time"] = current_time_s
                elif action == "sleep":
                    self._sanity_context["is_sleeping"] = True
                elif action == "wake":
                    self._sanity_context["is_sleeping"] = False

                return action, reasoning

            # Sanity failed
            all_failures.extend(failures)
            self._sanity_violations += len(failures)
            logger.debug(
                "[%s] Sanity check failed (attempt %d): %s",
                self.name,
                attempt + 1,
                failures,
            )

            # Remove the failed action and retry
            if action in available_actions and len(available_actions) > 1:
                available_actions = [a for a in available_actions if a != action]
                prompt += (
                    f"\n\nNote: '{action}' is not available right now. "
                    f"Choose from: {', '.join(available_actions)}"
                )

        # Exhausted retries — use first available action
        fallback = available_actions[0] if available_actions else "wait"
        self.state.cognition.record_decision()
        self._actions_by_type[fallback] = (
            self._actions_by_type.get(fallback, 0) + 1
        )
        return fallback, f"Fallback after sanity failures: {all_failures}"

    def _record_trace(
        self,
        time_s: float,
        event_summary: str,
        loop_name: str,
        prompt: str,
        raw_response: str,
        decision: str,
        reasoning: str,
    ) -> None:
        """Store a decision trace."""
        self._traces.append(
            DecisionTrace(
                time_s=time_s,
                event_summary=event_summary,
                loop_used=loop_name,
                state_snapshot=self.state.snapshot(),
                prompt_summary=prompt[:200],
                raw_response=raw_response,
                decision=decision,
                reasoning=reasoning,
                llm_calls=1,
                model_used=self.backend.model_id,
            )
        )

    def _execute_action(
        self, action: str, event: Event
    ) -> list[Event] | None:
        """Call the registered action handler."""
        handler = self._action_handlers.get(action)
        if handler is None:
            logger.debug("[%s] No handler for action '%s'", self.name, action)
            return None

        result = handler(self, action, event)
        if result is None:
            return None
        if isinstance(result, Event):
            return [result]
        return result
