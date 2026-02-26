"""Tests for reasoning loop protocol and implementations."""

from happysimulator.components.llm_agent.backend import MockLLMBackend
from happysimulator.components.llm_agent.loops import (
    HeuristicLoop,
    ReactiveLoop,
    ReasoningLoop,
    _parse_action,
    select_loop,
)
from happysimulator.components.llm_agent.state import HumanState


class TestParseAction:
    def test_json_response(self):
        action, reasoning = _parse_action(
            '{"action": "eat", "reasoning": "hungry"}',
            ["eat", "wait"],
            "wait",
        )
        assert action == "eat"
        assert reasoning == "hungry"

    def test_first_line_match(self):
        action, reasoning = _parse_action(
            "eat\nBecause I'm hungry",
            ["eat", "wait"],
            "wait",
        )
        assert action == "eat"
        assert "hungry" in reasoning

    def test_mention_anywhere(self):
        action, reasoning = _parse_action(
            "I think I should eat something",
            ["eat", "wait"],
            "wait",
        )
        assert action == "eat"

    def test_fallback_to_default(self):
        action, reasoning = _parse_action(
            "I have no idea",
            ["eat", "wait"],
            "wait",
        )
        assert action == "wait"


class TestReactiveLoop:
    def test_returns_action_from_available(self):
        backend = MockLLMBackend(default_action="eat")
        loop = ReactiveLoop()
        action, reasoning = loop.run(backend, "prompt", ["eat", "wait"], 0.5)
        assert action == "eat"

    def test_satisfies_protocol(self):
        assert isinstance(ReactiveLoop(), ReasoningLoop)


class TestHeuristicLoop:
    def test_returns_action_with_reasoning(self):
        backend = MockLLMBackend(
            default_action="wait\nNeed to think about it."
        )
        loop = HeuristicLoop()
        action, reasoning = loop.run(backend, "prompt", ["wait", "act"], 0.8)
        assert action == "wait"

    def test_satisfies_protocol(self):
        assert isinstance(HeuristicLoop(), ReasoningLoop)


class TestSelectLoop:
    def test_reactive_when_depleted(self):
        state = HumanState()
        state.cognition.attention = 0.2
        state.cognition.decision_fatigue = 0.5
        # effective_capacity = 0.2 * 0.5 = 0.1
        loop = select_loop(state, stakes=0.5, involves_others=False)
        assert isinstance(loop, ReactiveLoop)

    def test_reactive_when_high_decision_fatigue(self):
        state = HumanState()
        state.cognition.attention = 0.8
        state.cognition.decision_fatigue = 0.85
        # effective_capacity = 0.8 * 0.15 = 0.12, and decision_fatigue > 0.8
        loop = select_loop(state, stakes=0.5, involves_others=False)
        assert isinstance(loop, ReactiveLoop)

    def test_heuristic_when_capable(self):
        state = HumanState()
        state.cognition.attention = 1.0
        state.cognition.decision_fatigue = 0.0
        loop = select_loop(state, stakes=0.5, involves_others=False)
        assert isinstance(loop, HeuristicLoop)

    def test_arousal_can_boost_capacity(self):
        state = HumanState()
        state.cognition.attention = 0.4
        state.cognition.decision_fatigue = 0.3
        # effective_capacity = 0.4 * 0.7 = 0.28 (below 0.3 → reactive without boost)
        state.emotion.arousal = 0.8
        # With arousal: 0.28 * 1.3 = 0.364 (above 0.3, may get reactive still at 0.364 < 0.5)
        loop = select_loop(state, stakes=0.5, involves_others=False)
        assert isinstance(loop, ReactiveLoop)

    def test_fully_rested_gets_heuristic(self):
        state = HumanState()  # All defaults: fresh
        loop = select_loop(state, stakes=0.5, involves_others=True)
        assert isinstance(loop, HeuristicLoop)
