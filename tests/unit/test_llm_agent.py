"""Tests for the HumanAgent entity."""

from happysimulator.components.llm_agent.agent import HumanAgent
from happysimulator.components.llm_agent.backend import MockLLMBackend
from happysimulator.components.llm_agent.memory import EpisodicMemory
from happysimulator.components.llm_agent.sanity import NoDoubleEating
from happysimulator.components.llm_agent.state import HumanState
from happysimulator.components.llm_agent.stats import HumanAgentStats
from happysimulator.components.llm_agent.trace import DecisionTrace


class TestHumanAgentConstruction:
    def test_defaults(self):
        backend = MockLLMBackend()
        agent = HumanAgent("Alice", "a friendly person", backend)
        assert agent.name == "Alice"
        assert agent.persona == "a friendly person"
        assert isinstance(agent.state, HumanState)
        assert isinstance(agent.memory, EpisodicMemory)

    def test_custom_state(self):
        backend = MockLLMBackend()
        state = HumanState()
        state.physiology.hunger = 0.8
        agent = HumanAgent("Bob", "a person", backend, state=state)
        assert agent.state.physiology.hunger == 0.8

    def test_heartbeat_interval(self):
        backend = MockLLMBackend()
        agent = HumanAgent("Alice", "a person", backend, heartbeat_interval=5.0)
        assert agent.heartbeat_interval == 5.0


class TestHumanAgentStats:
    def test_initial_stats(self):
        backend = MockLLMBackend()
        agent = HumanAgent("Alice", "a person", backend)
        stats = agent.stats
        assert isinstance(stats, HumanAgentStats)
        assert stats.events_received == 0
        assert stats.decisions_made == 0
        assert stats.llm_calls == 0

    def test_stats_frozen(self):
        stats = HumanAgentStats(events_received=5)
        assert stats.events_received == 5


class TestActionHandlerRegistration:
    def test_register_handler(self):
        backend = MockLLMBackend()
        agent = HumanAgent("Alice", "a person", backend)

        called = []
        agent.on_action("greet", lambda a, act, e: called.append(act))
        assert "greet" in agent._action_handlers


class TestDecisionTrace:
    def test_trace_is_frozen(self):
        trace = DecisionTrace(
            time_s=1.0,
            event_summary="test event",
            loop_used="ReactiveLoop",
        )
        assert trace.time_s == 1.0
        assert trace.loop_used == "ReactiveLoop"
        assert trace.escalated is False
        assert trace.sanity_failures == ()

    def test_trace_with_all_fields(self):
        trace = DecisionTrace(
            time_s=1.0,
            event_summary="bribe",
            loop_used="HeuristicLoop",
            state_snapshot={"hunger": 0.5},
            prompt_summary="You are...",
            raw_response="stall",
            decision="stall",
            reasoning="buying time",
            llm_calls=1,
            model_used="mock-llm",
            escalated=False,
            sanity_failures=("too tired",),
        )
        assert trace.decision == "stall"
        assert trace.sanity_failures == ("too tired",)
