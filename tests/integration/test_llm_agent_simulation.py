"""Integration tests for LLM-powered human agents in simulations."""

from happysimulator.components.llm_agent.agent import HumanAgent
from happysimulator.components.llm_agent.backend import MockLLMBackend
from happysimulator.components.llm_agent.loops import HeuristicLoop, ReactiveLoop
from happysimulator.components.llm_agent.sanity import NoDoubleEating
from happysimulator.components.llm_agent.state import HumanState
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class Collector(Entity):
    """Simple entity that collects events for assertions."""

    def __init__(self):
        super().__init__("collector")
        self.events: list[Event] = []

    def handle_event(self, event):
        self.events.append(event)
        return None


class TestSingleAgentStimulus:
    def test_agent_processes_stimulus_events(self):
        """A single agent processes stimulus events in a simulation."""
        backend = MockLLMBackend(default_action="wait")
        agent = HumanAgent(
            "Alice", "a cautious person", backend, action_delay=0.0
        )
        agent.on_action("wait", lambda a, act, e: None)

        sim = Simulation(entities=[agent], duration=5.0)

        # Schedule two stimulus events
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "A loud noise"},
            )
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(3.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "A bright light"},
            )
        )

        sim.run()

        stats = agent.stats
        assert stats.events_received == 2
        assert stats.decisions_made == 2
        assert stats.llm_calls == 2

    def test_agent_action_produces_downstream_events(self):
        """Agent action handlers can produce downstream events."""
        collector = Collector()
        backend = MockLLMBackend(default_action="greet")
        agent = HumanAgent(
            "Alice", "a friendly person", backend, action_delay=0.0
        )

        def greet_handler(a, act, event):
            return Event(
                time=a.now,
                event_type="Greeting",
                target=collector,
                context={"from": a.name},
            )

        agent.on_action("greet", greet_handler)

        sim = Simulation(entities=[agent, collector], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "Someone arrived"},
            )
        )

        sim.run()

        assert len(collector.events) == 1
        assert collector.events[0].event_type == "Greeting"


class TestConversation:
    def test_two_agents_converse(self):
        """Two agents exchange Talk events (turn-taking)."""
        backend_a = MockLLMBackend(default_action="respond")
        backend_b = MockLLMBackend(default_action="respond")

        agent_a = HumanAgent("Alice", "a talkative person", backend_a, action_delay=0.0)
        agent_b = HumanAgent("Bob", "a quiet person", backend_b, action_delay=0.0)

        # Track turns
        turns: list[str] = []

        def respond_a(agent, act, event):
            turns.append(f"{agent.name} responds")
            if len(turns) < 4:  # Limit turns
                return Event(
                    time=agent.now,
                    event_type="Talk",
                    target=agent_b,
                    context={
                        "message": f"Hello from {agent.name}",
                        "speaker": agent.name,
                        "conversation_id": "conv1",
                        "tone": "positive",
                    },
                )
            return None

        def respond_b(agent, act, event):
            turns.append(f"{agent.name} responds")
            if len(turns) < 4:
                return Event(
                    time=agent.now,
                    event_type="Talk",
                    target=agent_a,
                    context={
                        "message": f"Hello from {agent.name}",
                        "speaker": agent.name,
                        "conversation_id": "conv1",
                        "tone": "positive",
                    },
                )
            return None

        agent_a.on_action("respond", respond_a)
        agent_b.on_action("respond", respond_b)

        sim = Simulation(entities=[agent_a, agent_b], duration=10.0)

        # Alice initiates
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Talk",
                target=agent_a,
                context={
                    "message": "Start conversation",
                    "speaker": "System",
                    "conversation_id": "conv1",
                    "tone": "neutral",
                },
            )
        )

        sim.run()

        assert len(turns) == 4
        assert agent_a.stats.conversations_participated >= 1
        assert agent_b.stats.conversations_participated >= 1

    def test_talk_updates_social_state(self):
        """Talk events update the social relationship state."""
        backend = MockLLMBackend(default_action="stay_silent")
        agent = HumanAgent("Alice", "a person", backend, action_delay=0.0)
        agent.on_action("stay_silent", lambda a, act, e: None)

        sim = Simulation(entities=[agent], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Talk",
                target=agent,
                context={
                    "message": "Hey there!",
                    "speaker": "Bob",
                    "conversation_id": "c1",
                    "tone": "positive",
                },
            )
        )

        sim.run()

        rel = agent.state.get_relation("Bob")
        assert rel.familiarity > 0.0


class TestHeartbeat:
    def test_heartbeat_ticks_state_over_time(self):
        """Heartbeat events drive physiological state changes."""
        backend = MockLLMBackend()
        agent = HumanAgent(
            "Alice", "a person", backend,
            heartbeat_interval=1.0,
            action_delay=0.0,
        )

        sim = Simulation(entities=[agent], duration=5.0)

        # Schedule initial heartbeat
        hb = agent.schedule_first_heartbeat(Instant.from_seconds(0.0))
        if hb:
            sim.schedule(hb)

        sim.run()

        # Heartbeats should have ticked state
        assert agent.stats.events_received >= 4  # ~4 heartbeats in 5s


class TestSanityCheck:
    def test_sanity_check_prevents_invalid_action(self):
        """Sanity check rejects eating too soon, agent re-decides."""
        # First call returns "eat", second call returns "wait"
        call_count = [0]
        original_complete = None

        class SequentialBackend(MockLLMBackend):
            def complete(self, prompt, *, temperature=0.7, max_tokens=500):
                call_count[0] += 1
                self._call_count += 1
                self._prompts.append(prompt)
                if call_count[0] == 1:
                    return "eat"
                return "wait"

        backend = SequentialBackend()
        agent = HumanAgent(
            "Alice",
            "a person",
            backend,
            sanity_checks=[NoDoubleEating()],
            action_delay=0.0,
        )
        agent.on_action("eat", lambda a, act, e: None)
        agent.on_action("wait", lambda a, act, e: None)

        # Set up context so eating is blocked
        agent._sanity_context["last_eat_time"] = 50.0

        sim = Simulation(entities=[agent], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),  # 1.0s, too close to 50.0? No, 1.0 < 50.0+1800
                event_type="Stimulus",
                target=agent,
                context={
                    "description": "Food available",
                    "metadata": {"choices": ["eat", "wait"]},
                },
            )
        )

        # Override sanity context timing to trigger the check
        agent._sanity_context["last_eat_time"] = 0.5  # Just 0.5s ago

        sim.run()

        assert agent.stats.sanity_violations >= 1


class TestStateInfluencesLoop:
    def test_exhausted_agent_uses_reactive_loop(self):
        """An exhausted agent should be assigned the reactive loop."""
        backend = MockLLMBackend(default_action="wait")
        state = HumanState()
        state.cognition.attention = 0.2
        state.cognition.decision_fatigue = 0.8

        agent = HumanAgent(
            "Alice", "an exhausted person", backend,
            state=state, action_delay=0.0,
        )
        agent.on_action("wait", lambda a, act, e: None)

        sim = Simulation(entities=[agent], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "Something happened"},
            )
        )

        sim.run()

        # Check the trace to see which loop was used
        assert len(agent.traces) == 1
        assert agent.traces[0].loop_used == "ReactiveLoop"

    def test_fresh_agent_uses_heuristic_loop(self):
        """A fresh, well-rested agent should use the heuristic loop."""
        backend = MockLLMBackend(default_action="wait")
        state = HumanState()
        # Defaults: attention=1.0, decision_fatigue=0.0

        agent = HumanAgent(
            "Alice", "a well-rested person", backend,
            state=state, action_delay=0.0,
        )
        agent.on_action("wait", lambda a, act, e: None)

        sim = Simulation(entities=[agent], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "Something happened"},
            )
        )

        sim.run()

        assert len(agent.traces) == 1
        assert agent.traces[0].loop_used == "HeuristicLoop"


class TestMemoryIntegration:
    def test_events_are_added_to_memory(self):
        """Events are recorded in the agent's episodic memory."""
        backend = MockLLMBackend(default_action="wait")
        agent = HumanAgent("Alice", "a person", backend, action_delay=0.0)
        agent.on_action("wait", lambda a, act, e: None)

        sim = Simulation(entities=[agent], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Talk",
                target=agent,
                context={
                    "message": "Hello!",
                    "speaker": "Bob",
                    "conversation_id": "c1",
                    "tone": "neutral",
                },
            )
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(2.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "A bell rang"},
            )
        )

        sim.run()

        # Memory should contain entries for both events
        assert len(agent.memory.buffer) == 2
        summaries = [e.summary for e in agent.memory.buffer]
        assert any("Bob" in s for s in summaries)
        assert any("bell" in s for s in summaries)


class TestDelayedDecision:
    def test_action_delay_creates_generator(self):
        """Nonzero action_delay uses generator-based delayed decision."""
        collector = Collector()
        backend = MockLLMBackend(default_action="greet")
        agent = HumanAgent(
            "Alice", "a person", backend, action_delay=0.5
        )

        def greet_handler(a, act, event):
            return Event(
                time=a.now,
                event_type="Greeting",
                target=collector,
                context={"from": a.name},
            )

        agent.on_action("greet", greet_handler)

        sim = Simulation(entities=[agent, collector], duration=5.0)
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Stimulus",
                target=agent,
                context={"description": "Someone arrived"},
            )
        )

        sim.run()

        # Greeting should arrive at t=1.5 (1.0 + 0.5 delay)
        assert len(collector.events) == 1
        greeting_time = collector.events[0].time.to_seconds()
        assert abs(greeting_time - 1.5) < 0.01
