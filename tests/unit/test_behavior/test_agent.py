"""Unit tests for behavior agent module."""

import random

from happysimulator import Simulation, Instant, Event
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.traits import PersonalityTraits
from happysimulator.components.behavior.state import AgentState
from happysimulator.components.behavior.decision import (
    Choice,
    DecisionContext,
    UtilityModel,
)


class TestAgent:
    def _make_agent(self, **kwargs):
        def utility(choice, ctx):
            return {"buy": 0.9, "wait": 0.1}.get(choice.action, 0)

        defaults = dict(
            name="test_agent",
            traits=PersonalityTraits.big_five(),
            decision_model=UtilityModel(utility_fn=utility),
            seed=42,
        )
        defaults.update(kwargs)
        return Agent(**defaults)

    def test_stimulus_makes_decision(self):
        agent = self._make_agent()
        action_results = []

        def buy_handler(ag, choice, event):
            action_results.append(choice.action)
            return None

        agent.on_action("buy", buy_handler)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[agent],
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="Stimulus",
            target=agent,
            context={"metadata": {
                "choices": [
                    Choice(action="buy"),
                    Choice(action="wait"),
                ],
            }},
        ))
        sim.run()

        assert agent.stats.events_received == 1
        assert agent.stats.decisions_made == 1
        assert action_results == ["buy"]

    def test_social_message_updates_beliefs(self):
        agent = self._make_agent()
        agent.state.beliefs["topic1"] = 0.0

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[agent],
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="SocialMessage",
            target=agent,
            context={"metadata": {
                "topic": "topic1",
                "opinion": 1.0,
                "credibility": 1.0,
            }},
        ))
        sim.run()

        # Belief should move toward 1.0 based on agreeableness (0.5) * credibility (1.0)
        assert agent.state.beliefs["topic1"] > 0.0
        assert agent.stats.social_messages_received == 1

    def test_no_decision_model_returns_none(self):
        agent = Agent(name="passive", decision_model=None, seed=42)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[agent],
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="Stimulus",
            target=agent,
            context={"metadata": {
                "choices": [Choice(action="buy")],
            }},
        ))
        sim.run()
        assert agent.stats.decisions_made == 0

    def test_action_handler_returns_events(self):
        from happysimulator import Counter

        counter = Counter("result_counter")
        agent = self._make_agent()

        def buy_handler(ag, choice, event):
            return [Event(
                time=ag.now,
                event_type="Purchased",
                target=counter,
            )]

        agent.on_action("buy", buy_handler)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[agent, counter],
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="Stimulus",
            target=agent,
            context={"metadata": {
                "choices": [Choice(action="buy"), Choice(action="wait")],
            }},
        ))
        sim.run()

        assert counter.total == 1
        assert counter.by_type.get("Purchased") == 1

    def test_action_delay(self):
        agent = self._make_agent(action_delay=0.5)
        action_times = []

        def buy_handler(ag, choice, event):
            action_times.append(ag.now.to_seconds())
            return None

        agent.on_action("buy", buy_handler)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[agent],
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="Stimulus",
            target=agent,
            context={"metadata": {
                "choices": [Choice(action="buy")],
            }},
        ))
        sim.run()

        assert len(action_times) == 1
        assert abs(action_times[0] - 1.5) < 1e-9  # 1.0 + 0.5 delay

    def test_memory_recorded(self):
        agent = self._make_agent()
        agent.on_action("buy", lambda ag, c, e: None)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[agent],
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="Stimulus",
            target=agent,
            context={"metadata": {
                "choices": [Choice(action="buy")],
                "valence": 0.5,
            }},
        ))
        sim.run()

        memories = agent.state.recent_memories(5)
        assert len(memories) == 1
        assert memories[0].valence == 0.5

    def test_stats_tracking(self):
        agent = self._make_agent()
        agent.on_action("buy", lambda ag, c, e: None)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[agent],
        )
        for t in [1.0, 2.0]:
            sim.schedule(Event(
                time=Instant.from_seconds(t),
                event_type="Stimulus",
                target=agent,
                context={"metadata": {
                    "choices": [Choice(action="buy")],
                }},
            ))
        sim.run()

        assert agent.stats.events_received == 2
        assert agent.stats.decisions_made == 2
        assert agent.stats.actions_by_type["buy"] == 2
