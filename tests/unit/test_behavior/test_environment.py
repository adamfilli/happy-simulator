"""Unit tests for behavior environment module."""

from happysimulator import Event, Instant, Simulation
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.decision import Choice, UtilityModel
from happysimulator.components.behavior.environment import Environment
from happysimulator.components.behavior.influence import DeGrootModel
from happysimulator.components.behavior.social_network import SocialGraph


def _make_utility():
    return UtilityModel(utility_fn=lambda c, ctx: 1.0 if c.action == "buy" else 0.0)


class TestEnvironment:
    def test_broadcast_reaches_all_agents(self):
        agents = [Agent(name=f"a{i}", decision_model=_make_utility(), seed=i) for i in range(5)]
        for a in agents:
            a.on_action("buy", lambda ag, c, e: None)

        env = Environment(name="env", agents=agents, seed=42)

        all_entities = [env, *agents]
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=all_entities,
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="BroadcastStimulus",
                target=env,
                context={
                    "metadata": {
                        "stimulus_type": "Promo",
                        "choices": [Choice(action="buy"), Choice(action="wait")],
                    }
                },
            )
        )
        sim.run()

        for a in agents:
            assert a.stats.events_received >= 1
        assert env.stats.broadcasts_sent == 1

    def test_targeted_stimulus(self):
        a1 = Agent(name="target1", decision_model=_make_utility(), seed=1)
        a2 = Agent(name="target2", decision_model=_make_utility(), seed=2)
        a3 = Agent(name="excluded", decision_model=_make_utility(), seed=3)
        for a in [a1, a2, a3]:
            a.on_action("buy", lambda ag, c, e: None)

        env = Environment(name="env", agents=[a1, a2, a3], seed=42)
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[env, a1, a2, a3],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="TargetedStimulus",
                target=env,
                context={
                    "metadata": {
                        "stimulus_type": "Offer",
                        "targets": ["target1", "target2"],
                        "choices": [Choice(action="buy")],
                    }
                },
            )
        )
        sim.run()

        assert a1.stats.events_received >= 1
        assert a2.stats.events_received >= 1
        assert a3.stats.events_received == 0
        assert env.stats.targeted_sends == 1

    def test_influence_propagation(self):
        a1 = Agent(name="leader", seed=1)
        a2 = Agent(name="follower", seed=2)
        a1.state.beliefs["topic"] = 1.0
        a2.state.beliefs["topic"] = 0.0

        graph = SocialGraph()
        graph.add_edge("leader", "follower", weight=1.0, trust=1.0)

        env = Environment(
            name="env",
            agents=[a1, a2],
            social_graph=graph,
            influence_model=DeGrootModel(self_weight=0.0),
            seed=42,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[env, a1, a2],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="InfluencePropagation",
                target=env,
                context={"metadata": {"topic": "topic"}},
            )
        )
        sim.run()

        # Follower should have moved toward leader's opinion
        assert a2.state.beliefs["topic"] > 0.0
        assert env.stats.influence_rounds == 1

    def test_state_change(self):
        env = Environment(name="env", seed=42)
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[env],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="StateChange",
                target=env,
                context={"metadata": {"key": "price", "value": 9.99}},
            )
        )
        sim.run()

        assert env.shared_state["price"] == 9.99
        assert env.stats.state_changes == 1

    def test_clock_propagation(self):
        a1 = Agent(name="a1", seed=1)
        env = Environment(name="env", agents=[a1], seed=42)
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[env, a1],
        )
        sim.run()
        # Both should have clocks
        assert env._clock is not None
        assert a1._clock is not None
