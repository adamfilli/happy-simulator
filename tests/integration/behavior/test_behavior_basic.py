"""Integration test: basic behavioral simulation.

10 agents receive constant stimulus, verify action counts match
expected distribution given utility function.
"""

import random

from happysimulator import Simulation, Instant, Event
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.traits import PersonalityTraits
from happysimulator.components.behavior.decision import Choice, UtilityModel
from happysimulator.components.behavior.environment import Environment


class TestBehaviorBasic:
    def test_ten_agents_constant_stimulus(self):
        random.seed(42)

        # Utility: strongly prefer "buy"
        def utility(choice, ctx):
            return {"buy": 0.95, "wait": 0.05}.get(choice.action, 0)

        model = UtilityModel(utility_fn=utility)
        agents = [
            Agent(
                name=f"agent_{i}",
                traits=PersonalityTraits.big_five(),
                decision_model=model,
                seed=i,
            )
            for i in range(10)
        ]
        for a in agents:
            a.on_action("buy", lambda ag, c, e: None)
            a.on_action("wait", lambda ag, c, e: None)

        env = Environment(name="market", agents=agents, seed=42)

        all_entities = [env] + agents
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            entities=all_entities,
        )

        # Schedule 10 broadcasts manually
        for t in range(1, 11):
            sim.schedule(Event(
                time=Instant.from_seconds(float(t)),
                event_type="BroadcastStimulus",
                target=env,
                context={"metadata": {
                    "stimulus_type": "Offer",
                    "choices": [
                        Choice(action="buy", context={"price": 10}),
                        Choice(action="wait"),
                    ],
                }},
            ))
        summary = sim.run()

        # Each agent should have received ~10 stimuli
        total_decisions = sum(a.stats.decisions_made for a in agents)
        assert total_decisions > 0

        # With utility 0.95 for buy, vast majority should buy
        total_buys = sum(a.stats.actions_by_type.get("buy", 0) for a in agents)
        total_waits = sum(a.stats.actions_by_type.get("wait", 0) for a in agents)
        assert total_buys > total_waits

    def test_agents_without_handlers_do_nothing(self):
        model = UtilityModel(utility_fn=lambda c, ctx: 1.0)
        agents = [
            Agent(name=f"passive_{i}", decision_model=model, seed=i)
            for i in range(5)
        ]
        env = Environment(name="env", agents=agents, seed=42)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=2.0,
            entities=[env] + agents,
        )
        sim.schedule(Event(
            time=Instant.from_seconds(1.0),
            event_type="BroadcastStimulus",
            target=env,
            context={"metadata": {
                "stimulus_type": "Test",
                "choices": [Choice(action="act")],
            }},
        ))
        sim.run()

        # Decisions made but no handlers means no downstream events
        for a in agents:
            assert a.stats.events_received == 1
            assert a.stats.decisions_made == 1
