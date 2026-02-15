"""Integration test: market adoption simulation.

Population with innovator/majority segments. Price change stimulus.
Verify innovators adopt before majority (adoption curve).
"""

import random

from happysimulator import Event, Instant, Simulation
from happysimulator.components.behavior.decision import Choice, UtilityModel
from happysimulator.components.behavior.environment import Environment
from happysimulator.components.behavior.population import DemographicSegment, Population
from happysimulator.components.behavior.stimulus import price_change
from happysimulator.components.behavior.traits import NormalTraitDistribution


def _innovator_utility(choice, ctx):
    """Innovators: high willingness to buy."""
    if choice.action == "buy":
        openness = ctx.traits.get("openness")
        return 0.5 + 0.5 * openness  # high openness -> high buy utility
    return 0.2


def _majority_utility(choice, ctx):
    """Majority: moderate willingness, influenced by peers."""
    if choice.action == "buy":
        peer_buys = ctx.social_context.get("peer_actions", {}).get("buy", 0)
        # More likely to buy when peers have bought
        return 0.3 + min(0.5, peer_buys * 0.05)
    return 0.4


class TestBehaviorMarket:
    def test_innovators_adopt_before_majority(self):
        random.seed(42)

        innovator_dist = NormalTraitDistribution(
            means={
                "openness": 0.9,
                "conscientiousness": 0.5,
                "extraversion": 0.7,
                "agreeableness": 0.5,
                "neuroticism": 0.3,
            },
        )
        majority_dist = NormalTraitDistribution(
            means={
                "openness": 0.4,
                "conscientiousness": 0.5,
                "extraversion": 0.5,
                "agreeableness": 0.6,
                "neuroticism": 0.5,
            },
        )

        segments = [
            DemographicSegment(
                name="innovators",
                fraction=0.15,
                trait_distribution=innovator_dist,
                decision_model_factory=lambda: UtilityModel(utility_fn=_innovator_utility),
                seed=1,
            ),
            DemographicSegment(
                name="majority",
                fraction=0.85,
                trait_distribution=majority_dist,
                decision_model_factory=lambda: UtilityModel(utility_fn=_majority_utility),
                seed=2,
            ),
        ]

        pop = Population.from_segments(
            total_size=50,
            segments=segments,
            graph_type="small_world",
            seed=42,
        )

        # Register buy handler for all agents
        for agent in pop.agents:
            agent.on_action("buy", lambda ag, c, e: None)
            agent.on_action("wait", lambda ag, c, e: None)
            agent.on_action("switch", lambda ag, c, e: None)

        env = Environment(
            name="market",
            agents=pop.agents,
            social_graph=pop.social_graph,
            seed=42,
        )

        # Schedule two price changes
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            entities=[env, *pop.agents],
        )

        sim.schedule(price_change(1.0, env, "ProductX", 100.0, 80.0))
        sim.schedule(price_change(3.0, env, "ProductX", 80.0, 70.0))
        sim.run()

        # Count buys by segment
        innovator_count = int(0.15 * 50)
        innovator_buys = sum(
            pop.agents[i].stats.actions_by_type.get("buy", 0) for i in range(innovator_count)
        )
        majority_buys = sum(
            pop.agents[i].stats.actions_by_type.get("buy", 0) for i in range(innovator_count, 50)
        )

        # Innovators should have higher per-capita adoption
        innovator_rate = innovator_buys / max(1, innovator_count)
        majority_rate = majority_buys / max(1, 50 - innovator_count)

        assert innovator_rate >= majority_rate, (
            f"Innovator rate {innovator_rate:.2f} should >= majority rate {majority_rate:.2f}"
        )

    def test_population_stats_after_simulation(self):
        pop = Population.uniform(size=10, seed=42)
        for a in pop.agents:
            a.on_action("buy", lambda ag, c, e: None)

        model = UtilityModel(utility_fn=lambda c, ctx: 1.0)
        for a in pop.agents:
            a.decision_model = model

        env = Environment(name="env", agents=pop.agents, seed=42)
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=2.0,
            entities=[env, *pop.agents],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="BroadcastStimulus",
                target=env,
                context={
                    "metadata": {
                        "stimulus_type": "Test",
                        "choices": [Choice(action="buy")],
                    }
                },
            )
        )
        sim.run()

        stats = pop.stats
        assert stats.size == 10
        assert stats.total_events == 10
        assert stats.total_decisions == 10
        assert stats.total_actions.get("buy", 0) == 10
