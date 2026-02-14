"""Product adoption simulation example.

Demonstrates how innovators adopt a product early while the majority
follows later, influenced by social connections. Uses segmented
populations, price-change stimuli, and a small-world social graph.

Usage:
    python examples/product_adoption.py
"""

import random

from happysimulator import Simulation, Instant, Event, Source
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.traits import NormalTraitDistribution
from happysimulator.components.behavior.decision import Choice, UtilityModel
from happysimulator.components.behavior.environment import Environment
from happysimulator.components.behavior.population import DemographicSegment, Population
from happysimulator.components.behavior.stimulus import broadcast_stimulus, price_change
from happysimulator.load import ConstantArrivalTimeProvider, EventProvider


# ---------------------------------------------------------------------------
# Decision models
# ---------------------------------------------------------------------------

def innovator_utility(choice, ctx):
    """Innovators value novelty — high openness boosts buy utility."""
    if choice.action == "buy":
        return 0.6 + 0.4 * ctx.traits.get("openness")
    if choice.action == "wait":
        return 0.2
    return 0.1  # switch


def majority_utility(choice, ctx):
    """Majority are more conservative — social proof matters."""
    if choice.action == "buy":
        peers = ctx.social_context.get("peer_actions", {})
        peer_buys = peers.get("buy", 0)
        return 0.25 + min(0.6, peer_buys * 0.03)
    if choice.action == "wait":
        return 0.45
    return 0.15


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    random.seed(42)

    # Define population segments
    innovator_traits = NormalTraitDistribution(
        means={"openness": 0.85, "conscientiousness": 0.5,
               "extraversion": 0.7, "agreeableness": 0.5,
               "neuroticism": 0.3},
        stds={"openness": 0.1, "conscientiousness": 0.15,
              "extraversion": 0.15, "agreeableness": 0.15,
              "neuroticism": 0.15},
    )

    majority_traits = NormalTraitDistribution(
        means={"openness": 0.4, "conscientiousness": 0.55,
               "extraversion": 0.5, "agreeableness": 0.6,
               "neuroticism": 0.45},
    )

    segments = [
        DemographicSegment(
            name="innovators",
            fraction=0.15,
            trait_distribution=innovator_traits,
            decision_model_factory=lambda: UtilityModel(utility_fn=innovator_utility),
            seed=1,
        ),
        DemographicSegment(
            name="majority",
            fraction=0.85,
            trait_distribution=majority_traits,
            decision_model_factory=lambda: UtilityModel(utility_fn=majority_utility),
            seed=2,
        ),
    ]

    pop = Population.from_segments(
        total_size=100,
        segments=segments,
        graph_type="small_world",
        seed=42,
    )

    # Register action handlers
    adoption_log: list[tuple[float, str, str]] = []

    for agent in pop.agents:
        def make_handler(ag_name):
            def handler(ag, choice, event):
                adoption_log.append((ag.now.to_seconds(), ag_name, choice.action))
                return None
            return handler

        agent.on_action("buy", make_handler(agent.name))
        agent.on_action("wait", make_handler(agent.name))
        agent.on_action("switch", make_handler(agent.name))

    env = Environment(
        name="market",
        agents=pop.agents,
        social_graph=pop.social_graph,
        seed=42,
    )

    # Build simulation
    end_time = Instant.from_seconds(20.0)
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        entities=[env] + pop.agents,
    )

    # Schedule stimuli: periodic price drops and influence rounds
    for t in [2.0, 6.0, 10.0, 14.0]:
        sim.schedule(price_change(t, env, "GadgetX", 100.0, 100.0 - t * 2))

    for t in range(1, 20):
        sim.schedule(Event(
            time=Instant.from_seconds(float(t)),
            event_type="InfluencePropagation",
            target=env,
            context={"metadata": {"topic": "product_sentiment"}},
        ))

    # Run
    summary = sim.run()

    # Report
    print("=" * 60)
    print("PRODUCT ADOPTION SIMULATION")
    print("=" * 60)
    print(f"Population: {pop.size} agents (15% innovators, 85% majority)")
    print(f"Duration:   {summary.duration_s:.1f}s simulated")
    print(f"Events:     {summary.total_events_processed} processed")
    print()

    # Adoption stats
    innovator_count = int(0.15 * 100)
    innovator_buys = sum(
        pop.agents[i].stats.actions_by_type.get("buy", 0)
        for i in range(innovator_count)
    )
    majority_buys = sum(
        pop.agents[i].stats.actions_by_type.get("buy", 0)
        for i in range(innovator_count, 100)
    )

    print(f"Innovator buys: {innovator_buys} ({innovator_buys / max(1, innovator_count):.1f} per capita)")
    print(f"Majority buys:  {majority_buys} ({majority_buys / max(1, 100 - innovator_count):.1f} per capita)")
    print()

    # Timeline
    buy_events = [(t, name, action) for t, name, action in adoption_log if action == "buy"]
    if buy_events:
        print(f"First buy:  t={buy_events[0][0]:.1f}s by {buy_events[0][1]}")
        print(f"Last buy:   t={buy_events[-1][0]:.1f}s by {buy_events[-1][1]}")
    print()

    # Population stats
    stats = pop.stats
    print(f"Total decisions: {stats.total_decisions}")
    print(f"Actions: {dict(stats.total_actions)}")


if __name__ == "__main__":
    main()
