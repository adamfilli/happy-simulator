"""Unit tests for behavior population module."""

from happysimulator.components.behavior.decision import UtilityModel
from happysimulator.components.behavior.population import (
    DemographicSegment,
    Population,
)
from happysimulator.components.behavior.state import AgentState
from happysimulator.components.behavior.traits import NormalTraitDistribution


class TestPopulation:
    def test_uniform(self):
        pop = Population.uniform(size=20, seed=42)
        assert pop.size == 20
        assert len(pop.agents) == 20
        assert pop.social_graph.nodes == {a.name for a in pop.agents}

    def test_uniform_complete_graph(self):
        pop = Population.uniform(size=5, graph_type="complete", seed=42)
        assert pop.size == 5
        # Complete graph: each node connects to 4 others
        for agent in pop.agents:
            assert len(pop.social_graph.neighbors(agent.name)) == 4

    def test_uniform_random_graph(self):
        pop = Population.uniform(size=10, graph_type="random", seed=42)
        assert pop.size == 10

    def test_from_segments(self):
        def utility(choice, ctx):
            return 1.0

        segments = [
            DemographicSegment(
                name="early_adopters",
                fraction=0.3,
                decision_model_factory=lambda: UtilityModel(utility_fn=utility),
                seed=1,
            ),
            DemographicSegment(
                name="majority",
                fraction=0.7,
                decision_model_factory=lambda: UtilityModel(utility_fn=utility),
                seed=2,
            ),
        ]
        pop = Population.from_segments(total_size=100, segments=segments, seed=42)
        assert pop.size == 100

    def test_from_segments_with_custom_traits(self):
        dist = NormalTraitDistribution(
            means={
                "openness": 0.9,
                "conscientiousness": 0.5,
                "extraversion": 0.8,
                "agreeableness": 0.5,
                "neuroticism": 0.3,
            },
        )
        segments = [
            DemographicSegment(
                name="innovators",
                fraction=1.0,
                trait_distribution=dist,
                seed=42,
            ),
        ]
        pop = Population.from_segments(total_size=50, segments=segments, seed=42)
        assert pop.size == 50
        # Innovators should have high openness on average
        avg_openness = sum(a.traits.get("openness") for a in pop.agents) / len(pop.agents)
        assert avg_openness > 0.7

    def test_from_segments_initial_state(self):
        def make_state():
            s = AgentState()
            s.beliefs["product"] = -0.5
            return s

        segments = [
            DemographicSegment(
                name="skeptics",
                fraction=1.0,
                initial_state_factory=make_state,
                seed=42,
            ),
        ]
        pop = Population.from_segments(total_size=10, segments=segments, seed=42)
        for agent in pop.agents:
            assert agent.state.beliefs["product"] == -0.5

    def test_stats(self):
        pop = Population.uniform(size=5, seed=42)
        stats = pop.stats
        assert stats.size == 5
        assert stats.total_events == 0
        assert stats.total_decisions == 0
