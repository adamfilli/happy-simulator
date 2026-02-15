"""Integration test: opinion dynamics with DeGroot model.

Complete graph of agents starting with varied opinions. Verify
convergence to consensus within a bounded number of rounds.
"""

from happysimulator import Event, Instant, Simulation
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.environment import Environment
from happysimulator.components.behavior.influence import DeGrootModel
from happysimulator.components.behavior.social_network import SocialGraph


class TestOpinionDynamics:
    def test_degroot_convergence(self):
        """Complete graph + DeGroot should converge to consensus."""
        n = 10
        agents = [Agent(name=f"node_{i}", seed=i) for i in range(n)]

        # Assign initial opinions: spread from -1 to 1
        for i, agent in enumerate(agents):
            agent.state.beliefs["policy"] = -1.0 + 2.0 * i / (n - 1)

        names = [a.name for a in agents]
        graph = SocialGraph.complete(names, weight=1.0, trust=1.0)

        # self_weight=0 means full averaging
        env = Environment(
            name="society",
            agents=agents,
            social_graph=graph,
            influence_model=DeGrootModel(self_weight=0.3),
            seed=42,
        )

        # Schedule 20 rounds of influence propagation
        all_entities = [env, *agents]
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=25.0,
            entities=all_entities,
        )
        for t in range(1, 21):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(float(t)),
                    event_type="InfluencePropagation",
                    target=env,
                    context={"metadata": {"topic": "policy"}},
                )
            )
        sim.run()

        # After 20 rounds, opinions should be near consensus
        opinions = [a.state.beliefs.get("policy", 0) for a in agents]
        mean_opinion = sum(opinions) / len(opinions)
        max_deviation = max(abs(o - mean_opinion) for o in opinions)

        # Should converge close to consensus (max deviation < 0.1)
        assert max_deviation < 0.1, f"Max deviation {max_deviation} too large: {opinions}"

    def test_bounded_confidence_clustering(self):
        """Bounded confidence with epsilon=0.3 should form clusters."""
        from happysimulator.components.behavior.influence import BoundedConfidenceModel

        n = 20
        agents = [Agent(name=f"n_{i}", seed=i) for i in range(n)]

        # Two clusters: opinions near 0.2 and 0.8
        for i, agent in enumerate(agents):
            if i < n // 2:
                agent.state.beliefs["issue"] = 0.2
            else:
                agent.state.beliefs["issue"] = 0.8

        names = [a.name for a in agents]
        graph = SocialGraph.complete(names, weight=1.0, trust=1.0)

        env = Environment(
            name="polarized",
            agents=agents,
            social_graph=graph,
            influence_model=BoundedConfidenceModel(epsilon=0.3, self_weight=0.3),
            seed=42,
        )

        all_entities = [env, *agents]
        sim = Simulation(
            start_time=Instant.Epoch,
            duration=15.0,
            entities=all_entities,
        )
        for t in range(1, 11):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(float(t)),
                    event_type="InfluencePropagation",
                    target=env,
                    context={"metadata": {"topic": "issue"}},
                )
            )
        sim.run()

        # Opinions should still be clustered (epsilon=0.3 < |0.8-0.2|=0.6)
        opinions = [a.state.beliefs.get("issue", 0.5) for a in agents]
        low_cluster = [o for o in opinions if o < 0.5]
        high_cluster = [o for o in opinions if o >= 0.5]
        assert len(low_cluster) > 0
        assert len(high_cluster) > 0

        # Clusters should be separated
        if low_cluster and high_cluster:
            assert max(low_cluster) < min(high_cluster)
