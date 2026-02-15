"""Opinion dynamics simulation example.

Demonstrates DeGroot consensus convergence on a complete graph and
bounded-confidence clustering (Hegselmann-Krause model).

Usage:
    python examples/opinion_dynamics.py
"""

from happysimulator import Event, Instant, Simulation
from happysimulator.components.behavior.agent import Agent
from happysimulator.components.behavior.environment import Environment
from happysimulator.components.behavior.influence import BoundedConfidenceModel, DeGrootModel
from happysimulator.components.behavior.social_network import SocialGraph


def run_degroot_consensus():
    """DeGroot model on a complete graph: converge to consensus."""
    print("=" * 60)
    print("DEGROOT CONSENSUS MODEL")
    print("=" * 60)

    n = 10
    agents = [Agent(name=f"node_{i}", seed=i) for i in range(n)]

    # Spread initial opinions from -1 to 1
    for i, agent in enumerate(agents):
        agent.state.beliefs["policy"] = -1.0 + 2.0 * i / (n - 1)

    initial = [a.state.beliefs["policy"] for a in agents]
    print(f"Initial opinions: {[f'{o:.2f}' for o in initial]}")

    names = [a.name for a in agents]
    graph = SocialGraph.complete(names, weight=1.0, trust=1.0)

    env = Environment(
        name="society",
        agents=agents,
        social_graph=graph,
        influence_model=DeGrootModel(self_weight=0.3),
        seed=42,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=25.0,
        entities=[env, *agents],
    )

    # 20 rounds of influence
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

    final = [a.state.beliefs["policy"] for a in agents]
    mean = sum(final) / len(final)
    max_dev = max(abs(o - mean) for o in final)

    print(f"Final opinions:   {[f'{o:.2f}' for o in final]}")
    print(f"Mean: {mean:.4f}, Max deviation: {max_dev:.6f}")
    print(f"Converged: {'Yes' if max_dev < 0.01 else 'No'}")
    print()


def run_bounded_confidence():
    """Bounded confidence model: opinions cluster instead of converging."""
    print("=" * 60)
    print("BOUNDED CONFIDENCE (HEGSELMANN-KRAUSE)")
    print("=" * 60)

    n = 20
    agents = [Agent(name=f"citizen_{i}", seed=i) for i in range(n)]

    # Three initial clusters: 0.1, 0.5, 0.9
    for i, agent in enumerate(agents):
        if i < 7:
            agent.state.beliefs["tax_policy"] = 0.1
        elif i < 14:
            agent.state.beliefs["tax_policy"] = 0.5
        else:
            agent.state.beliefs["tax_policy"] = 0.9

    initial = [a.state.beliefs["tax_policy"] for a in agents]
    print(f"Initial clusters: {sorted(set(initial))}")
    print(f"Cluster sizes:    {[initial.count(v) for v in sorted(set(initial))]}")

    names = [a.name for a in agents]
    graph = SocialGraph.complete(names, weight=1.0, trust=1.0)

    # epsilon=0.25 means clusters 0.1 and 0.5 are too far apart
    env = Environment(
        name="town_hall",
        agents=agents,
        social_graph=graph,
        influence_model=BoundedConfidenceModel(epsilon=0.25, self_weight=0.3),
        seed=42,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=15.0,
        entities=[env, *agents],
    )

    for t in range(1, 11):
        sim.schedule(
            Event(
                time=Instant.from_seconds(float(t)),
                event_type="InfluencePropagation",
                target=env,
                context={"metadata": {"topic": "tax_policy"}},
            )
        )
    sim.run()

    final = sorted(a.state.beliefs["tax_policy"] for a in agents)
    print(f"Final opinions:   {[f'{o:.3f}' for o in final]}")

    # Count clusters (opinions within 0.05 of each other)
    clusters: list[list[float]] = []
    for o in final:
        placed = False
        for cluster in clusters:
            if abs(o - cluster[0]) < 0.05:
                cluster.append(o)
                placed = True
                break
        if not placed:
            clusters.append([o])

    print(f"Number of clusters: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i + 1}: center={sum(cluster) / len(cluster):.3f}, size={len(cluster)}")
    print()


def main():
    run_degroot_consensus()
    run_bounded_confidence()


if __name__ == "__main__":
    main()
