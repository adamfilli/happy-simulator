# Behavioral Modeling

Model individual human agents with personality, decision-making, and social dynamics.

## Agent

```python
from happysimulator.components.behavior import (
    Agent, PersonalityTraits, UtilityModel, Choice,
)

agent = Agent(
    name="consumer_1",
    traits=PersonalityTraits.big_five(openness=0.8, agreeableness=0.6),
    decision_model=UtilityModel(utility_fn=lambda c, ctx: 0.9 if c.action == "buy" else 0.1),
    seed=42,
    action_delay=0.1,
    heartbeat_interval=5.0,
)
agent.on_action("buy", lambda ag, choice, event: [Event(...)])
```

## Decision Models

| Model | Theory | Usage |
|-------|--------|-------|
| `UtilityModel(fn, temperature=0)` | Rational choice | Maximize utility, optional softmax |
| `RuleBasedModel(rules, default)` | Heuristics | Priority-ordered if-then rules |
| `BoundedRationalityModel(fn, aspiration)` | Satisficing | First option above threshold |
| `SocialInfluenceModel(fn, conformity)` | Conformity | Weighted by peer behavior |
| `CompositeModel([(model, weight), ...])` | Hybrid | Weighted voting |

## Population

Create large populations with demographic segments:

```python
from happysimulator.components.behavior import Population, DemographicSegment

pop = Population.uniform(size=1000, decision_model=model, graph_type="small_world", seed=42)

pop = Population.from_segments(total_size=1000, segments=[
    DemographicSegment("innovators", fraction=0.15,
                       trait_distribution=dist, decision_model_factory=fn),
    DemographicSegment("majority", fraction=0.85, ...),
])

pop.agents        # list of Agent
pop.social_graph  # SocialGraph
pop.size          # int
pop.stats         # PopulationStats
```

## Social Graph

```python
from happysimulator.components.behavior import SocialGraph

graph = SocialGraph.complete(names)
graph = SocialGraph.small_world(names, k=4, p_rewire=0.1, rng=rng)
graph = SocialGraph.random_erdos_renyi(names, p=0.1, rng=rng)

graph.neighbors("alice")
graph.influencers("bob")
graph.influence_weights("bob")
```

## Environment & Stimuli

```python
from happysimulator.components.behavior import (
    Environment, DeGrootModel,
    broadcast_stimulus, price_change, influence_propagation,
)

env = Environment(
    name="market",
    agents=pop.agents,
    social_graph=pop.social_graph,
    influence_model=DeGrootModel(self_weight=0.3),
    seed=42,
)

sim.schedule(broadcast_stimulus(1.0, env, "Promo", choices=["buy", "wait"]))
sim.schedule(price_change(5.0, env, "ProductX", 100.0, 80.0))
sim.schedule(influence_propagation(2.0, env, topic="product_sentiment"))
```

### Influence Models

| Model | Behavior |
|-------|----------|
| `DeGrootModel(self_weight)` | Weighted average convergence (consensus) |
| `BoundedConfidenceModel(epsilon, self_weight)` | Only consider nearby opinions (clustering) |
| `VoterModel()` | Random adoption from one neighbor |

## Next Steps

- [Industrial Simulation](industrial-simulation.md) — operations research components
- [Examples: Behavior](../examples/behavior.md) — product adoption and opinion dynamics
