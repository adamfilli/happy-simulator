# Using with Claude Code

happy-simulator ships with [Claude Code](https://claude.ai/claude-code) skills that help you build, debug, and analyze simulations through natural language. If you have Claude Code installed, these skills are available automatically when you work in a happy-simulator project.

## Available Skills

| Skill | Description |
|-------|-------------|
| `/happy-sim-scaffold` | Generate a complete simulation from a description |
| `/happy-sim-diagnose` | Troubleshoot a broken or misbehaving simulation |
| `/happy-sim-add-instrumentation` | Add probes, trackers, and charts to existing code |
| `/happy-sim-explain-example` | Walk through any library example with explanation |
| `/happy-sim-component-guide` | Interactive wizard to pick the right components |
| `/happy-sim-analyze` | Run a simulation and interpret its results |

## `/happy-sim-scaffold` — Generate a Simulation

Describe what you want to simulate and get a complete, runnable Python file.

```
/happy-sim-scaffold
> "A coffee shop with 2 baristas, customers arriving every 30 seconds,
>  and drinks taking 1-3 minutes to prepare"
```

The generated file includes:

- A `Config` dataclass with tunable parameters
- Entity classes using the appropriate components (`QueuedResource`, `Source.poisson()`, etc.)
- A `Sink` collecting latency metrics
- A `run()` function with summary output

This is the fastest way to go from idea to running simulation.

## `/happy-sim-diagnose` — Fix a Broken Simulation

When your simulation isn't behaving as expected — events aren't being processed, queues grow without bound, or you're getting errors — point `/happy-sim-diagnose` at your file.

```
/happy-sim-diagnose my_simulation.py
```

It checks for the most common issues:

- **Entities not registered** in `Simulation(entities=[...])` — silent failure
- **Missing `target`** on Events
- **`has_capacity()` not overridden** — queue never builds up
- **Arrival rate > service rate** — unbounded growth
- **Generator yield mistakes** — yielding `Instant` instead of `float`, returning events mid-generator

## `/happy-sim-add-instrumentation` — Add Observability

You've built a simulation that runs. Now you need to understand what's happening inside it.

```
/happy-sim-add-instrumentation my_simulation.py
```

This reads your code, identifies entities worth monitoring, and adds:

- **`Probe`** for queue depth over time
- **`LatencyTracker`** for end-to-end latency (p50, p99)
- **`ThroughputTracker`** for events/sec
- **`Chart`** definitions for the visual debugger (optional)
- **matplotlib plots** as a fallback

## `/happy-sim-explain-example` — Learn from Examples

The library includes 78 examples across 10 categories. Pick one and get a guided walkthrough.

```
/happy-sim-explain-example
> "Show me the Raft leader election example"
```

The explanation covers:

- **Overview** — what system is being simulated
- **Architecture** — entities, connections, and event flow
- **Key patterns** — generator yields, SimFuture, probes, etc.
- **What to watch for** — what the output demonstrates

### Example Categories

| Category | Count | Highlights |
|----------|-------|-----------|
| Queuing | 7 | M/M/1, metastable failure, retry amplification |
| Distributed | 12 | Raft, Paxos, CRDTs, chain replication |
| Industrial | 20 | Bank, hospital, manufacturing, warehouse |
| Infrastructure | 7 | CPU scheduling, disk I/O, stream processing |
| Storage | 7 | LSM trees, WAL, bloom filters, transactions |
| Deployment | 7 | Canary, rolling, saga, service mesh |
| Performance | 8 | Auto-scaler, cold start, burst suppression |
| Behavior | 3 | Product adoption, opinion dynamics |
| Load Balancing | 4 | Consistent hashing, virtual nodes |
| Visual | 1 | Browser-based debugger demo |

## `/happy-sim-component-guide` — Choose the Right Components

With 350+ components, finding the right ones can be overwhelming. Describe your scenario and get targeted recommendations.

```
/happy-sim-component-guide
> "I need to model a web service with retry logic and circuit breaking"
```

The guide maps your description to components using a decision tree:

- **Queue + processing?** → `QueuedResource`
- **Network of nodes?** → `Network` + link conditions
- **Resilience patterns?** → `CircuitBreaker`, `Bulkhead`, `TimeoutWrapper`
- **Rate limiting?** → `RateLimitedEntity` + policy, or `Inductor`
- **Industrial scenario?** → `ConveyorBelt`, `BatchProcessor`, `ShiftSchedule`, etc.
- **Agent-based?** → `Agent`, `Population`, `Environment`

Each recommendation includes a minimal wiring example and pointers to relevant example files.

## `/happy-sim-analyze` — Interpret Results

After running a simulation, get a plain-English interpretation of the results.

```
/happy-sim-analyze my_simulation.py
```

The analysis covers:

- **Health summary** — healthy, stressed, or failing
- **Key metrics** — throughput, latency (mean/p50/p99), queue depth, utilization
- **Phases** — warmup, steady-state, overload, recovery
- **Stability** — stable, metastable, or unstable
- **Bottleneck** — which entity is the constraint
- **Recommendations** — concrete suggestions for improvement

It uses the library's built-in `analyze()` and `detect_phases()` functions under the hood.

## Workflow

These skills are designed to cover the full simulation lifecycle:

```
/happy-sim-component-guide  →  /happy-sim-scaffold  →  /happy-sim-diagnose  →  /happy-sim-add-instrumentation  →  /happy-sim-analyze
   "What do I         "Build       "Why isn't      "Add metrics"         "What do the
    need?"              it"          it working?"                          results mean?"
```

At any point, use `/happy-sim-explain-example` to learn from the 78 built-in examples.
