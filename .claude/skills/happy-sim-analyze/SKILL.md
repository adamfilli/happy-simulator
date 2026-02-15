---
name: happy-sim-analyze
description: Analyze simulation results and provide insights
---

# Analyze Simulation

Run a simulation and interpret its results using happysimulator's built-in analysis tools.

## Instructions

1. Identify what to analyze:
   - If the user points to a simulation file, read it
   - If the user has simulation output (console text, summary dict), work with that
   - If neither, ask what simulation to analyze

2. Read the simulation file to understand its structure: what entities exist, what metrics are collected, and what the simulation models.

3. Run the simulation if needed:
   ```
   python <file>
   ```
   Capture the output.

4. Analyze results using these approaches:

### From SimulationSummary (returned by `sim.run()`)

Key fields to examine:
- `summary.duration_s` — simulated time elapsed
- `summary.total_events_processed` — total event count
- `summary.events_per_second` — simulation throughput
- `summary.wall_clock_seconds` — real time taken
- `summary.entities` — per-entity stats (events handled, queue depth if applicable)

### From Sink / LatencyTracker

```python
sink.latency_stats()  # {count, avg, min, max, p50, p99}
tracker.mean_latency()
tracker.p50()
tracker.p99()
tracker.data.between(start, end).mean()  # slice a time range
```

### From Data / Probe

```python
data.mean()                          # overall average
data.percentile(0.99)                # p99
data.between(30.0, 60.0).mean()      # steady-state only (skip warmup)
data.rate(window_s=1.0)              # events per second over time

buckets = data.bucket(window_s=1.0)
buckets.times()                      # time axis
buckets.means()                      # mean per window
buckets.p99s()                       # p99 per window
```

### Phase detection

```python
from happysimulator.analysis import detect_phases
phases = detect_phases(data, window_s=5.0, threshold=2.0)
# Returns list of phases: warmup, steady-state, anomaly, etc.
```

### Full analysis

```python
from happysimulator.analysis import analyze
analysis = analyze(summary, latency=tracker.data, queue_depth=probe_data)
print(analysis.to_prompt_context(max_tokens=2000))
```

5. Provide a plain-English interpretation covering:

   **Health Summary** — Is the system healthy, stressed, or failing? One sentence.

   **Key Metrics** — Report the most important numbers:
   - Throughput (events/sec or requests/sec)
   - Latency (mean, p50, p99)
   - Queue depth (mean, max)
   - Utilization (if available)

   **Phases** — Did the simulation have distinct phases? (warmup, steady-state, overload, recovery)

   **Stability** — Is the system stable (queues bounded), metastable (appears stable but fragile), or unstable (unbounded growth)?

   **Bottleneck** — Where is the bottleneck? Which entity has the highest utilization or deepest queue?

   **Recommendations** — Concrete suggestions:
   - "Increase server capacity from 1 to 2 to handle the arrival rate"
   - "Add a circuit breaker to prevent cascade failure"
   - "The queue stabilizes at ~12 items — safe but close to the tipping point at 80% utilization"

6. If the simulation doesn't have enough instrumentation to analyze, suggest using `/happy-sim-add-instrumentation` first.
