# Observability

## Data Class

`Data` is the core time-series container. Record values over time, then slice and aggregate.

```python
from happysimulator import Data

data = Data()
data.record(1.0, 42.0)   # record value 42.0 at time 1.0s
data.record(2.0, 55.0)

data.mean()                          # average across all points
data.percentile(0.99)                # p99
data.between(1.0, 2.0).mean()       # slice by time range
data.rate(window_s=1.0)             # count/sec per window
```

### BucketedData

`Data.bucket()` groups points into time windows:

```python
bucketed = data.bucket(window_s=1.0)
bucketed.times()    # bucket start times
bucketed.means()    # mean per bucket
bucketed.p99s()     # p99 per bucket
bucketed.to_dict()  # export
```

## Probe

`Probe` periodically samples an entity attribute and records it to a `Data` object:

```python
from happysimulator import Probe

probe, depth_data = Probe.on(server, "depth", interval=0.1)
```

Register the probe: `Simulation(probes=[probe])`. For multiple metrics on the same target:

```python
probes, data = Probe.on_many(server, ["depth", "in_flight"], interval=0.1)
# data["depth"], data["in_flight"] are the Data objects
```

## Collectors

### LatencyTracker

```python
from happysimulator import LatencyTracker

tracker = LatencyTracker("Sink")
# Route events to tracker — it reads context['created_at']
tracker.p50()
tracker.p99()
tracker.mean_latency()
tracker.data          # raw Data object
tracker.summary()
```

### ThroughputTracker

```python
from happysimulator import ThroughputTracker

tp = ThroughputTracker("Throughput")
tp.throughput(window_s=1.0)
```

## SimulationSummary

`sim.run()` returns a `SimulationSummary`:

```python
summary = sim.run()
summary.duration_s                  # simulation time elapsed
summary.total_events_processed      # event count
summary.events_per_second           # throughput
summary.wall_clock_seconds          # real time elapsed
summary.entities                    # dict of EntitySummary
summary.to_dict()                   # export
```

## Analysis

```python
from happysimulator.analysis import analyze, detect_phases

# Detect steady-state vs transient phases
phases = detect_phases(data, window_s=5.0, threshold=2.0)

# Full analysis for LLM consumption
analysis = analyze(sim.summary, latency=tracker.data, queue_depth=depth_data)
analysis.to_prompt_context(max_tokens=2000)
```

## Next Steps

- [Simulation Control](simulation-control.md) — pause, step, and breakpoints
- [Visual Debugger](visual-debugger.md) — browser-based dashboard
