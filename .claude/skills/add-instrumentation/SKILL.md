---
name: add-instrumentation
description: Add observability (probes, trackers, charts) to a simulation
---

# Add Instrumentation

Add metrics collection, probes, and optional visual debugger charts to an existing simulation.

## Instructions

1. Read the user's simulation file to understand its structure — identify entities, sources, and the sink/output.

2. Ask the user what they want to observe if not obvious. Common choices:
   - Queue depth over time
   - End-to-end latency (p50, p99)
   - Throughput (events/sec)
   - Utilization
   - All of the above

3. Add instrumentation by modifying the existing file. Use these patterns:

### Latency tracking

Replace or augment a `Sink` with a `LatencyTracker`:

```python
from happysimulator import LatencyTracker

tracker = LatencyTracker("Latency")
# Use tracker as the terminal entity instead of (or alongside) Sink
# Events must have context["created_at"] set by the source
```

After `sim.run()`:
```python
print(f"Mean latency: {tracker.mean_latency():.3f}s")
print(f"P50: {tracker.p50():.3f}s, P99: {tracker.p99():.3f}s")
```

### Queue depth probing

```python
from happysimulator import Data, Probe

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1)
# Add probe to Simulation: Simulation(..., probes=[depth_probe])
```

After `sim.run()`:
```python
buckets = depth_data.bucket(window_s=1.0)
print(f"Mean queue depth: {depth_data.mean():.1f}")
print(f"Max queue depth: {depth_data.percentile(1.0):.0f}")
```

### Throughput tracking

```python
from happysimulator import ThroughputTracker

tp = ThroughputTracker("Throughput")
# Place tp in the pipeline where you want to measure throughput
```

### Visual debugger charts (optional)

If the user wants interactive visualization:

```python
from happysimulator.visual import serve, Chart

serve(sim, charts=[
    Chart(depth_data, title="Queue Depth", y_label="items"),
    Chart(depth_data, title="P99 Queue Depth", transform="p99", window_s=1.0, y_label="items"),
    Chart(tracker.data, title="Latency", y_label="seconds"),
])
# Note: serve() replaces sim.run() — it runs the sim interactively in the browser
```

Available chart transforms: `"raw"`, `"mean"`, `"p50"`, `"p99"`, `"p999"`, `"max"`, `"rate"`

### Matplotlib plots (no extra deps)

If the user prefers static plots:

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

buckets = depth_data.bucket(window_s=1.0)
plt.figure(figsize=(10, 4))
plt.plot(buckets.times(), buckets.means(), label="Mean depth")
plt.xlabel("Time (s)")
plt.ylabel("Queue Depth")
plt.legend()
plt.savefig("queue_depth.png", dpi=150)
```

4. Verify entities and probes are registered in `Simulation(entities=[...], probes=[...])`.

5. Run the simulation to verify the instrumentation works: `python <file>`

6. Summarize what was added and what metrics are now available.
