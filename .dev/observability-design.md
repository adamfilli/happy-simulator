# Observability & Query API for Reasoning About Simulations

## Goal
Enable external observers (AI models, humans) to query simulation data and reason about emergent system properties through a unified API.

## Current State Problems
1. **No Query API** - Cannot ask "average queue depth in [10s, 20s]?" without custom code
2. **Post-simulation only** - Data only accessible after `sim.run()` completes
3. **Scattered patterns** - Each example reinvents latency tracking, sinks, etc.
4. **No external observer support** - No pause/resume, no interactive queries
5. **Inflexible data model** - `Data` class is just a list with no indexing or aggregation

## Design Overview

```
                    +------------------+
                    |  ObservationBus  |
                    | (Central Router) |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------v-------+    +-------v-------+    +-------v-------+
|  MetricStore  |    |  TraceStore   |    |  EventStore   |
| (time-indexed)|    | (span-indexed)|    | (event-indexed)|
+---------------+    +---------------+    +---------------+
        |                    |                    |
        +--------------------+--------------------+
                             |
                    +--------v---------+
                    |   QueryEngine    |
                    | (Unified Access) |
                    +--------+---------+
                             |
        +--------------------+--------------------+
        |                    |                    |
+-------v-------+    +-------v-------+    +-------v-------+
| Subscriptions |    |   Snapshots   |    |    Exports    |
|  (real-time)  |    |(point-in-time)|    | (JSON/CSV/DF) |
+---------------+    +---------------+    +---------------+
```

## Core Abstractions

### 1. Observation (Universal record)
```python
@dataclass(frozen=True)
class Observation:
    time: Instant
    kind: ObservationKind  # METRIC, SPAN, EVENT, STATE
    source: str            # Entity name or "simulation"
    name: str              # Metric name, span kind, or event type
    value: Any
    tags: dict[str, str]
```

### 2. Query Builder (Fluent API)
```python
result = (sim.query
    .metrics("queue_depth")
    .from_entity("Server")
    .between(t1, t2)
    .aggregate("p99")
    .execute())
```

### 3. SimulationControl (Pause/Step/Resume)
```python
sim.control.pause()
sim.control.step(10)  # Process 10 events
state = sim.control.get_state()
sim.control.resume()
```

### 4. Breakpoints
```python
sim.control.add_breakpoint(Breakpoint(
    metric_condition=MetricCondition(
        source="Server", name="queue_depth", operator="gt", threshold=50
    )
))
```

## File Organization

```
happysimulator/
├── observability/
│   ├── __init__.py              # Public API exports
│   ├── bus.py                   # ObservationBus, Observation, ObservationKind
│   ├── config.py                # ObservabilityConfig
│   ├── query.py                 # Query, QueryResult, QueryEngine
│   ├── observer.py              # SimulationObserver protocol
│   ├── control.py               # SimulationControl, Breakpoint
│   ├── integration.py           # Simulation integration hooks
│   ├── instrumented.py          # InstrumentedEntity base
│   ├── auto_instrument.py       # Auto-instrumentation utilities
│   ├── anomaly.py               # AnomalyDetector
│   ├── export.py                # Exporter (JSON, CSV, DataFrame)
│   └── stores/
│       ├── base.py              # ObservationStore protocol
│       ├── metric_store.py      # MetricStore (time-indexed, binary search)
│       ├── trace_store.py       # TraceStore (span-indexed)
│       ├── event_store.py       # EventStore (event-indexed)
│       └── streaming_store.py   # StreamingMetricStore (bounded memory)
```

## Critical Files to Modify

| File | Changes |
|------|---------|
| `happysimulator/core/simulation.py` | Add observability integration directly |
| `happysimulator/instrumentation/probe.py` | Emit to ObservationBus |
| `happysimulator/instrumentation/recorder.py` | Emit to ObservationBus |
| `happysimulator/core/event.py` | Integrate traces with TraceStore |
| `happysimulator/__init__.py` | Export new public API |

## Implementation Phases

### Phase 1: Core Infrastructure
1. Create `happysimulator/observability/` directory
2. Implement `Observation`, `ObservationKind`, `ObservationBus`
3. Implement `MetricStore` with time-range queries and binary search
4. Implement `Query` builder and `QueryEngine`
5. Implement `ObservabilityConfig`

### Phase 2: Simulation Integration
6. Modify `Simulation` class to integrate `ObservationBus`
7. Implement `SimulationControl` (pause/step/resume)
8. Implement `Breakpoint` conditions
9. Implement `SimulationObserver` protocol
10. Add `query` and `control` properties to `Simulation`

### Phase 3: Tracing Integration
11. Implement `TraceStore` with trace reconstruction
12. Implement `Span` dataclass
13. Update `TraceRecorder` to emit to ObservationBus
14. Bridge existing `event.context["trace"]` to TraceStore

### Phase 4: Auto-Instrumentation
15. Implement `InstrumentedEntity` base class
16. Implement `instrument_queued_resource()` utility
17. Auto-instrument standard components (QueuedResource, etc.)

### Phase 5: Export & AI Integration
18. Implement `Exporter` (JSON, CSV, DataFrame)
19. Implement `AnomalyDetector`
20. Create example: AI reasoning about metastable failure
21. Update CLAUDE.md with observability documentation

### Phase 6: Streaming & Performance
22. Implement `StreamingMetricStore` for large simulations
23. Add sampling support for high-throughput scenarios
24. Performance benchmarks and memory profiling

## Example Usage

### Human Debugging Metastable Failure
```python
sim = Simulation(
    sources=[source],
    entities=[server, sink],
    observability=ObservabilityConfig.full(),
)

# Break when queue builds up
sim.control.add_breakpoint(Breakpoint(
    metric_condition=MetricCondition("Server", "queue_depth", "gt", 50)
))

sim.run()  # Pauses at breakpoint

# Investigate
state = sim.control.get_state()
history = sim.query.query().metrics("queue_depth").from_entity("Server").between(
    state.current_time - 10.0, state.current_time
).execute()

# Step through
events = sim.control.step(10)
sim.control.resume()
```

### AI Model Reasoning
```python
sim = Simulation(...)
sim.run()

# Export structured data for AI consumption
exporter = Exporter(sim.query)
analysis_data = {
    "queue_behavior": exporter.to_json(
        sim.query.query().metrics("queue_depth").limit(1000)
    ),
    "latency_p99": sim.query.query()
        .metrics("latency")
        .aggregate("p99")
        .execute().scalar,
}

# Feed to AI model for reasoning
prompt = f"Analyze this simulation: {json.dumps(analysis_data)}"
```

## Backwards Compatibility

- **Breaking change**: `Simulation` class gains observability features directly
- Existing `Data`/`Probe` classes still work (bridge to new stores)
- Observability enabled by default but configurable via `ObservabilityConfig`
- Disable with `Simulation(..., observability=ObservabilityConfig.disabled())`

## Verification Plan

1. **Unit tests**: Each store, query builder, and control mechanism
2. **Integration test**: Run M/M/1 queue with full observability, verify:
   - Metrics collected at expected intervals
   - Traces reconstructable by request ID
   - Pause/step/resume works correctly
   - JSON export parseable and complete
3. **Performance test**: 100k events with observability vs without
4. **AI integration test**: Export data, verify format suitable for LLM consumption
5. **Example update**: Update `examples/m_m_1_queue.py` to demonstrate new API
