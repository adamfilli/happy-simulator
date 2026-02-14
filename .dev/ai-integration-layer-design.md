# AI Integration Layer for happysimulator

## Context

AI models can reason more effectively about distributed systems, queuing theory, and performance when they can *run simulations as part of their thinking process*. Today, using happysimulator has avoidable boilerplate in several areas: Probe/Data wiring (8 lines per metric), custom EventProvider classes for simple cases (15+ lines), downstream event forwarding (5 lines each time), and Simulation constructor ceremony. Across 7 representative examples, ~72% of lines are setup overhead vs. interesting logic.

This revised plan takes a different approach from the original Scenario builder: instead of introducing a parallel fluent API that can only express a subset of simulations, we **reduce friction in the existing API** so that all simulations — from simple queues to Raft consensus — benefit. Then we layer analysis tooling and an MCP server on top of the improved core.

Three layers:
1. **Ergonomic improvements** — Reduce boilerplate in existing classes (Probe, Server, Entity, Simulation)
2. **Analysis tooling** — `SimulationResult` wrapper with comparison, recommendations, and `to_prompt_context()`
3. **MCP Server** — Expose simulation as callable tools for AI models

---

## Phase 1: Ergonomic API Improvements

### 1a. `Probe.on()` factory method

**Problem**: Every probe requires creating a `Data()` first, then a 5-argument `Probe()` constructor:
```python
queue_depth_data = Data()
queue_probe = Probe(target=server, metric="depth", data=queue_depth_data, interval=0.1, start_time=Instant.Epoch)
```

**Solution**: Add a class method that creates both and returns the `Data`:
```python
# Returns (Probe, Data) — Probe is registered, Data is what you use
queue_probe, queue_depth_data = Probe.on(server, "depth", interval=0.1)

# Multiple metrics at once
probes, metrics = Probe.on_many(server, ["depth", "utilization", "in_flight"], interval=0.1)
# metrics is a dict: {"depth": Data, "utilization": Data, "in_flight": Data}
```

**File**: `happysimulator/instrumentation/probe.py`

**Implementation**:
```python
@classmethod
def on(cls, target: Entity, metric: str, interval: float = 1.0) -> tuple["Probe", Data]:
    """Create a Probe and its Data container in one call.

    Returns:
        (probe, data) tuple. Pass probe to Simulation(probes=[...]),
        use data for post-simulation analysis.
    """
    data = Data()
    probe = cls(target=target, metric=metric, data=data, interval=interval)
    return probe, data

@classmethod
def on_many(cls, target: Entity, metrics: list[str], interval: float = 1.0) -> tuple[list["Probe"], dict[str, Data]]:
    """Create Probes for multiple metrics on the same target.

    Returns:
        (probes_list, data_dict) where data_dict is keyed by metric name.
    """
    probes = []
    data_dict = {}
    for metric in metrics:
        probe, data = cls.on(target, metric, interval=interval)
        probes.append(probe)
        data_dict[metric] = data
    return probes, data_dict
```

### 1b. `Server.downstream` parameter

**Problem**: `Server.handle_queued_event()` returns `None` — it doesn't forward events. Every example that needs a pipeline must subclass `QueuedResource` and manually create downstream events (5+ lines per forwarding).

**Solution**: Add an optional `downstream` parameter to `Server`. When set, completed requests are automatically forwarded:
```python
sink = LatencyTracker("Sink")
server = Server("WebServer", concurrency=4, service_time=ExponentialLatency(0.05), downstream=sink)
# After processing, server automatically sends event to sink with context preserved
```

**File**: `happysimulator/components/server/server.py`

**Implementation**: Add `downstream: Entity | None = None` to `__init__`, modify `handle_queued_event()` to return a forwarded event when downstream is set:
```python
def __init__(self, name, concurrency=1, service_time=None, queue_policy=None,
             queue_capacity=None, downstream=None):
    # ... existing init ...
    self._downstream = downstream

@property
def downstream(self) -> Entity | None:
    return self._downstream

@downstream.setter
def downstream(self, target: Entity | None) -> None:
    self._downstream = target

def handle_queued_event(self, event):
    # ... existing processing logic ...

    # After yield + stats, forward if downstream is set
    if self._downstream is not None:
        return [Event(
            time=self.now,
            event_type=event.event_type,
            target=self._downstream,
            context=event.context,
        )]
    return None
```

### 1c. `Entity.forward()` helper

**Problem**: Downstream event forwarding is 5 lines of boilerplate that appears 10+ times across examples:
```python
completed = Event(time=self.now, event_type="Completed", target=self.downstream, context=event.context)
return [completed]
```

**Solution**: Add a helper on `Entity`:
```python
return [self.forward(event, target=self.downstream)]
return [self.forward(event, target=self.downstream, event_type="Completed")]
```

**File**: `happysimulator/core/entity.py`

**Implementation**:
```python
def forward(self, event: Event, target: "Entity", event_type: str | None = None) -> Event:
    """Create a forwarding event that preserves context.

    Args:
        event: The original event to forward.
        target: The downstream entity to receive the event.
        event_type: Override the event type (default: keep original).

    Returns:
        A new Event with the same context, targeted at the downstream entity.
    """
    return Event(
        time=self.now,
        event_type=event_type or event.event_type,
        target=target,
        context=event.context,
    )
```

### 1d. `Simulation` accepts `duration` (float)

**Problem**: Every simulation wraps a float in `Instant.from_seconds()`, and `start_time=Instant.Epoch` is the default 100% of the time:
```python
sim = Simulation(
    start_time=Instant.Epoch,
    end_time=Instant.from_seconds(100),
    sources=[source],
    entities=[server, sink],
    probes=[p1, p2],
)
```

**Solution**: Accept a `duration` float as a convenience:
```python
sim = Simulation(
    duration=100,
    sources=[source],
    entities=[server, sink],
    probes=[p1, p2],
)
```

**File**: `happysimulator/core/simulation.py`

**Implementation**: Add `duration: float | None = None` parameter. If provided and `end_time` is not, set `end_time = Instant.from_seconds(duration)`. Raise `ValueError` if both `end_time` and `duration` are provided.

```python
def __init__(
    self,
    start_time: Instant = None,
    end_time: Instant = None,
    duration: float | None = None,
    sources: list[Source] = None,
    entities: list[Simulatable] = None,
    probes: list[Source] = None,
    trace_recorder: TraceRecorder | None = None,
    fault_schedule: 'FaultSchedule | None' = None,
):
    if duration is not None and end_time is not None:
        raise ValueError("Cannot specify both 'duration' and 'end_time'")

    self._start_time = start_time or Instant.Epoch

    if duration is not None:
        self._end_time = Instant.from_seconds(duration)
    elif end_time is not None:
        self._end_time = end_time
    else:
        self._end_time = Instant.Infinity
    # ... rest unchanged ...
```

### 1e. Make `_SimpleEventProvider` public

**Problem**: `_SimpleEventProvider` exists inside `source.py` but is private. Examples rewrite it because they can't import it. When users need to customize context fields, they must write a full `EventProvider` subclass (15+ lines).

**Solution**:
1. Rename to `SimpleEventProvider` and export it
2. Add a `context_fn` parameter for custom context generation

**File**: `happysimulator/load/source.py`

**Implementation**:
```python
class SimpleEventProvider(EventProvider):
    """Event provider that creates targeted events with auto-incrementing IDs.

    For simple cases, use directly. For custom context, pass a context_fn:

        provider = SimpleEventProvider(
            target=server,
            context_fn=lambda time, count: {"created_at": time, "request_id": count, "priority": "high"}
        )

    Args:
        target: Entity to receive generated events.
        event_type: Type string for generated events.
        stop_after: Stop generating after this time.
        context_fn: Optional function(time: Instant, count: int) -> dict to generate custom context.
    """

    def __init__(self, target, event_type="Request", stop_after=None, context_fn=None):
        self._target = target
        self._event_type = event_type
        self._stop_after = stop_after
        self._context_fn = context_fn
        self._generated: int = 0

    def get_events(self, time):
        if self._stop_after is not None and time > self._stop_after:
            return []
        self._generated += 1
        if self._context_fn is not None:
            context = self._context_fn(time, self._generated)
        else:
            context = {"created_at": time, "request_id": self._generated}
        return [Event(time=time, event_type=self._event_type, target=self._target, context=context)]
```

Keep `_SimpleEventProvider` as an alias pointing to `SimpleEventProvider` so existing internal usage doesn't break.

### 1f. `Source.constant()` and `Source.poisson()` accept `event_provider`

**Problem**: The factory methods `Source.constant()` and `Source.poisson()` are convenient but only support the simple case (target + event_type). When you need a custom `EventProvider`, you fall back to the full 3-step constructor.

**Solution**: Allow passing an `event_provider` directly, making `target` and `event_type` optional when a provider is given:
```python
# Simple case (unchanged)
source = Source.constant(rate=10, target=server)

# Custom provider (new)
source = Source.constant(rate=10, event_provider=my_provider)
```

**File**: `happysimulator/load/source.py`

**Implementation**: In `constant()` and `poisson()`, if `event_provider` is passed, use it directly instead of creating a `SimpleEventProvider`. Raise `ValueError` if neither `target` nor `event_provider` is given.

```python
@classmethod
def constant(
    cls,
    rate: float,
    target: Entity | None = None,
    event_type: str = "Request",
    *,
    name: str = "Source",
    stop_after: float | Instant | None = None,
    event_provider: EventProvider | None = None,
) -> Source:
    if event_provider is None and target is None:
        raise ValueError("Either 'target' or 'event_provider' must be provided")

    if event_provider is None:
        stop_instant = cls._resolve_stop_after(stop_after)
        event_provider = SimpleEventProvider(target, event_type, stop_instant)

    return cls(
        name=name,
        event_provider=event_provider,
        arrival_time_provider=ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=rate), start_time=Instant.Epoch,
        ),
    )
```

Same pattern for `poisson()` and `with_profile()`.

---

## Phase 2: Analysis Tooling (`happysimulator/ai/`)

### 2a. `SimulationResult` — rich post-run wrapper

A convenience wrapper that bundles `SimulationSummary` + `SimulationAnalysis` + metric data. Users can construct it manually from any simulation's output.

```python
from happysimulator.ai import SimulationResult

# After running any simulation
summary = sim.run()
result = SimulationResult.from_run(
    summary,
    latency=tracker.data,
    queue_depth={"Server": depth_data},
)

# Or build directly
result = SimulationResult(
    summary=summary,
    analysis=analysis,
    latency=tracker.data,
    queue_depth={"Server": depth_data},
)
```

**File**: `happysimulator/ai/result.py`

```python
@dataclass
class SimulationResult:
    """Rich simulation result with analysis, comparison, and AI-friendly output.

    Works with any simulation — not tied to a specific builder pattern.
    """
    summary: SimulationSummary
    analysis: SimulationAnalysis
    latency: Data | None = None
    queue_depth: dict[str, Data] = field(default_factory=dict)
    throughput: Data | None = None

    @classmethod
    def from_run(
        cls,
        summary: SimulationSummary,
        *,
        latency: Data | None = None,
        queue_depth: dict[str, Data] | None = None,
        throughput: Data | None = None,
        **named_metrics: Data,
    ) -> "SimulationResult":
        """Create a SimulationResult by running analyze() automatically."""
        analysis = analyze(
            summary,
            latency=latency,
            queue_depth=next(iter(queue_depth.values())) if queue_depth else None,
            throughput=throughput,
            **named_metrics,
        )
        return cls(
            summary=summary,
            analysis=analysis,
            latency=latency,
            queue_depth=queue_depth or {},
            throughput=throughput,
        )

    def to_prompt_context(self, max_tokens: int = 2000) -> str:
        """Generate AI-optimized summary text."""
        # Delegates to analysis.to_prompt_context() + adds recommendations

    def to_dict(self) -> dict[str, Any]:
        """Structured data for programmatic access."""

    def compare(self, other: "SimulationResult") -> "SimulationComparison":
        """Compare this result with another."""
```

### 2b. `SimulationComparison`

```python
@dataclass
class SimulationComparison:
    result_a: SimulationResult
    result_b: SimulationResult
    metric_diffs: dict[str, MetricDiff]

    def to_prompt_context(self, max_tokens: int = 2000) -> str
    def to_dict(self) -> dict[str, Any]

@dataclass
class MetricDiff:
    name: str
    mean_a: float
    mean_b: float
    mean_change_pct: float
    p99_a: float
    p99_b: float
    p99_change_pct: float
```

Example `comparison.to_prompt_context()` output:
```
## Simulation Comparison

| Metric | Run A | Run B | Change |
|--------|-------|-------|--------|
| latency (mean) | 0.0523s | 0.0312s | -40.3% |
| latency (p99) | 0.5234s | 0.1205s | -77.0% |
| queue_depth (mean) | 12.3 | 3.1 | -74.8% |
| throughput | 10.0/s | 10.0/s | +0.0% |

## Key Differences
- Run B has 77% lower tail latency (p99)
- Queue depth reduced significantly, suggesting less congestion
```

### 2c. `SweepResult`

For comparing multiple runs across a parameter range:

```python
@dataclass
class SweepResult:
    parameter_name: str
    parameter_values: list[Any]
    results: list[SimulationResult]

    def to_prompt_context(self, max_tokens: int = 2000) -> str
    def to_dict(self) -> dict[str, Any]
    def best_by(self, metric: str = "latency", stat: str = "p99") -> SimulationResult
```

### 2d. Recommendations Engine

Rules-based analysis that generates actionable suggestions:

**File**: `happysimulator/ai/insights.py`

```python
def generate_recommendations(result: SimulationResult) -> list[Recommendation]:
    """Analyze results and suggest improvements."""

@dataclass
class Recommendation:
    category: str       # "capacity", "architecture", "configuration"
    description: str
    confidence: str     # "high", "medium", "low"
    suggested_change: str
```

Rules:
- **Queue saturation**: If mean queue depth in last 20% of sim > 2x first 20%, recommend more capacity
- **Underutilization**: If utilization < 30%, recommend fewer servers
- **Tail latency**: If p99/p50 ratio > 10x, recommend investigating variance or adding concurrency
- **Phase transitions**: If `analysis.phases` contains a "degraded" phase, recommend capacity planning

Recommendations are included automatically in `SimulationResult.to_prompt_context()`.

---

## Phase 3: Export to `__init__`

### Files to modify
- `happysimulator/ai/__init__.py` — export `SimulationResult`, `SimulationComparison`, `SweepResult`, `Recommendation`, `generate_recommendations`
- `happysimulator/__init__.py` — add `SimulationResult` and `SimpleEventProvider` to top-level exports

---

## Phase 4: MCP Server (`happysimulator/mcp/`)

An MCP server exposing simulation as callable tools. Each tool accepts JSON parameters, constructs entities using the real API (with the ergonomic improvements), runs the simulation, wraps in `SimulationResult`, and returns structured output.

### Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `simulate_queue` | Run an M/M/1 or M/M/c queue | `arrival_rate`, `service_rate`, `servers`, `duration`, `seed` |
| `simulate_pipeline` | Run a multi-stage pipeline | `stages` (list of {name, concurrency, service_time}), `source_rate`, `duration` |
| `sweep_parameter` | Run parametric sweep | `base` (queue config), `parameter`, `values` |
| `compare_scenarios` | Compare two queue configs | `scenario_a`, `scenario_b` |
| `list_distributions` | List available service time distributions | (none) |

### Tool Response Format

```json
{
  "prompt_context": "## Simulation Results\n...",
  "data": { "summary": {...}, "analysis": {...}, "recommendations": [...] }
}
```

### Implementation

Each MCP tool handler is a thin function that:
1. Constructs `Server`, `Source`, `LatencyTracker`, `Probe` using the real API
2. Runs `Simulation(duration=..., ...)`
3. Wraps output in `SimulationResult.from_run()`
4. Returns `result.to_prompt_context()` + `result.to_dict()`

Example `simulate_queue` handler (showing how the ergonomic improvements help):
```python
async def simulate_queue(arrival_rate, service_rate, servers=1, duration=100, seed=None):
    if seed is not None:
        random.seed(seed)

    tracker = LatencyTracker("Sink")
    server = Server("Server", concurrency=servers,
                    service_time=ExponentialLatency(1.0 / service_rate),
                    downstream=tracker)  # Phase 1b!
    source = Source.poisson(rate=arrival_rate, target=server)  # already existed
    probe, depth_data = Probe.on(server, "depth", interval=0.5)  # Phase 1a!

    summary = Simulation(duration=duration, sources=[source],  # Phase 1d!
                         entities=[server, tracker], probes=[probe]).run()

    result = SimulationResult.from_run(summary, latency=tracker.data,
                                       queue_depth={"Server": depth_data})
    return {"prompt_context": result.to_prompt_context(), "data": result.to_dict()}
```

That's ~12 lines of real API code — no parallel builder abstraction needed.

- Uses `mcp` Python SDK (`pip install mcp`)
- Entry point: `python -m happysimulator.mcp`
- Server uses stdio transport

### Key files to create
- `happysimulator/mcp/__init__.py`
- `happysimulator/mcp/server.py` — MCP server with tool definitions
- `happysimulator/mcp/__main__.py` — entry point

---

## File Summary

### New files
| File | Description |
|------|-------------|
| `happysimulator/ai/__init__.py` | Package exports |
| `happysimulator/ai/result.py` | `SimulationResult`, `SimulationComparison`, `SweepResult`, `MetricDiff` |
| `happysimulator/ai/insights.py` | `generate_recommendations()`, `Recommendation` |
| `happysimulator/mcp/__init__.py` | Package exports |
| `happysimulator/mcp/server.py` | MCP server with tool definitions |
| `happysimulator/mcp/__main__.py` | `python -m happysimulator.mcp` entry point |
| `tests/unit/test_ergonomic_api.py` | Tests for Probe.on, Server.downstream, Entity.forward, Simulation(duration=) |
| `tests/unit/test_simulation_result.py` | Result/comparison/sweep tests |
| `tests/unit/test_insights.py` | Recommendations tests |
| `tests/unit/test_mcp_server.py` | MCP tool tests |

### Modified files
| File | Change |
|------|--------|
| `happysimulator/instrumentation/probe.py` | Add `Probe.on()` and `Probe.on_many()` |
| `happysimulator/components/server/server.py` | Add `downstream` parameter |
| `happysimulator/core/entity.py` | Add `Entity.forward()` helper |
| `happysimulator/core/simulation.py` | Add `duration` parameter |
| `happysimulator/load/source.py` | Make `SimpleEventProvider` public, add `event_provider` to factory methods |
| `happysimulator/__init__.py` | Export `SimulationResult`, `SimpleEventProvider` |

---

## Implementation Order

1. **Phase 1**: Ergonomic improvements (Probe.on, Server.downstream, Entity.forward, Simulation duration, SimpleEventProvider)
2. **Phase 2**: Analysis tooling (SimulationResult, SimulationComparison, SweepResult, recommendations)
3. **Phase 3**: Wire into `__init__.py`
4. **Phase 4**: MCP server
5. **Phase 5**: Tests
6. **Phase 6**: Run full test suite to verify no regressions

## Before/After: M/M/1 Queue Setup

### Before (current API)
```python
from happysimulator import *

sink = LatencyTracker("Sink")

class MM1Server(QueuedResource):
    def __init__(self, name, mean_service_time, downstream):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.mean_service_time = mean_service_time
    def handle_queued_event(self, event):
        yield random.expovariate(1.0 / self.mean_service_time)
        return [Event(time=self.now, event_type="Done", target=self.downstream, context=event.context)]

server = MM1Server("Server", 1/12, sink)

class RequestProvider(EventProvider):
    def __init__(self, target):
        self._target = target
        self._count = 0
    def get_events(self, time):
        self._count += 1
        return [Event(time=time, event_type="Request", target=self._target,
                      context={"created_at": time, "request_id": self._count})]

provider = RequestProvider(server)
profile = ConstantRateProfile(rate=10)
arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1, start_time=Instant.Epoch)

sim = Simulation(start_time=Instant.Epoch, end_time=Instant.from_seconds(100),
                 sources=[source], entities=[server, sink], probes=[depth_probe])
summary = sim.run()
```
**~30 lines**

### After (with ergonomic improvements)
```python
from happysimulator import *

tracker = LatencyTracker("Sink")
server = Server("Server", concurrency=1, service_time=ExponentialLatency(1/12), downstream=tracker)
source = Source.poisson(rate=10, target=server)
depth_probe, depth_data = Probe.on(server, "depth", interval=0.1)

sim = Simulation(duration=100, sources=[source], entities=[server, tracker], probes=[depth_probe])
summary = sim.run()
```
**7 lines** — same real API, no new abstraction layer, works for all simulation types.

## Verification

1. **Unit tests**: Test each ergonomic addition independently
   - `Probe.on()`: creates correct Probe + Data pair
   - `Server.downstream`: forwards events with context preserved
   - `Entity.forward()`: produces correct Event
   - `Simulation(duration=)`: sets correct end_time
   - `SimpleEventProvider`: generates correct events with custom context_fn
   - `SimulationResult`: `to_prompt_context()`, `to_dict()`, `compare()`
   - Recommendations: each rule fires correctly
   - MCP: tool handlers return correct JSON
2. **Integration test**: Run the "after" M/M/1 example end-to-end, verify results match existing behavior
3. **Existing tests**: `pytest -q` to verify no regressions (~1719 tests)
