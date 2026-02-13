# AI Integration Layer for happysimulator

## Context

AI models can reason more effectively about distributed systems, queuing theory, and performance when they can *run simulations as part of their thinking process*. Today, using happysimulator requires writing 30-80+ lines of Python with custom Entity subclasses, EventProviders, Profiles, manual Probe/Data wiring, and post-hoc analysis calls. This friction makes it impractical for an AI to use the simulator as a "thinking tool" during a conversation.

This plan adds three layers:
1. **Scenario Builder** — Express common simulations in 1-5 lines
2. **ScenarioResult** — Rich result object with auto-analysis and comparison
3. **MCP Server** — Expose simulation as callable tools for AI models

## Phase 1: Scenario Builder (`happysimulator/ai/scenario.py`)

A fluent API that auto-wires entities, instrumentation, sources, and analysis. Leverages existing `Server`, `Source`, `LatencyTracker`, `Probe`, `Data`, and `analyze()`.

### API Design

```python
from happysimulator.ai import Scenario

# Quick M/M/1 queue
result = Scenario.queue(arrival_rate=10, service_rate=12, duration=100).run()

# M/M/c queue
result = Scenario.queue(arrival_rate=50, service_rate=12, servers=5, duration=100).run()

# Multi-stage pipeline
result = (Scenario.pipeline()
    .source(rate=10, poisson=True)
    .server("WebServer", concurrency=4, service_time=0.05)
    .server("Database", concurrency=2, service_time=0.02)
    .duration(100)
    .run())

# Load balancer + server pool
result = (Scenario.pipeline()
    .source(rate=100)
    .load_balancer("LB", algorithm="round_robin")
    .server_pool("Workers", count=5, concurrency=2, service_time=0.1)
    .duration(60)
    .run())

# Custom processing via callback (no Entity subclass needed)
result = (Scenario.pipeline()
    .source(rate=10)
    .processor("Filter", fn=lambda event: 0.01 if event.context.get("priority") == "high" else None)
    .server("Backend", service_time=0.05)
    .duration(60)
    .run())

# Load profiles (spikes, ramps)
result = (Scenario.queue(service_rate=12, duration=120)
    .arrival_profile(base_rate=5, spike_rate=25, spike_start=30, spike_end=50)
    .run())

# Parametric sweep
results = (Scenario.queue(service_rate=12, duration=100)
    .sweep("arrival_rate", [8, 10, 12, 14, 16])
    .run())

# Seed for reproducibility
result = Scenario.queue(arrival_rate=10, service_rate=12, duration=100, seed=42).run()
```

### Implementation

The `Scenario` class builds a configuration (list of stages, source config, duration), then `.run()` constructs the real entities, wires instrumentation automatically, runs the `Simulation`, and calls `analyze()`:

- Auto-creates `LatencyTracker` as the terminal sink
- Auto-creates `Probe` for queue depth on every `Server`/`QueuedResource`
- Auto-creates `Data` objects for all tracked metrics
- Auto-runs `analyze(summary, latency=..., queue_depth=...)` after simulation
- Returns a `ScenarioResult` (not raw `SimulationSummary`)

### Key files to create
- `happysimulator/ai/__init__.py` — exports `Scenario`, `ScenarioResult`
- `happysimulator/ai/scenario.py` — `Scenario` class with factory methods and builder
- `happysimulator/ai/builders.py` — Internal builder logic that constructs entities from config

### Key files to reuse
- `happysimulator/components/server/server.py` — `Server` (already has concurrency + service_time distribution)
- `happysimulator/load/source.py:150-258` — `Source.constant()`, `Source.poisson()`, `Source.with_profile()`
- `happysimulator/instrumentation/collectors.py` — `LatencyTracker`
- `happysimulator/instrumentation/probe.py` — `Probe`
- `happysimulator/instrumentation/data.py` — `Data`
- `happysimulator/analysis/report.py:201-294` — `analyze()`
- `happysimulator/components/common.py` — `Sink`, `Counter`
- `happysimulator/distributions/` — `ConstantLatency`, `ExponentialLatency`
- `happysimulator/load/profile.py` — `SpikeProfile`, `LinearRampProfile`, `ConstantRateProfile`

## Phase 2: ScenarioResult (`happysimulator/ai/result.py`)

Rich result object that wraps `SimulationSummary` + `SimulationAnalysis` + raw metric data.

```python
@dataclass
class ScenarioResult:
    summary: SimulationSummary
    analysis: SimulationAnalysis

    # Direct metric access (auto-collected)
    latency: Data
    queue_depth: dict[str, Data]  # keyed by server name

    # Convenience
    def to_prompt_context(self, max_tokens: int = 2000) -> str
    def to_dict(self) -> dict[str, Any]

    # Comparison
    def compare(self, other: ScenarioResult) -> ScenarioComparison
```

### ScenarioComparison

Structural diff between two simulation results:

```python
@dataclass
class ScenarioComparison:
    """Side-by-side comparison of two scenario runs."""
    result_a: ScenarioResult
    result_b: ScenarioResult
    metric_diffs: dict[str, MetricDiff]  # per-metric deltas

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

Example output from `comparison.to_prompt_context()`:
```
## Scenario Comparison

| Metric | Scenario A | Scenario B | Change |
|--------|-----------|-----------|--------|
| latency (mean) | 0.0523s | 0.0312s | -40.3% |
| latency (p99) | 0.5234s | 0.1205s | -77.0% |
| queue_depth (mean) | 12.3 | 3.1 | -74.8% |
| throughput | 10.0/s | 10.0/s | +0.0% |

## Key Differences
- Scenario B has 77% lower tail latency (p99)
- Queue depth reduced significantly, suggesting less congestion
```

### SweepResult

For parametric sweeps:

```python
@dataclass
class SweepResult:
    parameter_name: str
    parameter_values: list[Any]
    results: list[ScenarioResult]

    def to_prompt_context(self, max_tokens: int = 2000) -> str
    def to_dict(self) -> dict[str, Any]
    def best_by(self, metric: str = "latency", stat: str = "p99") -> ScenarioResult
```

Example output from `sweep.to_prompt_context()`:
```
## Parameter Sweep: arrival_rate

| arrival_rate | latency_mean | latency_p99 | queue_depth_mean | throughput |
|-------------|-------------|-------------|-----------------|-----------|
| 8 | 0.012s | 0.035s | 0.3 | 8.0/s |
| 10 | 0.052s | 0.523s | 12.3 | 10.0/s |
| 12 | 0.891s | 4.210s | 89.2 | 11.8/s |  <-- saturation
| 14 | 2.340s | 12.10s | 234.0 | 11.9/s |

## Observations
- System saturates between arrival_rate=10 and arrival_rate=12
- At arrival_rate=12 (100% utilization), p99 latency increases 12x
```

### Key files to create
- `happysimulator/ai/result.py` — `ScenarioResult`, `ScenarioComparison`, `SweepResult`, `MetricDiff`

## Phase 3: Enhanced Analysis (`happysimulator/ai/insights.py`)

Build on the existing `analyze()` to add AI-specific reasoning helpers:

### Recommendations Engine

```python
def generate_recommendations(result: ScenarioResult) -> list[Recommendation]:
    """Analyze results and suggest improvements."""

@dataclass
class Recommendation:
    category: str  # "capacity", "architecture", "configuration"
    description: str
    confidence: str  # "high", "medium", "low"
    suggested_change: str  # e.g. "increase servers from 3 to 5"
```

Rules-based recommendations:
- **Queue saturation**: If queue depth grows monotonically, recommend lower arrival rate or more servers
- **Underutilization**: If utilization < 30%, recommend fewer servers
- **Tail latency**: If p99/p50 ratio > 10x, recommend investigating variance
- **Phase transitions**: If degraded phases detected, recommend capacity planning around the transition point

The recommendations are included in `ScenarioResult.to_prompt_context()` automatically.

### Key files to create
- `happysimulator/ai/insights.py` — `generate_recommendations()`, `Recommendation`

### Key files to modify
- `happysimulator/ai/result.py` — integrate recommendations into `to_prompt_context()`

## Phase 4: Export to `__init__` and add `Scenario` to main API

### Key files to modify
- `happysimulator/__init__.py` — add `Scenario`, `ScenarioResult` to exports
- `happysimulator/ai/__init__.py` — module exports

## Phase 5: MCP Server (`happysimulator/mcp/`)

An MCP server exposing simulation tools. Each tool accepts JSON parameters, runs a simulation using the Scenario builder, and returns structured results.

### Tools

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `simulate_queue` | Run an M/M/1 or M/M/c queue simulation | `arrival_rate`, `service_rate`, `servers`, `duration`, `seed` |
| `simulate_pipeline` | Run a multi-stage pipeline | `stages` (list of {name, concurrency, service_time}), `rate`, `duration` |
| `sweep_parameter` | Run parametric sweep | `base_scenario`, `parameter`, `values` |
| `compare_scenarios` | Compare two scenario configurations | `scenario_a`, `scenario_b` |
| `list_distributions` | List available service time distributions | (none) |

### Tool Response Format

Every tool returns:
```json
{
  "prompt_context": "## Simulation Summary\n...",
  "data": { "summary": {...}, "analysis": {...}, "metrics": {...} }
}
```

The `prompt_context` field is the primary interface for AI reasoning. The `data` field provides structured access for follow-up analysis.

### Example Tool Call

```json
{
  "tool": "simulate_queue",
  "parameters": {
    "arrival_rate": 10,
    "service_rate": 12,
    "duration": 100,
    "seed": 42
  }
}
```

Returns:
```json
{
  "prompt_context": "## Simulation Summary\n- Duration: 100.00s\n- Events processed: 1003\n...\n\n## Recommendations\n- System is operating at 83% utilization...",
  "data": {
    "summary": {"duration_s": 100.0, "total_events_processed": 1003, ...},
    "analysis": {"phases": {...}, "anomalies": [...], ...},
    "recommendations": [...]
  }
}
```

### Implementation

- Use the `mcp` Python SDK (`pip install mcp`)
- Entry point: `python -m happysimulator.mcp`
- Each tool is a thin wrapper around `Scenario` builder methods
- Server uses stdio transport (standard for MCP)

### Key files to create
- `happysimulator/mcp/__init__.py`
- `happysimulator/mcp/server.py` — MCP server with tool definitions
- `happysimulator/mcp/__main__.py` — entry point

## File Summary

### New files
| File | Description |
|------|-------------|
| `happysimulator/ai/__init__.py` | Package exports |
| `happysimulator/ai/scenario.py` | `Scenario` builder class |
| `happysimulator/ai/builders.py` | Internal entity construction from config |
| `happysimulator/ai/result.py` | `ScenarioResult`, `ScenarioComparison`, `SweepResult` |
| `happysimulator/ai/insights.py` | `generate_recommendations()` |
| `happysimulator/mcp/__init__.py` | Package exports |
| `happysimulator/mcp/server.py` | MCP server with tool definitions |
| `happysimulator/mcp/__main__.py` | `python -m happysimulator.mcp` entry point |
| `tests/unit/test_scenario.py` | Scenario builder tests |
| `tests/unit/test_scenario_result.py` | Result/comparison tests |
| `tests/unit/test_insights.py` | Recommendations tests |
| `tests/unit/test_mcp_server.py` | MCP tool tests |
| `examples/ai_scenario_builder.py` | Example showcasing the new API |

### Modified files
| File | Change |
|------|--------|
| `happysimulator/__init__.py` | Add `Scenario`, `ScenarioResult` to exports |

## Implementation Order

1. **Phase 1**: `Scenario` builder + internal builders — the foundation everything else depends on
2. **Phase 2**: `ScenarioResult`, `ScenarioComparison`, `SweepResult` — make results useful
3. **Phase 3**: Recommendations engine — enrich output for AI consumption
4. **Phase 4**: Wire into `__init__.py` — make it importable from top level
5. **Phase 5**: MCP server — expose as tools
6. **Phase 6**: Tests + example

## Verification

1. **Unit tests**: Each phase gets dedicated tests
   - Scenario builder: test that `.queue()`, `.pipeline()`, `.sweep()` produce correct entity graphs
   - ScenarioResult: test `to_prompt_context()`, `to_dict()`, `compare()`
   - Recommendations: test each rule fires correctly
   - MCP: test tool handlers return correct JSON
2. **Integration test**: Run `Scenario.queue(arrival_rate=10, service_rate=12, duration=100, seed=42).run()` end-to-end and verify deterministic results
3. **Example**: Run `examples/ai_scenario_builder.py` and verify output
4. **Existing tests**: `pytest -q` to verify no regressions (~1719 tests)
