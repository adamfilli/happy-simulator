# Inductor Burst Suppression — Visual Debugger Example

## Context

The Inductor is a novel component that smooths bursty traffic using EWMA rate estimation — unlike traditional rate limiters, it has **no throughput cap** and only resists rapid rate *changes*. The existing integration test (`tests/integration/test_inductor.py`) validates this behavior under multiple load profiles and generates static PNG plots. This example brings those same scenarios to the interactive browser-based visual debugger so users can step through and observe the inductor's behavior in real-time.

## New File

`examples/inductor_burst_suppression.py` (~90 lines)

## Design

### Pipeline

```
Source (LoadGenerator) → InputCounter → Inductor (τ=2s) → ThroughputTracker (OutputTracker)
                                             ↑
                                    Probe: estimated_rate
                                    Probe: queue_depth
```

- **InputCounter** — Minimal passthrough entity (defined inline, ~10 lines). Records `1.0` per event to its own `Data` object, then forwards the event to the Inductor. Needed because `ThroughputTracker` is terminal (returns `[]`), so we can't use it as a passthrough.
- **ThroughputTracker** — Existing collector from `happysimulator.instrumentation.collectors`. Serves as the downstream sink and records output throughput.

### Multi-Phase Profile (`InductorShowcaseProfile`)

A single `Profile` subclass with three demo phases separated by 10s cooldowns at base rate:

| Time | Phase | Rate | What it demonstrates |
|------|-------|------|---------------------|
| 0–15s | Step-up (low) | 10 req/s | Baseline |
| 15–40s | Step-up (high) | 50 req/s | Output slowly approaches 50; queue builds during transition |
| 40–50s | Cooldown | 10 req/s | EWMA resets toward baseline |
| 50–100s | Linear ramp | 10→50 req/s | Output tracks input with no limiting; queue stays near 0 |
| 100–110s | Cooldown | 10 req/s | EWMA resets toward baseline |
| 110–180s | Periodic bursts | 10/80 req/s (period=10s, burst=2s) | Full up/down dampening visible |

Total simulation: **180s**. Uses `poisson=False` for deterministic arrivals (matching the integration tests).

### Charts (4 total)

| # | Title | Data Source | Transform | Color | Purpose |
|---|-------|------------|-----------|-------|---------|
| 1 | Input Throughput | `InputCounter.data` | `rate`, window=1s | `#6366f1` (indigo) | Raw input rate — what the load generator produces |
| 2 | Output Throughput | `ThroughputTracker.data` | `rate`, window=1s | `#10b981` (emerald) | Smoothed output — what passes through the inductor |
| 3 | EWMA Rate Estimate | Probe `estimated_rate` | `raw` | `#f59e0b` (amber) | The inductor's internal rate model |
| 4 | Queue Depth | Probe `queue_depth` | `raw` | `#ef4444` (red) | Buffering during rate suppression |

Both probes sample at 0.1s intervals.

### Key Parameters

- `time_constant=2.0` — matches integration tests; provides visible smoothing without being overly aggressive (~6-8s full adaptation)
- `queue_capacity=10_000` — default, more than sufficient for this demo

## Files to Reference

- `examples/visual_debugger.py` — pattern for inline entities, serve() call, Chart wiring
- `happysimulator/components/rate_limiter/inductor.py` — `estimated_rate`, `queue_depth` properties
- `happysimulator/instrumentation/collectors.py` — `ThroughputTracker`
- `happysimulator/visual/dashboard.py` — `Chart` API and transforms

## Verification

1. Run `python examples/inductor_burst_suppression.py`
2. Browser opens at `http://127.0.0.1:8765`
3. Click "Play" and observe all 4 charts updating through the three phases
4. Verify Phase 1: Input jumps to 50, output rises slowly, queue spikes
5. Verify Phase 2: Input ramps linearly, output tracks closely, queue near 0
6. Verify Phase 3: Input oscillates 10/80, output dampened, queue pulses
