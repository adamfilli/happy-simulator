# Performance Testing Infrastructure

## Context

The codebase review (`.dev/codebase-review.md`) identified several performance bottlenecks in the simulation engine's hot path — event memory footprint, cancelled event heap bloat, logging overhead, unbounded context stack growth, etc. Before fixing these, we need a repeatable way to measure performance so we can quantify improvements and catch regressions.

This plan creates a benchmark suite with both automated reporting (before/after comparison) and interactive profiling support, using **only stdlib tools** (no new dependencies).

## File Structure

```
tests/perf/
├── __init__.py
├── __main__.py              # CLI entry point: python -m tests.perf
├── runner.py                # Orchestrates scenarios, collects results, generates report
├── scenarios/
│   ├── __init__.py
│   ├── throughput.py        # Pure event loop speed (1M events, no instrumentation)
│   ├── generator_heavy.py   # Multi-yield generators (ProcessContinuation allocation)
│   ├── instrumented.py      # Same workload + LatencyTracker + Probes (Data collection cost)
│   ├── memory_footprint.py  # Event object memory measurement (tracemalloc)
│   ├── large_heap.py        # Many concurrent scheduled events (heap scaling)
│   └── cancellation.py      # Heavy timeout/cancel pattern (cancelled event bloat)
└── baseline.json            # Saved baseline for comparison (gitignored)
```

## Benchmark Scenarios

Each scenario is a function that returns a standardized `BenchmarkResult` dataclass.

### 1. `throughput` — Pure Event Loop Speed
- **What**: Simple M/M/1 queue, 500K events, no probes/trackers, deterministic (seeded)
- **Why**: Baseline for core event loop: `EventHeap.pop()` → `Event.invoke()` → `Entity.handle_event()` → `EventHeap.push()`
- **Measures**: events/sec, wall clock time, peak memory (via `tracemalloc`)
- **Entity**: Minimal `QueuedResource` subclass with `yield 0.0` (near-zero service time to maximize event throughput)
- **Key files exercised**: `simulation.py:_run_loop()`, `event_heap.py`, `event.py:invoke()/__lt__()`, `temporal.py`

### 2. `generator_heavy` — Generator Yield Overhead
- **What**: Same setup but each `handle_event` yields 5 times (creating 5 `ProcessContinuation` objects per event)
- **Why**: Measures overhead of `ProcessContinuation` allocation per yield, which the review flagged
- **Measures**: events/sec, wall clock time, peak memory
- **Key files exercised**: `event.py:ProcessContinuation`, `_normalize_yield()`

### 3. `instrumented` — Instrumentation Overhead
- **What**: Same as throughput but with `LatencyTracker` on sink + `Probe` sampling queue depth every 0.01s
- **Why**: Measures cost of `Data.add_stat()` on every event, unbounded sample list growth
- **Measures**: events/sec, wall clock time, peak memory (memory diff vs throughput scenario shows instrumentation cost)
- **Key files exercised**: `instrumentation/data.py`, `instrumentation/probe.py`, `instrumentation/latency_tracker.py`

### 4. `memory_footprint` — Event Object Size
- **What**: Create 100K Event objects in a list, measure total memory via `tracemalloc`
- **Why**: Directly measures the "~500 bytes/event" concern from the review — lazy context/on_complete/trace fields should reduce this
- **Measures**: bytes/event, total memory for N events
- **Key files exercised**: `event.py:Event.__init__()`, `temporal.py:Instant`

### 5. `large_heap` — Heap Scaling
- **What**: Schedule 100K future events spanning a wide time range, then process them all
- **Why**: Tests heap performance when the heap is large (many concurrent pending events)
- **Measures**: events/sec, peak heap size, wall clock time
- **Key files exercised**: `event_heap.py:push()/pop()`, `Event.__lt__()`

### 6. `cancellation` — Cancelled Event Bloat
- **What**: Entity that schedules timeout events and cancels ~80% of them before they fire
- **Why**: Directly measures the "cancelled events stay in heap" issue from the review
- **Measures**: events/sec, peak heap size vs actual live events, wall clock time, memory
- **Key files exercised**: `event_heap.py`, `event.py:cancel()`

## Data Model

```python
@dataclass
class BenchmarkResult:
    name: str
    events_processed: int
    wall_clock_s: float
    events_per_second: float
    peak_memory_mb: float
    extra: dict[str, float]  # scenario-specific metrics (bytes/event, heap_size, etc.)
```

## Runner & Report (`runner.py`)

The runner:
1. Discovers all scenarios from the `scenarios/` package
2. Runs each scenario, collecting `BenchmarkResult`
3. Prints a console table
4. Optionally compares against a saved baseline (`baseline.json`)
5. Optionally saves current results as the new baseline

### Console Report Format

```
========================================================================
  HAPPY-SIMULATOR PERFORMANCE REPORT
  Python 3.13.x | 2026-02-14 12:34:56
========================================================================

  Scenario              Events/sec    Peak Mem (MB)   Wall (s)   vs Baseline
  --------------------  ----------    -------------   --------   -----------
  throughput              845,230          45.2         0.592     +12.3%
  generator_heavy         312,100          78.4         1.602     +8.1%
  instrumented            523,400         124.8         0.956     -2.1%
  memory_footprint             —          38.7            —      -15.4%
  large_heap              678,900          52.1         0.147     (new)
  cancellation            401,200          67.3         1.245     (new)

  Extra Metrics:
    memory_footprint: 387 bytes/event
    large_heap: peak_heap_size=100000
    cancellation: cancelled_ratio=0.80, heap_bloat_ratio=2.4x

========================================================================
```

When a baseline exists, the `vs Baseline` column shows percentage change in events/sec (or peak memory for memory_footprint), with improvement shown as positive.

## CLI Interface (`__main__.py`)

```
python -m tests.perf                     # run all scenarios, print report
python -m tests.perf --scenario throughput  # run one scenario
python -m tests.perf --save-baseline     # save results as baseline.json
python -m tests.perf --compare           # compare against saved baseline (default when baseline exists)
python -m tests.perf --profile           # run with cProfile, save .prof files to test_output/perf/
python -m tests.perf --tracemalloc-top 20  # show top 20 memory allocators after each scenario
python -m tests.perf --scale 2.0         # multiply event counts by factor (for longer/shorter runs)
python -m tests.perf --json              # output results as JSON to stdout
```

### Interactive Profiling Mode (`--profile`)

When `--profile` is passed:
- Wraps each scenario in `cProfile.Profile()`
- Saves `.prof` file to `test_output/perf/<scenario_name>.prof`
- Prints top 20 cumulative-time functions to console
- `.prof` files can be visualized with `snakeviz` (`pip install snakeviz && snakeviz test_output/perf/throughput.prof`) or converted to flame graphs

### Memory Profiling Mode (`--tracemalloc-top N`)

When `--tracemalloc-top` is passed:
- Uses `tracemalloc.take_snapshot()` after each scenario
- Prints top N memory-allocating lines
- Useful for finding where memory is being consumed (e.g., Event.__init__, Data.add_stat)

## Implementation Details

### No New Dependencies

Everything uses Python stdlib:
- `time.perf_counter()` for wall clock timing
- `tracemalloc` for memory measurement (already in stdlib)
- `cProfile` / `pstats` for CPU profiling (already in stdlib)
- `json` for baseline save/load

### Determinism

All scenarios use `random.seed(42)` and `Source.constant()` or `ConstantArrivalTimeProvider` for deterministic timing. This ensures results are comparable across runs (variance comes only from system load).

### Warm-up

Each scenario runs a small warm-up (1000 events) before the measured run to ensure JIT/import/caching effects don't skew results. Only the main run is measured.

### Baseline File

`baseline.json` stores results as:
```json
{
  "timestamp": "2026-02-14T12:34:56",
  "python_version": "3.13.1",
  "results": {
    "throughput": {"events_per_second": 845230, "peak_memory_mb": 45.2, ...},
    ...
  }
}
```

This file is gitignored since baselines are machine-specific.

## Files to Create

1. `tests/perf/__init__.py` — empty
2. `tests/perf/__main__.py` — argparse CLI, imports runner
3. `tests/perf/runner.py` — `BenchmarkResult` dataclass, `run_all()`, `print_report()`, `save_baseline()`, `load_baseline()`, `compare()`
4. `tests/perf/scenarios/__init__.py` — scenario registry (list of scenario functions)
5. `tests/perf/scenarios/throughput.py` — `run() -> BenchmarkResult`
6. `tests/perf/scenarios/generator_heavy.py` — `run() -> BenchmarkResult`
7. `tests/perf/scenarios/instrumented.py` — `run() -> BenchmarkResult`
8. `tests/perf/scenarios/memory_footprint.py` — `run() -> BenchmarkResult`
9. `tests/perf/scenarios/large_heap.py` — `run() -> BenchmarkResult`
10. `tests/perf/scenarios/cancellation.py` — `run() -> BenchmarkResult`
11. `.gitignore` update — add `tests/perf/baseline.json`

## Files to Modify

- None — this is purely additive infrastructure

## Existing Patterns to Reuse

- `SimulationSummary` (`happysimulator/instrumentation/summary.py`) — already tracks `events_per_second`, `wall_clock_seconds`, `total_events_processed`
- `Source.constant()` — deterministic arrival times for reproducible benchmarks
- `QueuedResource` + `FIFOQueue` — standard server pattern from examples
- `LatencyTracker` / `Probe` / `Data` — for the instrumented scenario
- `test_output/` directory convention from `conftest.py` — profile output goes to `test_output/perf/`

## Verification

After implementation:
1. `python -m tests.perf` — should run all 6 scenarios and print the report table
2. `python -m tests.perf --save-baseline` — should save `baseline.json`
3. `python -m tests.perf --compare` — should show deltas against baseline (all ~0% since nothing changed)
4. `python -m tests.perf --scenario throughput --profile` — should produce `test_output/perf/throughput.prof` and print top functions
5. `python -m tests.perf --tracemalloc-top 10` — should show top memory allocators
6. `pytest -q` — existing tests still pass (no interference)
