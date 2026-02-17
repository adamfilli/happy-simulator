# Performance Audit & Optimization Plan — happysimulator Core Engine

## Context

The simulation core runs at **~50K events/sec** (Python 3.14). Profiling the `throughput` benchmark (0.2x scale, 54.3s) reveals that **~40% of CPU time** is spent on overhead unrelated to actual simulation logic: UUID generation (7.2%), application-level tracing (2.9%), dict lookups for trace context (5.8%), and run-loop branching/attribute lookups (17.7%). Memory is also bloated: 745 bytes/event, with `Instant` objects using 128 bytes each (vs 40 with `__slots__`).

The goal is a **25-40% throughput improvement** and **30-50% memory reduction** through targeted optimizations, plus a parallelism utility for multi-run workloads.

---

## Profiling Summary

| Hotspot | % of CPU | Root Cause |
|---------|----------|------------|
| `_run_loop` self-time | 17.7% | Attribute lookups, control checks, trace calls |
| `uuid.uuid4()` + `str()` | 7.2% | Called for every Event incl. ProcessContinuations |
| `dict.get()` (6.7M calls) | 5.8% | `context.get("id")` in trace.record() calls |
| `Event.invoke()` | 7.1% | Unconditional trace/stack building |
| `Event.__init__` | 6.6% | UUID + context setup + on_complete list |
| `Event.trace()` (2.1M calls) | 2.9% | Dict alloc + append per call, never read |
| `ProcessContinuation.__init__` | 3.7% | Calls full `Event.__init__`, wastes UUID |

---

## Implementation Plan

### Step 1: Save pre-optimization baseline
```bash
python -m tests.perf --save-baseline --checkpoint
```

### Step 2: Add `__slots__` to Instant and Duration
**File**: `happysimulator/core/temporal.py`

Add `__slots__ = ('nanoseconds',)` to both `Duration` and `Instant`. Add `__slots__ = ()` to `_InfiniteInstant` (subclass, no new attrs).

**Impact**: 128 → 40 bytes per Instant. Negligible speed change but significant memory reduction at scale.

**Risk**: Very low. Verified no code does `instant.__dict__`.

### Step 3: Replace UUID4 with integer counter for Event._id
**File**: `happysimulator/core/event.py`

- Remove `import uuid`
- Change `self._id = uuid.uuid4()` → `self._id = self._sort_index` (reuse the existing global counter)
- `context.setdefault("id", str(self._id))` remains — `str(int)` is 30x faster than `str(uuid4())`
- `__hash__` and `__eq__` work unchanged with int

**Consumers verified safe**: `visual/serializers.py:140` does `str(event._id)` (works with int). No tests reference `_id` as UUID. `context["id"]` remains a string.

**Impact**: ~7-10% throughput improvement. uuid4+str is 800K ops/sec vs counter: 24M ops/sec.

### Step 4: Guard NullTraceRecorder calls with boolean flag
**Files**: `happysimulator/core/event_heap.py`, `happysimulator/core/simulation.py`

Add `self._tracing_enabled = not isinstance(self._trace, NullTraceRecorder)` in both `EventHeap.__init__` and `Simulation.__init__`. Wrap all `self._trace.record(...)` calls with `if self._tracing_enabled:`.

This eliminates:
- kwargs dict construction for every `record()` call
- `event.context.get("id")` lookups (5.8% of CPU) when tracing is off
- 4 trace.record() calls per event in the hot loop

**Impact**: ~3-5% throughput improvement.

### Step 5: Cache frequently-accessed attributes as locals in _run_loop
**File**: `happysimulator/core/simulation.py`

At the top of `_run_loop`, cache as local variables:
```python
control = self._control
heap = self._event_heap
end_time = self._end_time
clock = self._clock
```

Use these throughout the loop. Python local access (LOAD_FAST) is significantly faster than attribute access (LOAD_ATTR).

**Impact**: ~1-2% throughput improvement.

### Step 6: Make Event.trace() and _ensure_stack() opt-in
**File**: `happysimulator/core/event.py`

Add a module-level `_event_tracing_enabled = False` flag with `enable_event_tracing()`/`disable_event_tracing()` functions. Guard all `self.trace()` and `self._ensure_stack()` calls in `Event.invoke()` and `ProcessContinuation.invoke()` behind this flag.

The visual debugger's `serve()` function will call `enable_event_tracing()`.

**Tests affected**: 3 files access traces (`tests/integration/tracing/test_tracing_basic.py`, `test_system_tracing.py`) and 1 file accesses stack (`tests/unit/test_event_context_stack.py`). These will need to enable tracing in setup.

**Impact**: Eliminates 2.1M dict creations + list appends per benchmark run. ~3-4% throughput improvement.

### Step 7: Optimize Event context initialization
**File**: `happysimulator/core/event.py`

For the common case (no user-provided context), use dict literal instead of empty dict + 3 setdefault calls:
```python
if context is not None:
    self.context = context
    context.setdefault("id", str(self._id))
    context.setdefault("created_at", self.time)
    context.setdefault("metadata", {})
else:
    self.context = {"id": str(self._id), "created_at": self.time, "metadata": {}}
```

**Impact**: ~0.5-1% throughput improvement.

### Step 8: Lightweight ProcessContinuation constructor
**File**: `happysimulator/core/event.py`

Bypass `Event.__init__` in `ProcessContinuation.__init__` — directly set slots instead of calling `super().__init__()`. The continuation shares its parent's context dict, so the three `setdefault` calls are always no-ops. The UUID (now counter) increment still happens for heap ordering but the str() conversion and context setup are skipped.

```python
def __init__(self, time, event_type, target, *, daemon, on_complete, context, process):
    self.time = time
    self.event_type = event_type
    self.target = target
    self.daemon = daemon
    self.on_complete = on_complete if on_complete is not None else []
    self._sort_index = _global_event_counter.__next__()
    self._id = self._sort_index
    self._cancelled = False
    self.context = context  # Shared — no setdefault needed
    self.process = process
    self._send_value = None
```

**Impact**: ~4-6% on generator_heavy benchmark. Eliminates wasted UUID/context work for every yield step.

### Step 9: Fast-path _run_loop for common case
**File**: `happysimulator/core/simulation.py`

Create `_run_loop_fast()` for simulations with no control surface, no trace recorder, and explicit end_time. This eliminates per-event: 3 control checks, 2 trace.record() calls, logger.isEnabledFor() checks, last_event assignment, and auto-termination check.

Selection logic at top of `_run_loop`:
```python
if control is None and not tracing_enabled and end_time != Instant.Infinity:
    return self._run_loop_fast()
```

**Impact**: ~8-10% throughput improvement on the most common run configuration.

### Step 10: ParallelRunner utility for multi-run workloads
**New file**: `happysimulator/parallel.py`

Provide `ParallelRunner` class using `concurrent.futures.ProcessPoolExecutor`:
- `run_sweep(configs)` — run different configurations in parallel
- `run_replicas(build_fn, n, seed)` — Monte Carlo replicas with different seeds

Uses `build_fn: Callable[[], Simulation]` pattern so simulations are constructed in subprocesses (avoids pickle issues with entities/generators).

Export from `happysimulator/__init__.py`.

**PDES note**: True parallel discrete-event simulation (conservative/optimistic) is **not recommended** for this engine. Generator-based handlers can't be rolled back, zero-delay event chains defeat partitioning, and SimFuture uses a global heap context. Process-level parallelism for independent runs is the practical approach.

---

## Expected Cumulative Results

| Metric | Before | After (est.) | Improvement |
|--------|--------|-------------|-------------|
| throughput (events/sec) | ~50K | ~70-85K | +40-70% |
| bytes per event | 745 | ~400-500 | -33-46% |
| generator_heavy (events/sec) | ~57K | ~80-95K | +40-67% |

## Verification

After each step:
1. `pytest -q` — all ~2755 tests pass
2. `python -m tests.perf --compare --checkpoint` — verify improvement, no regression

Final validation: compare checkpoint against pre-optimization baseline.

## Files Modified

| File | Steps |
|------|-------|
| `happysimulator/core/temporal.py` | 2 |
| `happysimulator/core/event.py` | 3, 6, 7, 8 |
| `happysimulator/core/event_heap.py` | 4 |
| `happysimulator/core/simulation.py` | 4, 5, 9 |
| `happysimulator/parallel.py` (new) | 10 |
| `happysimulator/__init__.py` | 10 |
| `tests/integration/tracing/test_tracing_basic.py` | 6 |
| `tests/integration/tracing/test_system_tracing.py` | 6 |
| `tests/unit/test_event_context_stack.py` | 6 |
