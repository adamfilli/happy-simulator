# Simulation Control — Implementation Plan

## Context

The simulation currently runs to completion with no way to pause, inspect, step through events, or halt on conditions. This makes debugging emergent behaviors (metastable failures, queue buildup) difficult — you must run the full simulation, then reason backwards from results.

This plan adds interactive simulation control: pause/resume, stepping, breakpoints, event hooks, and heap introspection. The design follows the sketch in `.dev/observability-design.md` but is scoped to the control features only (not the full observability bus/query system).

## Architecture Decision

**`SimulationControl` as a separate class, accessed via `sim.control`** (lazy-created on first access).

- Keeps `Simulation` focused on the core loop
- Zero overhead when unused (`if self._control is not None` guard)
- Follows the composition pattern used elsewhere in the library
- Easy to test in isolation

## File Organization

```
happysimulator/core/control/
├── __init__.py          # Re-exports public API
├── control.py           # SimulationControl class
├── state.py             # SimulationState, BreakpointContext dataclasses
└── breakpoints.py       # Breakpoint protocol + 5 concrete implementations

tests/unit/control/
├── __init__.py
├── test_control.py      # pause/resume/step/reset/hooks
└── test_breakpoints.py  # All breakpoint types

tests/integration/
└── test_simulation_control.py  # End-to-end workflows
```

## Changes to Existing Files

### `happysimulator/core/simulation.py`

1. **Promote loop-local variables to instance state** so they survive between `run()` calls:
   - `_current_time: Instant` (was `current_time`)
   - `_events_processed: int` (was `events_processed`)
   - `_is_running: bool` (new)
   - `_wall_start: float | None` (was `wall_start`)
   - `_last_event: Event | None` (new, for state inspection)

2. **Add lazy `control` property**:
   ```python
   @property
   def control(self) -> SimulationControl:
       if self._control is None:
           self._control = SimulationControl(self)
       return self._control
   ```

3. **Refactor `run()` to be re-entrant** — calling `run()` on a paused sim resumes it:
   - First call: initialize wall clock, set `_is_running = True`
   - Subsequent calls: skip initialization, resume from where we paused
   - Insert 3 control check points in the loop (all guarded by `if self._control is not None`):
     - **Before pop**: `_should_pause()` — handles pause requests and step counting
     - **After time advance**: `_notify_time_advance()` — fires time hooks
     - **After invoke+push**: `_notify_event_processed()` + `_check_breakpoints()` — fires event hooks and evaluates breakpoint conditions
   - On pause: return partial `SimulationSummary` (same type, same fields)
   - On completion: set `_is_running = False`, return final summary as today

### `happysimulator/__init__.py`

Add exports: `SimulationControl`, `SimulationState`, `BreakpointContext`, `TimeBreakpoint`, `EventCountBreakpoint`, `ConditionBreakpoint`, `MetricBreakpoint`, `EventTypeBreakpoint`

## New Classes

### `SimulationState` (`state.py`)

```python
@dataclass(frozen=True)
class SimulationState:
    current_time: Instant
    events_processed: int
    heap_size: int
    primary_events_remaining: int
    is_paused: bool
    is_running: bool
    is_complete: bool
    last_event: Event | None
    wall_clock_elapsed: float
```

### `BreakpointContext` (`state.py`)

```python
@dataclass(frozen=True)
class BreakpointContext:
    current_time: Instant
    events_processed: int
    last_event: Event
    simulation: Simulation   # read-only access to entities
```

### Breakpoints (`breakpoints.py`)

Protocol + 5 concrete classes:

| Class | Triggers when | `one_shot` default |
|-------|--------------|-------------------|
| `TimeBreakpoint(time)` | `current_time >= time` | `True` |
| `EventCountBreakpoint(count)` | `events_processed >= count` | `True` |
| `ConditionBreakpoint(fn, description)` | `fn(context)` returns `True` | `False` |
| `MetricBreakpoint(entity_name, attr, op, threshold)` | entity attribute crosses threshold | `False` |
| `EventTypeBreakpoint(event_type)` | last event matches type | `False` |

All implement: `should_break(context: BreakpointContext) -> bool` and `one_shot: bool` property.

### `SimulationControl` (`control.py`)

```python
class SimulationControl:
    # --- Execution control ---
    def pause() -> None
    def resume() -> SimulationSummary    # calls sim.run() internally
    def step(n=1) -> SimulationSummary   # processes n events, returns summary
    def get_state() -> SimulationState
    def reset() -> None                  # re-prime heap, reset clock & counters

    # --- Breakpoints ---
    def add_breakpoint(bp) -> str        # returns ID
    def remove_breakpoint(id) -> None
    def list_breakpoints() -> list[tuple[str, Breakpoint]]
    def clear_breakpoints() -> None

    # --- Event hooks ---
    def on_event(callback) -> str        # called after each event, returns ID
    def on_time_advance(callback) -> str # called when sim time advances, returns ID
    def remove_hook(id) -> None

    # --- Heap introspection (only when paused) ---
    def peek_next(n=1) -> list[Event]
    def find_events(predicate) -> list[Event]

    # --- Properties ---
    is_paused: bool
    is_running: bool
```

**`resume()` and `step()` return `SimulationSummary`** — they delegate to `sim.run()` which always returns a summary (partial if paused again, final if complete). This keeps the API consistent: every execution action returns a summary.

**`reset()`** clears the heap, resets the clock to `start_time`, re-calls `source.start()` and `probe.start()` to re-prime, and resets counters. Does NOT reset entity state (entities own their state).

### Speed Control

Added as a stretch goal, not in the critical path. If included:

```python
def set_speed(multiplier: float | None) -> None  # None = unlimited
```

Implemented via `time.sleep()` between events, scaled by multiplier. Only meaningful for demos/visualization — most users will want unlimited speed.

## Example Usage

### Debugging a Metastable Failure

```python
sim = Simulation(
    sources=[source],
    entities=[server, sink],
    end_time=Instant.from_seconds(120),
)

# Break when queue builds up
sim.control.add_breakpoint(MetricBreakpoint(
    entity_name="Server",
    attribute="depth",
    operator="gt",
    threshold=50,
))

summary = sim.run()  # Pauses at breakpoint

# Investigate
state = sim.control.get_state()
print(f"Paused at t={state.current_time}, queue depth={server.depth}")

# Preview upcoming events
upcoming = sim.control.peek_next(5)
for e in upcoming:
    print(f"  {e.time}: {e.event_type} -> {e.target.name}")

# Step through a few events
summary = sim.control.step(10)

# Resume to completion
summary = sim.control.resume()
```

### Event Monitoring

```python
events_by_type: dict[str, int] = {}

def track_events(event: Event) -> None:
    events_by_type[event.event_type] = events_by_type.get(event.event_type, 0) + 1

sim.control.on_event(track_events)
sim.run()
print(events_by_type)
```

## Implementation Phases

### Phase 1: State & Structure
- Create `happysimulator/core/control/` package
- Implement `SimulationState` and `BreakpointContext` dataclasses
- Promote `run()` local variables to instance attributes on `Simulation`
- Add `_control` field + lazy `control` property
- Verify all existing tests still pass (no behavioral change yet)

### Phase 2: Pause / Resume / Step
- Implement `SimulationControl` with `pause()`, `resume()`, `step()`, `get_state()`
- Refactor `run()` to check `_should_pause()` before each pop
- Make `run()` re-entrant (resume from paused state)
- Write unit tests for pause/resume/step
- Write integration test: pause mid-simulation, inspect state, resume

### Phase 3: Breakpoints
- Implement `Breakpoint` protocol
- Implement all 5 concrete breakpoint classes
- Add `_check_breakpoints()` to the run loop (after invoke+push)
- Implement `add/remove/list/clear_breakpoints()`
- Write unit tests for each breakpoint type (including one-shot removal)
- Write integration test: breakpoint on queue depth during M/M/1 queue

### Phase 4: Hooks & Introspection
- Implement `on_event()`, `on_time_advance()`, `remove_hook()`
- Add `_notify_event_processed()` and `_notify_time_advance()` to run loop
- Implement `peek_next(n)` using `heapq.nsmallest()`
- Implement `find_events(predicate)` as linear scan of heap
- Write tests for hooks and introspection

### Phase 5: Reset & Polish
- Implement `reset()` — re-prime heap, reset clock, reset counters
- Add exports to `happysimulator/__init__.py`
- Write integration test: run, reset, run again, compare results
- Run full test suite, verify zero overhead when `control` not accessed

## Verification

1. **Existing tests pass unchanged** — `python -m pytest -q` (all ~1592 tests)
2. **New unit tests** for every control method and breakpoint type
3. **Integration test**: Set up M/M/1 queue, add `MetricBreakpoint(entity_name="Server", attribute="depth", operator="gt", threshold=10)`, run, verify it pauses, inspect state, step 5 events, resume to completion
4. **Zero overhead test**: Run a simulation without accessing `.control`, verify performance matches baseline (the only added cost is `if self._control is not None` per event, which is negligible)

## Backwards Compatibility

- Existing `sim.run()` works identically — returns `SimulationSummary`
- Control features are opt-in (only activated when `sim.control` is accessed)
- No new required parameters to `Simulation.__init__`
- Zero overhead when control is not used
