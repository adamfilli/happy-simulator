# Rewind / Step-by-Step Execution for happysimulator

## Context

Enable interactive debugging and visualization replay by letting users step forward and backward through simulation events one at a time. This supports inspecting entity state at any point and scrubbing through a completed simulation like a timeline.

**Mechanism**: Journal + Replay. Forward execution records each event in a journal. Backward stepping restores initial state (via Snapshottable protocol) and replays forward to the target step. This cleanly avoids the generator serialization problem since generators are reconstructed naturally during replay.

## API

```python
from happysimulator.stepping import SteppableSimulation

sim = SteppableSimulation(sources=[source], entities=[sink], end_time=...)

sim.step()              # Process one event, returns True/False
sim.step_back()         # Rewind one step (replay from initial state)
sim.run_to(30.0)        # Step forward until t >= 30s
sim.rewind_to(20.0)     # Rewind to t=20s
sim.reset()             # Back to t=0
sim.current_time        # Current Instant
sim.current_step        # Step index (0-based)
sim.inspect(entity)     # Entity state dict via Snapshottable
sim.journal             # EventJournal with all processed entries
```

## Implementation Steps

### Step 1: Snapshottable Protocol

**File**: `happysimulator/core/protocols.py` (add to existing)

Add `Snapshottable` protocol alongside existing `Simulatable`:

```python
@runtime_checkable
class Snapshottable(Protocol):
    def get_snapshot(self) -> dict[str, Any]: ...
    def restore_snapshot(self, snapshot: dict[str, Any]) -> None: ...
```

**File**: `happysimulator/__init__.py` — export `Snapshottable`

### Step 2: ResettableCounter

**File**: `happysimulator/core/event.py` (modify)

Replace `_global_event_counter = count()` with a resettable wrapper:

```python
class _ResettableCounter:
    def __init__(self): self._value = 0
    def __next__(self): v = self._value; self._value += 1; return v
    def get_state(self) -> int: return self._value
    def set_state(self, v: int): self._value = v

_global_event_counter = _ResettableCounter()
```

Add module-level helpers `_get_event_counter_state()` / `_set_event_counter_state()` for `SteppableSimulation` to save/restore the counter during replay.

The existing `Event._sort_index = field(default_factory=_global_event_counter.__next__, ...)` works unchanged.

### Step 3: Implement Snapshottable on Core Entities

Each entity's `get_snapshot()` returns a dict of its mutable fields; `restore_snapshot()` writes them back.

| Entity | File | Mutable State to Capture |
|--------|------|--------------------------|
| `Sink` | `components/common.py` | `events_received`, `latencies_s` (copy), `completion_times` (serialize as nanoseconds) |
| `Counter` | `components/common.py` | `total`, `by_type` (copy) |
| `Queue` | `components/queue.py` | `stats_dropped`, `stats_accepted`, delegates to `policy.get_snapshot()` |
| `FIFOQueue` | `components/queue_policy.py` | `list(self._queue)` (copy deque) |
| `LIFOQueue` | `components/queue_policy.py` | `list(self._queue)` (copy deque) |
| `PriorityQueue` | `components/queue_policy.py` | `list(self._heap)`, `_insert_counter` |
| `Source` | `load/source.py` | `_nmb_generated`, delegates to `_time_provider.get_snapshot()` and `_event_provider.get_snapshot()` |
| `ConstantArrivalTimeProvider` | `load/constant_arrival.py` | `current_time` (ns), `_integral_value` |
| `PoissonArrivalTimeProvider` | `load/poisson_arrival.py` | Same as Constant (global RNG handled separately) |
| `_SimpleEventProvider` | `load/source.py` (inner class) | `_generated` counter |
| `Data` | `instrumentation/data.py` | `list(self._samples)` |
| `LatencyTracker` | `instrumentation/collectors.py` | `count`, delegates to `data.get_snapshot()` |
| `ThroughputTracker` | `instrumentation/collectors.py` | `count`, delegates to `data.get_snapshot()` |
| `Inductor` | `components/rate_limiter/inductor.py` | `stats.*`, `_smoothed_interval`, `_last_arrival_time`, `_last_output_time`, `_poll_scheduled`, queue contents, timestamp lists |

### Step 4: EventJournal

**New file**: `happysimulator/stepping/event_journal.py`

```python
@dataclass
class JournalEntry:
    step: int
    time: Instant
    event_type: str
    target_name: str
    produced_count: int

class EventJournal:
    def record(self, step: int, event: Event, produced_count: int) -> None: ...
    def entries(self) -> list[JournalEntry]: ...
    def truncate_after(self, step: int) -> None: ...
    def __len__(self) -> int: ...
```

Lightweight metadata only — not full event copies.

### Step 5: SteppableSimulation

**New file**: `happysimulator/stepping/steppable_simulation.py`

Constructor matches `Simulation` signature. Internally:

**Init flow:**
1. Create `Clock(start_time)`, inject into all entities
2. Capture pre-bootstrap state:
   - `_initial_entity_snapshots` = `{name: entity.get_snapshot()}` for all Snapshottable entities
   - `_initial_event_counter` = `_get_event_counter_state()`
   - `_initial_rng_state` = `(random.getstate(), np.random.get_state())`
3. Call `source.start()` and `probe.start()` → push to heap (mirrors `Simulation.__init__`)

**`step()` method:**
1. Pop event from heap
2. Advance clock
3. `event.invoke()` → get new events
4. Push new events to heap
5. Record in journal
6. Increment `_current_step`

(Same pop-invoke-push as `Simulation.run()` but exactly one iteration.)

**`_reset_to_initial()` method:**
1. Restore all entity snapshots from `_initial_entity_snapshots`
2. Clear event heap (`self._heap.clear()` — needs `EventHeap.clear()` method)
3. Restore event counter via `_set_event_counter_state()`
4. Restore RNG state via `random.setstate()` + `np.random.set_state()`
5. Re-call `source.start()` / `probe.start()` → push to heap
6. Reset `_current_step = 0`, clear journal

**`step_back()` method:**
1. Target = `current_step - 1`
2. Call `_reset_to_initial()`
3. Call `step()` target times

**`rewind_to(time)` method:**
1. Scan journal to find last step at-or-before target time
2. `_reset_to_initial()` + replay to that step

**`run_to(time)` method:**
1. Loop `step()` until `current_time >= time` or heap exhausted

**`inspect(entity)` method:**
1. If entity implements `Snapshottable`, return `entity.get_snapshot()`
2. Else return `None`

### Step 6: EventHeap.clear()

**File**: `happysimulator/core/event_heap.py` (add method)

```python
def clear(self) -> None:
    self._heap.clear()
    self._primary_event_count = 0
    self._current_time = Instant.Epoch
```

### Step 7: Package Init + Exports

**New file**: `happysimulator/stepping/__init__.py`
- Export `SteppableSimulation`, `EventJournal`, `JournalEntry`

**File**: `happysimulator/__init__.py`
- Add `SteppableSimulation` to public API

### Step 8: Tests

**`tests/unit/test_snapshottable.py`**
- Test `get_snapshot()` / `restore_snapshot()` round-trip for each entity
- Verify restored state matches original

**`tests/unit/test_event_journal.py`**
- Record entries, retrieve, truncate

**`tests/integration/test_steppable_simulation.py`**
- **Basic stepping**: Source → Sink, step 10 times, verify Sink.events_received == expected
- **Step back**: Step 10, step_back 3 times, verify state at step 7 matches original step 7
- **Deterministic replay**: Step 100, rewind_to step 50, step forward again, verify same entity states
- **With generators**: Source → QueuedResource → Sink, step through generator continuations, rewind, verify consistency
- **run_to / rewind_to**: Time-based navigation
- **Reset**: Verify returns to step 0 with initial entity state

**`tests/integration/test_stepping_determinism.py`**
- Use PoissonArrivalTimeProvider, verify replay produces identical results

### Step 9: Example

**New file**: `examples/stepping_debugger.py`
- Source → Sink setup
- Step forward 5, print state, step back 2, print state, run_to, reset

## Key Design Decisions

**Why replay instead of undo/checkpoints?**
- `ProcessContinuation` holds live Python generators that can't be serialized
- Replay reconstructs generators naturally during re-execution
- Simpler to implement correctly; performance can be improved later with periodic checkpoints (V2)

**Why capture snapshots BEFORE source.start()?**
- source.start() modifies Source/ArrivalTimeProvider state and creates heap events
- By snapshotting pre-start and re-calling start() during reset, we avoid serializing events (which hold entity references, closures, generators)

**Why ResettableCounter?**
- `Event._sort_index` determines tie-breaking order for simultaneous events
- During replay, events must get the same sort indices → counter must be restorable
- Simple class replacement, no API change for Event users

**Generator handling:**
- During replay, `handle_event()` returns a new generator each time
- `ProcessContinuation` events are created fresh with new generator objects
- No need to serialize/deserialize generator state

## Future Work (V2): Periodic Checkpoints

For long simulations, replay-from-start becomes expensive. V2 would add:

- `CheckpointManager` that periodically captures full state (entities + heap + RNG + counter)
- Checkpoints only taken at "clean" moments (no `ProcessContinuation` in heap)
- `step_back()` replays from nearest checkpoint instead of from start
- Trade-off: memory usage vs rewind speed

## Verification Plan

1. Run `pytest tests/unit/test_snapshottable.py -q` — all entities round-trip correctly
2. Run `pytest tests/integration/test_steppable_simulation.py -q` — stepping + rewind works
3. Run `python examples/stepping_debugger.py` — interactive demo produces expected output
4. Run `pytest -q` — existing tests still pass (ResettableCounter is drop-in compatible)

## Files Summary

| Action | File |
|--------|------|
| Modify | `happysimulator/core/protocols.py` — add Snapshottable |
| Modify | `happysimulator/core/event.py` — ResettableCounter |
| Modify | `happysimulator/core/event_heap.py` — add clear() |
| Modify | `happysimulator/components/common.py` — Sink/Counter snapshots |
| Modify | `happysimulator/components/queue.py` — Queue snapshot |
| Modify | `happysimulator/components/queue_policy.py` — policy snapshots |
| Modify | `happysimulator/load/source.py` — Source + _SimpleEventProvider snapshots |
| Modify | `happysimulator/load/constant_arrival.py` — snapshot |
| Modify | `happysimulator/load/poisson_arrival.py` — snapshot |
| Modify | `happysimulator/instrumentation/data.py` — Data snapshot |
| Modify | `happysimulator/instrumentation/collectors.py` — tracker snapshots |
| Modify | `happysimulator/components/rate_limiter/inductor.py` — Inductor snapshot |
| Modify | `happysimulator/__init__.py` — exports |
| Create | `happysimulator/stepping/__init__.py` |
| Create | `happysimulator/stepping/event_journal.py` |
| Create | `happysimulator/stepping/steppable_simulation.py` |
| Create | `tests/unit/test_snapshottable.py` |
| Create | `tests/unit/test_event_journal.py` |
| Create | `tests/integration/test_steppable_simulation.py` |
| Create | `tests/integration/test_stepping_determinism.py` |
| Create | `examples/stepping_debugger.py` |
