# Ergonomic API Improvements for happy-simulator

> **Status:** Draft
> **Created:** 2026-02-07

## Context

The core simulation API (Entity, Event, Simulation) is well-designed and minimal. However, examining all 7 examples reveals a **50-70% boilerplate-to-logic ratio**. The most common pain points are:

1. **Load generation requires 4 separate objects** (Profile + ArrivalTimeProvider + EventProvider + Source) even for the most basic use case
2. **Trivial `EventProvider` subclasses** are reimplemented identically in 6/7 examples (~25 lines each)
3. **`LatencyTrackingSink`** is copy-pasted into every example (~25 lines each)
4. **`Instant.from_seconds(x)`** wrapping is required in Simulation constructors
5. **Follow-up event creation** is verbose: `Event(time=self.now, event_type=..., target=..., context=...)`

All proposed changes are **purely additive** — no existing API is modified or deprecated.

---

## Phase 1: Source Factory Methods

**Impact: HIGH** — eliminates ~50 lines of boilerplate per example (EventProvider class + 4-object wiring)

**File:** `happysimulator/load/source.py`

Add a private `_SimpleEventProvider` class and three `@classmethod` factories to `Source`:

### `Source.constant()`
```python
Source.constant(rate=10, target=server, event_type="Request")
```
Creates a Source with `ConstantRateProfile` + `ConstantArrivalTimeProvider` + auto-generated `EventProvider`.

### `Source.poisson()`
```python
Source.poisson(rate=10, target=server, event_type="Request")
```
Same but with `PoissonArrivalTimeProvider` for stochastic arrivals.

### `Source.with_profile()`
```python
Source.with_profile(profile=MetastableLoadProfile(), target=server, poisson=True)
```
For custom rate profiles (SpikeProfile, LinearRampProfile, etc.) but still auto-generating the EventProvider.

All three accept `stop_after: float | Instant | None` to control when generation stops.

### `_SimpleEventProvider` (internal)
Module-level private class that creates targeted events with auto-incrementing request IDs and `created_at` timestamps in context.

**Also update:**
- `happysimulator/__init__.py` — no change needed (Source already exported)

---

## Phase 2: Accept Floats in Simulation Constructor

**Impact: MEDIUM** — eliminates `Instant.from_seconds()` wrapping in setup code

**File:** `happysimulator/core/simulation.py`

Change `start_time` and `end_time` type hints to `float | Instant | None` and auto-convert floats at the top of `__init__`:

```python
# Before
sim = Simulation(start_time=Instant.Epoch, end_time=Instant.from_seconds(120))

# After
sim = Simulation(end_time=120.0)
```

Deliberate non-change: `Event.time` stays as `Instant` only — event time should be explicit.

---

## Phase 3: `Entity.send()` Helper

**Impact: MEDIUM** — reduces follow-up event creation from 5 lines to 1

**File:** `happysimulator/core/entity.py`

Add a convenience method:

```python
def send(self, event_type: str, target: Simulatable, context: dict | None = None, delay: float = 0.0) -> Event:
```

Uses `self.now` (+ optional delay) as the event time.

```python
# Before
return [Event(time=self.now, event_type="Completed", target=self.downstream, context=event.context)]

# After
return [self.send("Completed", target=self.downstream, context=event.context)]
```

---

## Phase 4: Built-in `Sink` and `Counter`

**Impact: MEDIUM** — eliminates the most-copied entity from every example

**New file:** `happysimulator/components/common.py`

### `Sink`
Event collector with latency tracking. Computes latency from `context["created_at"]`. Provides:
- `events_received: int`
- `latencies_s: list[float]`
- `latency_time_series_seconds() -> tuple[list[float], list[float]]`
- `latency_stats() -> dict` (count, avg, min, max, p50, p99)

### `Counter`
Simple event counter by type. Provides `total: int` and `by_type: dict[str, int]`.

**Also update:**
- `happysimulator/components/__init__.py` — add `Sink`, `Counter` imports
- `happysimulator/__init__.py` — add `Sink`, `Counter` to imports and `__all__`

---

## Before/After Comparison

### Setting up a basic simulation (current — 50+ lines):
```python
class LatencyTrackingSink(Entity):
    def __init__(self, name): ...        # 25 lines
class RequestProvider(EventProvider):
    def __init__(self, target, ...): ... # 25 lines

sink = LatencyTrackingSink(name="Sink")
server = MyServer(downstream=sink)
provider = RequestProvider(server, stop_after=Instant.from_seconds(60))
arrival = PoissonArrivalTimeProvider(ConstantRateProfile(rate=10), start_time=Instant.Epoch)
source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)
sim = Simulation(start_time=Instant.Epoch, end_time=Instant.from_seconds(70), sources=[source], entities=[server, sink])
```

### After (6 lines, zero custom classes for infrastructure):
```python
sink = Sink()
server = MyServer(downstream=sink)
source = Source.poisson(rate=10, target=server, stop_after=60.0)
sim = Simulation(end_time=70.0, sources=[source], entities=[server, sink])
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `happysimulator/load/source.py` | Add `_SimpleEventProvider`, `Source.constant()`, `.poisson()`, `.with_profile()` |
| `happysimulator/core/simulation.py` | Accept `float` for `start_time`/`end_time` |
| `happysimulator/core/entity.py` | Add `Entity.send()` |
| `happysimulator/components/common.py` | **New:** `Sink`, `Counter` |
| `happysimulator/components/__init__.py` | Add `Sink`, `Counter` exports |
| `happysimulator/__init__.py` | Add `Sink`, `Counter` to imports and `__all__` |

---

## Tests

| Test File | What it covers |
|-----------|----------------|
| `tests/unit/test_source_factories.py` | Source.constant(), .poisson(), .with_profile(), stop_after behavior, backward compat |
| `tests/unit/test_simulation_float_time.py` | Float start_time/end_time, backward compat with Instant |
| `tests/unit/test_entity_send.py` | send() creates correct events, delay offset, context passthrough |
| `tests/unit/test_common_entities.py` | Sink latency tracking, stats, Counter by-type counting |

---

## Verification

1. `pytest -q` — all existing tests pass (no breaking changes)
2. `pytest tests/unit/test_source_factories.py tests/unit/test_simulation_float_time.py tests/unit/test_entity_send.py tests/unit/test_common_entities.py -q` — new tests pass
3. Manual: run `python examples/m_m_1_queue.py` to confirm no regressions

---

## Deferred

**`@entity` decorator** — Deferred because: (1) `@simulatable` already exists with known type-checker limitations, (2) function-based entities can't use generators, limiting utility to cases already covered by `Sink`/`Counter`, (3) clock access (`self.now`) is awkward in a function API.

**Example refactoring** — Updating the 7 examples to use the new APIs should be a separate follow-up to keep this change focused on the framework itself.
