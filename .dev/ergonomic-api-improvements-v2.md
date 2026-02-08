# Ergonomic API Improvements (Breaking Changes Allowed)

> **Status:** Draft
> **Created:** 2026-02-07
> **Supersedes:** `.dev/ergonomic-api-improvements.md` (additive-only version)

## Context

The core simulation API works well, but examples show a 50-70% boilerplate-to-logic ratio. The previous plan was additive-only. Now that breaking changes are allowed, we can:
- **Remove dead code** (`start_time` on `ArrivalTimeProvider`)
- **Rewrite all 7 examples** to use the new APIs (proves the value, wasn't possible before)
- **Fix tests** that break from the above

The core phases (Source factories, Simulation float times, Entity.send, Sink/Counter) remain the same — they were already the right design. What changes is scope and sequencing.

---

## Breaking Change Decisions

| Change | Verdict | Rationale |
|--------|---------|-----------|
| Remove `start_time` from `ArrivalTimeProvider` | **YES** | Dead parameter — always `Instant.Epoch`, immediately overwritten by `Source.start()`. 90+ call sites all pass the same value. |
| Update all 7 examples | **YES** | Previously deferred. Now we prove the API and eliminate stale patterns. |
| Change `Source.__init__` signature | **NO** | Factory classmethods are cleaner than overloaded constructors. Existing `__init__` serves advanced use cases. |
| Accept floats in `Event.time` | **NO** | Most Event constructions use `self.now` (already Instant). Float discipline in events is intentional — `Entity.send()` handles the ergonomic case. |

---

## Phase 1: Remove `start_time` from ArrivalTimeProvider

**Why first:** Phase 2 (Source factories) constructs ArrivalTimeProviders internally. Clean this up first so new code doesn't embed the dead parameter.

**File: `happysimulator/load/arrival_time_provider.py`** (line 37-39)
```python
# Before
def __init__(self, profile: Profile, start_time: Instant):
    self.profile = profile
    self.current_time = start_time

# After
def __init__(self, profile: Profile):
    self.profile = profile
    self.current_time = Instant.Epoch  # Overwritten by Source.start()
```

**Mechanical updates (~90 sites):** Delete `start_time=Instant.Epoch` from every `ConstantArrivalTimeProvider(...)` and `PoissonArrivalTimeProvider(...)` call across examples, tests, and `happysimulator/instrumentation/probe.py`.

---

## Phase 2: Source Factory Methods

**File: `happysimulator/load/source.py`**

Add module-level `_SimpleEventProvider` (creates targeted events with auto-incrementing request IDs and `created_at` timestamps) and three `@classmethod` factories:

```python
Source.constant(rate=10, target=server, event_type="Request", stop_after=60.0)
Source.poisson(rate=10, target=server, event_type="Request", stop_after=60.0)
Source.with_profile(profile=my_profile, target=server, poisson=True, stop_after=60.0)
```

All accept `name: str | None` (auto-generates if omitted) and `stop_after: float | Instant | None`.

No changes to existing `Source.__init__` — it remains for advanced use cases with custom EventProviders.

---

## Phase 3: Accept Floats in Simulation Constructor

**File: `happysimulator/core/simulation.py`** (line 44-59)

Change `start_time`/`end_time` type hints to `float | Instant | None`. Auto-convert floats via `Instant.from_seconds()` at the top of `__init__`.

```python
sim = Simulation(end_time=130.0)  # instead of Instant.from_seconds(130)
```

---

## Phase 4: `Entity.send()` Helper

**File: `happysimulator/core/entity.py`**

```python
def send(self, event_type: str, target: Simulatable,
         context: dict | None = None, delay: float = 0.0) -> Event:
```

Uses `self.now` + optional delay. Replaces the verbose `Event(time=self.now, event_type=..., target=..., context=...)` pattern.

---

## Phase 5: Built-in `Sink` and `Counter`

**New file: `happysimulator/components/common.py`**

- **`Sink`**: Event collector with latency tracking from `context["created_at"]`. Provides `events_received`, `latencies_s`, `latency_time_series_seconds()`, `latency_stats()`. Default name `"Sink"`.
- **`Counter`**: Event counter by type. Provides `total` and `by_type: dict[str, int]`. Default name `"Counter"`.

**Also update:**
- `happysimulator/components/__init__.py` — add imports
- `happysimulator/__init__.py` — add to imports and `__all__`

---

## Phase 6: Rewrite Examples

Update all 7 examples to use the new APIs:

| Example | Remove custom EventProvider? | Use Sink? | Use Source factory? |
|---------|----------------------------|-----------|---------------------|
| `m_m_1_queue.py` | YES (RequestProvider) | YES | `Source.with_profile()` |
| `increasing_queue_depth.py` | YES | YES | `Source.with_profile()` |
| `cold_start.py` | NO (uses DistributedFieldProvider) | YES | NO |
| `dual_path_queue_latency.py` | PARTIAL (needs source_id context) | NO (custom sink) | PARTIAL |
| `retrying_client.py` | YES | N/A (client tracks) | `Source.constant()` |
| `metastable_state.py` | YES | N/A (client tracks) | `Source.with_profile()` |
| `load_aware_routing.py` | NO (custom providers) | N/A | NO |

Also use `Entity.send()` in entity `handle_event()` methods and float times in Simulation constructors across all examples.

---

## Before/After (m_m_1_queue.py setup)

**Before (~20 lines of infrastructure):**
```python
class RequestProvider(EventProvider): ...        # 15 lines
class LatencyTrackingSink(Entity): ...           # 20 lines

sink = LatencyTrackingSink(name="Sink")
server = MM1Server(name="Server", downstream=sink)
provider = RequestProvider(server, stop_after=Instant.from_seconds(120))
arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)
sim = Simulation(start_time=Instant.Epoch, end_time=Instant.from_seconds(130), ...)
```

**After (4 lines, zero custom infrastructure classes):**
```python
sink = Sink()
server = MM1Server(name="Server", downstream=sink)
source = Source.with_profile(profile, target=server, poisson=True, stop_after=120.0)
sim = Simulation(end_time=130.0, ...)
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `happysimulator/load/arrival_time_provider.py` | Remove `start_time` param |
| `happysimulator/load/source.py` | Add `_SimpleEventProvider`, `.constant()`, `.poisson()`, `.with_profile()` |
| `happysimulator/core/simulation.py` | Accept `float \| Instant \| None` |
| `happysimulator/core/entity.py` | Add `Entity.send()` |
| `happysimulator/components/common.py` | **New:** `Sink`, `Counter` |
| `happysimulator/components/__init__.py` | Add exports |
| `happysimulator/__init__.py` | Add exports |
| `happysimulator/instrumentation/probe.py` | Remove `start_time` from ArrivalTimeProvider construction |
| `examples/*.py` (7 files) | Rewrite to use new APIs |
| `tests/**/*.py` (~35 files) | Fix `start_time` removal, update as needed |

---

## Tests

| Test File | Covers |
|-----------|--------|
| `tests/unit/test_source_factories.py` | `.constant()`, `.poisson()`, `.with_profile()`, `stop_after`, auto-naming |
| `tests/unit/test_simulation_float_time.py` | Float `start_time`/`end_time`, backward compat with Instant |
| `tests/unit/test_entity_send.py` | `send()` creates correct events, delay, context |
| `tests/unit/test_common_entities.py` | Sink latency tracking + stats, Counter by-type |

---

## Verification

1. `pytest -q` — all tests pass after mechanical updates
2. New unit tests pass
3. `python examples/m_m_1_queue.py` — runs successfully with new APIs
