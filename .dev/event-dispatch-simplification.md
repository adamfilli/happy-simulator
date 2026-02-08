# Event Dispatch Simplification Design Document

> **Status:** Implemented (Approach A)
> **Last Updated:** 2026-02-08

## Problem Statement

The `Event` class supports two mutually exclusive dispatch paths:

```python
# Path 1: Model-style — target an Entity
Event(time=t, event_type="Request", target=my_server)

# Path 2: Callback-style — invoke a function directly
Event(time=t, event_type="Ping", callback=lambda e: print("pong"))
```

This duality imposes costs throughout the codebase:

### 1. Validation Branching in `__post_init__`

Every event creation pays for a two-way exclusivity check (`event.py:86-92`):

```python
def __post_init__(self):
    if self.target is None and self.callback is None:
        raise ValueError("Event must have EITHER a 'target' OR a 'callback'.")
    if self.target is not None and self.callback is not None:
        raise ValueError("Event cannot have BOTH 'target' and 'callback'.")
```

### 2. Dispatch Branching in `invoke()`

The hot path of the simulation loop (`event.py:139-171`) branches on every invocation:

```python
def invoke(self) -> List[Event]:
    # ...
    if self.callback:             # Path 1
        raw_result = self.callback(self)
    elif self.target:             # Path 2
        raw_result = self.target.handle_event(self)
    else:
        raise ValueError(...)     # Defensive — should never hit
```

### 3. Dual `__repr__`

The repr branches again to show the right label (`event.py:101-108`):

```python
def __repr__(self):
    if self.target is not None:
        target_name = getattr(self.target, "name", None) or type(self.target).__name__
        return f"Event({self.time!r}, {self.event_type!r}, target={target_name})"
    else:
        callback_name = getattr(self.callback, "__qualname__", ...)
        return f"Event({self.time!r}, {self.event_type!r}, callback={callback_name})"
```

### 4. `ProcessContinuation` Propagates Both Fields

When a generator is wrapped, the continuation must preserve whichever field was set (`event.py:214-222`, `event.py:300-308`):

```python
continuation = ProcessContinuation(
    time=self.time,
    target=self.target,      # might be None
    callback=self.callback,  # might be None
    process=gen,
    ...
)
```

### 5. Subclasses Must Navigate the Duality

`SourceEvent` explicitly passes `callback=None` to satisfy the validation (`source_event.py:34-35`):

```python
super().__init__(
    target=source_entity,
    callback=None)    # Must explicitly pass None
```

### 6. Cognitive Tax

Every new contributor must learn and remember the "exactly one of two" rule. Every new Event subclass and every code review must verify compliance.

---

## Usage Census

A comprehensive scan of the codebase reveals that **target-style dispatch dominates overwhelmingly**.

### Production Code (`happysimulator/`)

| Dispatch Style | Count | Where |
|---|---|---|
| `target=` | 93 | Queue events, driver events, source events, entity-to-entity dispatch |
| `callback=` | **1** | `probe.py:75-84` — Probe creates callback events to sample metrics |
| `callback=None` (explicit) | 1 | `source_event.py:35` — SourceEvent satisfies validation |

The sync primitives (`mutex.py`, `semaphore.py`, `barrier.py`, `condition.py`, `rwlock.py`) use `callback=` on internal `_Waiter` dataclasses, not on `Event` objects — so they are not relevant to this discussion.

**The single production callback user:**

```python
# probe.py:75-84  (_ProbeEventProvider.get_events)
def get_events(self, time: Instant) -> List[Event]:
    callback = self._create_measurement_callback()
    return [
        Event(
            time=time,
            daemon=True,
            event_type="probe_event",
            target=None,
            callback=callback)
        ]
```

### Examples (`examples/`)

| Dispatch Style | Count | Where |
|---|---|---|
| `target=` | ~61 | All entity-to-entity dispatch |
| `callback=` | **2** | `cold_start.py:462`, `fleet_change_comparison.py:324` |

Both example callbacks are **one-shot scheduled actions** — things that happen once at a specific time:

```python
# cold_start.py — Reset server cache at t=30s
def reset_callback(_e: Event) -> list[Event]:
    server.reset_cache()
    return []

Event(time=Instant.from_seconds(30), event_type="CacheReset", callback=reset_callback)

# fleet_change_comparison.py — Add a server at t=60s
Event(time=Instant.from_seconds(60), event_type="FleetChange", callback=fleet_change_callback)
```

### Tests (`tests/`)

| Dispatch Style | Count | Where |
|---|---|---|
| `target=` | ~481 | Entity tests, integration tests, queue tests |
| `callback=` | ~64 | Mostly `lambda e: None` placeholder events in messaging tests |

Most test callbacks are **lightweight placeholders** where creating a full Entity would be verbose boilerplate. A few are integration tests specifically exercising the callback dispatch path.

### Summary

| Area | `target=` | `callback=` | Ratio |
|---|---|---|---|
| Production | 93 | 1 | **99:1** |
| Examples | ~61 | 2 | **30:1** |
| Tests | ~481 | ~64 | **7.5:1** |
| **Total** | **~635** | **~67** | **~9.5:1** |

Production code has a single genuine callback user (Probe). The callback path exists to serve <1% of the production event volume.

---

## Approach A: Target-Only

**Core idea:** Remove `callback` from Event. All events must have a `target`. Provide lightweight entities and a static constructor for the callback use case.

### Design

#### 1. Remove `callback` from Event

```python
@dataclass
class Event:
    time: Instant
    event_type: str
    target: Simulatable          # Required, no longer Optional
    daemon: bool = False
    on_complete: list[CompletionHook] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def invoke(self) -> list[Event]:
        # Single dispatch path — no branching
        raw_result = self.target.handle_event(self)
        if isinstance(raw_result, Generator):
            return self._start_process(raw_result)
        normalized = self._normalize_return(raw_result)
        completion_events = self._run_completion_hooks(self.time)
        return normalized + completion_events
```

#### 2. Add `CallbackEntity` — A Reusable Adapter

```python
class CallbackEntity(Entity):
    """Lightweight entity that delegates handle_event to a callback function.

    Bridges the gap between function-based actions and the entity-based
    event system. Use Event.once() for one-shot convenience.
    """
    def __init__(self, name: str, fn: Callable[[Event], list[Event] | Event | None]):
        super().__init__(name)
        self._fn = fn

    def handle_event(self, event: Event) -> list[Event] | Event | None:
        return self._fn(event)
```

#### 3. Add `Event.once()` — Convenience Static Constructor

```python
@staticmethod
def once(
    time: Instant,
    event_type: str,
    fn: Callable[[Event], list[Event] | Event | None],
    *,
    daemon: bool = False,
    context: dict[str, Any] | None = None,
) -> Event:
    """Create a one-shot event that invokes a function.

    Wraps the function in a CallbackEntity automatically.
    This is the replacement for the old callback= parameter.
    """
    entity = CallbackEntity(name=f"once:{event_type}", fn=fn)
    return Event(
        time=time,
        event_type=event_type,
        target=entity,
        daemon=daemon,
        context=context or {},
    )
```

#### 4. Add `NullEntity` — For Fire-and-Forget Events

```python
class NullEntity(Entity):
    """Entity that accepts and discards all events. Singleton."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            super().__init__("NullEntity")
            self._initialized = True

    def handle_event(self, event: Event) -> None:
        return None
```

### Before / After

**Probe (`probe.py`):**

```python
# BEFORE
Event(
    time=time,
    daemon=True,
    event_type="probe_event",
    target=None,
    callback=callback,
)

# AFTER — Option 1: Use Event.once()
Event.once(
    time=time,
    event_type="probe_event",
    fn=callback,
    daemon=True,
)

# AFTER — Option 2: Use explicit CallbackEntity
probe_entity = CallbackEntity("probe_sampler", fn=callback)
Event(time=time, event_type="probe_event", target=probe_entity, daemon=True)
```

**Cold Start Example:**

```python
# BEFORE
Event(
    time=Instant.from_seconds(config.cold_start_time_s),
    event_type="CacheReset",
    callback=reset_callback,
)

# AFTER
Event.once(
    time=Instant.from_seconds(config.cold_start_time_s),
    event_type="CacheReset",
    fn=reset_callback,
)
```

**SourceEvent:**

```python
# BEFORE
super().__init__(
    target=source_entity,
    callback=None)    # Noise — needed only to satisfy validation

# AFTER
super().__init__(
    target=source_entity)   # Clean — callback field doesn't exist
```

**Test placeholder events:**

```python
# BEFORE
Event(time=t, event_type="msg", callback=lambda e: None)

# AFTER
Event.once(time=t, event_type="msg", fn=lambda e: None)
# or
Event(time=t, event_type="msg", target=NullEntity())
```

### Trade-offs

| Dimension | Assessment |
|---|---|
| **Architectural purity** | Strong. Every event targets an entity. The simulation is uniformly entity-based. |
| **Debugging** | Improved. `repr` always shows a target name. Stack traces always route through `handle_event()`. |
| **CallbackEntity clock** | `CallbackEntity` instances created via `Event.once()` won't have a clock injected (no `Simulation` registration). Callbacks that need `self.now` must be registered in `entities=[]`. For Probe's use case this is fine — the callback already captures what it needs. |
| **Migration cost** | Low. Only 1 production file, 2 examples, and ~64 test files need changes. `Event.once()` makes migration nearly mechanical. |
| **ProcessContinuation** | Simplified. Only needs to propagate `target`, not two optional fields. |
| **Test ergonomics** | Slightly more verbose for placeholder events (`Event.once(...)` vs `callback=lambda e: None`), but `NullEntity()` is just as concise. |
| **Backward compatibility** | Breaking change. All code using `callback=` must be updated in one release. |

### Affected Files

**Production (must change):**

| File | Change |
|---|---|
| `happysimulator/core/event.py` | Remove `callback` field, simplify `invoke()`, `__repr__`, `__post_init__`, add `Event.once()`. Remove `EventCallback` type alias. |
| `happysimulator/core/event.py` (ProcessContinuation) | Remove `callback=self.callback` propagation. |
| `happysimulator/instrumentation/probe.py` | Use `Event.once()` or `CallbackEntity` in `_ProbeEventProvider.get_events()`. |
| `happysimulator/load/source_event.py` | Remove `callback=None` argument. |
| New: `happysimulator/core/callback_entity.py` | `CallbackEntity` and `NullEntity` classes. |

**Examples (must change):**

| File | Change |
|---|---|
| `examples/cold_start.py` | `callback=reset_callback` → `Event.once(...)` |
| `examples/load-balancing/fleet_change_comparison.py` | `callback=fleet_change_callback` → `Event.once(...)` |

**Tests (~64 files with callback usage, mechanical changes).**

---

## Approach B: Callback-Only with Static Constructors

**Core idea:** Make `callback` the single dispatch mechanism. Events always call a function. Provide `Event.to()` as a static constructor that creates events targeting entities by wrapping `entity.handle_event` as the callback.

### Design

#### 1. Make `callback` Required

```python
@dataclass
class Event:
    time: Instant
    event_type: str
    callback: EventCallback       # Required
    daemon: bool = False
    on_complete: list[CompletionHook] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    # Metadata for debugging — stores the target entity ref if created via Event.to()
    _target_entity: Simulatable | None = field(default=None, repr=False)

    def invoke(self) -> list[Event]:
        # Single dispatch path
        raw_result = self.callback(self)
        if isinstance(raw_result, Generator):
            return self._start_process(raw_result)
        normalized = self._normalize_return(raw_result)
        completion_events = self._run_completion_hooks(self.time)
        return normalized + completion_events
```

#### 2. Add `Event.to()` — Entity-Targeting Constructor

```python
@staticmethod
def to(
    target: Simulatable,
    *,
    time: Instant,
    event_type: str,
    daemon: bool = False,
    context: dict[str, Any] | None = None,
    on_complete: list[CompletionHook] | None = None,
) -> Event:
    """Create an event that targets an entity's handle_event method.

    This is the primary way to create events in model-style simulations.
    """
    return Event(
        time=time,
        event_type=event_type,
        callback=target.handle_event,
        daemon=daemon,
        context=context or {},
        on_complete=on_complete or [],
        _target_entity=target,
    )
```

#### 3. Keep `target` as Read-Only Property for Debugging

```python
@property
def target(self) -> Simulatable | None:
    """The entity this event is addressed to (if created via Event.to)."""
    return self._target_entity
```

### Before / After

**Creating entity-targeted events:**

```python
# BEFORE
Event(
    time=self.now,
    event_type="Request",
    target=my_server,
    context={"customer_id": 42},
)

# AFTER
Event.to(
    my_server,
    time=self.now,
    event_type="Request",
    context={"customer_id": 42},
)
```

**Probe:**

```python
# BEFORE
Event(
    time=time,
    daemon=True,
    event_type="probe_event",
    target=None,
    callback=callback,
)

# AFTER
Event(
    time=time,
    event_type="probe_event",
    callback=callback,
    daemon=True,
)
```

**QueueDriver retargeting:**

```python
# BEFORE (queue_driver.py:72-74)
target_event = payload
target_event.time = self.now
target_event.target = self.target     # Direct mutation

# AFTER — Must rewrap the callback
target_event = Event.to(
    self.target,
    time=self.now,
    event_type=payload.event_type,
    context=payload.context,
    on_complete=payload.on_complete,
)
```

**SourceEvent:**

```python
# BEFORE
super().__init__(target=source_entity, callback=None)

# AFTER — Inheriting from Event becomes awkward; SourceEvent must accept callback
super().__init__(callback=source_entity.handle_event, _target_entity=source_entity)
```

### Trade-offs

| Dimension | Assessment |
|---|---|
| **Architectural purity** | Mixed. Callback is a simpler primitive, but the entity model is the framework's core metaphor. `Event.to()` is now the standard path, making callback feel like the "real" mechanism that entities are layered on top of. |
| **Debugging** | Worse. `repr` shows a bound method `Server.handle_event` instead of `target=Server`. Need `_target_entity` metadata for meaningful debugging. Stack traces gain an extra layer of indirection. |
| **Retargeting** | Breaking. QueueDriver's `event.target = self.target` mutation pattern no longer works. Must reconstruct events or add a `.retarget()` method. |
| **Event subclasses** | Breaking. `QueuePollEvent`, `QueueNotifyEvent`, `QueueDeliverEvent`, `SourceEvent` all use `target=` in their constructors. All must be updated. |
| **Migration cost** | High. **Every `target=` call site must change** (~635 occurrences). The 93 production sites, 61 example sites, and 481 test sites all need updating. |
| **Ergonomics** | The common case (targeting an entity) becomes more verbose: `Event.to(server, time=..., ...)` vs `Event(time=..., target=server)`. |
| **ProcessContinuation** | Simplified. Only propagates `callback`. But loses direct target reference — must propagate `_target_entity` separately. |
| **Simulatable protocol** | The protocol's `handle_event` method is still relevant, but its role shifts from "dispatch target" to "callback source". |

### Affected Files

**Production (must change):**

| File | Change |
|---|---|
| `happysimulator/core/event.py` | Remove `target`, make `callback` required, add `Event.to()`, update `__repr__`, simplify `invoke()`. |
| `happysimulator/core/event.py` (ProcessContinuation) | Propagate `callback` only, add `_target_entity`. |
| `happysimulator/components/queue.py` | All `QueuePollEvent`, `QueueNotifyEvent`, `QueueDeliverEvent` subclasses must switch to callback-style. |
| `happysimulator/components/queue_driver.py` | Retargeting pattern must be rewritten. |
| `happysimulator/load/source_event.py` | Constructor rewritten. |
| `happysimulator/instrumentation/probe.py` | Remove `target=None`. |
| ~20 other production files | Every `target=entity` must become `Event.to(entity, ...)`. |

**Examples (all ~61 target usages must change).**

**Tests (~481 target usages must change).**

---

## Comparison Table

| Dimension | Approach A: Target-Only | Approach B: Callback-Only |
|---|---|---|
| **Core metaphor** | "Events are messages to entities" | "Events are scheduled function calls" |
| **Alignment with framework** | Strong — Entity is the central abstraction | Weaker — Functions are the primitive, entities are wrappers |
| **Production files changed** | ~3 | ~25+ |
| **Example files changed** | 2 | All |
| **Test files changed** | ~64 (callback users) | ~400+ (target users) |
| **Total call sites changed** | ~67 | ~635 |
| **Migration direction** | Migrate minority (callbacks → entities) | Migrate majority (entities → callbacks) |
| **Event.target mutation** | Still works (`event.target = ...`) | Broken — requires `.retarget()` or reconstruction |
| **Event subclasses** | No changes needed | All must be rewritten |
| **Debugging (repr)** | Always shows entity name | Shows bound method name; needs `_target_entity` for entity name |
| **Debugging (stack traces)** | `invoke() → handle_event()` — clean | `invoke() → callback() → handle_event()` — extra frame |
| **ProcessContinuation** | Simpler (one field) | Simpler (one field) + metadata field |
| **Clock injection** | Natural — entities registered with Simulation | Unchanged for entities; callbacks don't get clocks |
| **Ergonomics (common case)** | `Event(time=t, target=s)` — unchanged | `Event.to(s, time=t)` — slightly different |
| **Ergonomics (rare case)** | `Event.once(time=t, fn=f)` — new syntax | `Event(time=t, callback=f)` — direct |
| **New concepts** | `CallbackEntity`, `NullEntity`, `Event.once()` | `Event.to()`, `_target_entity`, `event.retarget()` |
| **Risk** | Low — changes concentrated in minority path | High — changes touch nearly every file |

---

## Recommendation

**Approach A (Target-Only) is strongly preferred** for these reasons:

1. **Migration cost:** 10x fewer call sites to change (67 vs 635).
2. **Alignment:** The framework's core metaphor is entities processing events. Making `target` the sole dispatch mechanism reinforces this.
3. **No breaking patterns:** The QueueDriver retargeting pattern (`event.target = self.target`) continues to work unchanged.
4. **Event subclasses:** No changes needed for QueuePollEvent, QueueNotifyEvent, QueueDeliverEvent, or SourceEvent.
5. **Debugging quality:** `repr` and stack traces are cleaner.
6. **Minimal new surface area:** `CallbackEntity` and `Event.once()` are simple, composable primitives.

---

## Implementation Plan (Approach A)

### Phase 1: Add New Primitives (Non-Breaking)

**Goal:** Introduce `CallbackEntity`, `NullEntity`, and `Event.once()` without removing anything.

1. Create `happysimulator/core/callback_entity.py`:
   - `CallbackEntity(Entity)` — wraps a function as an entity
   - `NullEntity(Entity)` — singleton that discards all events

2. Add `Event.once()` static method to `Event` class.

3. Export from `happysimulator/__init__.py`:
   - `CallbackEntity`, `NullEntity`

4. Write unit tests for all new primitives.

**Verification:** All existing tests still pass. New primitives work in isolation.

### Phase 2: Migrate Callback Users

**Goal:** Convert all existing `callback=` usage to the new target-based primitives.

1. **`happysimulator/instrumentation/probe.py`** — Use `Event.once()` in `_ProbeEventProvider.get_events()`.

2. **`examples/cold_start.py`** — Replace `callback=reset_callback` with `Event.once(...)`.

3. **`examples/load-balancing/fleet_change_comparison.py`** — Replace `callback=fleet_change_callback` with `Event.once(...)`.

4. **`happysimulator/load/source_event.py`** — Remove `callback=None` from `super().__init__()`.

5. **Tests** — Mechanical replacement of `callback=lambda e: None` with `Event.once(...)` or `target=NullEntity()`.

**Verification:** All tests pass with no `callback=` usage remaining except in `Event` class internals.

### Phase 3: Remove Callback from Event (Breaking)

**Goal:** Delete the callback dispatch path entirely.

1. Remove `callback` field from `Event` dataclass.
2. Remove `callback` parameter from `ProcessContinuation`.
3. Remove the `EventCallback` type alias.
4. Simplify `Event.invoke()` to single target dispatch.
5. Simplify `Event.__repr__()` to single target format.
6. Simplify `Event.__post_init__()` — validate that `target` is not None.
7. Simplify `ProcessContinuation.invoke()` — only propagate `target`.
8. Update `Event._start_process()` — remove `callback=self.callback`.

**Verification:** Full test suite passes. No references to `event.callback` remain in codebase.

### Phase 4: Clean Up

1. Update `CLAUDE.md` — Remove callback documentation, update Event examples.
2. Update module docstring in `event.py` — Remove "scripting-style" references.
3. Review and update any remaining design documents that reference the old duality.

---

## Appendix: Files Reference

| File | Role in This Design |
|---|---|
| `happysimulator/core/event.py` | Primary target — Event class, ProcessContinuation, invoke() dispatch |
| `happysimulator/core/entity.py` | Entity base class — unchanged |
| `happysimulator/core/protocols.py` | Simulatable protocol — unchanged |
| `happysimulator/core/simulation.py` | Simulation loop — unchanged (calls event.invoke()) |
| `happysimulator/core/event_heap.py` | Event heap — unchanged |
| `happysimulator/instrumentation/probe.py` | Only production callback user — must migrate |
| `happysimulator/load/source_event.py` | SourceEvent passes `callback=None` — must clean up |
| `happysimulator/components/queue.py` | Event subclasses (QueuePollEvent, etc.) — unchanged in Approach A |
| `happysimulator/components/queue_driver.py` | Retargeting pattern — unchanged in Approach A |
| New: `happysimulator/core/callback_entity.py` | CallbackEntity, NullEntity — new file |

---

## References

- `.dev/COMPONENTLIB.md` — Component library design philosophy
- `happysimulator/core/event.py` — Current Event implementation
- `happysimulator/core/protocols.py` — Simulatable protocol definition
