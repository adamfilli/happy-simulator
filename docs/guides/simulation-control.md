# Simulation Control

Lazy-created via `sim.control` — zero overhead when unused.

## Basic Controls

```python
sim.control.pause()                    # pause before or during run
summary = sim.run()                    # runs until pause/breakpoint/end

sim.control.step(5)                    # process exactly 5 events
sim.control.resume()                   # resume to completion

state = sim.control.get_state()        # SimulationState
state.current_time                     # Instant
state.events_processed                 # int
state.is_paused                        # bool
```

### Peeking

While paused, inspect upcoming events without consuming them:

```python
upcoming = sim.control.peek_next(3)    # list of next 3 events
```

### Reset

```python
sim.control.reset()   # clear heap, re-prime sources (does NOT reset entity state)
```

## Breakpoints

Breakpoints automatically pause the simulation when conditions are met.

```python
from happysimulator import (
    TimeBreakpoint, EventCountBreakpoint, ConditionBreakpoint,
    MetricBreakpoint, EventTypeBreakpoint, Instant,
)

# Pause at specific simulation time
sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(30.0)))

# Pause after N events
sim.control.add_breakpoint(EventCountBreakpoint(count=1000))

# Pause on custom condition
sim.control.add_breakpoint(ConditionBreakpoint(
    fn=lambda ctx: ctx.current_time > Instant.from_seconds(10),
    description="After warmup",
))

# Pause when entity metric crosses threshold
sim.control.add_breakpoint(MetricBreakpoint(
    entity_name="Server",
    attribute="depth",
    operator="gt",
    threshold=100,
))

# Pause on specific event type
sim.control.add_breakpoint(EventTypeBreakpoint(event_type="Timeout"))
```

### Managing Breakpoints

```python
bp_id = sim.control.add_breakpoint(...)
sim.control.remove_breakpoint(bp_id)
sim.control.list_breakpoints()
sim.control.clear_breakpoints()
```

## Hooks

Hooks fire on every event without pausing — useful for logging or metrics collection.

```python
hook_id = sim.control.on_event(lambda event: print(event))
sim.control.on_time_advance(lambda t: print(f"Time: {t}"))
sim.control.remove_hook(hook_id)
```

## Re-entrant Runs

`sim.run()` is re-entrant — it resumes from the pause point:

```python
sim.control.pause()
summary1 = sim.run()     # runs until pause
# inspect state...
sim.control.resume()
summary2 = sim.run()     # continues from where it left off
```

## Next Steps

- [Visual Debugger](visual-debugger.md) — browser-based step/play/pause with charts
- [Breakpoints](../reference/core/control.md) — API reference
