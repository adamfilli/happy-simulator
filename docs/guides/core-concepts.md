# Core Concepts

## Instant & Duration

All simulation time is represented by `Instant` (a point in time) and `Duration` (a span of time).

```python
from happysimulator import Instant, Duration

t = Instant.from_seconds(1.5)
t2 = t + Duration.from_seconds(0.5)  # Instant + Duration → Instant
dt = t2 - t                           # Instant - Instant → Duration
```

**Special values:**

- `Instant.Epoch` — time zero (the start of simulation)
- `Instant.Infinity` — used for open-ended simulations

!!! warning
    Always use `Instant.from_seconds(n)`, never raw floats for event times. Floats are only used for `yield` delays inside generators.

## Event

Every event must have a `target` entity. Events are the messages that flow between entities.

```python
from happysimulator import Event, Instant

# Target-based event (most common)
event = Event(
    time=Instant.from_seconds(1.0),
    event_type="Request",
    target=server,
    context={"payload": "hello"},
)

# Function-based event (convenience wrapper)
event = Event.once(
    time=Instant.from_seconds(1.0),
    event_type="Ping",
    fn=lambda e: print("pong"),
)
```

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `time` | `Instant` | When the event fires |
| `event_type` | `str` | Identifier for routing/filtering |
| `target` | `Entity` | The entity that will handle this event |
| `context` | `dict` | Arbitrary payload data |

## Entity

Entities are stateful actors — the building blocks of every simulation. Override `handle_event()` to define behavior.

```python
from happysimulator import Entity, Event

class Server(Entity):
    def handle_event(self, event):
        # Process and forward
        return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

**Return types from `handle_event()`:**

| Return | Meaning |
|--------|---------|
| `None` | No follow-up events |
| `Event` | Schedule one event |
| `list[Event]` | Schedule multiple events |
| `Generator` | Coroutine with delays (see [Generators & Futures](generators-and-futures.md)) |

**Useful properties:**

- `self.now` — current simulation time (`Instant`)
- `self.name` — entity name (set in constructor)

## Simulation

The `Simulation` class ties everything together. It manages the event heap, injects clocks into entities, and runs the main loop.

```python
from happysimulator import Simulation

sim = Simulation(
    entities=[source, server, sink],   # register all entities
    duration=100,
)
summary = sim.run()
```

**Key points:**

- All entities must be registered in the `entities` list — this is how they receive clock injection
- `sim.schedule(event)` adds events before or during the run
- `sim.run()` returns a `SimulationSummary` with timing and per-entity statistics
- The simulation processes events in chronological order until `end_time` or the heap is empty

### Lifecycle

1. **Construction** — entities registered, clocks injected
2. **Scheduling** — initial events added via `sim.schedule()` or Source auto-priming
3. **Running** — `sim.run()` pops events from the heap and dispatches to entities
4. **Completion** — returns `SimulationSummary`

## Clock Injection

When entities are registered with a `Simulation`, they receive a clock via `set_clock()`. This is how `self.now` works — it reads from the simulation's clock.

!!! note
    Always register entities with the simulation. An entity that isn't registered won't have a working `self.now`.

## Next Steps

- [Generators & Futures](generators-and-futures.md) — yield-based delays and SimFuture
- [Load Generation](load-generation.md) — Source factories and arrival patterns
