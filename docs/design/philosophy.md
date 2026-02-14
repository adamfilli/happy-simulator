# Design Philosophy

## Core Principles

### Composition Over Inheritance

Complex systems are built by composing smaller entities into pipelines, not by deep inheritance hierarchies. A server is a queue + processing logic. A rate limiter is a policy + entity wrapper.

```python
# Compose: Source → RateLimiter → Server → Sink
sink = Sink()
server = Server("server", downstream=sink)
limiter = RateLimitedEntity("limiter", policy=TokenBucketPolicy(100, 10), downstream=server)
source = Source.constant(rate=200, target=limiter)
```

### Protocol-Based Design

The `Simulatable` protocol enables duck-typing — any object with a `handle_event()` method can participate in the simulation. No mandatory base class required (though `Entity` is convenient).

### Generator-Friendly

Generators let you express multi-step processing naturally. `yield` delays instead of callbacks. `yield` futures instead of completion handlers. This keeps entity logic linear and readable.

```python
def handle_event(self, event):
    yield 0.01                          # delay
    yield 0.0, [Event(...)]             # delay + side-effect
    response = yield future             # park until resolved
    return [Event(...)]                 # final output
```

### Clock Injection

Entities receive time through `set_clock()`, called automatically when registered with a `Simulation`. This enables per-node clock skew, drift, and logical clocks without changing entity code.

### Transparent Internals

Components hide implementation complexity. A `QueuedResource` manages its own queue, dequeue scheduling, and capacity checks internally. Callers just send events to it.

## Architecture

### Event Loop

```
EventHeap.pop() → Entity.handle_event() → schedule returned Event(s) → repeat
```

The simulation processes events in strict chronological order. Generators are stepped forward through the heap — each `yield` creates a continuation event.

### Entity Graph

Entities form a directed graph through their `target` references. Events flow through the graph. The visual debugger renders this topology automatically.

### Time Model

- **True time** (`self.now`): monotonically advancing simulation clock, used for scheduling
- **Local time** (`self.local_now`): per-node perceived time with optional skew/drift
- **Logical time**: Lamport/Vector/HLC clocks for causal ordering

### No Global State

Entities communicate only through events. There is no shared mutable state, no global bus, no service locator. This makes simulations deterministic and reproducible.

## Python 3.13+

The library targets Python 3.13+ and uses modern Python features:

- Type hints with `|` union syntax
- Dataclasses for immutable value types
- `typing.Protocol` for structural subtyping
- Google-style docstrings

## Documentation

- Comment the "why", not the "what"
- Document yield semantics in generator methods
- Design docs for major features in `.dev/` directory
