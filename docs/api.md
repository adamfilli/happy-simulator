# API Reference

## Core Classes

### Simulation

```python
from happysimulator import Simulation
```

The main simulation engine that manages the event loop.

**Parameters:**

- `end_time: Instant` - When the simulation should stop

**Methods:**

- `run()` - Start the simulation
- `schedule(event: Event)` - Add an event to the queue

---

### Event

```python
from happysimulator import Event
```

Represents a scheduled occurrence in the simulation.

**Parameters:**

- `time: Instant` - When the event occurs
- `event_type: str` - Identifier for the event type
- `target: Entity` (optional) - Entity to handle the event
- `callback: Callable` (optional) - Function to call when event fires

---

### Instant

```python
from happysimulator import Instant
```

Represents a point in simulation time.

**Class Methods:**

- `Instant.from_seconds(seconds: float)` - Create from seconds
- `Instant.Epoch` - Time zero
- `Instant.Infinity` - Used for auto-termination

---

### Entity

```python
from happysimulator import Entity
```

Base class for stateful simulation objects.

**Methods:**

- `handle_event(event: Event)` - Override to process events

---

## Load Generation

### Source

```python
from happysimulator.load import Source
```

Generates events at specified intervals.

### ArrivalTimeProvider

```python
from happysimulator.load import ConstantArrivalTimeProvider
```

Controls the timing between generated events.
