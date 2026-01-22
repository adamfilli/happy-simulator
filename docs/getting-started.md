# Getting Started

## Installation

### From PyPI

```bash
pip install happysim
```

### From Source

```bash
git clone https://github.com/adamfilli/happy-simulator.git
cd happy-simulator
pip install -e .
```

## Core Concepts

### Events

Events are the fundamental unit of work in happy-simulator. Each event has a scheduled time and either a target entity or a callback function.

```python
from happysimulator import Event, Instant

# Callback-style event
event = Event(
    time=Instant.from_seconds(1),
    event_type="MyEvent",
    callback=lambda e: print("Event fired!")
)
```

### Simulation

The `Simulation` class runs the event loop, processing events in chronological order.

```python
from happysimulator import Simulation, Instant

sim = Simulation(end_time=Instant.from_seconds(100))
sim.run()
```

### Entities

Entities are stateful objects that handle events via the `handle_event()` method.

```python
from happysimulator import Entity

class MyServer(Entity):
    def handle_event(self, event):
        # Process the event
        pass
```

## Next Steps

See the [API Reference](api.md) for detailed documentation.
