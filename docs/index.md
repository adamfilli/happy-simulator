# happy-simulator

A discrete-event simulation library for Python 3.13+, inspired by MATLAB SimEvents.

## Overview

happy-simulator models systems using an event-driven architecture where a central `EventHeap` schedules and executes `Event` objects until the simulation ends.

## Installation

```bash
pip install happysim
```

## Quick Example

```python
from happysimulator import Simulation, Event, Instant

def on_ping(event):
    print(f"Pong at {event.time}")

sim = Simulation(end_time=Instant.from_seconds(10))
sim.schedule(Event(
    time=Instant.from_seconds(1),
    event_type="Ping",
    callback=on_ping
))
sim.run()
```

## Next Steps

- [Getting Started](getting-started.md) - Learn the basics
- [API Reference](api.md) - Full API documentation
