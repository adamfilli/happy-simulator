# happy-simulator

A discrete-event simulation library for Python 3.13+, built for modeling distributed systems, queuing networks, and complex real-world processes.

---

<div class="grid cards" markdown>

-   **Event-Driven Core**

    ---

    Central `EventHeap` schedules and executes `Event` objects through `Entity` handlers. Generators let you express delays with `yield`.

-   **200+ Components**

    ---

    Queues, servers, networks, consensus protocols, storage engines, industrial processes, behavioral models, and more.

-   **Observability Built In**

    ---

    `Data`, `Probe`, `LatencyTracker`, `SimulationSummary`, and bucketed time-series analysis out of the box.

-   **Visual Debugger**

    ---

    Browser-based dashboard with entity graphs, charts, event logs, and step/play/pause controls.

</div>

## Quick Example

```python
from happysimulator import (
    Simulation, Event, Entity, Instant, Source, Sink,
)

class Server(Entity):
    def __init__(self, name, downstream):
        super().__init__(name)
        self.downstream = downstream

    def handle_event(self, event):
        yield 0.1  # simulate 100ms processing
        return [Event(time=self.now, event_type="Done", target=self.downstream)]

sink = Sink()
server = Server("server", downstream=sink)
source = Source.constant(rate=5, target=server)

sim = Simulation(
    entities=[source, server, sink],
    duration=10,
)
summary = sim.run()
print(f"Processed {sink.events_received} requests")
print(f"Avg latency: {sink.latency_stats()['avg']:.3f}s")
```

## Next Steps

- [Installation](installation.md) — install from PyPI or source
- [Getting Started](guides/getting-started.md) — build your first simulation
- [API Reference](reference/index.md) — full auto-generated API docs
- [Examples](examples/index.md) — 78 runnable examples across 10 categories
