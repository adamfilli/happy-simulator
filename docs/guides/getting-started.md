# Getting Started

This guide walks you through building your first discrete-event simulation — from a single event to a full M/M/1 queuing model.

## Your First Event

The simplest simulation schedules one event using `Event.once()`:

```python
from happysimulator import Simulation, Event, Instant

sim = Simulation(duration=10)
sim.schedule(Event.once(
    time=Instant.from_seconds(1.0),
    event_type="Ping",
    fn=lambda e: print(f"Pong at {e.time}"),
))
sim.run()
```

`Event.once()` wraps a function in a lightweight entity so you don't need to define a class for simple one-shot events.

## Your First Entity

Entities are stateful actors that receive events via `handle_event()`. Here's a server that processes requests with a 100ms delay:

```python
from happysimulator import Entity, Event

class Server(Entity):
    def __init__(self, name):
        super().__init__(name)
        self.processed = 0

    def handle_event(self, event):
        yield 0.1  # pause for 100ms of simulation time
        self.processed += 1
        print(f"[{self.now}] Processed request #{self.processed}")
```

The `yield 0.1` pauses the generator for 0.1 seconds of *simulation* time — the simulation engine resumes it at the right moment.

## Wiring Entities Together

Entities communicate by returning events that target other entities:

```python
from happysimulator import Entity, Event, Sink

class Server(Entity):
    def __init__(self, name, downstream):
        super().__init__(name)
        self.downstream = downstream

    def handle_event(self, event):
        yield 0.1  # processing time
        return [Event(time=self.now, event_type="Done", target=self.downstream)]
```

The returned event is automatically scheduled by the simulation engine.

## Adding Load Generation

Use `Source` to generate a stream of events. The factory methods handle timing automatically:

```python
from happysimulator import Source

source = Source.constant(rate=5, target=server, event_type="Request")
```

This generates one `Request` event every 0.2 seconds (rate = 5/sec).

## Complete Example: Source → Server → Sink

Putting it all together into a pipeline:

```python
from happysimulator import (
    Simulation, Event, Entity, Instant, Source, Sink,
)

class Server(Entity):
    def __init__(self, name, downstream):
        super().__init__(name)
        self.downstream = downstream

    def handle_event(self, event):
        yield 0.1  # 100ms service time
        return [Event(time=self.now, event_type="Done", target=self.downstream)]

# Build the pipeline
sink = Sink()
server = Server("server", downstream=sink)
source = Source.constant(rate=5, target=server)

# Run the simulation
sim = Simulation(
    entities=[source, server, sink],
    duration=10,
)
summary = sim.run()

# Inspect results
print(f"Events processed: {summary.total_events_processed}")
print(f"Requests completed: {sink.events_received}")
print(f"Latency stats: {sink.latency_stats()}")
```

## M/M/1 Queue Model

For a proper queuing model with buffering, use `QueuedResource`:

```python
from happysimulator import (
    Simulation, Event, Instant, Source, Sink,
    QueuedResource, FIFOQueue, ExponentialLatency,
)

class MMOneServer(QueuedResource):
    def __init__(self, name, downstream, service_rate):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.service_time = ExponentialLatency(1.0 / service_rate)

    def handle_queued_event(self, event):
        yield self.service_time.sample()
        return [Event(time=self.now, event_type="Done", target=self.downstream)]

sink = Sink()
server = MMOneServer("server", downstream=sink, service_rate=6)
source = Source.poisson(rate=5, target=server)

sim = Simulation(
    entities=[source, server, sink],
    duration=1000,
)
summary = sim.run()

stats = sink.latency_stats()
print(f"Avg latency: {stats['avg']:.3f}s")
print(f"P99 latency: {stats['p99']:.3f}s")
```

With arrival rate 5 and service rate 6, utilization is ~83% — expect moderate queuing.

## Next Steps

- [Core Concepts](core-concepts.md) — deeper dive into Instant, Event, Entity, and Simulation
- [Generators & Futures](generators-and-futures.md) — advanced yield patterns and SimFuture
- [Load Generation](load-generation.md) — Source factories, profiles, and custom providers
