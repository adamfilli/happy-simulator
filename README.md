<p align="center">
  <!-- Replace with project banner/logo (recommended: 1200x400px) -->
  <img src="docs/assets/banner-placeholder.png" alt="happy-simulator" width="100%" />
</p>

<h1 align="center">happy-simulator</h1>

<p align="center">
  <strong>A discrete-event simulation library for Python 3.13+</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/happysim/"><img src="https://img.shields.io/pypi/v/happysim" alt="PyPI" /></a>
  <a href="https://github.com/adamfilli/happy-simulator/actions/workflows/tests.yml"><img src="https://github.com/adamfilli/happy-simulator/actions/workflows/tests.yml/badge.svg" alt="Tests" /></a>
  <a href="https://adamfilli.github.io/happy-simulator/"><img src="https://github.com/adamfilli/happy-simulator/actions/workflows/docs.yml/badge.svg" alt="Docs" /></a>
  <a href="https://github.com/adamfilli/happy-simulator/blob/main/license"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License" /></a>
</p>

<p align="center">
  <a href="https://adamfilli.github.io/happy-simulator/">Documentation</a> &middot;
  <a href="https://adamfilli.github.io/happy-simulator/examples/">Examples</a> &middot;
  <a href="https://adamfillion.com/posts/simulation-enhanced-reasoning/">Blog Post</a>
</p>

---

> **Alpha** — Still in active development. APIs may change between releases.

happy-simulator is a composable, generator-based discrete-event simulation engine built for modeling queuing systems, distributed systems, manufacturing lines, and human behavior. It ships with 70+ ready-to-run examples and a browser-based visual debugger.

## Features

- **Generator-driven entities** — model delays and async coordination with `yield`
- **Rich component library** — queues, load balancers, rate limiters, circuit breakers, consensus protocols, industrial components, and more
- **Network simulation** — topology, latency models, partitions, and per-node clock skew/drift
- **Behavioral modeling** — agents with personality traits, decision models, social graphs, and influence propagation
- **Observability built in** — probes, latency/throughput trackers, bucketed time series
- **Simulation control** — pause, step, breakpoints (time, event count, metric threshold, conditional)
- **Visual debugger** — browser-based UI with entity graph, live dashboards, event log, and step-through debugging

## Quick Start

```bash
pip install happysim
```

```python
from happysimulator import Simulation, Source, Sink, Event, Instant
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.components.queue_policies import FIFOQueue

class Server(QueuedResource):
    def __init__(self, downstream):
        super().__init__("Server", policy=FIFOQueue())
        self.downstream = downstream

    def handle_queued_event(self, event):
        yield 0.1  # process for 100ms
        return [Event(time=self.now, event_type="Done", target=self.downstream)]

sink = Sink()
server = Server(downstream=sink)
source = Source.poisson(rate=8, target=server)

sim = Simulation(entities=[source, server, sink], end_time=Instant.from_seconds(60))
summary = sim.run()

print(f"Processed {summary.total_events_processed} events in {summary.wall_clock_seconds:.2f}s")
print(f"Latency: {sink.latency_stats()}")
```

## Visual Debugger

```bash
pip install happysim[visual]
```

```python
from happysimulator.visual import serve
serve(sim)  # opens browser at http://127.0.0.1:8765
```

Step through events, inspect entity state, pin live charts to the graph, and set breakpoints — all from the browser.

## Installation

```bash
pip install happysim              # core library
pip install happysim[visual]      # + browser debugger (FastAPI + uvicorn)
pip install happysim[dev]         # + testing & docs tools
```

Or install from source:

```bash
git clone https://github.com/adamfilli/happy-simulator.git
cd happy-simulator
pip install -e ".[dev]"
```

## Documentation

Full guides, API reference, and example walkthroughs at **[adamfilli.github.io/happy-simulator](https://adamfilli.github.io/happy-simulator/)**.

## License

[Apache 2.0](license)
