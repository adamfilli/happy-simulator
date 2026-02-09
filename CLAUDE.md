# CLAUDE.md

> **Last Updated:** 2026-02-09

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick Reference

| Aspect | Summary |
|--------|---------|
| **What** | Discrete-event simulation library for Python 3.13+ |
| **Core Loop** | `EventHeap` pop → `Entity.handle_event()` → schedule returned `Event`s |
| **Key Invariant** | Events always have a `target` (Entity); use `Event.once()` for function-based dispatch |
| **Time** | Use `Instant.from_seconds(n)`, not raw floats |
| **Generators** | Yield delays (float seconds) or `SimFuture`; return events on completion |
| **Load Gen** | `Source.poisson(rate=10, target=server)` for quick setup; full constructor for advanced cases |
| **Control** | `sim.control.pause()` / `.step()` / `.add_breakpoint()` for interactive debugging |
| **Testing** | Use `Source.constant()` or `ConstantArrivalTimeProvider` for deterministic timing |

---

## Reading Order for New Contributors

### Phase 1: Orientation
1. **`CLAUDE.md`** - Entry point (this file)
2. **`happysimulator/__init__.py`** - Public API surface

### Phase 2: Core Concepts
3. **`happysimulator/core/instant.py`** - Time representation (`Instant` class)
4. **`happysimulator/core/event.py`** - Event structure and lifecycle
5. **`happysimulator/core/entity.py`** - Actor pattern, `handle_event()` method
6. **`happysimulator/core/simulation.py`** - Main loop, event scheduling
7. **`happysimulator/core/sim_future.py`** - SimFuture, any_of, all_of

### Phase 3: Interactive Control
7. **`happysimulator/core/control/control.py`** - Pause/resume, stepping, breakpoints
8. **`happysimulator/core/control/breakpoints.py`** - Breakpoint protocol and implementations

### Phase 4: Patterns
9. **`happysimulator/load/source.py`** - Self-perpetuating event generation
10. **`happysimulator/components/queue.py`** - Queue/Driver pattern
11. **`happysimulator/components/queued_resource.py`** - Resource abstraction

### Phase 5: Examples
12. **`examples/m_m_1_queue.py`** - Full M/M/1 queue workflow with visualization
13. Corresponding integration tests in `tests/integration/`

### Phase 6: Design Context
14. **`.dev/COMPONENTLIB.md`** - Component library design philosophy
15. **`.dev/zipf-distribution-design.md`** - Feature design document template

---

## Development Commands

```bash
# Setup (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .

# Run tests
pytest -q                                    # all tests
pytest tests/integration/test_queue.py -q    # single file

# Run examples
python examples/m_m_1_queue.py

# Enable logging (silent by default)
python -c "import happysimulator; happysimulator.enable_console_logging('DEBUG')"
```

---

## Logging

By default, happysimulator is **silent** (follows Python library best practices). Enable logging explicitly:

```python
import happysimulator

# Console logging
happysimulator.enable_console_logging(level="DEBUG")

# Rotating file logging (prevents disk space issues)
happysimulator.enable_file_logging("simulation.log", max_bytes=10_000_000)

# JSON logging for log aggregation (ELK, Datadog, etc.)
happysimulator.enable_json_logging()

# Configure from environment variables
happysimulator.configure_from_env()

# Per-module control
happysimulator.set_module_level("core.simulation", "DEBUG")
happysimulator.set_module_level("distributions", "WARNING")

# Silence completely
happysimulator.disable_logging()
```

**Environment variables** (used with `configure_from_env()`):
- `HS_LOGGING`: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `HS_LOG_FILE`: Path to log file (enables rotating file handler)
- `HS_LOG_JSON`: Set to "1" for JSON output

---

## Core Abstractions

### Instant (Time Representation)

`Instant` represents a point in simulation time with nanosecond precision internally.

```python
from happysimulator import Instant

# Creation
t = Instant.from_seconds(1.5)       # 1.5 seconds from epoch
t = Instant.Epoch                    # Time zero
t = Instant.Infinity                 # For auto-termination / never

# Arithmetic
t2 = t + Duration.from_seconds(0.5)  # Add duration
delta = t2 - t                       # Get Duration between instants
seconds = delta.to_seconds()         # Convert to float

# Comparison
if t < t2:
    print("t is earlier")
```

### Event (Unit of Work)

Events are the fundamental unit of work in the simulation. They carry information about *when* something happens and *what* should happen.

**Key Invariant**: Every Event must have a `target` (Entity). For function-based dispatch, use `Event.once()` which wraps a function in a `CallbackEntity`.

```python
from happysimulator import Event, Instant

# Standard: target an Entity
request_event = Event(
    time=Instant.from_seconds(1.0),
    event_type="Request",
    target=my_server,
    context={"customer_id": 42},
)

# Function-based: use Event.once() for one-shot callbacks
ping_event = Event.once(
    time=Instant.from_seconds(1.0),
    event_type="Ping",
    fn=lambda e: print("pong"),
)
```

### Entity (Actor Pattern)

Entities are stateful actors that process events. They implement `handle_event(event)` which can return:
- `None` - No follow-up events
- `Event` - Single follow-up event
- `list[Event]` - Multiple follow-up events
- `Generator[float, None, list[Event]]` - Multi-step process with delays

```python
from happysimulator import Entity, Event
from typing import Generator

class Server(Entity):
    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self.downstream = downstream
        self.processed = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        # Yield delays (in seconds) to simulate processing time
        yield 0.1  # Wait 100ms

        self.processed += 1

        # Return follow-up events
        return [Event(
            time=self.now,  # Current simulation time
            event_type="Completed",
            target=self.downstream,
            context=event.context,
        )]
```

### Generator Semantics (Multi-Step Processes)

Generators allow expressing sequential processes with delays naturally:

```python
def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
    # Each yield pauses the process for that many seconds
    yield 0.05  # Wait 50ms (e.g., network latency)

    # Can yield computed values
    service_time = random.expovariate(1.0 / 0.1)  # Exponential ~100ms
    yield service_time

    # Can yield (delay, side_effect_events) tuples for immediate side effects
    yield 0.01, [Event(...)]  # Wait 10ms AND schedule side-effect event NOW

    # Can yield (delay, None) explicitly
    yield 1.0, None  # Equivalent to just `yield 1.0`

    # Final return sends events after all yields complete
    return [completion_event]
```

**Yield Forms**:
- `yield delay` - Pause for `delay` seconds
- `yield delay, None` - Same as above (explicit)
- `yield delay, event` - Pause AND schedule a side-effect event immediately
- `yield delay, [events]` - Pause AND schedule multiple side-effect events
- `yield future` - Park until `future.resolve(value)` is called (see SimFuture below)

**Under the hood**: The runtime wraps generators as `ProcessContinuation` events that reschedule after each yield. For `SimFuture` yields, the process parks instead of scheduling a time-based continuation.

### SimFuture (Yield on Events, Not Just Delays)

`SimFuture` enables generators to pause until an external condition is met, rather than only pausing for fixed time delays. This unlocks natural request-response modeling, resource acquisition, timeout races, and quorum waits.

```python
from happysimulator import SimFuture, any_of, all_of

# Basic: request-response pattern
class Client(Entity):
    def handle_event(self, event):
        future = SimFuture()
        # Send request with the future so server can resolve it
        yield 0.0, [Event(
            time=self.now, event_type="Request", target=self.server,
            context={"reply_future": future},
        )]
        response = yield future  # Park until server resolves
        # response is the value passed to future.resolve(value)

class Server(Entity):
    def handle_event(self, event):
        yield 0.1  # Processing time
        event.context["reply_future"].resolve({"status": "ok"})

# Timeout race with any_of
response_future = SimFuture()
timeout_future = SimFuture()
yield 0.0, [
    Event(time=self.now, event_type="Req", target=server,
          context={"future": response_future}),
    Event.once(time=Instant.from_seconds(self.now.to_seconds() + 5.0),
               event_type="Timeout",
               fn=lambda e: timeout_future.resolve("timeout")),
]
idx, value = yield any_of(response_future, timeout_future)
# idx=0 → response won; idx=1 → timeout won

# Quorum wait with all_of
f1, f2, f3 = SimFuture(), SimFuture(), SimFuture()
yield 0.0, [
    Event(time=self.now, event_type="Write", target=r1, context={"ack": f1}),
    Event(time=self.now, event_type="Write", target=r2, context={"ack": f2}),
    Event(time=self.now, event_type="Write", target=r3, context={"ack": f3}),
]
results = yield all_of(f1, f2, f3)  # [value1, value2, value3]

```

**Key behaviors**:
- `resolve(value)` resumes the parked generator with `value` via `gen.send(value)`
- Pre-resolved futures work: yielding an already-resolved future resumes immediately
- Each `SimFuture` can only be yielded by one generator
- `any_of(*futures)` resolves with `(index, value)` when the first input resolves
- `all_of(*futures)` resolves with `[values]` when all inputs resolve

---

## Architecture

### Core Simulation Loop (`happysimulator/core/simulation.py`)

The `Simulation` class:
1. Initializes sources (calls `source.start()` to get initial events)
2. Runs a pop-invoke-push loop on the `EventHeap`
3. Continues until `end_time` is reached or heap is exhausted

```
┌─────────────────────────────────────────────────────────┐
│                    SIMULATION LOOP                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌─────────┐    pop     ┌─────────────┐                │
│   │ EventHeap │─────────►│   Event     │                │
│   │ (min-heap │          └──────┬──────┘                │
│   │  by time) │                 │                        │
│   └─────────┘                   ▼                        │
│       ▲                         │                        │
│       │                         ▼                        │
│       │                  ┌─────────────┐                │
│       │                  │   Entity    │                │
│       │                  │ handle_     │                │
│       │                  │  event()   │                │
│       │                  └──────┬──────┘                │
│       │                         │                        │
│       │                         ▼                        │
│       │         ┌──────────────────────────┐            │
│       │         │ Result: None | Event |   │            │
│       │         │ list[Event] | Generator │            │
│       │         └───────────┬──────────────┘            │
│       │                     │                            │
│       └─────────────────────┘                            │
│               push new events                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Event Dispatch

All events dispatch through a `target` Entity's `handle_event()` method. For function-based dispatch, `Event.once()` wraps a function in a `CallbackEntity`. This uniform target-based model simplifies the dispatch path and debugging.

### Load Generation (`happysimulator/load/`)

- **`Source`**: Self-perpetuating entity that generates events at intervals
  - **Factory methods** for common cases (see Common Patterns below):
    - `Source.constant(rate, target, ...)` - Fixed-rate traffic
    - `Source.poisson(rate, target, ...)` - Stochastic arrivals
    - `Source.with_profile(profile, target, ...)` - Custom rate profiles
- **`EventProvider`**: Creates payload events at each tick
- **`ArrivalTimeProvider`** implementations:
  - `ConstantArrivalTimeProvider`: Fixed intervals (deterministic)
  - `PoissonArrivalTimeProvider`: Poisson-distributed arrivals
- **`Profile`**: Defines rate over time (e.g., `ConstantRateProfile`, `SpikeProfile`)

---

## Component Design Principles

All components follow these patterns (extracted from `.dev/COMPONENTLIB.md`):

### Composition over Inheritance
Combine smaller entities into larger abstractions rather than deep inheritance hierarchies.

```python
# Good: Compose a QueuedResource from Queue + Driver
class Server(QueuedResource):
    def __init__(self, name: str):
        super().__init__(name, policy=FIFOQueue())
```

### Protocol-based Design
Use the `Simulatable` protocol for duck-typing compatibility.

```python
from happysimulator import Simulatable, simulatable

# Classes implement Simulatable protocol
class MyComponent:
    def set_clock(self, clock: Clock) -> None: ...

# Or use decorator for simple cases
@simulatable
class SimpleComponent: ...
```

### Generator-friendly
Express delays naturally with `yield` statements in `handle_event()`.

### Clock Injection
Components receive simulation time via `set_clock()`. Use `self.now` to get current time.

```python
class MyEntity(Entity):
    def handle_event(self, event: Event):
        current_time = self.now  # Instant from injected clock
        # ...
```

### Completion Hooks
Enable loose coupling between components using callbacks/events on completion.

```python
class Server(QueuedResource):
    def __init__(self, name: str, on_complete: Callable[[Event], None] | None = None):
        self._on_complete = on_complete
```

### Transparent Internals
Hide implementation complexity from external callers. Expose simple APIs.

---

## Key Directories

```
happysimulator/
├── core/                    # Core simulation engine
│   ├── instant.py          # Time representation
│   ├── event.py            # Event structure
│   ├── callback_entity.py  # CallbackEntity, NullEntity
│   ├── sim_future.py       # SimFuture, any_of, all_of
│   ├── entity.py           # Entity base class
│   ├── simulation.py       # Main simulation loop (re-entrant)
│   ├── clock.py            # Clock abstraction
│   ├── protocols.py        # Simulatable protocol
│   └── control/            # Interactive simulation control
│       ├── control.py      # SimulationControl (pause/resume/step/breakpoints)
│       ├── state.py        # SimulationState, BreakpointContext
│       └── breakpoints.py  # Breakpoint protocol + 5 implementations
│
├── load/                    # Load generation
│   ├── source.py           # Self-perpetuating source
│   ├── profile.py          # Rate profiles (ConstantRateProfile, etc.)
│   ├── event_provider.py   # EventProvider base class
│   └── providers/          # EventProvider implementations
│       └── distributed_field.py  # Zipf/distribution-based
│
├── components/              # Reusable simulation components
│   ├── common.py           # Sink (latency tracking), Counter
│   ├── queue.py            # Queue implementations
│   ├── queued_resource.py  # Queue + processing
│   ├── random_router.py    # Load balancing
│   ├── rate_limiter/       # Rate limiting
│   │   ├── inductor.py    # Inductor (EWMA burst suppression)
│   │   └── ...            # Policies + RateLimitedEntity
│   ├── network/            # Network simulation
│   ├── server/             # Server models
│   ├── client/             # Client models
│   ├── resilience/         # Circuit breakers, bulkheads
│   ├── messaging/          # Queues, topics, DLQ
│   ├── datastore/          # KV stores, caches
│   ├── sync/               # Mutexes, semaphores
│   └── queue_policies/     # CoDel, RED, fair queuing
│
├── distributions/           # Probability distributions
│   ├── latency_distribution.py  # Base for latency dists
│   ├── constant.py         # ConstantLatency
│   ├── exponential.py      # ExponentialLatency
│   ├── value_distribution.py  # Base for discrete dists
│   ├── zipf.py             # ZipfDistribution
│   └── uniform.py          # UniformDistribution
│
├── instrumentation/         # Metrics and probing
│   ├── data.py             # Time-series data collection
│   ├── collectors.py       # LatencyTracker, ThroughputTracker
│   └── probe.py            # Periodic metric sampling
│
└── utils/                   # Utilities

examples/                    # Runnable example scenarios
├── m_m_1_queue.py          # Metastable failure demo
├── basic_client_server.py  # Simple request/response
└── ...

tests/
├── unit/                   # Unit tests
├── integration/            # Integration tests
└── conftest.py             # Shared fixtures

.dev/                       # Design documents
├── COMPONENTLIB.md         # Component library plan
├── event-dispatch-simplification.md  # Target-only dispatch (implemented)
├── ergonomic-api-improvements.md     # Source factories, Sink/Counter
├── logging-design.md       # Logging system design (implemented)
├── observability-design.md # Observability API design (implemented)
├── inductor-design.md      # Inductor entity design (implemented)
├── simulation-control-design.md     # Simulation control design (implemented)
├── sketching-algorithms-design.md    # Sketching data structures
└── zipf-distribution-design.md      # Feature design template

archive/                    # Reference implementations
                           # (not fully integrated, useful for patterns)
```

---

## Testing Patterns

### Test Organization
- **Unit tests**: `tests/unit/` - Isolated component tests
- **Integration tests**: `tests/integration/` - Full simulation workflows

### Deterministic Testing
Use `ConstantArrivalTimeProvider` and `ConstantLatency` for reproducible tests:

```python
from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Instant,
)
from happysimulator.distributions import ConstantLatency

# Deterministic arrival: exactly 10 req/s
arrival = ConstantArrivalTimeProvider(
    ConstantRateProfile(rate=10.0),
    start_time=Instant.Epoch,
)

# Deterministic service time: always 100ms
service_time = ConstantLatency(0.1)
```

### Test Fixtures
Available fixtures in `tests/conftest.py`:

```python
def test_my_simulation(test_output_dir):
    """test_output_dir provides a directory for test artifacts."""
    # Saves to: test_output/<module>/<test_name>/
    plt.savefig(test_output_dir / "my_plot.png")

def test_with_history(timestamped_output_dir):
    """timestamped_output_dir keeps history of multiple runs."""
    # Saves to: test_output/<module>/<test>/<timestamp>/
```

### Visualization Tests
Use `matplotlib.use("Agg")` for headless rendering:

```python
def test_visualization(test_output_dir):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    # ... run simulation ...

    plt.plot(times, values)
    plt.savefig(test_output_dir / "results.png", dpi=150)
    plt.close()
```

### Key Integration Tests (Reference)

| Test File | What It Demonstrates |
|-----------|---------------------|
| `test_queued_resource.py` | End-to-end QueuedResource with probes |
| `test_simulation_basic_yield.py` | Generator yields with side effects |
| `test_queue.py` | Queue/Driver interaction with tracing |
| `test_visualization_example.py` | Saving plots, CSV, JSON artifacts |
| `test_zipf_distribution_visualization.py` | Distribution verification |
| `test_compare_lifo_fifo.py` | Comparing queue policies |
| `test_simulation_control.py` | Pause/resume, breakpoints, stepping |

---

## Example Patterns

### Standard Example Structure

Examples follow a consistent pattern (see `examples/m_m_1_queue.py`):

```python
"""Short description of what this example demonstrates.

Longer explanation of the scenario, including:
- What behavior to observe
- Key parameters
- Expected outcomes

## Architecture Diagram (ASCII art)
"""

from __future__ import annotations
# ... imports ...

# =============================================================================
# Configuration / Profiles
# =============================================================================

@dataclass(frozen=True)
class MyProfile(Profile):
    """Defines load pattern for this scenario."""
    ...

# =============================================================================
# Custom Entities (if needed)
# =============================================================================

class MySink(Entity):
    """Collects results for analysis."""
    ...

# =============================================================================
# Simulation Setup
# =============================================================================

def run_simulation(...) -> SimulationResult:
    """Run the simulation with given parameters."""
    ...

def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations."""
    ...

def print_summary(result: SimulationResult) -> None:
    """Print summary statistics."""
    ...

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(...)
    # ... parse args, run simulation, visualize ...
```

---

## Common Patterns

### Creating a One-Shot Function Event

```python
Event.once(
    time=Instant.from_seconds(1),
    event_type="Ping",
    fn=lambda e: print("pong"),
)
```

### Generator-based Process (multi-step with delays)

```python
def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
    yield 0.05  # Wait 50ms (simulated network latency)
    yield self._compute_latency()  # Wait for processing

    # Return events at the end
    return [Event(
        time=self.now,
        event_type="Completed",
        target=self.downstream,
        context=event.context,
    )]
```

### Starting a Source (Factory Methods)

For most use cases, use the factory class methods which auto-generate the `EventProvider`:

```python
# Fixed-rate traffic (deterministic)
source = Source.constant(rate=10, target=server, event_type="Request")

# Stochastic arrivals (Poisson process)
source = Source.poisson(rate=10, target=server, event_type="Request")

# Custom rate profile with stochastic arrivals
source = Source.with_profile(
    profile=MetastableLoadProfile(),
    target=server,
    event_type="Request",
    poisson=True,
)

# All factories accept stop_after (float seconds or Instant)
source = Source.poisson(rate=10, target=server, stop_after=60.0)
```

For advanced cases (custom `EventProvider` with field distributions, etc.), use the full constructor:

```python
source = Source(
    name="Traffic",
    event_provider=my_provider,
    arrival_time_provider=ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=100),
    ),
)
# Source.start() is called automatically by Simulation
```

### Using Sink and Counter

`Sink` and `Counter` eliminate common boilerplate for collecting simulation results:

```python
from happysimulator import Sink, Counter

# Sink: collects events and tracks latency (from context['created_at'])
sink = Sink()
# ... wire as downstream entity ...
sink.events_received       # total count
sink.latency_stats()       # dict with count, avg, min, max, p50, p99

# Counter: counts events by type
counter = Counter()
counter.total              # total events
counter.by_type            # dict[str, int]
```

### Using Distributions

```python
from happysimulator import ZipfDistribution, DistributedFieldProvider

# Zipf distribution for realistic hot/cold key patterns
customer_dist = ZipfDistribution(range(1000), s=1.0, seed=42)

provider = DistributedFieldProvider(
    target=my_server,
    event_type="Request",
    field_distributions={"customer_id": customer_dist},
)
```

### QueuedResource Pattern

`QueuedResource` combines a queue with processing logic. Override `handle_queued_event()` and optionally `has_capacity()`:

```python
from happysimulator import QueuedResource, FIFOQueue

class MyServer(QueuedResource):
    def __init__(self, name: str, downstream: Entity, concurrency: int = 1):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.concurrency = concurrency
        self._in_flight = 0

    def has_capacity(self) -> bool:
        """Return True if server can accept more work."""
        return self._in_flight < self.concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._in_flight += 1
        try:
            yield 0.1  # 100ms service time
        finally:
            self._in_flight -= 1

        return [Event(
            time=self.now,
            event_type="Done",
            target=self.downstream,
            context=event.context,
        )]
```

**Key methods**:
- `handle_queued_event(event)` - Process events from the queue (generator)
- `has_capacity()` - Controls when queue driver pulls next item
- `self.depth` - Current queue depth (read-only property)

### Inductor (Burst Suppression)

`Inductor` smooths bursty traffic using EWMA of inter-arrival times. Unlike rate limiters, it has **no throughput cap** — it resists rate *changes*, not absolute rate.

```python
from happysimulator import Inductor

# Place between source and server to smooth bursts
inductor = Inductor(
    name="Smoother",
    downstream=server,
    time_constant=1.0,    # higher = more smoothing
    queue_capacity=10000, # max buffered events
)
source = Source.poisson(rate=100, target=inductor)

# Observability
inductor.stats             # InductorStats: received, forwarded, queued, dropped
inductor.estimated_rate    # current EWMA-estimated rate
inductor.queue_depth       # current buffer depth
```

---

## Observability & Analysis API

### Data Class (Enriched)

The `Data` class now supports slicing, aggregation, and bucketing:

```python
from happysimulator import Data, Probe

# Existing: collect samples via Probe or manually
data = Data()
data.add_stat(value, time)

# Slicing
subset = data.between(30.0, 60.0)  # samples in [30s, 60s)

# Aggregations
data.mean()           # mean of all values
data.min() / data.max()
data.percentile(0.99) # p99
data.count()
data.sum()
data.std()            # population std dev

# Time-windowed bucketing
buckets = data.bucket(window_s=1.0)  # BucketedData
buckets.times()   # bucket start times
buckets.means()   # per-bucket means
buckets.p50s()    # per-bucket medians
buckets.p99s()    # per-bucket p99
buckets.counts()  # samples per bucket
buckets.to_dict() # {"time_s": [...], "mean": [...], ...}

# Chaining
avg_depth = queue_data.between(30, 60).mean()
p99_lat = latency_data.between(55, 65).percentile(0.99)

# Convenience
data.times()       # just timestamps
data.raw_values()  # just values
data.rate(window_s=1.0)  # count/sec per window
```

### Built-in Collectors

`LatencyTracker` and `ThroughputTracker` replace custom boilerplate sinks:

```python
from happysimulator import LatencyTracker, ThroughputTracker

# LatencyTracker: records end-to-end latency from event context['created_at']
sink = LatencyTracker("Sink")
# ... wire as downstream entity ...
sink.p50()          # 50th percentile latency
sink.p99()          # 99th percentile latency
sink.mean_latency()
sink.data           # underlying Data for custom analysis
sink.summary(window_s=1.0)  # BucketedData

# ThroughputTracker: counts events per time window
tp = ThroughputTracker("Throughput")
tp.throughput(window_s=1.0)  # BucketedData with counts
```

### SimulationSummary

`Simulation.run()` now returns a `SimulationSummary`:

```python
summary = sim.run()
print(summary)                    # human-readable
summary.duration_s                # simulation time
summary.total_events_processed
summary.events_per_second
summary.wall_clock_seconds        # real elapsed time
summary.entities                  # dict[str, EntitySummary]
summary.to_dict()                 # JSON-serializable
```

### Analysis Package

For reasoning about simulation behavior:

```python
from happysimulator.analysis import analyze, detect_phases

# Phase detection on any Data time series
phases = detect_phases(latency_data, window_s=5.0, threshold=2.0)
for p in phases:
    print(f"[{p.label}] {p.start_s}s-{p.end_s}s: mean={p.mean:.4f}")

# Full analysis pipeline
analysis = analyze(
    sim.summary,
    latency=tracker.data,
    queue_depth=probe_data,
)
analysis.phases      # per-metric phase detection
analysis.metrics     # MetricSummary with per-phase breakdown
analysis.anomalies   # detected anomalies
analysis.causal_chains  # correlated degradation patterns

# LLM-optimized output
prompt = analysis.to_prompt_context(max_tokens=2000)
json_data = analysis.to_dict()
```

---

## Simulation Control (Interactive Debugging)

The `sim.control` property provides interactive debugging for simulations. It is **lazy-created** on first access — when never used, the run loop incurs zero overhead (a single `if self._control is not None` guard per event).

### Basic Usage

```python
from happysimulator import Simulation, Source, Counter, Instant

counter = Counter("sink")
source = Source.constant(rate=10, target=counter, event_type="Ping")
sim = Simulation(
    end_time=Instant.from_seconds(60.0),
    sources=[source],
    entities=[counter],
)

# Pause before running
sim.control.pause()
summary = sim.run()  # Returns immediately with 0 events processed

# Step through events one at a time
summary = sim.control.step(5)  # Process 5 events, then pause

# Inspect state while paused
state = sim.control.get_state()
state.current_time        # Instant
state.events_processed    # int
state.heap_size           # int
state.is_paused           # True
state.last_event          # Event

# Peek at upcoming events (only while paused)
upcoming = sim.control.peek_next(3)
matches = sim.control.find_events(lambda e: e.event_type == "Ping")

# Resume to completion
final_summary = sim.control.resume()
```

### Breakpoints

Five breakpoint types pause the simulation when conditions are met:

```python
from happysimulator import (
    TimeBreakpoint,
    EventCountBreakpoint,
    ConditionBreakpoint,
    MetricBreakpoint,
    EventTypeBreakpoint,
)

# Pause at simulation time t=30s (one-shot by default)
sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(30.0)))

# Pause after 1000 events processed
sim.control.add_breakpoint(EventCountBreakpoint(count=1000))

# Pause when a custom condition is met (repeatable by default)
sim.control.add_breakpoint(ConditionBreakpoint(
    fn=lambda ctx: ctx.last_event.event_type == "Error",
    description="error occurred",
))

# Pause when an entity attribute crosses a threshold
sim.control.add_breakpoint(MetricBreakpoint(
    entity_name="Server",
    attribute="depth",
    operator="gt",     # gt, ge, lt, le, eq, ne
    threshold=100,
))

# Pause when a specific event type is processed
sim.control.add_breakpoint(EventTypeBreakpoint(event_type="Timeout"))
```

**One-shot vs repeatable**: `TimeBreakpoint` and `EventCountBreakpoint` default to `one_shot=True` (auto-removed after triggering). Others default to `one_shot=False` (persist and can trigger again after resume).

**Management**:
```python
bp_id = sim.control.add_breakpoint(...)
sim.control.remove_breakpoint(bp_id)
sim.control.list_breakpoints()  # [(id, breakpoint), ...]
sim.control.clear_breakpoints()
```

### Event Hooks

Hooks fire on every event without pausing — useful for logging, metrics, or custom tracing:

```python
# Called after each event is processed
hook_id = sim.control.on_event(lambda event: print(event.event_type))

# Called when simulation time advances
hook_id = sim.control.on_time_advance(lambda t: print(f"Time: {t}"))

# Remove a hook
sim.control.remove_hook(hook_id)
```

### Reset

Reset clears the heap, resets counters, and re-primes sources/probes. Entity internal state is **not** reset (entities own their state).

```python
sim.run()               # Run to completion
sim.control.reset()     # Clear heap, reset clock, re-prime sources
sim.run()               # Run again from scratch
```

### Architecture Notes

- `SimulationControl` is a separate class accessed via `sim.control` (composition pattern)
- `Simulation.run()` is re-entrant: calling it on a paused sim resumes from the pause point
- Both `resume()` and `step()` return `SimulationSummary` (partial if paused again, final if complete)
- Three control check points in the loop, all guarded by `if self._control is not None`:
  1. **Before pop**: checks pause requests and step counting
  2. **After time advance**: fires time hooks
  3. **After invoke+push**: fires event hooks and evaluates breakpoints

---

## Troubleshooting

### Common Issues

**"Event must have a 'target'"**
- Every `Event` must have a `target=` entity. Use `Event.once()` for function-based dispatch.

**Generator not progressing**
- Check that you're yielding `float` values (seconds), not `Instant`
- Ensure the simulation `end_time` is long enough

**Non-deterministic test results**
- Use `ConstantArrivalTimeProvider` instead of `PoissonArrivalTimeProvider`
- Set `random.seed(42)` for reproducible random values
- Use `seed=` parameter on distributions

**Queue builds up forever**
- Check that arrival rate < service rate (utilization < 100%)
- Verify `handle_queued_event()` returns/yields correctly

**Events not being processed**
- Verify entities are passed to `Simulation(entities=[...])`
- Check that `Source.start()` is being called (automatic if in `sources=[]`)

### Debugging Tips

1. **Enable debug logging** (library is silent by default):
   ```python
   import happysimulator
   happysimulator.enable_console_logging(level="DEBUG")
   ```

   Or via environment variables:
   ```powershell
   $env:HS_LOGGING='DEBUG'
   ```
   ```python
   import happysimulator
   happysimulator.configure_from_env()
   ```

2. **Add probes** to monitor queue depth, latency, etc.:
   ```python
   from happysimulator import Probe, Data

   data = Data()
   probe = Probe(target=server, metric="depth", data=data, interval=0.1)
   sim = Simulation(..., probes=[probe])
   ```

3. **File logging with rotation** (prevents disk space issues):
   ```python
   happysimulator.enable_file_logging("simulation.log", max_bytes=10_000_000)
   ```

4. **Use simulation control** to pause and inspect mid-run:
   ```python
   sim.control.add_breakpoint(MetricBreakpoint(
       entity_name="Server", attribute="depth", operator="gt", threshold=50,
   ))
   sim.run()  # Pauses when queue depth > 50
   state = sim.control.get_state()
   upcoming = sim.control.peek_next(10)
   ```

---

## .dev Documentation Conventions

The `.dev/` directory contains design documents for major features and architectural decisions.

### When to Create a .dev Document

- New major feature or component design
- Architectural decisions with trade-offs
- Multi-phase implementation plans
- Design alternatives that were considered

### Document Template

```markdown
# [Feature Name] Design Document

## Overview
Brief description of the feature.

## Motivation
Why is this needed? What problem does it solve?

## Requirements
### Functional Requirements
### Non-Functional Requirements

## Design
### New Concepts
### Integration Points
### File Organization

## Examples
Code examples showing usage.

## Testing Strategy
How to test this feature.

## Alternatives Considered
What other approaches were evaluated and why they were rejected.

## Implementation Plan
Phased approach to building the feature.

## References
External resources, papers, prior art.
```

### Naming Conventions

| Type | Format | Example |
|------|--------|---------|
| Feature designs | `feature-name-design.md` | `zipf-distribution-design.md` |
| Major plans | `FEATURENAME.md` | `COMPONENTLIB.md` |
| Decisions | `ADR-NNN-title.md` | `ADR-001-time-representation.md` |

### Document Lifecycle

1. **Draft** - Initial design, open for feedback
2. **Approved** - Design accepted, ready for implementation
3. **Implemented** - Code complete, document is reference
4. **Superseded** - Replaced by newer design (link to successor)

---

## Code Style

- Use modern Python 3.13+ features (type hints, dataclasses, `|` union syntax)
- Google-style docstrings with Args/Returns/Raises sections
- Comment the "why", not the "what"; trust clean code and good naming
- For generators, document yield semantics (what the yielded value represents)

---

## Skills and Plugins

The following skills are available for use in this project:

| Skill | Description |
|-------|-------------|
| `/commit` | Create a git commit with conventional message |
| `/commit-push-pr` | Commit, push, and open a pull request |
| `/code-review` | Review a pull request for issues and improvements |
| `/gen-test` | Generate a pytest test following project conventions |
| `/run-example` | Run a simulation example and analyze the output |
| `/line-count` | Count lines of code using cloc (source, tests, examples) |
| `/update-claudemd` | Review recent changes and update CLAUDE.md if needed |
| `/update-pypi` | Bump the package version for PyPI release |
| `/claude-automation-recommender` | Analyze codebase and recommend Claude Code automations |

### Subagents (Internal)

Claude Code uses specialized subagents internally:
- **simulation-reviewer**: Specialized code reviewer for discrete-event simulation code
- **code-simplifier**: Simplifies and refines code for clarity and maintainability
- **Explore**: Fast codebase exploration and search
- **Plan**: Implementation planning and architecture design
