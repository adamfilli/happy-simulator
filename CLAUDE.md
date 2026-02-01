# CLAUDE.md

> **Last Updated:** 2026-01-31

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Quick Reference

| Aspect | Summary |
|--------|---------|
| **What** | Discrete-event simulation library for Python 3.13+ |
| **Core Loop** | `EventHeap` pop → `Entity.handle_event()` → schedule returned `Event`s |
| **Key Invariant** | Events have EITHER `target` (Entity) OR `callback` (function) - never both, never neither |
| **Time** | Use `Instant.from_seconds(n)`, not raw floats |
| **Generators** | Yield delays (float seconds); return events on completion |
| **Testing** | Use `ConstantArrivalTimeProvider` for deterministic timing |

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

### Phase 3: Patterns
7. **`happysimulator/load/source.py`** - Self-perpetuating event generation
8. **`happysimulator/components/queue.py`** - Queue/Driver pattern
9. **`happysimulator/components/queued_resource.py`** - Resource abstraction

### Phase 4: Examples
10. **`examples/m_m_1_queue.py`** - Full M/M/1 queue workflow with visualization
11. Corresponding integration tests in `tests/integration/`

### Phase 5: Design Context
12. **`.dev/COMPONENTLIB.md`** - Component library design philosophy
13. **`.dev/zipf-distribution-design.md`** - Feature design document template

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

**Key Invariant**: An Event must have EITHER a `target` (Entity) OR a `callback` (function) - never both, never neither.

```python
from happysimulator import Event, Instant

# Model-style: target an Entity
request_event = Event(
    time=Instant.from_seconds(1.0),
    event_type="Request",
    target=my_server,
    context={"customer_id": 42},
)

# Callback-style: invoke a function
ping_event = Event(
    time=Instant.from_seconds(1.0),
    event_type="Ping",
    callback=lambda e: print("pong"),
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

**Under the hood**: The runtime wraps generators as `ProcessContinuation` events that reschedule after each yield.

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
│       ▲                  ┌─────────────┐                │
│       │                  │ Has target? │                │
│       │                  └──────┬──────┘                │
│       │                    yes/   \no                    │
│       │              ┌────────┐  ┌────────┐             │
│       │              │ Entity │  │Callback│             │
│       │              │handle_ │  │invoke  │             │
│       │              │ event()│  └───┬────┘             │
│       │              └───┬────┘      │                   │
│       │                  │           │                   │
│       │                  └─────┬─────┘                   │
│       │                        ▼                         │
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

### Two Event Handling Styles

1. **Model-style**: Event has a `target` Entity; the Entity's `handle_event()` method processes it
2. **Callback-style**: Event has a `callback` function that is invoked directly

### Load Generation (`happysimulator/load/`)

- **`Source`**: Self-perpetuating entity that generates events at intervals
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
│   ├── entity.py           # Entity base class
│   ├── simulation.py       # Main simulation loop
│   ├── clock.py            # Clock abstraction
│   └── protocols.py        # Simulatable protocol
│
├── load/                    # Load generation
│   ├── source.py           # Self-perpetuating source
│   ├── profile.py          # Rate profiles (ConstantRateProfile, etc.)
│   ├── event_provider.py   # EventProvider base class
│   └── providers/          # EventProvider implementations
│       └── distributed_field.py  # Zipf/distribution-based
│
├── components/              # Reusable simulation components
│   ├── queue.py            # Queue implementations
│   ├── queued_resource.py  # Queue + processing
│   ├── random_router.py    # Load balancing
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
└── zipf-distribution-design.md  # Feature design template

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

### Creating an Event with Callback

```python
Event(
    time=Instant.from_seconds(1),
    event_type="Ping",
    callback=lambda e: print("pong"),
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

### Starting a Source

```python
source = Source(
    name="Traffic",
    event_provider=my_provider,
    arrival_time_provider=ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=100),
        start_time=Instant.Epoch,
    ),
)
# Source.start() is called automatically by Simulation
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

---

## Troubleshooting

### Common Issues

**"Event must have either target or callback"**
- Ensure every `Event` has exactly one of `target=` or `callback=` set

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
| `/update-claudemd` | Review recent changes and update CLAUDE.md if needed |
| `/claude-automation-recommender` | Analyze codebase and recommend Claude Code automations |

### Subagents (Internal)

Claude Code uses specialized subagents internally:
- **simulation-reviewer**: Specialized code reviewer for discrete-event simulation code
- **code-simplifier**: Simplifies and refines code for clarity and maintainability
- **Explore**: Fast codebase exploration and search
- **Plan**: Implementation planning and architecture design
