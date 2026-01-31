# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

happy-simulator is a discrete-event simulation library for Python 3.13+, similar to Matlab SimEvent. It models systems using an event-driven architecture where a central `EventHeap` schedules and executes `Event` objects until the simulation ends.

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
python examples/basic_client_server.py

# Debugging (set logging level)
$env:HS_LOGGING='DEBUG'   # PowerShell
# Logs written to happysimulator.log in repo root
```

## Architecture

### Core Simulation Loop (`happysimulator/simulation.py`)
The `Simulation` class initializes sources and runs a pop-invoke-push loop on the `EventHeap` until `end_time` is reached or the heap is exhausted.

### Event System (`happysimulator/events/event.py`)
Events are the fundamental unit of work. Key invariants:
- An Event must have EITHER a `target` (Entity) OR a `callback` (function) - never both, never neither
- Events can return: `None`, `Event`, `list[Event]`, or a **Generator** (for multi-step processes)
- Generators yield delays (`float` seconds) or `(delay, side_effect_events)` tuples; the runtime wraps them as `ProcessContinuation` events

### Two Event Handling Styles
1. **Model-style**: Event has a `target` Entity; the Entity's `handle_event()` method processes it
2. **Callback-style**: Event has a `callback` function that is invoked directly

### Entity Pattern (`happysimulator/entities/entity.py`)
Entities implement `handle_event(event)` and can return immediate events or generators for sequential processes that yield delays.

### Time Semantics (`happysimulator/utils/instant.py`)
All scheduling uses `Instant` (nanoseconds internally). Use `Instant.from_seconds()` for creation. Special values: `Instant.Epoch` (time zero), `Instant.Infinity` (for auto-termination).

### Load Generation (`happysimulator/load/`)
- `Source`: Self-perpetuating entity that generates events at intervals
- `EventProvider`: Creates payload events at each tick
- `ArrivalTimeProvider` implementations: `ConstantArrivalTimeProvider`, Poisson-based providers

## Key Directories

- `happysimulator/` - Core library
- `examples/` - Runnable example scenarios
- `archive/` - Reference implementations for patterns like measurements, queued servers, rate limiters (not fully integrated but useful for guidance)
- `tests/` - pytest tests; unit tests use `ConstantArrivalTimeProvider` and `Instant.from_seconds()` for deterministic timing

## Code Style

- Use modern Python 3.13+ features (type hints, dataclasses, `|` union syntax)
- Google-style docstrings with Args/Returns/Raises sections
- Comment the "why", not the "what"; trust clean code and good naming
- For generators, document yield semantics (what the yielded value represents)

## Common Patterns

### Creating an Event with Callback
```python
Event(time=Instant.from_seconds(1), event_type="Ping", callback=lambda e: print("pong"))
```

### Generator-based Process (multi-step with delays)
```python
def handle_request(self, request: Request) -> Generator[float, None, None]:
    yield 0.05  # wait 50ms (simulated network latency)
    yield self._compute_latency()  # wait for processing
    # optionally return events at the end
```

### Starting a Source
```python
source = Source(name="Traffic", event_provider=my_provider, arrival_time_provider=ConstantArrivalTimeProvider(...))
# Source.start() returns initial SourceEvent to prime the simulation
```


## Skills and plugins

/code-review:code-review
/claude-automation-recommender
code simplifier?? Subagents - I invoke them internally when working on tasks
/commit commit commands