# Copilot / AI Agent instructions for happy-simulator

Goal: Help an AI coding agent become productive quickly by describing the project's "shape", critical files and patterns, and common developer workflows.

## Big picture (what this project is)
- "happy-simulator" is a discrete-event simulation library and a set of example scenarios.
- Core simulation loop: `happysimulator.simulation.Simulation` uses an `EventHeap` to pop the next `Event` and invoke it until the end time.
- Two coding styles supported for events:
  - The Model-style: Events target an `Entity` (see `happysimulator.entities.entity.Entity`) and the entity handles the event.
  - The Callback-style: Events carry a `callback` that is invoked directly (see `happysimulator.events.event.Event`).
- Long-running processes are written as Python generators (yielding delays). The system wraps/continues these as `ProcessContinuation` events.

## Key files & where to look first 🔎
- `happysimulator/events/event.py` — Event contract: must have EITHER `target` OR `callback` (not both), returns list[Event] or a generator process. Read `invoke()`, `_start_process()`, and `ProcessContinuation`.
- `happysimulator/event_heap.py` — Simple heap using objects' ordering (`time`, then insert order) to schedule events.
- `happysimulator/simulation.py` — The main loop that initializes sources and runs events until `end_time`.
- `happysimulator/load/` — Source / providers. Important classes:
  - `Source` (bootstrap + self-perpetuating ticks)
  - `EventProvider` (creates payload events at a tick)
  - `ArrivalTimeProvider` implementations (e.g., `ConstantArrivalTimeProvider`, Poisson providers)
- `happysimulator/entities/entity.py` — Implement `handle_event(self, event)`; return either immediate events or a generator for processes.
- `happysimulator/utils/instant.py` — `Instant.from_seconds()` and `Instant.Infinity` singleton. Use these for times and arithmetic.
- `examples/` and `archive/` — runnable examples and older but useful code (measurement implementations, complex scenarios). Treat `archive/` as a reference for patterns that may not be fully integrated.

## Project-specific conventions & important gotchas ⚠️
- Use modern python, do things in a pythonic way, and prefer to use the new and improved ways to do things
versus legacy / backwards compatible ways.
- Logging: Configure logging level via HS_LOGGING env var. The package sets up a file `happysimulator.log` and a stream handler in `happysimulator.__init__`.
  - Example (PowerShell): `$env:HS_LOGGING='DEBUG'`
- Event invariants:
  - An Event must have EITHER a `target` OR a `callback` (checked in `Event.__post_init__`).
  - Events can return generators; the runtime will schedule continuations automatically.
- Time semantics: All scheduling uses `Instant` (nanoseconds under the hood). Use `Instant.from_seconds()` in tests/examples.
- The `archive/` directory contains many working examples (measurements, rate limiting, queued servers). Some example code imports from `archive.*` and shows how to setup measurements — use it for guidance when adding or restoring features.
- The README shows a slightly different public API (e.g., `Generator` or `Measurement` usage). Validate examples against current package contents. Some top-level re-exports are commented out in `happysimulator/__init__.py`.
- Python requirement: `setup.py` states `python_requires='>=3.13'` — check your environment; code also uses modern type hinting and dataclasses.

## Inline code comment guidelines 📝
Follow these rules when adding comments to maintain consistency and readability across the codebase.

### When to comment
- **Do comment:**
  - Complex algorithms or non-obvious logic (explain the "why", not the "what")
  - Workarounds, hacks, or edge-case handling with context
  - Important invariants or assumptions that aren't obvious from the code
  - Public API docstrings for modules, classes, and functions
  - TODOs with context: `# TODO(username): description of what and why`
- **Don't comment:**
  - Obvious code that is self-explanatory (e.g., `# increment counter` before `counter += 1`)
  - Every line or block — trust clean code and good naming to communicate intent
  - Commented-out code — delete it; version control preserves history

### Docstring conventions (PEP 257 + Google style)
Use triple double-quotes for all docstrings. Prefer Google-style formatting for parameters and returns.

```python
def schedule_event(self, event: Event, delay: Instant) -> Event:
    """Schedule an event to fire after a delay.

    Adds the event to the heap with its time adjusted by the given delay.
    If the event already has a time set, the delay is added to it.

    Args:
        event: The event to schedule.
        delay: Time offset from now (use Instant.from_seconds()).

    Returns:
        The scheduled event (same object, mutated with new time).

    Raises:
        ValueError: If delay is negative.
    """
```

- **One-liner docstrings** are acceptable for simple, obvious functions:
  ```python
  def is_empty(self) -> bool:
      """Return True if the heap has no pending events."""
  ```

### Inline comment style
- Use `#` with a single space: `# This is a comment`
- Place comments on their own line above the code they describe, not at end-of-line (except for very short clarifications)
- Keep comments up to date — stale comments are worse than no comments
- Use sentence case; periods are optional for single-sentence comments

```python
# Calculate the next arrival using the inverse-transform method.
# This ensures proper statistical distribution of inter-arrival times.
next_arrival = self._arrival_provider.next_arrival(current_time)

x = value * SCALE_FACTOR  # normalize to [0, 1]
```

### Section and block comments
For logical sections within long functions or modules, use a blank line and a comment:

```python
def run(self) -> SimulationResult:
    # --- Initialization ---
    self._initialize_sources()
    
    # --- Main simulation loop ---
    while self._event_heap:
        event = self._event_heap.pop()
        ...
```

### Type hints vs. comments
- Prefer type hints over comments for documenting parameter/return types — this project uses modern Python 3.13+ typing
- Use `# type: ignore` sparingly and only with a reason: `# type: ignore[arg-type]  # third-party lib has incorrect stubs`

### Generator and process comments
Since this project uses generators for multi-step processes, clearly document yield semantics:

```python
def handle_request(self, request: Request) -> Generator[Instant, None, None]:
    """Process a request with simulated latency.
    
    Yields:
        Instant: Delay to wait before continuing (simulates processing time).
    """
    # Simulate network latency before processing.
    yield Instant.from_seconds(0.05)
    
    # Process the request (CPU-bound work simulation).
    yield self._compute_latency()
```

### What NOT to do
```python
# Bad: Redundant comment
i = 0  # set i to zero

# Bad: Comment instead of better naming
x = t * 3.6  # convert m/s to km/h
# Better: 
speed_kmh = speed_ms * MS_TO_KMH

# Bad: Commented-out code left in
# old_value = calculate_old_way()
# if old_value > threshold:
#     do_something()

# Bad: Stale/incorrect comment
# Returns the first event (this actually returns all events now)
def get_events(self) -> list[Event]:
    ...
```

## How to run & debug locally (developer flow) ✅
1. Create a venv and install (Windows PS example):
   - `python -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
   - `pip install -e .`
2. Run tests: `pytest -q` (or a single test file: `pytest tests/test_simulation_basic_counter.py -q`).
3. Run examples: `python examples/basic_client_server.py` (check `examples/` for more scenarios).
4. Debugging:
   - Increase logging: `$env:HS_LOGGING='DEBUG'` and re-run (inspect `happysimulator.log` in repo root for trace messages).
   - For timing/scheduling bugs, examine `Event.invoke()` and `ProcessContinuation.invoke()` to trace generator yields and scheduling of the next continuation.

## How to implement changes safely (developer hints) 🔧
- When adding new event types or entity behavior: prefer returning generator-based processes for multi-step flows (yields represent delays) rather than cookie-cutter threading or sleep.
- When creating a new Source/Provider: subclass `EventProvider` / `ArrivalTimeProvider` and validate with a short unit test that ensures the Source produces expected number of ticks (`source._nmb_generated`).
- When adding measurements or scheduled callbacks, review `archive/measurement_event.py` and `archive/measurement.py` for patterns (these show how measurement events are scheduled and handled).

## Recommended tests & quick checks
- Unit tests live under `tests/` and use `pytest`.
- Small, deterministic tests often use `ConstantArrivalTimeProvider` and `Instant.from_seconds()` to make the timing assertions clear.
- Validate event ordering by creating simultaneous events and ensuring FIFO among same-time events (Event._sort_index behavior).

## Example snippets to reference
- Event with callback:
  - `Event(time=Instant.from_seconds(1), event_type="Ping", callback=lambda e: ...)`
- Source bootstrap:
  - `Source.start()` returns a `SourceEvent` at the computed first arrival time (see `Source.start`).

## When in doubt (where to ask/look)
- Check `archive/` for real-world examples and measurement handlers.
- Read `happysimulator/events/event.py` first to understand the event lifecycle; most bugs relate to incorrect return values (None vs Event vs list vs Generator).

---
If you'd like, I can: (a) open a PR adding a few sample unit tests around generator-based processes, or (b) add a short `CONTRIBUTING.md` checklist adapted from this file. Which would you prefer? (I can iterate on wording or add more examples.)
