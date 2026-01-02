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
