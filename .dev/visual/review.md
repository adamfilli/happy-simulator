# Visual Simulation Editor — Design Review

> **Reviewed:** 2026-02-08

This document is a critique of the design documents in `.dev/visual/` (schema.md, Render.md, Editor.md, PLAN.md).

---

## Overall Assessment

The plan is well-structured with a clear 4-phase progression, good separation of concerns, and a solid schema design. The main issues are: (1) the proposed `SimulationController` duplicates existing infrastructure, (2) the connection-wiring mechanism is underspecified, and (3) there's a security concern with embedded Python. Tech stack choices are sound.

---

## 1. SimulationController Duplicates SimulationControl (Major)

This is the biggest issue in the plan. `Render.md` proposes a new `SimulationController` that reimplements step/pause/resume and directly accesses `Simulation` internals (`_event_heap`, `_clock`):

```python
# Proposed in Render.md — reaches into private state
event = self._sim._event_heap.pop()
self._sim._clock.update(event.time)
new_events = event.invoke()
```

We already have `SimulationControl` with `step()`, `resume()`, `pause()`, `get_state()`, `peek_next()`, breakpoints, and event hooks — all properly integrated with the simulation loop's control checkpoints. The proposed `SimulationController.step()` duplicates and bypasses all of this, meaning:

- Generator-based processes (`ProcessContinuation`) won't work correctly (the controller's `step()` doesn't handle them)
- Breakpoints won't fire
- Event hooks won't fire
- Trace recording won't happen
- The `end_time` guard won't be checked

**Recommendation**: Delete `SimulationController`. Instead, create a thin `SimulationSession` that wraps `Simulation` + `sim.control` and adds only what's missing: the entity ID registry and the JSON-friendly `get_state_snapshot()`. Let `sim.control.step()` do the actual stepping.

```python
class SimulationSession:
    """Wraps a loaded simulation for the visual editor API."""

    def __init__(self, sim: Simulation, entity_registry: dict[str, Entity]):
        self._sim = sim
        self._registry = entity_registry
        # Pause before first run so stepping works
        sim.control.pause()
        sim.run()  # primes the heap, pauses immediately

    def step(self, count: int = 1) -> SimulationSummary:
        return self._sim.control.step(count)

    def get_state_snapshot(self) -> dict:
        state = self._sim.control.get_state()
        return {
            "time": state.current_time.to_seconds(),
            "events_processed": state.events_processed,
            ...
        }
```

---

## 2. Connection Wiring Is the Hard Part (Major)

`Render.md` calls `self._wire_connection(conn, entities)` but never implements it. This is actually the most complex part of the loader, because existing entities don't have a uniform "set downstream" interface:

- `QueuedResource` subclasses set `self.downstream` in their constructor
- `RandomRouter` takes a list of targets
- `Source` sets its target via `EventProvider`
- Some entities have multiple output ports

The plan needs to specify:

**a)** How built-in entity types expose their downstream wiring. Options:
  - A `set_downstream(entity)` method on a `Connectable` protocol
  - Constructor parameter injection (build entities in dependency order)
  - Post-construction mutation via a known attribute name

**b)** How multi-output entities work (Router has N outputs, Fork duplicates to all outputs). The schema supports `"to": ["server1", "server2"]` but doesn't clarify whether this means "router chooses one" vs "fork sends to all."

**Recommendation**: Add a `Connectable` protocol or mixin to the built-in entity types:

```python
class Connectable(Protocol):
    def set_downstream(self, target: Entity | list[Entity]) -> None: ...
```

And specify the wiring order in the loader: create all entities first, then wire connections, then create sources (since sources need their target wired at construction).

---

## 3. Embedded Python Is a Security Risk (Major)

The schema allows:

```json
{
  "type": "Custom",
  "params": {
    "handler": "def handle_event(self, event):\n    import os; os.system('rm -rf /')"
  }
}
```

If the API server ever faces the network (or even untrusted local input), this is arbitrary code execution.

**Recommendation**: Drop embedded Python from the schema entirely. The "external Python class" approach (`"type": "mymodule.MyCustomServer"`) is sufficient and already scoped to importable modules. If in-browser customization is needed later, consider a restricted DSL or expression language rather than raw Python.

---

## 4. Source-Edge Asymmetry Will Cause Bugs (Medium)

In the schema, Source→Entity connections live in `source.target`, while Entity→Entity connections live in the `connections` array. The serialization code special-cases this:

```ts
if (sourceNode?.type === 'Source') continue; // Skip in connections
```

This means:
- Deleting an edge from a Source in the editor must update the Source node's data, not just remove the edge
- Reconnecting a Source to a different target is a node-data update, not an edge update
- The user sees one consistent interaction (draw edge) but the data model is split

**Recommendation**: Consider normalizing — put all connections in the `connections` array and have Sources reference their target from there. Or accept the asymmetry but document it explicitly in the serialization layer with clear bidirectional sync logic.

---

## 5. WebSocket Streaming Will Block the Event Loop (Medium)

```python
# Render.md — this blocks the async event loop
while not controller.is_complete:
    results = controller.run_events(100)  # synchronous, CPU-bound
    await websocket.send_json(...)
    await asyncio.sleep(0.016)
```

`run_events(100)` is synchronous and potentially long-running. In an `async def` handler, this blocks the entire FastAPI server. Other requests (state queries, second simulations) will stall.

**Recommendation**: Run the simulation in a background thread:

```python
@app.websocket("/simulations/{sim_id}/stream")
async def stream(websocket: WebSocket, sim_id: str):
    await websocket.accept()
    session = simulations[sim_id]

    while not session.is_complete:
        # Run sim work in thread pool
        snapshot = await asyncio.to_thread(session.run_batch, 100)
        await websocket.send_json(snapshot)
```

---

## 6. No Simulation Lifecycle Management (Medium)

Simulations are stored in a global `dict[str, SimulationController]` with no cleanup:

```python
simulations: dict[str, SimulationController] = {}
```

Every `POST /simulations` allocates memory that never gets freed. A fleet of 1000 servers at 100 req/s for 300s could use substantial memory.

**Recommendation**: Add a `DELETE /simulations/{id}` endpoint and consider a TTL-based cleanup (e.g., remove simulations not accessed for 10 minutes). A simple approach:

```python
from dataclasses import dataclass, field
from time import time

@dataclass
class SimEntry:
    session: SimulationSession
    last_accessed: float = field(default_factory=time)
```

---

## 7. Missing CORS Configuration (Minor)

Frontend on `localhost:5173` and backend on `localhost:8000` need CORS headers. This will be the first thing that breaks during development.

Add to Phase 1 checklist:

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], ...)
```

---

## 8. Schema Feedback (Minor)

**Good decisions:**
- `position` for round-tripping editor state
- Distribution as a discriminated union (`type` field)
- Fleet concept with visual flattening
- Probes as first-class schema objects

**Suggestions:**
- `drain_time` in the simulation config has no implementation specified in the loader. Clarify: does it translate to `end_time = source_duration + drain_time`? Or does it set `Source.stop_after` while keeping the sim running longer?
- The `arrival` and `profile` on Source overlap — if you specify `arrival: { type: "poisson", rate: 15 }` and also `profile: { type: "constant", rate: 50 }`, which rate wins? Document the precedence or make them mutually exclusive.
- Consider adding a `version` field to the schema for forward compatibility.

---

## 9. Tech Stack (Positive)

The frontend choices are solid:
- **React Flow** is the right tool for node-based editors
- **Zustand** is lightweight and pairs well with React Flow
- **Vite** is the standard for React projects now
- **shadcn/ui + Tailwind** is pragmatic for rapid UI development

One note: React Flow rebranded to **xyflow** and the npm package is now `@xyflow/react` (v12+). The plan references the old `reactflow` package. Check which version to target.

---

## 10. What's Missing from the Plan

**a) Error handling**: No API error responses defined. What happens when you POST invalid JSON? Step a completed simulation? Reference a nonexistent entity?

**b) Testing strategy**: The plan has an "end-to-end test" in Phase 1 but it's manual. Consider:
- Python unit tests for `SimulationLoader` (JSON in, verify entity graph)
- API integration tests with `httpx.AsyncClient`
- Frontend: at minimum, serialization round-trip tests

**c) How the editor handles simulation results**: Phase 3 says "live metrics overlay" but doesn't specify what happens when the simulation completes — is there a results panel? Do you keep the final state? Can you re-run with different parameters?

**d) `SimulationLoader` entity creation**: The loader needs to resolve how to pass `params` to entity constructors. Different entity types have different constructor signatures. A factory pattern or a convention where all visual-editor entities accept `**params` is needed.

---

## Summary of Recommendations

| Priority | Issue | Action |
|----------|-------|--------|
| **P0** | Controller duplicates Control | Use `sim.control` instead of new Controller |
| **P0** | Connection wiring unspecified | Design `Connectable` protocol, specify wire order |
| **P1** | Embedded Python security | Remove from schema, keep module-path imports only |
| **P1** | Async blocking | Run sim in thread pool via `asyncio.to_thread` |
| **P1** | Source-edge asymmetry | Normalize or add explicit sync logic |
| **P2** | No simulation cleanup | Add DELETE endpoint + TTL |
| **P2** | Missing CORS | Add to Phase 1 checklist |
| **P2** | Schema versioning | Add `version` field |
| **P3** | reactflow → @xyflow/react | Update package references |
| **P3** | arrival/profile overlap | Document precedence |

The foundation is strong — the phased approach is right, the schema design is clean, and leveraging the existing simulation control infrastructure will save significant work. The main risk is the Controller reimplementation; fixing that one issue eliminates a large class of bugs.
