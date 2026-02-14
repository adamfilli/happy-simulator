# Rendering and Running Simulations

This document describes the architecture for loading JSON simulation definitions, running them interactively, and querying live state.

## Architecture Overview

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   JSON Schema   │ ---> │ SimulationLoader│ ---> │   Controller    │
│   (from editor) │      │  (builds objs)  │      │ (play/pause/step)│
└─────────────────┘      └─────────────────┘      └─────────────────┘
                                                          │
                                                          ▼
                                                  ┌─────────────────┐
                                                  │   Simulation    │
                                                  │   (existing)    │
                                                  └─────────────────┘
```

---

## Key Components

### 1. SimulationLoader

Parses JSON, instantiates entities/sources/probes, wires up connections:

```python
class SimulationLoader:
    """Builds a runnable simulation from a JSON definition."""

    # Registry of built-in entity types
    ENTITY_TYPES = {
        "QueuedServer": QueuedServer,
        "RandomRouter": RandomRouter,
        "Sink": Sink,
        # ...
    }

    def load(self, definition: dict) -> "SimulationController":
        """Parse JSON and build simulation objects."""

        entities = {}  # id -> entity instance (or list for fleets)

        # 1. Create entities (expand fleets)
        for entity_def in definition.get("entities", []):
            entity = self._create_entity(entity_def)
            entities[entity_def["id"]] = entity

        # 2. Wire up connections (set downstream references)
        for conn in definition.get("connections", []):
            self._wire_connection(conn, entities)

        # 3. Create sources
        sources = [self._create_source(s, entities) for s in definition.get("sources", [])]

        # 4. Create probes
        probes = [self._create_probe(p, entities) for p in definition.get("probes", [])]

        # 5. Build simulation
        sim_config = definition.get("simulation", {})
        simulation = Simulation(
            end_time=Instant.from_seconds(sim_config.get("end_time", float("inf"))),
            sources=sources,
            entities=list(self._flatten_entities(entities)),
            probes=probes,
        )

        return SimulationController(simulation, entities)
```

### 2. SimulationController

Wraps simulation with interactive controls:

```python
@dataclass
class StepResult:
    """What happened during a simulation step."""
    time: float
    event_type: str
    entity_id: str | None
    events_produced: int
    is_complete: bool


class SimulationController:
    """Interactive wrapper around Simulation."""

    def __init__(self, simulation: Simulation, entities: dict[str, Entity]):
        self._sim = simulation
        self._entities = entities  # For querying by ID
        self._events_processed = 0
        self._is_started = False
        self._is_complete = False

    # --- Execution Control ---

    def step(self) -> StepResult:
        """Execute exactly one event. Returns what happened."""
        if self._is_complete:
            raise SimulationComplete()

        if not self._sim._event_heap.has_events():
            self._is_complete = True
            return StepResult(..., is_complete=True)

        # Pop-invoke-push for one event
        event = self._sim._event_heap.pop()
        self._sim._clock.update(event.time)
        new_events = event.invoke()
        if new_events:
            self._sim._event_heap.push(new_events)

        self._events_processed += 1
        return StepResult(
            time=event.time.to_seconds(),
            event_type=event.event_type,
            entity_id=self._get_entity_id(event),
            events_produced=len(new_events) if new_events else 0,
            is_complete=False,
        )

    def run_until(self, target_time: float) -> list[StepResult]:
        """Run until simulation time reaches target. Returns all step results."""
        results = []
        target = Instant.from_seconds(target_time)
        while not self._is_complete:
            if self._sim._clock.now() >= target:
                break
            # Peek next event time
            if self._sim._event_heap.peek().time > target:
                break
            results.append(self.step())
        return results

    def run_events(self, count: int) -> list[StepResult]:
        """Run exactly N events. Returns all step results."""
        results = []
        for _ in range(count):
            if self._is_complete:
                break
            results.append(self.step())
        return results

    # --- State Queries ---

    @property
    def current_time(self) -> float:
        return self._sim._clock.now().to_seconds()

    @property
    def events_processed(self) -> int:
        return self._events_processed

    @property
    def pending_events(self) -> int:
        return self._sim._event_heap.size()

    @property
    def is_complete(self) -> bool:
        return self._is_complete

    def get_entity(self, entity_id: str) -> Entity:
        """Get entity by ID (supports fleet indexing: 'servers[42]')."""
        return self._entities[entity_id]

    def query_metric(self, entity_id: str, metric: str) -> Any:
        """Query a metric from an entity (e.g., 'depth', 'stats_processed')."""
        entity = self.get_entity(entity_id)
        return getattr(entity, metric)

    def query_fleet_metric(
        self,
        fleet_id: str,
        metric: str,
        aggregation: str = "sum"
    ) -> float:
        """Query aggregated metric from a fleet."""
        instances = self._get_fleet_instances(fleet_id)
        values = [getattr(inst, metric) for inst in instances]

        match aggregation:
            case "sum": return sum(values)
            case "avg": return sum(values) / len(values)
            case "min": return min(values)
            case "max": return max(values)
            case "each": return values

    def get_state_snapshot(self) -> dict:
        """Full state snapshot for UI sync."""
        return {
            "time": self.current_time,
            "events_processed": self.events_processed,
            "pending_events": self.pending_events,
            "is_complete": self.is_complete,
            "entities": {
                eid: self._entity_state(e)
                for eid, e in self._entities.items()
            },
        }
```

### 3. Web API (FastAPI)

```python
from fastapi import FastAPI, WebSocket

app = FastAPI()
simulations: dict[str, SimulationController] = {}

@app.post("/simulations")
async def create_simulation(definition: dict) -> dict:
    """Load a new simulation from JSON."""
    loader = SimulationLoader()
    controller = loader.load(definition)
    sim_id = str(uuid4())
    simulations[sim_id] = controller
    return {"id": sim_id, "state": controller.get_state_snapshot()}

@app.post("/simulations/{sim_id}/step")
async def step(sim_id: str) -> dict:
    """Execute one event."""
    controller = simulations[sim_id]
    result = controller.step()
    return {"result": result, "state": controller.get_state_snapshot()}

@app.post("/simulations/{sim_id}/run")
async def run(sim_id: str, until_time: float = None, events: int = None) -> dict:
    """Run simulation until time or event count."""
    controller = simulations[sim_id]
    if until_time is not None:
        results = controller.run_until(until_time)
    elif events is not None:
        results = controller.run_events(events)
    return {"results": results, "state": controller.get_state_snapshot()}

@app.get("/simulations/{sim_id}/state")
async def get_state(sim_id: str) -> dict:
    """Get current simulation state."""
    return simulations[sim_id].get_state_snapshot()

@app.get("/simulations/{sim_id}/entity/{entity_id}")
async def get_entity(sim_id: str, entity_id: str) -> dict:
    """Get entity state."""
    controller = simulations[sim_id]
    return controller._entity_state(controller.get_entity(entity_id))

# WebSocket for streaming updates during continuous run
@app.websocket("/simulations/{sim_id}/stream")
async def stream(websocket: WebSocket, sim_id: str):
    """Stream state updates during continuous execution."""
    await websocket.accept()
    controller = simulations[sim_id]

    while not controller.is_complete:
        # Run a batch of events
        results = controller.run_events(100)
        await websocket.send_json({
            "results": results,
            "state": controller.get_state_snapshot(),
        })
        # Small delay to allow UI to render
        await asyncio.sleep(0.016)  # ~60fps
```

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Step granularity | Per-event | Maximum control, UI can batch as needed |
| State queries | Synchronous | Simulation is single-threaded, no locks needed |
| Fleet expansion | At load time | Simpler runtime, entities exist in registry |
| Streaming | WebSocket | Real-time updates during "play" mode |
| Batch size | Configurable | Balance between responsiveness and throughput |

---

## Interactive Features This Enables

1. **Step-by-step debugging**: Click through events one at a time
2. **Run until time T**: "Jump to t=10s"
3. **Play/pause**: Continuous execution with streaming updates
4. **Live metrics**: Query queue depths, throughput, latencies during run
5. **Breakpoints**: (future) Pause when condition is met
6. **Speed control**: Adjust batch size / delay between updates

---

## API Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/simulations` | POST | Load a new simulation from JSON |
| `/simulations/{id}/step` | POST | Execute one event |
| `/simulations/{id}/run` | POST | Run until time or event count |
| `/simulations/{id}/state` | GET | Get current simulation state |
| `/simulations/{id}/entity/{eid}` | GET | Get specific entity state |
| `/simulations/{id}/stream` | WS | Stream updates during play |

---

## State Snapshot Format

```json
{
  "time": 12.345,
  "events_processed": 1542,
  "pending_events": 23,
  "is_complete": false,
  "entities": {
    "server": {
      "id": "server",
      "type": "QueuedServer",
      "depth": 5,
      "stats_processed": 423
    },
    "servers[0]": {
      "id": "servers[0]",
      "type": "QueuedServer",
      "depth": 2,
      "stats_processed": 89
    }
  }
}
```

---

## Future Extensions

- **Breakpoints**: Pause when `entity.metric > threshold`
- **Time travel**: Snapshot/restore for rewinding simulation
- **Multiple simulations**: Run A/B comparisons in parallel
- **Parameter sweeps**: Run same topology with different params
- **Recording/playback**: Record event stream for replay
