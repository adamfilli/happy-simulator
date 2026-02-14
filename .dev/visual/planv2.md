# Simulation Debugger — Design Plan

## Context

The original visual editor plan (`/.dev/visual/`) designed a drag-and-drop editor for *creating* simulations. The new direction: simulations stay as Python code, but users get a browser-based **debugger/inspector** to step through and observe their simulation. This is simpler, more useful, and leverages the existing `SimulationControl` infrastructure.

**User code:**
```python
from happysimulator.visual import serve

sim = Simulation(sources=[source], entities=[server, sink])
serve(sim)  # opens browser, user steps through interactively
```

---

## Architecture Overview

```
Python Process                              Browser
┌────────────────────────────────┐     ┌──────────────────────┐
│  serve(sim)                    │     │  Single-page app     │
│    │                           │     │                      │
│    ├── SimulationBridge        │     │  ┌────────────────┐  │
│    │     wraps sim.control     │◄───►│  │ Graph View     │  │
│    │     records events        │ REST│  │ (React Flow)   │  │
│    │     serializes state      │  +  │  ├────────────────┤  │
│    │     discovers topology    │ WS  │  │ Inspector      │  │
│    │                           │     │  │ Event Log      │  │
│    ├── FastAPI server          │     │  │ Controls       │  │
│    │     /api/topology         │     │  └────────────────┘  │
│    │     /api/state            │     │                      │
│    │     /api/step             │     └──────────────────────┘
│    │     /api/ws (WebSocket)   │
│    │     serves static/        │
└────────────────────────────────┘
```

---

## Python Package Structure

```
happysimulator/visual/
  __init__.py          # exports serve()
  server.py            # FastAPI app + endpoints
  bridge.py            # SimulationBridge: wraps sim + control for API
  topology.py          # Entity graph discovery via attribute introspection
  serializers.py       # Entity state → JSON-safe dicts

happysimulator/visual/static/
  index.html           # Pre-built SPA (committed, no npm at runtime)
  app.js / app.css     # Bundled frontend

visual-frontend/       # Dev-only (not in Python package)
  package.json
  vite.config.ts
  src/
    App.tsx
    components/
      GraphView.tsx      # React Flow entity graph
      ControlBar.tsx     # Step / Play / Pause / Reset
      InspectorPanel.tsx # Entity state + event log
      EventLog.tsx
      TimelineBar.tsx
    hooks/
      useWebSocket.ts
      useSimState.ts     # Zustand store
    types.ts
```

---

## Key Components

### 1. `serve(sim)` — Entry Point

**File:** `happysimulator/visual/__init__.py`

```python
def serve(sim: Simulation, *, host="127.0.0.1", port=8765, open_browser=True) -> None:
```

Steps:
1. Assert sim hasn't been run yet (`sim._is_running is False`)
2. Create `SimulationBridge(sim)` — installs event hooks, discovers topology
3. Call `sim.control.pause()` then `sim.run()` — primes heap, pauses at t=0
4. Build FastAPI app, mount static files, register endpoints
5. Open browser to `http://{host}:{port}`
6. Start uvicorn (blocks)

### 2. SimulationBridge — Core Mediator

**File:** `happysimulator/visual/bridge.py`

Wraps `Simulation` + `sim.control` for the API layer. All simulation interaction goes through here.

**Responsibilities:**
- **Step/reset**: Delegates to `sim.control.step(n)` / `sim.control.reset()`
- **State snapshots**: Builds JSON-safe dict from `sim.control.get_state()` + per-entity state
- **Event recording**: `on_event` hook captures each processed event into a bounded deque (max 5000)
- **Edge discovery**: Tracks which entities produce events targeting other entities
- **Upcoming events**: Uses `sim.control.peek_next(n)` when paused

**Key method signatures:**
```python
def get_topology(self) -> dict          # nodes + edges
def get_state(self) -> dict             # sim state + entity states + upcoming events
def step(self, count=1) -> dict         # step + return new state + processed events
def reset(self) -> dict                 # reset sim to t=0
def get_event_log(self, last_n=100) -> list[dict]
```

### 3. Topology Discovery

**File:** `happysimulator/visual/topology.py`

**Two-phase hybrid approach:**

**Phase 1 — Static introspection (at serve-time):**
Scan entities and sources for known attribute patterns:

| Pattern | Used by |
|---------|---------|
| `.downstream` (property) | RateLimitedEntity, Inductor, DistributedRateLimiter, NullRateLimiter |
| `._downstream` | StreamProcessor, WorkStealingPool |
| `.targets` (list) | RandomRouter |
| `._target` (property) | Client, ConnectionPool, Timeout, CircuitBreaker, Bulkhead, Hedge |
| `._event_provider._target` | Source (via _SimpleEventProvider) |

```python
_DOWNSTREAM_ATTRS = ["downstream", "targets", "target", "_downstream", "_target"]

def discover(sim: Simulation) -> Topology:
    nodes = []
    edges = []
    # Scan sim._entities + sim._sources
    for entity in sim._entities + sim._sources:
        nodes.append(Node(id=entity.name, type=type(entity).__name__, category=classify(entity)))
        for downstream in _find_downstream(entity):
            edges.append(Edge(source=entity.name, target=downstream.name))
    # Also introspect source._event_provider._target
    for source in sim._sources:
        target = getattr(getattr(source, "_event_provider", None), "_target", None)
        if target:
            edges.append(Edge(source=source.name, target=target.name))
    return Topology(nodes, edges)
```

**Phase 2 — Dynamic observation (during stepping):**
The `on_event` hook tracks `last_handled_entity → event.target.name`. New edges discovered at runtime are sent to the UI via WebSocket `topology_update` messages.

### 4. Entity State Serialization

**File:** `happysimulator/visual/serializers.py`

**Type-aware registry with safe fallback:**

| Entity Type | Serialized Attributes |
|------------|----------------------|
| `QueuedResource` | `depth`, `stats_accepted`, `stats_dropped` |
| `Sink` | `events_received`, `average_latency()`, `latency_stats()` |
| `LatencyTracker` | `count`, `mean_latency()`, `p50()`, `p99()` |
| `ThroughputTracker` | `count` |
| `Counter` | `total`, `by_type` |
| `RateLimitedEntity` / `Inductor` | `queue_depth`, `estimated_rate`, stats |
| `Resource` | `available`, `utilization`, `waiters` |
| `Source` | event count from `_event_provider._generated` |

For unknown/custom entities: inspect `__dict__`, keep only primitive-valued public attributes.

### 5. Event Recording

The bridge's `on_event` hook captures each processed event:

```python
@dataclass
class RecordedEvent:
    time_s: float
    event_type: str
    target_name: str
    source_name: str | None   # inferred from last handler
    event_id: str
    is_internal: bool          # SourceEvent, QUEUE_POLL, etc.
```

Internal event types (`SourceEvent`, `QUEUE_POLL`, `QUEUE_NOTIFY`, `QUEUE_DELIVER`, `probe_event`, `inductor_poll::*`, `rate_limit_poll::*`) are flagged so the UI can filter/dim them.

---

## API Design

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/topology` | GET | Entity graph (nodes + edges) |
| `/api/state` | GET | Full sim state + entity states + upcoming events |
| `/api/step` | POST | Step N events, return new state + processed events |
| `/api/reset` | POST | Reset sim to t=0 |
| `/api/events` | GET | Event log (last N) |
| `/api/breakpoint` | POST | Add a breakpoint |
| `/api/breakpoint/{id}` | DELETE | Remove a breakpoint |

### WebSocket `/api/ws`

For play mode. Client sends commands, server pushes state updates:

- Client: `{"action": "play", "speed": 10}` — step 10 events per batch, push state, repeat
- Client: `{"action": "pause"}` — stop play loop
- Client: `{"action": "step", "count": 5}` — one-shot step
- Server: `{"type": "state_update", "state": {...}, "new_events": [...]}`
- Server: `{"type": "topology_update", "edges": [...]}` — new edges discovered
- Server: `{"type": "breakpoint_hit", ...}`

Play loop uses `asyncio.to_thread(bridge.step, speed)` to avoid blocking the event loop. A `threading.Event` signals pause requests.

---

## Frontend Design

### Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Graph | **React Flow** (`@xyflow/react`) + `elkjs` | Full React component nodes for maximum styling control, elkjs for auto-layout |
| UI | **React** + TypeScript | Familiar, good component model |
| State | **Zustand** | Lightweight, pairs well with React Flow |
| Build | **Vite** | Fast, outputs to `happysimulator/visual/static/` |
| Styling | **Tailwind CSS** | Utility-first CSS for clean, minimalist design |

**Why React Flow + elkjs?** React Flow renders nodes as full React components, giving complete control over aesthetics via CSS/Tailwind. This enables the minimalist, polished design we want without being constrained by a graph library's styling DSL. elkjs (Eclipse Layout Kernel) computes node positions from graph structure, solving the auto-layout problem since simulations are defined in code, not placed visually. If we later want interactive features (drag to rearrange, inline editing), React Flow already supports them.

### Layout

```
┌─ ControlBar ──────────────────────────────────────────────────┐
│ t=1.234s  │ Events: 142  │ ⏮ Step │ ▶ Play │ ⏸ Pause │ ↺ Reset │
├───────────────────────────────┬────────────────────────────────┤
│                               │  [Entity State] [Event Log]   │
│     Graph View (~65%)         │                               │
│                               │  Server (QueuedResource)      │
│  ┌──────┐    ┌──────┐        │  ──────────────────────        │
│  │Source │───▶│Server│───▶    │  depth: 3                     │
│  └──────┘    └──────┘    ┌─┐ │  accepted: 142                │
│                          │S│ │  dropped: 0                   │
│                          │i│ │                               │
│                          │n│ │  [Event Log tab]              │
│                          │k│ │  t=1.234 Request → Server     │
│                          └─┘ │  t=1.200 Response → Sink      │
│                               │  t=1.100 Request → Server     │
├───────────────────────────────┴────────────────────────────────┤
│ Upcoming: t=1.300 SourceEvent→Source │ t=1.345 Request→Server │
└───────────────────────────────────────────────────────────────-┘
```

### Graph Node Styling

| Category | Color | Examples |
|----------|-------|---------|
| `source` | Green | Source |
| `queued_resource` | Blue | QueuedResource subclasses |
| `sink` | Red/Gray | Sink, LatencyTracker, Counter |
| `rate_limiter` | Orange | RateLimitedEntity, Inductor |
| `router` | Purple | RandomRouter |
| `resource` | Brown | Resource |
| `other` | Gray | Custom entities |

Nodes display: name, type, and 1-2 live metrics (e.g., `depth: 3`). Active node (last to process an event) gets a brief highlight border.

---

## Dependencies

**Python** (optional extra `[visual]` in pyproject.toml):
```toml
[project.optional-dependencies]
visual = ["fastapi>=0.100", "uvicorn[standard]>=0.20"]
```

**Frontend** (dev-only, not needed at runtime):
- `@xyflow/react` (React Flow v12+)
- `elkjs` (auto-layout engine)
- `react`, `react-dom`, `zustand`
- `tailwindcss`
- `vite`

Built JS/CSS is committed to `happysimulator/visual/static/` — `pip install` includes everything, no npm needed at runtime.

---

## Implementation Phases

### Phase 1: Minimal Stepping Debugger (MVP)

**Goal:** `serve(sim)` opens browser showing entity graph. Step button advances simulation. Entity state updates live.

Build:
1. `visual/__init__.py` — `serve()` function
2. `visual/bridge.py` — SimulationBridge (step, state, event recording)
3. `visual/topology.py` — static introspection
4. `visual/serializers.py` — entity state → JSON
5. `visual/server.py` — FastAPI: GET /api/topology, GET /api/state, POST /api/step, POST /api/reset, static files
6. Frontend: React Flow graph (elkjs layout) + step button + entity inspector panel + event log

**Test against:** M/M/1 queue example.

**Success:** Graph shows Source → Server → Sink. Step advances sim. Entity metrics update. Event log populates.

### Phase 2: Play Mode + WebSocket

Add WebSocket `/api/ws`, play/pause controls, speed slider. Dynamic edge discovery via event flow hooks.

### Phase 3: Live Metrics + Polish

Metric badges on graph nodes. Timeline bar. Internal event filtering. Keyboard shortcuts (Space=step, Shift+Space=play).

### Phase 4: Breakpoints

Breakpoint management UI. Time/event-count/event-type breakpoints. Pause-on-breakpoint notification in UI.

---

## Files to Create/Modify

| File | Change |
|------|--------|
| `happysimulator/visual/__init__.py` | **New** — `serve()` entry point |
| `happysimulator/visual/server.py` | **New** — FastAPI app + endpoints |
| `happysimulator/visual/bridge.py` | **New** — SimulationBridge |
| `happysimulator/visual/topology.py` | **New** — topology discovery |
| `happysimulator/visual/serializers.py` | **New** — entity state serialization |
| `happysimulator/visual/static/*` | **New** — pre-built frontend assets |
| `visual-frontend/` | **New** — frontend source (dev-only) |
| `pyproject.toml` | Add `[visual]` optional dependency |

No changes to core library code — the debugger is purely additive, using only the existing `sim.control` public API (plus `sim._entities`/`sim._sources` private access for entity enumeration).

---

## Verification

1. Create a simple simulation (M/M/1 queue example)
2. Call `serve(sim)` — browser opens
3. Verify graph shows correct topology
4. Click Step — entity state updates, event appears in log
5. Click multiple times — queue depth changes, metrics update
6. Reset — sim returns to t=0
7. (Phase 2) Play — continuous updates at ~20fps
8. (Phase 4) Add breakpoint — sim pauses when condition met
