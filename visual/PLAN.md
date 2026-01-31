# Visual Simulation Editor - Implementation Plan

This document synthesizes the design documents ([schema.md](schema.md), [Render.md](Render.md), [Editor.md](Editor.md)) into an actionable implementation plan.

---

## Overview

Build a browser-based visual editor for creating and running discrete-event simulations, backed by the existing happy-simulator Python runtime.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              USER WORKFLOW                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. DESIGN                2. RUN                   3. ANALYZE           │
│   ────────────            ─────                    ─────────             │
│   Drag nodes              Click Play               View metrics          │
│   Connect edges           Step through             Export results        │
│   Set parameters          Watch live               Iterate               │
│                                                                          │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐       │
│   │   Editor    │  JSON   │   Python    │  State  │   Editor    │       │
│   │  (React)    │ ──────▶ │   Runtime   │ ──────▶ │  (Charts)   │       │
│   └─────────────┘         └─────────────┘         └─────────────┘       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## System Architecture

```
happysimulator/                # Core library
├── ...existing modules...
├── loader.py                  # SimulationLoader (new)
├── controller.py              # SimulationController (new)
└── entities/                  # Built-in entity types (new)
    ├── __init__.py            # Entity registry
    ├── queued_server.py
    ├── sink.py
    └── router.py

visual/
├── editor/                    # React + React Flow frontend
│   └── src/
│       ├── components/        # Editor UI components
│       ├── nodes/             # Custom node types
│       ├── serialization/     # JSON schema ↔ React Flow
│       └── api/               # Backend communication
│
├── server/                    # Thin FastAPI layer (new)
│   ├── api.py                 # REST + WebSocket endpoints
│   └── schema.py              # Pydantic models for API
│
├── schema.md                  # JSON schema reference
├── Render.md                  # Runtime design
├── Editor.md                  # Frontend design
└── PLAN.md                    # This file
```

---

## Implementation Phases

### Phase 1: Foundation

**Goal**: Minimal working pipeline from editor to running simulation.

#### 1.1 JSON Schema & Types
- [ ] Create TypeScript types from `schema.md`
- [ ] Create Pydantic models for API validation
- [ ] Write validation logic for both (or use JSON Schema validation)

**Deliverable**: `visual/editor/src/types/schema.ts`, `visual/server/schema.py`

#### 1.2 Core Library Extensions
- [ ] Implement `SimulationLoader.load(definition) -> SimulationController`
- [ ] Implement `SimulationController` with:
  - `step()` - execute one event
  - `run_until(time)` - run to simulation time
  - `get_state_snapshot()` - current state for UI
- [ ] Create entity registry for built-in types:
  - `QueuedServer` (exists in archive, adapt)
  - `Sink` (new, simple terminal)
  - `RandomRouter` (new, distributes to outputs)

**Deliverable**: `happysimulator/loader.py`, `happysimulator/controller.py`, `happysimulator/entities/`

#### 1.3 FastAPI Server
- [ ] `POST /simulations` - load JSON, return simulation ID
- [ ] `POST /simulations/{id}/step` - step one event
- [ ] `POST /simulations/{id}/run` - run until time/count
- [ ] `GET /simulations/{id}/state` - get current state

**Deliverable**: `visual/server/api.py` (thin layer importing from happysimulator)

#### 1.4 Minimal React Editor
- [ ] Scaffold Vite + React + TypeScript project
- [ ] Install React Flow, basic setup
- [ ] Create ONE node type (QueuedServer) with handles
- [ ] Wire drag-to-canvas from palette
- [ ] Connect nodes with edges
- [ ] Export button → console.log JSON

**Deliverable**: Editor that can create a simple graph and export JSON

#### 1.5 End-to-End Test
- [ ] Manual test: create graph in editor → export JSON → POST to server → step through
- [ ] Verify state snapshots come back correctly

---

### Phase 2: Full Editor

**Goal**: Complete visual editing experience.

#### 2.1 All Node Types
- [ ] `SourceNode` - green, shows arrival rate
- [ ] `SinkNode` - red, terminal
- [ ] `RouterNode` - orange, shows routing strategy
- [ ] `DelayNode` - gray, shows delay distribution
- [ ] Add node-specific icons/colors per Editor.md

#### 2.2 Properties Panel
- [ ] Show panel when node selected
- [ ] Generic fields: label
- [ ] Type-specific parameter editors:
  - Distribution editor (constant/exponential/uniform/normal)
  - Arrival editor (poisson/constant)
  - Profile editor (constant/ramp/spike)
- [ ] Fleet count input for entities

#### 2.3 Serialization
- [ ] Implement `serializeToSchema(nodes, edges)`
- [ ] Implement `deserializeFromSchema(schema)`
- [ ] Handle sources specially (target derived from edge)
- [ ] Handle fleet count

#### 2.4 Import/Export
- [ ] Save to file (download JSON)
- [ ] Load from file (upload JSON)
- [ ] Validate on import, show errors

#### 2.5 Editor Polish
- [ ] Undo/redo (Zustand middleware)
- [ ] Delete nodes/edges (keyboard + context menu)
- [ ] Copy/paste nodes
- [ ] Zoom/pan controls
- [ ] Grid snap

---

### Phase 3: Live Simulation

**Goal**: Run simulations and see live results.

#### 3.1 Simulation Controls
- [ ] Run button → POST to server, get simulation ID
- [ ] Step button → advance one event
- [ ] Play/Pause → continuous execution
- [ ] Speed slider → adjust batch size
- [ ] Reset button → reload simulation

#### 3.2 WebSocket Streaming
- [ ] Backend: `/simulations/{id}/stream` WebSocket endpoint
- [ ] Frontend: connect on Play, disconnect on Pause
- [ ] Receive state snapshots at ~60fps

#### 3.3 Live Metrics Overlay
- [ ] Display current metrics on nodes (depth, processed count)
- [ ] Utilization bar on server nodes
- [ ] Animated edges (show event flow)

#### 3.4 Time Display
- [ ] Current simulation time in toolbar
- [ ] Events processed counter
- [ ] Pending events counter

---

### Phase 4: Advanced Features

**Goal**: Power-user features and polish.

#### 4.1 Probes
- [ ] UI to add probes to entities
- [ ] Configure: metric, interval, aggregation
- [ ] Show probe data in sidebar chart

#### 4.2 Fleet Support
- [ ] Backend: expand `count > 1` entities at load time
- [ ] Backend: implement routing strategies (round_robin, random, least_loaded, hash)
- [ ] Backend: aggregate metrics across fleet instances
- [ ] Frontend: show fleet badge on nodes
- [ ] Frontend: routing selector on edges to fleets

#### 4.3 Validation
- [ ] Validate connections (e.g., Source must connect to entity)
- [ ] Warn on disconnected nodes
- [ ] Validate params (e.g., rate > 0)

#### 4.4 Results & Export
- [ ] Collect probe time series during run
- [ ] Display charts (queue depth over time, latency histogram)
- [ ] Export results to CSV

#### 4.5 Examples
- [ ] Bundled example simulations (M/M/1, load balancer, fleet)
- [ ] "Load Example" dropdown in toolbar

---

## Dependencies Between Components

```
                    ┌─────────────────┐
                    │  JSON Schema    │
                    │  (schema.md)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────────┐ ┌─────────────────────────────────────┐
     │ TypeScript     │ │          happysimulator/            │
     │ Types          │ │  ┌────────────┐  ┌────────────────┐ │
     └───────┬────────┘ │  │ Loader     │  │ Entities       │ │
             │          │  └─────┬──────┘  └───────┬────────┘ │
             │          │        │                 │          │
             │          │        ▼                 │          │
             │          │  ┌────────────────┐      │          │
             │          │  │ Controller     │◄─────┘          │
             │          │  └───────┬────────┘                 │
             │          └──────────┼──────────────────────────┘
             │                     │
             ▼                     ▼
     ┌────────────────┐    ┌────────────────┐
     │ Serialization  │    │  FastAPI       │
     │ (TS)           │    │  (visual/      │
     └───────┬────────┘    │   server/)     │
             │             └───────┬────────┘
             ▼                     │
     ┌────────────────┐            │
     │ Editor         │            │
     │ (React Flow)   │◄───────────┘
     └────────────────┘     WebSocket
```

---

## Entity Type Checklist

Built-in types to implement:

| Type | Python Class | React Node | Priority |
|------|--------------|------------|----------|
| QueuedServer | Adapt from archive | QueuedServerNode | P1 |
| Sink | New (simple) | SinkNode | P1 |
| RandomRouter | New | RouterNode | P1 |
| Source | Use existing | SourceNode | P1 |
| Delay | New | DelayNode | P2 |
| Filter | New | FilterNode | P3 |
| Fork | New | ForkNode | P3 |
| Join | New | JoinNode | P3 |
| Custom | New (eval handler) | CustomNode | P3 |

---

## File Inventory

### Core Library (happysimulator/)

| File | Purpose | Phase |
|------|---------|-------|
| `happysimulator/loader.py` | `SimulationLoader` class | 1 |
| `happysimulator/controller.py` | `SimulationController` class | 1 |
| `happysimulator/entities/__init__.py` | Entity registry | 1 |
| `happysimulator/entities/queued_server.py` | QueuedServer implementation | 1 |
| `happysimulator/entities/sink.py` | Sink implementation | 1 |
| `happysimulator/entities/router.py` | RandomRouter implementation | 1 |

### API Server (visual/server/)

| File | Purpose | Phase |
|------|---------|-------|
| `visual/server/__init__.py` | Package init | 1 |
| `visual/server/schema.py` | Pydantic models for API requests | 1 |
| `visual/server/api.py` | FastAPI routes (imports from happysimulator) | 1 |

### Frontend (TypeScript/React)

| File | Purpose | Phase |
|------|---------|-------|
| `editor/src/types/schema.ts` | TypeScript types | 1 |
| `editor/src/nodes/registry.ts` | Node type mapping | 1 |
| `editor/src/nodes/QueuedServerNode.tsx` | Server node component | 1 |
| `editor/src/nodes/SourceNode.tsx` | Source node component | 2 |
| `editor/src/nodes/SinkNode.tsx` | Sink node component | 2 |
| `editor/src/nodes/RouterNode.tsx` | Router node component | 2 |
| `editor/src/components/Editor.tsx` | Main canvas | 1 |
| `editor/src/components/Palette.tsx` | Draggable node list | 1 |
| `editor/src/components/PropertiesPanel.tsx` | Parameter editor | 2 |
| `editor/src/components/Toolbar.tsx` | Run/Save/Load buttons | 2 |
| `editor/src/components/DistributionEditor.tsx` | Distribution param editor | 2 |
| `editor/src/serialization/serialize.ts` | React Flow → JSON | 2 |
| `editor/src/serialization/deserialize.ts` | JSON → React Flow | 2 |
| `editor/src/api/client.ts` | Backend API calls | 3 |
| `editor/src/api/websocket.ts` | WebSocket connection | 3 |

---

## Getting Started

### 1. Extend the core library

```bash
# Use existing project venv
.\.venv\Scripts\Activate.ps1  # Windows

# Install additional dependencies
pip install fastapi uvicorn pydantic

# Create new files in happysimulator/
# - happysimulator/loader.py
# - happysimulator/controller.py
# - happysimulator/entities/
```

### 2. Set up the API server

```bash
cd visual
mkdir server

# Create visual/server/api.py (imports from happysimulator)

# Run server
cd ..  # back to repo root
python -m uvicorn visual.server.api:app --reload --port 8000
```

### 3. Set up the React editor

```bash
cd visual
npm create vite@latest editor -- --template react-ts
cd editor

# Install dependencies
npm install reactflow zustand
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Install UI components
npm install @radix-ui/react-select @radix-ui/react-popover

# Run dev server
npm run dev
```

### 4. Test the integration

```bash
# Terminal 1: Python server (from repo root)
python -m uvicorn visual.server.api:app --reload --port 8000

# Terminal 2: React dev server
cd visual/editor
npm run dev

# Open http://localhost:5173
```

---

## Success Criteria

### Phase 1 Complete When:
- [ ] Can create a Source → Server → Sink graph in editor
- [ ] Export produces valid JSON matching schema.md
- [ ] Server loads JSON and creates controller
- [ ] Can step through simulation via API
- [ ] State snapshots show queue depth changing

### Phase 2 Complete When:
- [ ] All P1 node types implemented with proper styling
- [ ] Properties panel edits all parameters
- [ ] Can save/load simulation files
- [ ] Undo/redo works

### Phase 3 Complete When:
- [ ] Play button runs simulation with live updates
- [ ] Nodes show live metrics
- [ ] Can pause/resume simulation
- [ ] Speed control works

### Phase 4 Complete When:
- [ ] Fleets work (count > 1 with routing)
- [ ] Probes collect and display time series
- [ ] Example simulations included
- [ ] Can export results to CSV

---

## Open Questions

1. **Authentication**: Should the server support multiple users/sessions?
   - Initial: No, single-user local development
   - Future: Consider adding session management

2. **Persistence**: Where to store saved simulations?
   - Initial: Browser local storage + file download
   - Future: Server-side database

3. **Custom entities**: How to handle user Python code?
   - Initial: Reference by module path (must be importable)
   - Future: Sandboxed execution, code editor in UI

4. **Performance**: How to handle very large fleets (10k+ entities)?
   - Initial: Trust the existing event heap performance
   - Future: Profile and optimize if needed

---

## Reference Documents

- [schema.md](schema.md) - JSON schema for simulation definitions
- [Render.md](Render.md) - Python runtime architecture (loader, controller, API)
- [Editor.md](Editor.md) - React frontend architecture (components, nodes, serialization)
