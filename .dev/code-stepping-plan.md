# Plan: Line-by-Line Code Stepping in Visual Debugger

## Context

The visual debugger currently supports **event-level** stepping but has no way to see what code is executing inside an entity's generator between yields. This plan adds code panels as draggable/resizable nodes on the ReactFlow graph canvas that show the entity's Python source, auto-animate execution through lines, and pause at breakpoints.

**Key insight**: Python's native frame tracing (`sys.settrace()` / `gen.gi_frame.f_trace`) intercepts each line of execution within entity event handlers — works for **all entity types** (generators AND non-generators). The trace function records lines for animation replay and only blocks the sim thread at breakpoints. No browser-based Python interpreter needed.

---

## Architecture

**Two modes in the trace function:**

1. **Recording** (default): Each line appends `(line_number, locals)` to a buffer. Zero blocking. After `handle_event()` / `gen.send()` completes, the buffer is sent to the frontend as a `code_trace` for animated replay.

2. **Breakpoint**: When a line matches a code breakpoint, the trace function blocks the sim thread via `threading.Event`. A `code_paused` message is sent. User clicks Continue/Step to resume.

Trace installation happens at two points to cover all entity types:
- **`Event.invoke()`**: `sys.settrace()` before `handle_event()` — covers non-generator entities AND the first yield of generators
- **`ProcessContinuation.invoke()`**: `gen.gi_frame.f_trace` before `gen.send()` — covers generator resumptions

```
Browser (React)                           Backend (Python)

CodePanelNode (on graph)                  CodeDebugger
  |                                           |
  |  --- Entity handles event ---             |
  |                                           |-- Event.invoke() or
  |                                           |   ProcessContinuation.invoke()
  |                                           |-- installs trace fn
  |                                           |
  |  --- Normal (no breakpoint) ---           |
  |                                           |-- trace fn records lines to buffer
  |                                           |-- handle_event() / gen.send() returns
  |<-- WS: state_update + code_traces ---     |
  |   (frontend animates line highlights)     |
  |                                           |
  |  --- Breakpoint hit ---                   |
  |                                           |-- trace fn blocks sim thread
  |<-- WS: "code_paused" -----------------   |
  |-- WS: "code_continue" --------------->   |
  |                                           |-- threading.Event.set(), resumes
```

---

## Implementation Steps

### Step 1: Create `happysimulator/visual/code_debugger.py`

Core classes: `CodeBreakpoint`, `CodeLocation`, `FrameSnapshot`, `LineRecord`, `ExecutionTrace`, `CodeDebugger`

CodeDebugger responsibilities:
- **Entity activation**: Track entities with open code panels. Only these get trace functions
- **Source retrieval**: `inspect.getsourcelines()`, cached per class. Tries `handle_queued_event` first (QueuedResource), falls back to `handle_event`. Also caches method code objects for frame filtering
- **Two trace installation methods**:
  - `install_call_trace(name, cls)`: Uses `sys.settrace()` — for `Event.invoke()`, works with ALL entity types
  - `install_generator_trace(gen, name, cls)`: Uses `gen.gi_frame.f_trace` — for `ProcessContinuation.invoke()`, generator resumptions only
- **Frame filtering**: The `sys.settrace()` trace fn filters by checking `frame.f_code` against the entity method's cached code object, so only relevant frames are recorded
- **Breakpoints**: Registry checked on each line event
- **Blocking**: `threading.Event` for breakpoint pauses, 30s deadman timeout
- **Serialization**: `frame.f_locals` → JSON-safe dict with depth limits
- **Trace draining**: `drain_completed_traces()` for bridge to include in step results

### Step 2: Instrument `Event.invoke()` AND `ProcessContinuation.invoke()` in `happysimulator/core/event.py`

- Add module-level `_active_code_debugger` context (same pattern as `_active_heap`/`_active_clock` in `sim_future.py`)
- **In `Event.invoke()`** (covers ALL entity types):
  - Before `handle_event()`: check if entity is active, install trace via `code_debugger.install_call_trace()`
  - After `handle_event()` (in `finally`): remove trace via `code_debugger.remove_call_trace()`, flushes buffer
  - This traces non-generator entities fully, and generator entities from method entry to first yield
- **In `ProcessContinuation.invoke()`** (covers generator resumptions):
  - Before `gen.send()`: check if entity is active, install trace via `code_debugger.install_generator_trace()`
  - After `gen.send()` (in `finally`): remove trace via `code_debugger.remove_generator_trace()`, flushes buffer
- Resolve `_QueuedResourceWorkerAdapter` indirection to get actual entity name/class in both methods

### Step 3: Wire up context in `happysimulator/core/simulation.py`

In `run()`, set/clear `_active_code_debugger` alongside existing `_set_active_context(heap, clock)`

### Step 4: Extend `happysimulator/visual/bridge.py`

- Create `CodeDebugger` in `__init__`, inject into simulation as `sim._code_debugger`
- Add methods: `get_entity_source()`, `activate_code_debug()`, `deactivate_code_debug()`, `set_code_breakpoint()`, `remove_code_breakpoint()`, `get_code_debug_state()`, `code_step()`, `code_step_over()`, `code_step_out()`, `code_continue()`
- Extend `step()`: after stepping, drain completed traces and include as `code_traces` in result
- Extend `reset()`: also reset code debugger

### Step 5: Extend `happysimulator/visual/server.py`

REST endpoints:
- `GET /api/entity/{name}/source`
- `POST /api/debug/code/activate`, `POST /api/debug/code/deactivate`
- `POST /api/debug/code/breakpoints`, `DELETE /api/debug/code/breakpoints/{id}`

WebSocket actions: `activate_code_debug`, `deactivate_code_debug`, `code_step`, `code_step_over`, `code_step_out`, `code_continue`

WebSocket messages: `code_debug_activated`, `code_debug_deactivated`, `code_paused`, and `code_traces` field in `state_update`

Run event-level step as background task when code debugging is active (so WS can receive `code_continue` during breakpoint blocks)

### Step 6: Frontend types and state

**`types.ts`**: Add `CodeTraceRecord`, `CodeTrace`, `CodePausedState`, `EntitySource`, `CodePanelConfig`

**`useSimState.ts`**: Add `codePanels: Map<string, CodePanelConfig>`, `codePausedEntity: string | null`, and actions: `openCodePanel`, `closeCodePanel`, `setCodeTrace`, `setCodePaused`, `toggleCodeBreakpoint`, `clearCodePaused`

**`useWebSocket.ts`**: Handle `code_traces` in `state_update`, `code_paused`, `code_debug_activated`, `code_debug_deactivated`

### Step 7: Create `CodePanelNode.tsx` (custom ReactFlow node)

A resizable, draggable code panel that lives on the graph canvas:
- Header: entity class + method name, close button, drag handle
- Breakpoint gutter: clickable red circles
- Source pane: monospace `<pre>`, current line highlighted amber
- Locals panel: collapsible bottom section, visible when paused at breakpoint
- Uses ReactFlow's `NodeResizer` for resize handles
- Animation: replays `code_trace` lines at ~100-200ms intervals, highlighting each line
- Connected to entity node with a dashed edge

### Step 8: Modify existing frontend components

**`EntityNode.tsx`**: Add `</>` icon button to open code panel (sends `activate_code_debug`)

**`GraphView.tsx`**: Register `codePanelNode` node type, manage code panel nodes/edges dynamically when activated/deactivated

**`ControlBar.tsx`**: Show breakpoint step controls (Step, Over, Out, Continue) when `codePausedEntity` is set

### Step 9: Build and test

- Frontend build: `cd visual-frontend && npm run build`
- Unit tests: `tests/unit/test_code_debugger.py` (source retrieval, trace recording, breakpoints, serialization, deadman timeout)
- Integration tests: `tests/integration/test_code_stepping.py` (full round-trip with generator stepping)
- Manual E2E: open code panels on graph, verify animation and breakpoints

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `happysimulator/visual/code_debugger.py` | **Create** |
| `happysimulator/core/event.py` | **Modify** - instrument ProcessContinuation.invoke() |
| `happysimulator/core/simulation.py` | **Modify** - wire up code debugger context |
| `happysimulator/visual/bridge.py` | **Modify** - add code debug methods, include traces in step results |
| `happysimulator/visual/server.py` | **Modify** - REST endpoints + WS actions |
| `visual-frontend/src/types.ts` | **Modify** |
| `visual-frontend/src/hooks/useSimState.ts` | **Modify** |
| `visual-frontend/src/hooks/useWebSocket.ts` | **Modify** |
| `visual-frontend/src/components/CodePanelNode.tsx` | **Create** |
| `visual-frontend/src/components/EntityNode.tsx` | **Modify** - add open code button |
| `visual-frontend/src/components/GraphView.tsx` | **Modify** - register code panel node type |
| `visual-frontend/src/components/ControlBar.tsx` | **Modify** - breakpoint step controls |
| `tests/unit/test_code_debugger.py` | **Create** |
| `tests/integration/test_code_stepping.py` | **Create** |

---

## Verification

1. `pytest tests/unit/test_code_debugger.py -q`
2. `pytest tests/integration/test_code_stepping.py -q`
3. `pytest -q` (no regressions)
4. Manual: run example with `serve()`, click `</>` on entity, verify code panel appears on graph with source, set breakpoint, play simulation, verify animation + breakpoint pause
