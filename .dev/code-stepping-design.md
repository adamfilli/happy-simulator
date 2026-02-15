# Design: Line-by-Line Code Stepping in Visual Debugger

## Overview

Add the ability to view and debug Python source code for any entity, directly on the graph canvas of the browser-based visual debugger. Users click an entity node, open its implementation as a draggable code panel on the graph, set breakpoints on specific lines, and watch execution animate through the code whenever that entity handles an event.

## Motivation

The visual debugger currently supports **event-level** stepping -- you can step through events one at a time, set breakpoints on time/event count/metrics, and inspect entity state. However, there's no way to see *what code is executing* inside an entity's generator between yields. For complex entity logic, this makes debugging opaque -- you see the inputs and outputs of each yield but not the decisions happening in between.

## Requirements

1. **Source display as graph nodes**: Each entity can spawn a code panel node on the ReactFlow graph canvas, showing the Python source of its `handle_event()` / `handle_queued_event()`
2. **Draggable and resizable**: Code panel nodes behave like other graph nodes -- draggable, resizable via corner handles
3. **Multiple panels**: Multiple code panels can be open simultaneously for different entities, each connected to its entity node with an edge
4. **Auto-animate execution**: When an entity handles an event, the current line highlights and animates through the code at readable speed -- no manual stepping required for normal flow
5. **Breakpoints**: Click line gutters to set/remove breakpoints. Execution pauses at breakpoints, showing locals
6. **Variable inspection**: Display local variables when paused at a breakpoint
7. **Per-entity activation**: Only trace entities with open code panels (performance)
8. **Integration**: Works alongside existing event-level stepping and breakpoints

## Design

### Key Insight: Python Frame Tracing

No browser-based Python interpreter is needed. Python's native frame tracing mechanism handles everything:

- `sys.settrace()` / `frame.f_trace` - Per-frame trace function fires on each line
- `gen.gi_frame` - Generator objects expose their execution frame
- `frame.f_lineno` - Current line number
- `frame.f_locals` - Local variables at current line
- `inspect.getsourcelines()` - Retrieve source code with line numbers

This works for **all entity types**, not just generators:

| Entity type | Where trace is installed | How it works |
|---|---|---|
| Non-generator (`return [Event(...)]`) | `Event.invoke()` via `sys.settrace()` | Traces the entire `handle_event()` call |
| Generator (first yield) | `Event.invoke()` via `sys.settrace()` | Traces from method entry until the first `yield` |
| Generator (subsequent yields) | `ProcessContinuation.invoke()` via `gen.gi_frame.f_trace` | Traces each `gen.send()` resumption |

For non-generator entities, the trace is installed before calling `self.target.handle_event(self)` and removed after it returns. The trace function filters by checking the frame's code object against the entity's method, so only relevant frames are recorded.

For generators, the initial call goes through the same `Event.invoke()` path, and subsequent resumptions go through `ProcessContinuation.invoke()` where the trace is installed on `gen.gi_frame.f_trace`.

### Architecture: Trace Recording + Breakpoint Blocking

The trace function operates in two modes:

1. **Recording mode** (default): On each line event, append `(line_number, serialized_locals)` to a buffer. No blocking, no performance penalty. After `gen.send()` completes (generator yields or returns), the full trace buffer is sent to the frontend as a `code_trace` message.

2. **Breakpoint mode**: When the current line matches a code breakpoint, the trace function blocks the sim thread via `threading.Event` (same pattern as existing pause/resume). A `code_paused` message is sent. The user can then manually step or continue.

The frontend replays the recorded trace as an animation -- the highlighted line moves through the code at a configurable speed. This gives the "watching execution flow through the code" experience without any backend timing dependency.

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
  |                                           |-- flush buffer to completed_traces
  |<-- WS: state_update + code_traces ---     |
  |   (frontend animates through lines)       |
  |                                           |
  |  --- Breakpoint hit ---                   |
  |                                           |-- trace fn blocks sim thread
  |<-- WS: "code_paused" -----------------   |
  |   { line, locals, source, stack }         |
  |                                           |
  |-- WS: "code_continue" --------------->   |
  |                                           |-- threading.Event.set()
  |                                           |-- continues recording
```

### Threading Model

1. User clicks "Step" (event-level) or "Play" in the browser
2. `bridge.step()` or play loop calls sim via `asyncio.to_thread` (sim thread)
3. Simulation processes event → `Event.invoke()` calls `handle_event()`
4. If entity has an open code panel, trace fn is installed:
   - Non-generator: `sys.settrace()` before `handle_event()` call
   - Generator (first call): same as above, traces until first `yield`
   - Generator (resumption): `gen.gi_frame.f_trace` before `gen.send()`
5. Trace fn records lines to buffer. If a breakpoint is hit, blocks sim thread
6. WebSocket handler is still alive (async, different context)
7. User clicks "Continue" → signals `threading.Event`, sim thread resumes
8. When `handle_event()` / `gen.send()` completes, trace buffer is flushed
9. Frontend receives traces and replays the line animation

Event-level step must run as a background task when code debugging is active so the WebSocket can still receive code commands during breakpoint pauses.

---

## Implementation

### 1. New File: `happysimulator/visual/code_debugger.py`

Core module containing all code-level debugging logic.

**Data classes:**

```python
@dataclass(frozen=True)
class CodeBreakpoint:
    entity_class_name: str   # e.g. "MM1Server"
    file_path: str           # absolute path to .py file
    line_number: int         # 1-based line number

@dataclass(frozen=True)
class CodeLocation:
    file_path: str
    line_number: int
    function_name: str
    entity_name: str
    entity_class_name: str

@dataclass
class FrameSnapshot:
    """Captured state when paused at a breakpoint."""
    location: CodeLocation
    locals: dict[str, Any]
    source_lines: list[str]
    source_start_line: int
    call_stack: list[CodeLocation]

@dataclass
class LineRecord:
    """Single line execution record for animation replay."""
    line_number: int
    locals: dict[str, Any]  # serialized locals at this line

@dataclass
class ExecutionTrace:
    """Full trace of lines executed during one gen.send() call."""
    entity_name: str
    entity_class_name: str
    records: list[LineRecord]
    paused_at: FrameSnapshot | None  # non-None if stopped at breakpoint
```

**CodeDebugger class:**

```python
class CodeDebugger:
    DEADMAN_TIMEOUT_S = 30.0  # safety valve for breakpoint blocking

    def __init__(self):
        self._active_entities: set[str] = set()       # entity names with open code panels
        self._breakpoints: dict[str, CodeBreakpoint] = {}
        self._source_cache: dict[type, tuple] = {}

        # Trace recording buffer (populated by trace fn, consumed after gen.send)
        self._trace_buffer: list[LineRecord] = []
        self._trace_entity_name: str | None = None
        self._trace_entity_class_name: str | None = None

        # Breakpoint pause state
        self._paused_frame: FrameSnapshot | None = None
        self._is_code_paused: bool = False
        self._continue_event = threading.Event()
        self._continue_event.set()  # start in running state

        # Completed traces queue (read by bridge after gen.send completes)
        self._completed_traces: list[ExecutionTrace] = []

        self._lock = threading.Lock()
```

**Key methods:**

| Method | Thread | Purpose |
|--------|--------|---------|
| `activate_entity(name)` | API | Start tracing for entity (code panel opened) |
| `deactivate_entity(name)` | API | Stop tracing (code panel closed) |
| `get_entity_source(cls)` | API | `inspect.getsourcelines()` with caching; also caches code object for frame filtering |
| `add_breakpoint(bp)` / `remove_breakpoint(id)` | API | Manage code breakpoints |
| `code_step()` | API | Advance one line from breakpoint pause |
| `code_step_over()` | API | Step over function calls from breakpoint |
| `code_step_out()` | API | Step out to caller from breakpoint |
| `code_continue()` | API | Continue execution from breakpoint |
| `install_call_trace(name, cls)` | Sim | `sys.settrace()` before `handle_event()` — works for ALL entity types |
| `install_generator_trace(gen, name, cls)` | Sim | `gen.gi_frame.f_trace` before `gen.send()` — for generator resumptions |
| `remove_call_trace()` | Sim | `sys.settrace(None)`, flush buffer |
| `remove_generator_trace(gen)` | Sim | Clear frame trace, flush buffer |
| `_trace_handler(...)` | Sim | Per-line handler: record line OR block at breakpoint |
| `drain_completed_traces()` | API | Pop completed traces for WS delivery |
| `_serialize_locals(locals)` | Sim | Safe JSON serialization with depth limits |

**Source retrieval logic:**

```python
def get_entity_source(self, entity_class):
    # Try handle_queued_event first (QueuedResource subclasses)
    for method_name in ("handle_queued_event", "handle_event"):
        method = getattr(entity_class, method_name, None)
        if method and not getattr(method, '__isabstractmethod__', False):
            source_lines, start_line = inspect.getsourcelines(method)
            file_path = inspect.getfile(method)
            return (source_lines, start_line, str(Path(file_path).resolve()))
    return None  # source unavailable
```

**Trace function logic (hybrid record + breakpoint):**

```python
def _trace_handler(self, frame, event, arg, *, entity_name, ...):
    if event == "call": self._current_depth += 1; return nested_trace
    if event == "return": self._current_depth -= 1; return None
    if event != "line": return nested_trace

    line = frame.f_lineno
    locals_snapshot = self._serialize_locals(frame.f_locals)

    # Always record the line
    self._trace_buffer.append(LineRecord(line, locals_snapshot))

    # Check for breakpoint
    if self._check_code_breakpoint(entity_class_name, file_path, line):
        snapshot = self._capture_frame(frame, ...)
        with self._lock:
            self._paused_frame = snapshot
            self._is_code_paused = True
            self._continue_event.clear()
        # BLOCK sim thread until step/continue command
        self._continue_event.wait(timeout=self.DEADMAN_TIMEOUT_S)

    return nested_trace
```

**Variable serialization:**

```python
def _serialize_locals(self, local_vars, depth=0, max_depth=2):
    result = {}
    for key, value in list(local_vars.items())[:50]:
        if key.startswith("__"): continue
        result[key] = self._serialize_value(value, depth, max_depth)
    return result

def _serialize_value(self, value, depth=0, max_depth=2):
    if value is None: return None
    if isinstance(value, (bool, int, float, str)): return value
    if depth >= max_depth: return f"<{type(value).__name__}>"
    if isinstance(value, (list, tuple)):
        return [self._serialize_value(v, depth+1, max_depth) for v in value[:20]]
    if isinstance(value, dict):
        return {str(k): self._serialize_value(v, depth+1, max_depth)
                for k, v in list(value.items())[:20]}
    if hasattr(value, 'name'):
        return f"<{type(value).__name__} name={value.name!r}>"
    return f"<{type(value).__name__}>"
```

### 2. Modify: `happysimulator/core/event.py`

**Add module-level context** (following `_active_heap`/`_active_clock` pattern):

```python
_active_code_debugger = None

def _set_active_code_debugger(debugger): ...
def _get_active_code_debugger(): ...
def _clear_active_code_debugger(): ...
```

**Instrument `Event.invoke()`** (around line 133) — covers ALL entity types:

This is the entry point for every event dispatch. The trace is installed here before
`handle_event()` is called, which covers both non-generator and generator entities
(for generators, this traces from method entry to the first yield):

```python
def invoke(self):
    code_debugger = _get_active_code_debugger()
    entity = self.target
    entity_name = getattr(entity, "name", None)

    # Resolve QueuedResource indirection (_QueuedResourceWorkerAdapter)
    resolved_entity = entity
    if hasattr(entity, '_resource'):
        resolved_entity = entity._resource
        entity_name = getattr(resolved_entity, "name", entity_name)

    should_trace = (
        code_debugger is not None
        and entity_name is not None
        and code_debugger.is_entity_active(entity_name)
    )

    if should_trace:
        code_debugger.install_call_trace(entity_name, type(resolved_entity))

    try:
        raw_result = self.target.handle_event(self)

        if isinstance(raw_result, Generator):
            # Generator path: trace continues via ProcessContinuation
            return self._start_process(raw_result)
        else:
            # Non-generator: trace is complete, flush buffer
            return self._normalize_return(raw_result)
    finally:
        if should_trace:
            code_debugger.remove_call_trace()
```

**Also instrument `ProcessContinuation.invoke()`** (around line 307) — covers generator resumptions:

When a generator resumes via `gen.send()`, this traces the code between the
previous yield and the next yield/return:

```python
def invoke(self):
    code_debugger = _get_active_code_debugger()
    entity = self.target
    entity_name = getattr(entity, "name", None)

    # Resolve QueuedResource indirection
    resolved_entity = entity
    if hasattr(entity, '_resource'):
        resolved_entity = entity._resource
        entity_name = getattr(resolved_entity, "name", entity_name)

    should_trace = (
        code_debugger is not None
        and entity_name is not None
        and code_debugger.is_entity_active(entity_name)
        and self.process.gi_frame is not None
    )

    if should_trace:
        code_debugger.install_generator_trace(
            self.process, entity_name, type(resolved_entity)
        )

    try:
        yielded_val = self.process.send(self._send_value)
        # ... rest unchanged ...
    finally:
        if should_trace:
            code_debugger.remove_generator_trace(self.process)
```

**Two trace installation methods on CodeDebugger:**

| Method | Used by | Mechanism |
|--------|---------|-----------|
| `install_call_trace(name, cls)` | `Event.invoke()` | `sys.settrace()` with frame-filtering trace fn |
| `install_generator_trace(gen, name, cls)` | `ProcessContinuation.invoke()` | `gen.gi_frame.f_trace` (per-frame, more targeted) |
| `remove_call_trace()` | `Event.invoke()` finally | `sys.settrace(None)`, flush buffer |
| `remove_generator_trace(gen)` | `ProcessContinuation.invoke()` finally | Clear `gen.gi_frame.f_trace`, flush buffer |

`install_call_trace` uses `sys.settrace()` which fires for all frames. The trace function
filters by checking `frame.f_code` against the entity's `handle_event` / `handle_queued_event`
method's code object (cached during source retrieval). Only matching frames are recorded.
Child frames from function calls within the handler are also traced for step-over/step-out support.

### 3. Modify: `happysimulator/core/simulation.py`

In `run()` (around line 231):

```python
from happysimulator.core.event import _set_active_code_debugger, _clear_active_code_debugger

_set_active_context(self._event_heap, self._clock)
if hasattr(self, '_code_debugger'):
    _set_active_code_debugger(self._code_debugger)
try:
    return self._run_loop()
finally:
    _clear_active_context()
    _clear_active_code_debugger()
```

### 4. Modify: `happysimulator/visual/bridge.py`

**In `__init__`:**

```python
from happysimulator.visual.code_debugger import CodeDebugger
self._code_debugger = CodeDebugger()
sim._code_debugger = self._code_debugger
```

**New methods:**

- `get_entity_source(entity_name)` -> `{entity_name, class_name, file_path, start_line, source, line_count}`
- `activate_code_debug(entity_name)` -> source info dict (opens code panel)
- `deactivate_code_debug(entity_name)` (closes code panel)
- `set_code_breakpoint(class_name, file_path, line)` -> breakpoint ID
- `remove_code_breakpoint(bp_id)`
- `get_code_debug_state()` -> frame snapshot dict or None (when paused at breakpoint)
- `code_step()`, `code_step_over()`, `code_step_out()`, `code_continue()` (breakpoint controls)
- `_find_entity(name)` helper

**Extend `step()` method:** After `sim.control.step()`, drain completed traces from the code debugger and include them in the step result:

```python
def step(self, count=1):
    # ... existing step logic ...
    result = { ... existing fields ... }

    # Include code execution traces for animation
    traces = self._code_debugger.drain_completed_traces()
    if traces:
        result["code_traces"] = [
            {
                "entity_name": t.entity_name,
                "class_name": t.entity_class_name,
                "lines": [{"line": r.line_number, "locals": r.locals} for r in t.records],
            }
            for t in traces
        ]
    return result
```

**On `reset()`:** also call `self._code_debugger.reset()`

### 5. Modify: `happysimulator/visual/server.py`

**New REST endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/entity/{name}/source` | GET | Get source code |
| `/api/debug/code/activate` | POST | Enable code debugging |
| `/api/debug/code/deactivate` | POST | Disable code debugging |
| `/api/debug/code/breakpoints` | POST | Add code breakpoint |
| `/api/debug/code/breakpoints/{id}` | DELETE | Remove breakpoint |
| `/api/debug/code/state` | GET | Get paused frame state |

**New WebSocket actions (frontend -> backend):**

| Action | Params | Response |
|--------|--------|----------|
| `activate_code_debug` | `entity_name` | `code_debug_activated` |
| `deactivate_code_debug` | `entity_name` | `code_debug_deactivated` |
| `code_step` | - | `code_paused` or `state_update` |
| `code_step_over` | - | `code_paused` or `state_update` |
| `code_step_out` | - | `code_paused` or `state_update` |
| `code_continue` | - | `code_paused` or `state_update` |

**New WebSocket messages (backend -> frontend):**

```json
// Included in state_update when code panels are open:
{"type": "state_update", ..., "code_traces": [
  {"entity_name": "Server", "class_name": "MM1Server",
   "lines": [{"line": 42, "locals": {...}}, {"line": 43, "locals": {...}}, ...]}
]}

// When execution pauses at a code breakpoint:
{"type": "code_paused", "line_number": 46, "locals": {...}, "source": "...",
 "source_start_line": 42, "entity_name": "Server", "class_name": "MM1Server",
 "call_stack": [{"file": "...", "line": 46, "function": "handle_queued_event"}]}

// When a code panel is opened:
{"type": "code_debug_activated", "entity_name": "Server", "class_name": "MM1Server",
 "source": "...", "start_line": 42, "file_path": "...", "line_count": 15}

{"type": "code_debug_deactivated", "entity_name": "Server"}
```

**Concurrent blocking fix:** When code debugging is active and a breakpoint is hit, the event-level step blocks in `asyncio.to_thread`. Run it as a background asyncio task so the WebSocket handler remains responsive to `code_continue` commands.

### 6. Frontend: Types and State

**`visual-frontend/src/types.ts`** additions:

```typescript
// Execution trace for animation replay
interface CodeTraceRecord {
  line: number;
  locals: Record<string, unknown>;
}

interface CodeTrace {
  entity_name: string;
  class_name: string;
  lines: CodeTraceRecord[];
}

// State when paused at a breakpoint
interface CodePausedState {
  line_number: number;
  locals: Record<string, unknown>;
  source: string;
  source_start_line: number;
  entity_name: string;
  class_name: string;
  call_stack: Array<{file: string; line: number; function: string}>;
}

// Source info for an entity
interface EntitySource {
  entity_name: string;
  class_name: string;
  file_path: string;
  start_line: number;
  source: string;
  line_count: number;
  error?: string;
}

// Code panel configuration (one per open code panel on graph)
interface CodePanelConfig {
  entityName: string;
  className: string;
  source: string;
  startLine: number;
  breakpoints: Set<number>;         // line numbers with breakpoints
  currentLine: number | null;       // currently highlighted line (animation or paused)
  animatingLines: CodeTraceRecord[] | null;  // pending animation
  pausedLocals: Record<string, unknown> | null;  // locals when paused at breakpoint
  isPaused: boolean;
}
```

**`visual-frontend/src/hooks/useSimState.ts`** additions:

```typescript
// New state fields
codePanels: Map<string, CodePanelConfig>;  // keyed by entity name
codePausedEntity: string | null;           // entity paused at breakpoint

// New actions
openCodePanel(entityName: string, source: EntitySource): void;
closeCodePanel(entityName: string): void;
setCodeTrace(entityName: string, trace: CodeTrace): void;
setCodePaused(state: CodePausedState): void;
toggleCodeBreakpoint(entityName: string, line: number): void;
clearCodePaused(): void;
```

**`visual-frontend/src/hooks/useWebSocket.ts`** - Handle new messages:
- `state_update` with `code_traces` → feed traces to animation queue per panel
- `code_paused` → pause the relevant code panel, show locals
- `code_debug_activated` → open code panel node on graph
- `code_debug_deactivated` → remove code panel node from graph

### 7. Frontend: CodePanelNode Component

**New file: `visual-frontend/src/components/CodePanelNode.tsx`**

A custom ReactFlow node type that renders as a code panel on the graph canvas:

```
+--------------------------------------------------------------+
| MM1Server.handle_queued_event()    [x] close   ≡ drag handle |
+--------------------------------------------------------------+
|    | 42| def handle_queued_event(self, event):                |
|    | 43|     service_time = random.expovariate(...)           |
|    | 44|     yield service_time                               |
| ●  | 45|                                                     |
| >> | 46|     self.stats_processed += 1           ← animating |
|    | 47|                                                     |
|    | 48|     if self.downstream is None:                      |
|    | 49|         return []                                    |
|    | 50|                                                     |
|    | 51|     completed = Event(                               |
|    | 52|         time=self.now,                               |
+--------------------------------------------------------------+
| Locals (when paused at breakpoint):                          |
| self: <MM1Server>  event: <Event>  service_time: 0.153       |
+--------------------------------------------------------------+
    ^                                                     ^
    resize handle                                resize handle
```

**Features:**
- **Header**: Entity class + method name, close button, drag handle
- **Breakpoint gutter**: Red circles on click (●), empty circles on hover
- **Line numbers**: Monospace, 1-based matching file line numbers
- **Current line indicator**: `>>` marker + amber background for the active line
- **Source pane**: Monospace `<pre>`, scrollable. No external syntax highlighting for MVP
- **Locals panel**: Collapsible bottom section, only visible when paused at a breakpoint. Shows key-value pairs of serialized locals
- **Resizable**: Using ReactFlow's `NodeResizer` component (built into @xyflow/react)
- **Connected to entity**: An edge connects this node to the entity it represents

**Animation behavior:**
- When a `code_trace` arrives for this entity, replay the lines as an animation
- Highlight moves from line to line at ~100-200ms interval (configurable)
- If the panel isn't visible (scrolled off), the animation still runs but isn't rendered
- Multiple traces queue up and replay sequentially
- When paused at breakpoint: line stays highlighted, locals panel expands

**ReactFlow integration:**
- Register `codePanelNode` as a custom node type in ReactFlow
- When user clicks "Open Implementation" on an entity node, add a new node of type `codePanelNode` to the graph
- Position it to the right/below of the entity node
- Add an edge from entity node → code panel node (dashed style, distinct color)
- When closed, remove the node and edge

### 8. Frontend: Entity Node Modification

**Modify `visual-frontend/src/components/EntityNode.tsx`:**

Add an "Open Implementation" button/icon to entity nodes. Options:
- A small code icon (`</>`) in the corner of the entity node
- Clicking it sends `activate_code_debug` via WebSocket
- Only show for entities that aren't Probes or Sources (or show for all and let the backend return an error if source unavailable)

### 9. Frontend: GraphView Integration

**Modify `visual-frontend/src/components/GraphView.tsx`:**

- Register `codePanelNode` as a custom node type: `nodeTypes={{ entity: EntityNode, codePanel: CodePanelNode }}`
- When `code_debug_activated` is received, add a new node to the nodes array with type `codePanel`
- Add a dashed edge from entity → code panel
- When `code_debug_deactivated`, remove the node and edge

### 10. Frontend: ControlBar Integration

**Modify `visual-frontend/src/components/ControlBar.tsx`:**

When `codePausedEntity` is non-null, show breakpoint controls inline:

```
| ... existing controls ... | 🔴 Paused: Server:46 | [Step] [Over] [Out] [Continue] |
```

These send `code_step`, `code_step_over`, `code_step_out`, `code_continue` via WebSocket.

---

## Examples

### User Flow

1. Run `serve(sim, charts=[...])`, open browser
2. Graph view shows entity topology (Source → Server → Sink)
3. Click the `</>` icon on the "Server" entity node
4. A code panel node appears on the graph, connected to Server with a dashed edge
5. Code panel shows `handle_queued_event()` source with line numbers
6. Click line 45 gutter to set a breakpoint (red dot appears)
7. Click "Play" at speed 1 to start the simulation
8. When Server handles its first event, the code panel animates: line 42 highlights, then 43, then 44 (yield — generator pauses)
9. On next event handling, animation resumes: 45 → breakpoint hit!
10. Animation stops, line 45 stays highlighted amber, locals panel shows variables
11. Click "Continue" in the control bar — execution resumes past breakpoint
12. Close the code panel by clicking [x] — tracing deactivates, simulation returns to full speed
13. Open code panels for both Server and Router simultaneously

---

## Testing

### Unit Tests (`tests/unit/test_code_debugger.py`)

- Source retrieval for Entity subclass, QueuedResource subclass, dynamically defined class (expect None)
- Trace recording: verify LineRecords are captured correctly
- Breakpoint matching and blocking
- Variable serialization (primitives, nested dicts, non-serializable objects, depth limits)
- Deadman timeout behavior
- Thread safety of concurrent activate/deactivate
- drain_completed_traces returns and clears buffer

### Integration Tests (`tests/integration/test_code_stepping.py`)

- **Non-generator entity**: Create entity with direct-return `handle_event()`, activate code debugging, run event, verify trace captures all lines
- **Generator entity**: Create entity with multi-line generator, activate code debugging, run events, verify trace contains correct line sequence across multiple yields
- **Code breakpoints**: Set breakpoint, run event, verify execution pauses at correct line with correct locals
- **Step/continue from breakpoint**: Verify execution resumes and trace continues
- **QueuedResource indirection**: Verify source retrieval and tracing works for QueuedResource subclasses
- **Multiple active entities**: Verify independent traces for each (generator + non-generator mix)
- **CallbackEntity / Event.once()**: Verify tracing works for callback-based entities

### Manual E2E

- Run example with `serve()`, open code panel on graph, verify animation
- Set breakpoints, verify pause behavior
- Open multiple code panels, verify independent operation

---

## Alternatives Considered

### Browser-based Python interpreter (Pyodide/Skulpt)
Would need to replicate the entire simulation engine in the browser. Massive complexity, slow performance, doesn't work with existing architecture. Rejected.

### AST transformation (insert yields at every line)
Transform generator source to add yield points at every line. Extremely fragile, changes semantics, hard to maintain, doesn't work with complex control flow. Rejected.

### Debug Adapter Protocol (DAP)
Standard IDE debugging protocol. Designed for IDEs not browsers, would need a DAP client in the browser, overcomplicated for our use case. Rejected.

### Python's `sys.settrace()` only (no per-frame)
Global trace function would fire for ALL Python code, not just our generators. Too expensive and noisy. Using per-frame `gen.gi_frame.f_trace` is much more targeted. We set a minimal `sys.settrace` to enable per-frame tracing to propagate, but all real logic is in per-frame handlers.

### Separate "Code" view tab (original proposal)
A dedicated full-screen code view alongside Graph and Dashboard. Rejected in favor of code panels as graph nodes because: (a) keeps code visible alongside the topology, (b) supports multiple panels for different entities, (c) more intuitive -- the code "belongs to" the entity on the graph.

### Block on every line (original proposal)
Require manual stepping through every line. Rejected in favor of auto-animate with breakpoint pausing because: (a) much smoother UX for understanding flow, (b) no manual intervention needed unless breakpoint is hit, (c) frontend controls animation speed, no backend timing dependency.

---

## Implementation Plan

1. **Backend core**: `code_debugger.py` (trace recording + breakpoint blocking) + `event.py` instrumentation + `simulation.py` wiring
2. **API layer**: `bridge.py` methods + `server.py` endpoints and WS actions
3. **Frontend types/state**: `types.ts` + `useSimState.ts` + `useWebSocket.ts`
4. **Frontend components**: `CodePanelNode.tsx` (custom ReactFlow node) + `EntityNode.tsx` modifications + `GraphView.tsx` integration + `ControlBar.tsx` breakpoint controls
5. **Build and test**: Frontend build + unit tests + integration tests + manual E2E

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `happysimulator/visual/code_debugger.py` | **Create** - CodeDebugger, trace recording, breakpoint blocking |
| `happysimulator/core/event.py` | **Modify** - Add code debugger context, instrument ProcessContinuation.invoke() |
| `happysimulator/core/simulation.py` | **Modify** - Wire up code debugger context in run() |
| `happysimulator/visual/bridge.py` | **Modify** - Add code debugging methods, include traces in step results |
| `happysimulator/visual/server.py` | **Modify** - Add REST endpoints and WebSocket actions |
| `visual-frontend/src/types.ts` | **Modify** - Add CodeTrace, CodePausedState, EntitySource, CodePanelConfig |
| `visual-frontend/src/hooks/useSimState.ts` | **Modify** - Add code panel state and actions |
| `visual-frontend/src/hooks/useWebSocket.ts` | **Modify** - Handle code_trace, code_paused, code_debug_activated/deactivated |
| `visual-frontend/src/components/CodePanelNode.tsx` | **Create** - Custom ReactFlow node for code display |
| `visual-frontend/src/components/EntityNode.tsx` | **Modify** - Add "Open Implementation" button |
| `visual-frontend/src/components/GraphView.tsx` | **Modify** - Register codePanelNode type, manage code panel nodes/edges |
| `visual-frontend/src/components/ControlBar.tsx` | **Modify** - Add breakpoint step controls when paused |
| `tests/unit/test_code_debugger.py` | **Create** - Unit tests |
| `tests/integration/test_code_stepping.py` | **Create** - Integration tests |

---

## Future Enhancements (Deferred)

- Syntax highlighting (Prism.js or highlight.js)
- Expression evaluation in paused frame context
- Conditional code breakpoints
- Watch expressions
- Multiple concurrent generator tracking per entity
- Full file view (not just the method)
- Hot-reload awareness for modified source files
- Animation speed control on code panels
- Minimap within large code panels
