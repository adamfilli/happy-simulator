# Design: Line-by-Line Code Stepping in Visual Debugger

## Overview

Add the ability to step through Python code line-by-line within entity `handle_event()` generators, directly in the browser-based visual debugger. Users can select an entity, see its source code, set breakpoints on specific lines, and step through execution like a real debugger.

## Motivation

The visual debugger currently supports **event-level** stepping -- you can step through events one at a time, set breakpoints on time/event count/metrics, and inspect entity state. However, there's no way to see *what code is executing* inside an entity's generator between yields. For complex entity logic, this makes debugging opaque -- you see the inputs and outputs of each yield but not the decisions happening in between.

## Requirements

1. **Source display**: Show the Python source code of an entity's `handle_event()` / `handle_queued_event()` in the browser
2. **Line-level breakpoints**: Click line numbers to set/remove breakpoints
3. **Step controls**: Step (next line), Step Over, Step Out, Continue
4. **Variable inspection**: Display local variables at the current paused line
5. **Call stack**: Show the current frame chain
6. **Per-entity activation**: Only trace entities the user explicitly selects (performance)
7. **Integration**: Works alongside existing event-level stepping and breakpoints

## Design

### Key Insight: Python Frame Tracing

No browser-based Python interpreter is needed. Python's native frame tracing mechanism handles everything:

- `gen.gi_frame` - Generator objects expose their execution frame
- `frame.f_trace` - Per-frame trace function fires on each line
- `frame.f_lineno` - Current line number
- `frame.f_locals` - Local variables at current line
- `inspect.getsourcelines()` - Retrieve source code with line numbers

When `ProcessContinuation.invoke()` calls `gen.send(value)`, the generator executes Python lines between yields. With a trace function installed on the generator's frame, we intercept each line.

### Architecture

```
Browser (React)                           Backend (Python)

CodeView component                        CodeDebugger
  |                                           |
  |-- WS: "code_step" ------------------->   |
  |                                           |-- threading.Event.set()
  |                                           |   (unblocks trace fn in sim thread)
  |                                           |-- trace fn fires on next line
  |                                           |-- captures frame (line, locals)
  |                                           |-- threading.Event.clear() (blocks)
  |<-- WS: "code_paused" -----------------   |
  |   { line, locals, source, stack }         |
```

The trace function runs **inside the simulation thread** (same thread as `gen.send()`). It blocks via `threading.Event` when paused, and the async WebSocket handler signals it to continue. This is the same threading model already used by pause/resume.

### Threading Model

1. User clicks "Step" (event-level) in the browser
2. `bridge.step()` calls `sim.control.step(1)` via `asyncio.to_thread` (sim thread)
3. During `ProcessContinuation.invoke()`, the trace function pauses the sim thread at a line
4. `asyncio.to_thread(bridge.step)` is blocked (sim thread is paused in trace fn)
5. WebSocket handler is still alive (async, different context)
6. User clicks "Step Line" in the code view
7. WebSocket handler signals `threading.Event`, sim thread resumes
8. Trace function fires on next line, blocks again
9. Eventually generator yields/returns, `bridge.step()` completes

Event-level step must run as a background task when code debugging is active so the WebSocket can still receive code step commands.

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
    location: CodeLocation
    locals: dict[str, Any]        # serialized local variables
    source_lines: list[str]       # full method source
    source_start_line: int        # 1-based file line of first source line
    call_stack: list[CodeLocation]
```

**CodeDebugger class:**

```python
class CodeDebugger:
    DEADMAN_TIMEOUT_S = 30.0  # safety valve

    def __init__(self):
        self._active_entities: set[str] = set()       # entity names
        self._breakpoints: dict[str, CodeBreakpoint] = {}
        self._source_cache: dict[type, tuple] = {}
        self._paused_frame: FrameSnapshot | None = None
        self._is_code_paused: bool = False
        self._continue_event = threading.Event()
        self._step_mode: str = "none"  # line|over|out|continue|none
        self._current_depth: int = 0
        self._lock = threading.Lock()
```

**Key methods:**

| Method | Thread | Purpose |
|--------|--------|---------|
| `activate_entity(name)` | API | Start code debugging for entity |
| `deactivate_entity(name)` | API | Stop code debugging |
| `get_entity_source(cls)` | API | `inspect.getsourcelines()` with caching |
| `add_breakpoint(bp)` | API | Register code breakpoint |
| `code_step()` | API | Signal trace fn to advance one line |
| `code_step_over()` | API | Advance to next line at same/shallower depth |
| `code_step_out()` | API | Run until call depth decreases |
| `code_continue()` | API | Run until code breakpoint or yield |
| `install_trace(gen, name, cls)` | Sim | Set `gen.gi_frame.f_trace` before `gen.send()` |
| `remove_trace(gen)` | Sim | Clear trace after `gen.send()` |
| `_trace_handler(frame, event, arg)` | Sim | Per-line handler; decides pause, captures state |
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

**Trace function logic:**

```python
def _trace_handler(self, frame, event, arg, *, entity_name, file_path, ...):
    if event == "call": self._current_depth += 1; return nested_trace
    if event == "return": self._current_depth -= 1; return None
    if event != "line": return nested_trace

    # Decide whether to pause based on step mode
    should_pause = False
    if mode == "line": should_pause = True
    elif mode == "over": should_pause = depth <= step_over_depth
    elif mode == "out": should_pause = depth <= step_out_depth
    elif mode == "continue": should_pause = check_code_breakpoints(line)
    elif mode == "none": should_pause = True  # first line after install

    if should_pause:
        snapshot = capture_frame(frame, ...)
        self._paused_frame = snapshot
        self._is_code_paused = True
        self._continue_event.clear()
        # BLOCK sim thread until step command arrives
        self._continue_event.wait(timeout=DEADMAN_TIMEOUT_S)

    return nested_trace
```

### 2. Modify: `happysimulator/core/event.py`

**Add module-level context** (following `_active_heap`/`_active_clock` pattern):

```python
_active_code_debugger = None

def _set_active_code_debugger(debugger): ...
def _get_active_code_debugger(): ...
def _clear_active_code_debugger(): ...
```

**Instrument `ProcessContinuation.invoke()`** (around line 307):

```python
def invoke(self):
    code_debugger = _get_active_code_debugger()
    entity = self.target
    entity_name = getattr(entity, "name", None)

    # Resolve QueuedResource indirection
    if hasattr(entity, '_resource'):
        entity = entity._resource
        entity_name = getattr(entity, "name", entity_name)

    should_trace = (
        code_debugger is not None
        and entity_name is not None
        and code_debugger.is_entity_active(entity_name)
        and self.process.gi_frame is not None
    )

    if should_trace:
        code_debugger.install_trace(self.process, entity_name, type(entity))

    try:
        yielded_val = self.process.send(self._send_value)
        # ... rest unchanged ...
    finally:
        if should_trace:
            code_debugger.remove_trace(self.process)
```

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
- `activate_code_debug(entity_name)` -> source info dict
- `deactivate_code_debug(entity_name)`
- `set_code_breakpoint(class_name, file_path, line)` -> breakpoint ID
- `remove_code_breakpoint(bp_id)`
- `get_code_debug_state()` -> frame snapshot dict or None
- `code_step()`, `code_step_over()`, `code_step_out()`, `code_continue()`
- `_find_entity(name)` helper

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
| `code_step` | - | `code_paused` or `state_update` |
| `code_step_over` | - | `code_paused` or `state_update` |
| `code_step_out` | - | `code_paused` or `state_update` |
| `code_continue` | - | `code_paused` or `state_update` |
| `activate_code_debug` | `entity_name` | `code_debug_activated` |
| `deactivate_code_debug` | `entity_name` | `code_debug_deactivated` |

**New WebSocket messages (backend -> frontend):**

```json
{"type": "code_paused", "line_number": 46, "locals": {...}, "source": "...",
 "source_start_line": 42, "entity_name": "Server", "class_name": "MM1Server",
 "call_stack": [{"file": "...", "line": 46, "function": "handle_queued_event"}]}

{"type": "code_debug_activated", "entity_name": "Server", "class_name": "MM1Server",
 "source": "...", "start_line": 42, "file_path": "...", "line_count": 15}

{"type": "code_debug_deactivated", "entity_name": "Server"}
```

**Concurrent blocking fix:** When code debugging is active, run event-level step as a background asyncio task so the WebSocket handler remains responsive to code step commands.

### 6. Frontend: Types and State

**`visual-frontend/src/types.ts`** additions:

```typescript
interface CodeDebugState {
  file_path: string;
  line_number: number;
  function_name: string;
  entity_name: string;
  class_name: string;
  locals: Record<string, unknown>;
  source: string;
  source_start_line: number;
  call_stack: Array<{file: string; line: number; function: string}>;
}

interface EntitySource {
  entity_name: string;
  class_name: string;
  file_path: string;
  start_line: number;
  source: string;
  line_count: number;
  error?: string;
}
```

**`visual-frontend/src/hooks/useSimState.ts`** additions:

```typescript
// New state fields
codeDebugEntity: string | null;
codeDebugState: CodeDebugState | null;
entitySource: EntitySource | null;
codeBreakpoints: Array<{id: string; line: number}>;

// New actions
setCodeDebugEntity, setCodeDebugState, setEntitySource,
addCodeBreakpoint, removeCodeBreakpoint
```

**`visual-frontend/src/hooks/useWebSocket.ts`** - Handle `code_paused`, `code_debug_activated`, `code_debug_deactivated`.

### 7. Frontend: CodeView Component

**New file: `visual-frontend/src/components/CodeView.tsx`**

New main view (third tab alongside Graph/Dashboard):

```
+------------------------------------------------------------------+
| Code Debug: MM1Server.handle_queued_event()          [Deactivate] |
+------------------------------------------------------------------+
| [Step Line] [Step Over] [Step Out] [Continue]    Entity: Server   |
+-------------------------------------------+----------------------+
| BP |   | Source Code                       | Variables            |
|    | 42| def handle_queued_event(self, ev): | self: <MM1Server>    |
|    | 43|     service_time = random.expo...  | event: <Event>       |
| *  | 44|     yield service_time             | service_time: 0.153  |
|    | 45|                                    |                      |
|  > | 46|     self.stats_processed += 1  <<< | stats_processed: 7   |
|    | 47|                                    |                      |
|    | 48|     if self.downstream is None:    | Call Stack            |
|    | 49|         return []                  | -------------------- |
|    | 50|                                    | > handle_queued_event |
|    | 51|     completed = Event(             |   line 46            |
+-------------------------------------------+----------------------+
```

**Components:**
- Breakpoint gutter (clickable red circles)
- Line numbers with current line indicator
- Source pane (monospace `<pre>`, current line highlighted amber)
- Variables panel (serialized locals as key-value pairs)
- Call stack panel
- Toolbar with step controls + entity info + deactivate button
- Auto-scroll to keep current line visible

### 8. Frontend: Integration

**`App.tsx`:**
- Add "Code" as third main view tab
- Auto-switch to Code view when `codeDebugEntity` is set
- Render `<CodeView />` when `activeView === "code"`

**`InspectorPanel.tsx`:**
- Add "Debug Code" button for non-Probe, non-Source entities

**`ControlBar.tsx`:**
- Show code-level step buttons (purple accent) when `codeDebugState` is non-null

---

## Examples

### User Flow

1. Run `serve(sim, charts=[...])`, open browser
2. Graph view shows entity topology
3. Click on "Server" entity in graph
4. Inspector panel shows server state + "Debug Code" button
5. Click "Debug Code"
6. View switches to Code view showing `handle_queued_event()` source
7. Click line 44 gutter to set breakpoint
8. Click "Step" (event level) to process next event
9. If the Server handles an event, execution pauses at line 44
10. Variables panel shows locals: `self`, `event`, `service_time`
11. Click "Step Line" -- execution moves to line 45
12. Click "Continue" -- runs to next breakpoint or yield
13. Generator yields -- event-level step completes, state_update sent
14. Click "Deactivate" to return to normal mode

---

## Testing

### Unit Tests (`tests/unit/test_code_debugger.py`)

- Source retrieval for Entity subclass, QueuedResource subclass, dynamically defined class (expect None)
- Trace installation/removal on generators
- Step modes: line, over, out, continue
- Code breakpoint matching
- Variable serialization (primitives, nested dicts, non-serializable objects, depth limits)
- Deadman timeout behavior
- Thread safety of concurrent activate/deactivate

### Integration Tests (`tests/integration/test_code_stepping.py`)

- Create entity with multi-line generator, activate code debugging, step through and verify line numbers + locals at each pause
- Code breakpoints: set breakpoint, run to it, verify pause location
- Interaction with event-level stepping: ensure code stepping and event stepping work together
- QueuedResource indirection: verify source retrieval works for QueuedResource subclasses
- Multiple generators: entity with concurrency > 1

### Manual E2E

- Run example with `serve()`, activate code debugging, step through, verify in browser

---

## Alternatives Considered

### Browser-based Python interpreter (Pyodide/Skulpt)
Would need to replicate the entire simulation engine in the browser. Massive complexity, slow performance, doesn't work with existing architecture. Rejected.

### AST transformation (insert yields at every line)
Transform generator source to add yield points at every line. Extremely fragile, changes semantics, hard to maintain, doesn't work with complex control flow. Rejected.

### Debug Adapter Protocol (DAP)
Standard IDE debugging protocol. Designed for IDEs not browsers, would need a DAP client in the browser, overcomplicated for our use case. Rejected.

### Python's `sys.settrace()` only (no per-frame)
Global trace function would fire for ALL Python code, not just our generators. Too expensive and noisy. Using per-frame `gen.gi_frame.f_trace` is much more targeted. We do set a minimal `sys.settrace` to enable per-frame tracing to propagate, but all real logic is in per-frame handlers.

---

## Implementation Plan

1. **Backend core**: `code_debugger.py` + `event.py` changes + `simulation.py` wiring
2. **API layer**: `bridge.py` + `server.py` extensions
3. **Frontend**: Types, store, WebSocket, CodeView component, integration
4. **Testing**: Unit + integration + manual E2E
5. **Build**: Frontend build into static/

---

## Future Enhancements (Deferred)

- Syntax highlighting (Prism.js or CodeMirror)
- Expression evaluation in paused frame context
- Conditional code breakpoints
- Watch expressions
- Multiple concurrent generator views
- Full file view (not just method)
- Hot-reload awareness for modified source files
