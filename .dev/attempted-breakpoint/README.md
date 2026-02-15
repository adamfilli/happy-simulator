# Attempted Breakpoint Implementation: Status Report

Date: February 2026

## Executive Summary

Line-by-line code stepping with breakpoints was implemented for the visual debugger. The code panel display (showing Python source code with line numbers and current-line highlighting) works well and provides genuine value for understanding entity behavior. However, the breakpoint pause/resume and step/over/out controls proved unreliable due to subtle race conditions and UI state management issues. The breakpoint/stepping UI has been removed from the frontend pending redesign, but the code display foundation remains.

---

## What Was Built

### Backend

**CodeDebugger (`happysimulator/visual/code_debugger.py`)**: Python frame tracing engine that:
- Intercepts execution of entity `handle_event()` / `handle_queued_event()` methods using `sys.settrace()` and per-frame `f_trace`
- Records each line number and local variables as code executes
- Detects when execution reaches a breakpoint and blocks the simulation thread via `threading.Event`
- Sends recorded traces to the frontend for animation replay
- Handles both generator and non-generator entities seamlessly

**Server Integration (`happysimulator/visual/server.py`)**: WebSocket handlers for code debugging commands
- `activate_code_debug`: Enable tracing for an entity (open code panel)
- `code_continue`: Resume from breakpoint
- `code_step`, `code_step_over`, `code_step_out`: Stepping commands (partially implemented)

**Bridge Methods (`happysimulator/visual/bridge.py`)**: API layer
- `get_entity_source()`: Retrieve Python source code for an entity
- `activate_code_debug()` / `deactivate_code_debug()`: Manage which entities are traced
- Code breakpoint management: add/remove/list breakpoints
- Step execution: `step()` method extended to drain and return completed traces

### Frontend

**CodePanelNode.tsx**: Custom ReactFlow node that displays code on the graph
- Draggable and resizable code panel showing entity's source code
- Line numbers matching the source file (1-based)
- Current line highlighted with `>>` marker and amber background
- Breakpoint gutter with clickable circles (red when set, empty on hover)
- Animation replay of line-by-line execution from backend traces
- Locals panel (collapsible) for inspecting variables when paused
- Connected to entity nodes with dashed edges

**ControlBar.tsx**: Extended to show breakpoint step controls
- Step, Step Over, Step Out, Continue buttons when paused at a breakpoint
- Displays which entity and line number the debugger is paused at

---

## Architecture

### Backend Architecture

```
Event.invoke() / ProcessContinuation.invoke()
    ↓
Check if entity is active (has open code panel)
    ↓
Install trace handler via sys.settrace() or gen.gi_frame.f_trace
    ↓
Execute handle_event() / gen.send()
    ↓
[TRACE HANDLER LOOP]
    for each line executed:
      - Record (line_number, locals) to buffer
      - Check if line matches breakpoint
      - If breakpoint hit:
        - Create FrameSnapshot (locals, call stack, source)
        - Set _is_code_paused = True
        - Send "code_paused" message to frontend
        - Block sim thread: threading.Event.wait() (30s timeout)
        - Continue when Event is set (from code_continue command)
    ↓
After handle_event() / gen.send() completes:
    - Flush trace buffer to completed_traces
    - Remove trace handler
    ↓
bridge.step() drains completed_traces, includes as code_traces in response
    ↓
Frontend receives state_update with code_traces, animates line highlights
```

**Trace Installation Points:**
- `Event.invoke()`: Uses `sys.settrace()` to intercept all frames. Covers non-generator entities and the first yield of generators. Trace function filters by checking `frame.f_code` against the entity's method code object (cached during source retrieval).
- `ProcessContinuation.invoke()`: Uses `gen.gi_frame.f_trace` (per-frame) to trace generator resumptions. More targeted than global tracing.

### Frontend Architecture

```
Browser
├── GraphView: Displays entities as nodes, code panels as separate draggable nodes
├── CodePanelNode: Custom ReactFlow node showing source code
│   ├── Header: Entity name, method name, close button
│   ├── Gutter: Clickable line numbers, red dots for breakpoints
│   ├── Source: Monospace code with current line highlighted
│   └── Locals: Key-value variable inspection (when paused)
├── ControlBar: Shows step/continue buttons when paused
└── useSimState: Zustand store managing code panel state per entity
    ├── codePanels: Map<entity_name, CodePanelConfig>
    ├── codePausedEntity: Current paused entity (null if running)
    └── Actions: openCodePanel, closeCodePanel, setCodeTrace, setCodePaused

WebSocket Loop:
  Receives state_update → extract code_traces → feed to animation queue
  Receives code_paused → mark entity as paused, highlight line, show locals
  Receives code_debug_activated → add code panel node to graph
  Receives code_debug_deactivated → remove code panel node from graph
```

### Key Design Decisions

1. **No browser Python interpreter**: Uses native Python frame tracing, not Pyodide/Skulpt. Drastically simpler and integrates with existing simulation.

2. **Recording + Animation**: Trace handler records lines without blocking (fast). Frontend animates the replay at configurable speed. Breakpoints block only when hit.

3. **Per-entity activation**: Only entities with open code panels get traced. Performance cost only for active entities.

4. **Hybrid trace modes**: The same trace function records normally OR blocks at breakpoints, depending on line matched against breakpoint registry.

---

## Key Files and Implementation Details

### Backend Files

#### `happysimulator/visual/code_debugger.py` (915 lines, fully implemented)

**Core classes:**
- `CodeBreakpoint`: Immutable tuple of (entity_class_name, file_path, line_number)
- `CodeLocation`: File, line, function, entity name/class
- `FrameSnapshot`: Captured state when paused (locals, call stack, source)
- `LineRecord`: (line_number, locals_dict) for animation
- `ExecutionTrace`: Full trace of lines from one gen.send() call
- `CodeDebugger`: Main orchestrator

**Key methods:**
- `activate_entity(name)` / `deactivate_entity(name)`: Enable/disable tracing
- `get_entity_source(cls)`: Retrieve source via `inspect.getsourcelines()`, cache it
- `install_call_trace(name, cls)`: Install `sys.settrace()` handler
- `install_generator_trace(gen, name, cls)`: Install per-frame trace
- `remove_call_trace()` / `remove_generator_trace()`: Cleanup and flush
- `add_breakpoint()` / `remove_breakpoint()`: Manage breakpoint registry
- `code_continue()` / `code_step()` / `code_step_over()` / `code_step_out()`: Resume from breakpoint
- `_trace_handler()`: Per-line callback that records or blocks
- `drain_completed_traces()`: Pop traces for bridge delivery

**Serialization:**
- `_serialize_locals(locals_dict)`: Convert frame locals to JSON-safe dict
- `_serialize_value(v, depth)`: Recursive with depth limits, max 50 keys per dict

#### `happysimulator/core/event.py` (modified)

**Module-level context** (lines ~25-35):
```python
_active_code_debugger = None

def _set_active_code_debugger(debugger): ...
def _get_active_code_debugger(): ...
def _clear_active_code_debugger(): ...
```

**`Event.invoke()` instrumentation** (lines ~133-145):
- Check if entity is active in code debugger
- Install trace before `handle_event()` call
- Flush trace after call (finally block)

**`ProcessContinuation.invoke()` instrumentation** (lines ~307-330):
- Check if entity is active AND generator frame exists
- Install per-frame trace before `gen.send()`
- Flush trace after send (finally block)

#### `happysimulator/core/simulation.py` (modified)

In `run()` method (lines ~231-236):
```python
_set_active_code_debugger(self._code_debugger)
try:
    return self._run_loop()
finally:
    _clear_active_code_debugger()
```

#### `happysimulator/visual/bridge.py` (modified, ~60 new lines)

**New attributes:**
- `self._code_debugger = CodeDebugger()`
- `sim._code_debugger = self._code_debugger` (inject for use in event.py)

**New methods:**
- `get_entity_source(entity_name)` → source dict or error
- `activate_code_debug(entity_name)` → source dict + activate tracing
- `deactivate_code_debug(entity_name)` → deactivate tracing
- `set_code_breakpoint(class_name, file_path, line)` → breakpoint ID
- `remove_code_breakpoint(bp_id)` → success/error
- `get_code_debug_state()` → paused frame snapshot or None
- `code_step()`, `code_step_over()`, `code_step_out()`, `code_continue()` → resume from breakpoint

**Extended `step()` method**:
```python
def step(self, count=1):
    # ... existing logic ...
    result = { ... }

    # New: include code execution traces
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

#### `happysimulator/visual/server.py` (modified, WebSocket handlers)

**REST endpoints:**
- `GET /api/entity/{name}/source` → source code + metadata
- `POST /api/debug/code/activate` + `POST /api/debug/code/deactivate`
- `POST /api/debug/code/breakpoints` + `DELETE /api/debug/code/breakpoints/{id}`

**WebSocket actions:**
- `activate_code_debug` → `bridge.activate_code_debug()` → send `code_debug_activated`
- `deactivate_code_debug` → `bridge.deactivate_code_debug()` → send `code_debug_deactivated`
- `code_continue`, `code_step`, `code_step_over`, `code_step_out` → call bridge methods

**WebSocket messages sent:**
- `code_debug_activated`: { entity_name, class_name, source, start_line, file_path, line_count }
- `code_debug_deactivated`: { entity_name }
- `code_paused`: { line_number, locals, source, source_start_line, entity_name, class_name, call_stack }
- `state_update` extended: include `code_traces` field when code panels are active

### Frontend Files

#### `visual-frontend/src/components/CodePanelNode.tsx` (280 lines)

**Layout:**
```
┌────────────────────────────────────────┐
│ MM1Server.handle_queued_event() [x] ≡  │  Header + close + drag
├────────────────────────────────────────┤
│  # │ def handle_queued_event(...):      │
│  43│     service_time = ...             │  Source display
│  >> │ 44│     yield service_time        │  Current line highlighted
│  45│     self.stats += 1                │
├────────────────────────────────────────┤
│ Locals: self, service_time, ...        │  Locals panel (when paused)
└────────────────────────────────────────┘
```

**Features:**
- Header: Entity class name + method name, close button, drag handle
- Breakpoint gutter: Click to toggle red dot (● for set, ○ for empty on hover)
- Line numbers: Monospace, 1-based matching file
- Current line marker: `>>` + amber background
- Source pane: `<pre>` with monospace font, scrollable, syntax highlighting placeholder
- Locals panel: Collapsible, displays key-value pairs when paused at breakpoint
- Resizable: Uses ReactFlow's built-in `NodeResizer` (initially, later causes issues)
- Animation: Replays `code_trace` lines at ~100ms intervals, smoothly highlights each line

**State management:**
- Breakpoints stored in Zustand store as `Set<number>` (line numbers)
- Current line from animation or pause state
- Locals from `CodePausedState` when paused

#### `visual-frontend/src/hooks/useSimState.ts` (extended)

**New state fields:**
```typescript
codePanels: Map<string, CodePanelConfig>;     // one per open code panel
codePausedEntity: string | null;               // entity paused at breakpoint
```

**New actions:**
- `openCodePanel(entityName, source)`: Add panel to map, create graph node
- `closeCodePanel(entityName)`: Remove from map, remove graph node
- `setCodeTrace(entityName, trace)`: Queue animation for panel
- `setCodePaused(state)`: Pause at breakpoint, update locals
- `toggleCodeBreakpoint(entityName, line)`: Add/remove line from breakpoint set
- `clearCodePaused()`: Resume execution

#### `visual-frontend/src/components/EntityNode.tsx` (modified)

Added `</>` button to open code panel on entity nodes (sends `activate_code_debug` WS message)

#### `visual-frontend/src/components/GraphView.tsx` (modified)

- Register `codePanelNode` custom node type in `nodeTypes`
- On `code_debug_activated`: Add node + dashed edge to graph
- On `code_debug_deactivated`: Remove node + edge

#### `visual-frontend/src/components/ControlBar.tsx` (modified)

Added conditional rendering:
```typescript
{codePausedEntity && (
  <div>
    Paused at {codePausedEntity}:{line}
    <button onClick={() => send('code_step')}>Step</button>
    <button onClick={() => send('code_step_over')}>Over</button>
    <button onClick={() => send('code_step_out')}>Out</button>
    <button onClick={() => send('code_continue')}>Continue</button>
  </div>
)}
```

---

## Issues Found

### 1. Deadlock: WebSocket Blocked During Breakpoint

**Problem:** When a breakpoint hit and the sim thread called `threading.Event.wait()`, the WebSocket receive loop was also blocked in `asyncio.to_thread(bridge.step(), ...)`. The frontend couldn't send `code_continue` because the receive loop was blocked waiting for the step to complete.

**Attempted Fix:** Extract `_step_with_code_debug()` polling loop that:
- Polls `sim.control.peek_next()` in a loop
- Reads WebSocket messages inline while waiting
- Processes `code_continue`/`code_step` commands to signal the blocked trace handler

**Partial Success**: The polling avoided the receive-loop block, but introduced new race conditions (see below).

### 2. Race Conditions: Frontend State Clearing vs. Backend Resume

**Problem:** Frontend would clear pause state (`codePausedEntity = null`, clear `breakpoint` marker) on receiving `state_update` after `code_continue`, but timing was unpredictable:
- If frontend cleared before backend confirmed pause was released: visual glitch
- If backend sent `state_update` with traces while paused: animation started while UI still showed "paused"
- Multiple `state_update` messages arriving during pause could race with `code_paused` message

**Manifestation**: Code panel would flicker between paused/running states, locals panel would disappear then reappear, or stepping would "skip" lines.

**Root cause**: No explicit synchronization on pause state transitions. Paused state lived in two places with no single source of truth:
- Backend: `_is_code_paused` flag in CodeDebugger
- Frontend: `codePausedEntity` in Zustand store

When state transitioned from paused → running, messages could arrive out of order depending on WebSocket timing and thread scheduling.

### 3. NodeResizer Infinite Render Loops

**Problem:** The `NodeResizer` component from `@xyflow/react` (for draggable resize handles) caused infinite render loops when combined with the animation state updates. Every time a new line was highlighted, the component would re-measure, triggering a re-render, which would update the animation state again.

**Fix**: Removed `NodeResizer` from CodePanelNode. Users can still drag code panels (all ReactFlow nodes are draggable by default), but cannot resize via corner handles.

### 4. Breakpoint Selector Inefficiency (Fixed)

**Problem:** Code that created a new array reference on every store update:
```typescript
breakpoints: entity.breakpoints.filter(bp => ...)  // new array, always
```

This caused unnecessary re-renders even when breakpoints didn't change.

**Fix**: Used `useShallow` hook from Zustand to do shallow equality on breakpoint Set:
```typescript
const breakpoints = store((s) => s.codePanels.get(entityName)?.breakpoints, useShallow);
```

### 5. Step Over / Step Out Not Fully Implemented

The backend code for `code_step_over()` and `code_step_out()` was stubbed but incomplete. The logic to track call depth and skip intermediate frames was not finished. Only `code_step()` (step one line) was reliable.

### 6. Deserialization of Complex Objects Failed

When locals contained circular references or non-JSON-serializable objects, the serialization would silently fail or create misleading `<ClassName>` placeholders. Users couldn't inspect the actual structure of complex objects like Event or Entity instances.

---

## Why Breakpoint UI Was Removed

The breakpoint stepping UI (Step, Over, Out, Continue buttons in ControlBar) was removed because:

1. **Unreliability**: Race conditions made it unpredictable whether pause/resume would work correctly
2. **Limited value**: The animation-based line-by-line display (without manual stepping) already showed most entity behavior
3. **Engineering debt**: Fixing the race conditions would require major refactoring of pause state management (move to single source of truth on backend)
4. **User experience**: Users would encounter hangs, stutters, and state inconsistency

However, the **code display foundation** (CodePanelNode, source retrieval, line animation) remains and is valuable on its own.

---

## What's Kept: Code Panel Display

The code panel display without breakpoint stepping provides genuine utility:

✅ **Open code panel on entity**: Click `</>` button on entity node
✅ **Auto-animate through lines**: Watch execution flow through code as events are processed
✅ **No manual stepping needed**: Just click Play and observe
✅ **Multiple panels**: Keep code for different entities visible simultaneously
✅ **Resizable and draggable**: Arrange panels on the graph as desired

**Use case**: Understanding complex multi-step entity behavior without breakpoints. Developers can see:
- Which lines execute in sequence
- Which branches are taken
- Approximate execution time (via animation speed)
- Local variables at key points (with future locals-at-line UI)

---

## What Could Be Redesigned

### Option 1: Simplified Breakpoints (future)

Instead of step/over/out with race conditions, use simple breakpoint count:
- "Pause after N breakpoint hits" rather than per-line stepping
- No mid-execution pause/resume from frontend
- Simpler threading model, no blocking event

### Option 2: Separate Debug Tab

Move code debugging to a dedicated Debug tab (not on graph). Would eliminate:
- Graph layout complexity (code panels are large)
- Race conditions (separate async flow)
- But loses the visual connection between code and entity topology

### Option 3: IDE Integration

Instead of in-browser stepping, integrate with VSCode's Debug Adapter Protocol. Would require:
- Separate debugger subprocess
- More complex multi-process coordination
- But gains all IDE debugging features for free

---

## Implementation Lessons Learned

1. **Thread synchronization is hard**: Blocking a simulation thread while keeping an async WebSocket loop responsive requires careful coordination. Simple flags aren't enough; need explicit state machine.

2. **Frontend/backend state split is error-prone**: Paused state living in two places (backend flag, frontend Zustand store) caused race conditions. Single source of truth (probably on backend) needed.

3. **Animation state management**: Interleaving pause state with animation state was confusing. Should separate:
   - "Is paused at breakpoint?" (pause state)
   - "Which line is highlighted?" (display state)

4. **ReactFlow node complexity**: Custom nodes with complex internal state (resizing, animation, locals panel) interact poorly with drag/drop and graph layout. Simpler components work better.

5. **Trace recording works well**: The core idea of `sys.settrace()` recording lines is solid and performs fine. The problem wasn't trace collection but pause/resume coordination.

---

## Files Modified

| File | Changes |
|------|---------|
| `happysimulator/visual/code_debugger.py` | **Created** - Full CodeDebugger implementation |
| `happysimulator/core/event.py` | **Modified** - Instrument Event.invoke() and ProcessContinuation.invoke() |
| `happysimulator/core/simulation.py` | **Modified** - Wire code debugger context |
| `happysimulator/visual/bridge.py` | **Modified** - Add code debug methods, extend step() |
| `happysimulator/visual/server.py` | **Modified** - Add WS actions for code debugging |
| `visual-frontend/src/types.ts` | **Modified** - Add CodeTrace, CodePausedState types |
| `visual-frontend/src/hooks/useSimState.ts` | **Modified** - Add codePanels, codePausedEntity state |
| `visual-frontend/src/hooks/useWebSocket.ts` | **Modified** - Handle code_paused, code_traces messages |
| `visual-frontend/src/components/CodePanelNode.tsx` | **Created** - Code panel React component |
| `visual-frontend/src/components/CodePanelContext.ts` | **Created** - Context for code debugging |
| `visual-frontend/src/components/EntityNode.tsx` | **Modified** - Add open code button |
| `visual-frontend/src/components/GraphView.tsx` | **Modified** - Register codePanelNode type |
| `visual-frontend/src/components/ControlBar.tsx` | **Modified** - Add breakpoint step controls |

---

## Future Work

1. **Fix pause/resume**: Redesign state machine for breakpoint pausing with explicit synchronization
2. **Complete step over/out**: Implement call depth tracking for stepping over function calls
3. **Better locals inspection**: Improve serialization of complex objects, add "click to expand" UI
4. **Line-at-breakpoint locals**: Show locals specifically at breakpoint line, not from step history
5. **Conditional breakpoints**: Add UI for condition expression evaluation
6. **Watch expressions**: Allow users to type expressions to evaluate in paused frame context
7. **Syntax highlighting**: Add Prism.js or highlight.js for code readability
8. **Performance profiling**: Show which lines take most time (requires trace timestamps)

---

## Conclusion

The code stepping implementation successfully demonstrated that Python's native frame tracing can provide browser-based code-level debugging. The code display feature (showing source, line numbers, animation) works reliably and adds value.

The breakpoint stepping (pause, resume, step) proved more challenging due to thread synchronization complexity. Rather than ship an unreliable implementation, the stepping UI was removed while preserving the solid foundation.

**Recommendation**: Rebuild the stepping feature with a clearer pause-state model and proper async coordination (possibly moving pause state entirely to the backend) before attempting to ship it.
