# Entity State History: Plan

## Context

The visual debugger inspector shows entity metrics as static key-value pairs. This feature adds time-series history tracking so users can see how metrics evolved over time via mini sparklines next to each numeric metric. Part of PR #65 (visual debugger improvements).

## Changes

### 1. Backend: Auto-snapshot in SimulationBridge (`happysimulator/visual/bridge.py`)

In `__init__`, add:
- `_entity_history: dict[str, list[tuple[float, dict]]]` — per-entity list of (time_s, serialized_state) snapshots
- `_last_snapshot_time: float = -1.0` — tracks last snapshot time
- `MAX_HISTORY_SAMPLES = 10_000` — cap per entity

In `_on_event()`, after recording the event, check if `event.time_s - _last_snapshot_time >= 0.1`. If so, iterate all entities, call `serialize_entity()`, and append `(time_s, state_dict)` to `_entity_history[name]`. When any entity's list exceeds 10K, downsample by keeping every other sample (halving).

In `reset()`, clear `_entity_history` and reset `_last_snapshot_time`.

Add `get_entity_history(entity_name: str) -> dict`:
- Look up `_entity_history[entity_name]`
- Extract each numeric metric into `{ metric_name: { times: [...], values: [...] } }`
- Return `{ entity: name, metrics: { ... } }`

### 2. Backend: REST endpoint (`happysimulator/visual/server.py`)

Add `GET /api/entity_history?entity=NAME` → calls `bridge.get_entity_history(entity)`, returns JSONResponse.

### 3. Frontend: MiniSparkline component (`visual-frontend/src/components/MiniSparkline.tsx`)

New file. A minimal 40px-tall canvas sparkline:
- Props: `values: number[]`, `color?: string` (default `#3b82f6`)
- No axes, labels, ticks, or interaction — just a line + subtle area fill
- Auto-scales Y to data range, X fills available width
- Uses ResizeObserver + devicePixelRatio for crisp rendering

### 4. Frontend: InspectorPanel update (`visual-frontend/src/components/InspectorPanel.tsx`)

- When an entity is selected and state updates, fetch `GET /api/entity_history?entity=NAME`
- Store history in local state, keyed by entity name + events_processed (avoid redundant fetches, same pattern as existing probe fetch)
- In the metrics list, for each numeric metric that has history data, render a `<MiniSparkline>` below the key-value row
- Layout: metric row stays `flex justify-between`, sparkline renders in a 40px-tall div below it spanning full width

## Files Modified

| File | Change |
|------|--------|
| `happysimulator/visual/bridge.py` | Add history tracking, snapshot logic, `get_entity_history()` |
| `happysimulator/visual/server.py` | Add `/api/entity_history` endpoint |
| `visual-frontend/src/components/MiniSparkline.tsx` | New file: minimal canvas sparkline |
| `visual-frontend/src/components/InspectorPanel.tsx` | Fetch history, render sparklines |

## Verification

1. `pytest tests/ -q` — ensure no regressions
2. `cd visual-frontend && npx tsc --noEmit` — TypeScript compiles
3. `cd visual-frontend && npm run build` — Vite build succeeds
4. Run `python examples/visual_debugger.py`, play sim, click entity in graph, verify sparklines appear next to numeric metrics and update as simulation progresses
