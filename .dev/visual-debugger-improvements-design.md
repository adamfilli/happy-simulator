# Visual Debugger UI Improvements Design

## Overview

The visual debugger has the right architecture (graph view + dashboard + inspector + event log) but every feature stops at "minimum viable." Charts have zero interactivity, the Debug button has no way to set breakpoints, events can't be filtered, and there's no timeline showing simulation progress. This document describes 8 improvements that transform it from a demo into a usable tool.

## Motivation

Users need interactive debugging capabilities to effectively inspect and understand simulation behavior. The current static charts, lack of breakpoint management, and unfiltered event logs make it difficult to find relevant information during debugging sessions.

## Requirements

1. **Chart Interactivity** — Hover crosshair/tooltip, click-drag zoom, double-click reset, current value dot
2. **Event Log Filtering** — Filter by event type, entity, text search, time range; auto-scroll toggle
3. **Data Export** — PNG/CSV export for charts, clipboard copy for event context
4. **Speed Control Fix** — Replace confusing batch-count speed with time-based (1x/10x/100x/Max)
5. **Breakpoint Management UI** — CRUD for breakpoints via REST + frontend panel
6. **Timeline/Scrubber Bar** — Progress bar with seek-to-time and breakpoint markers
7. **Graph Edge Animation** — Dynamic edge width/color based on throughput
8. **Entity State History** — Auto-record entity state snapshots, sparklines in inspector

## Design

### Phase 1: Frontend-Only

#### 1. Chart Interactivity (TimeSeriesChart.tsx)

- Track `mouseMove` on canvas, binary-search `times[]` for nearest point
- Draw vertical dashed line + tooltip box showing `t = X.XXs, value = Y.YY`
- Click-drag zoom: `mouseDown` record start, `mouseUp` compute time range
- Double-click reset: Clear `zoomRange` back to null
- Current value dot: Filled circle at last data point
- New state: `hoverX`, `zoomRange`, `dragStart`, `dragCurrent`

#### 3. Event Log Filtering (EventLog.tsx)

- Event type filter: Multi-select dropdown from unique types
- Entity filter: Text input substring matching
- Text search: Across JSON.stringify(context) + event_type + target_name
- Time range: Min/max number inputs
- Auto-scroll toggle: Checkbox to pause auto-scroll
- Clear filters button

#### 5. Data Export

- PNG: `canvas.toBlob()` → download via temporary `<a>` element
- CSV: `times[]/values[]` to `"time_s,value\n..."` → Blob → download
- Clipboard: `navigator.clipboard.writeText()` on event context
- Export utility functions in `visual-frontend/src/utils/export.ts`

### Phase 2: Full-Stack

#### 6. Speed Control Fix

- Frontend: Replace `SPEEDS = [1, 5, 10, 50, 100]` with `[1x, 10x, 100x, Max]`
- Backend: Time-based stepping using `time.monotonic()` for wall-clock measurement
- Max speed: Large batches (100 events) with minimal delay

#### 2. Breakpoint Management UI

- REST endpoints: GET/POST/DELETE `/api/breakpoints`
- Bridge methods: `list_breakpoints_json()`, `add_breakpoint_from_json()`, etc.
- Frontend panel: Table of breakpoints with add/delete, type-specific forms
- Enhanced `breakpoint_hit` message with description

#### 4. Timeline/Scrubber Bar

- Backend: Add `end_time_s` to `get_state()` response
- Frontend: Canvas bar with playhead, breakpoint markers, click-to-seek

### Phase 3: Infrastructure-Heavy

#### 8. Graph Edge Animation

- Backend: Track per-edge event flow counts and rates in `_on_event()`
- Frontend: Custom ReactFlow edge with dynamic strokeWidth and color

#### 9. Entity State History

- Backend: Auto-record entity state snapshots every 0.1 sim-seconds
- Frontend: Mini sparklines per metric in inspector panel
- New endpoint: `GET /api/entity_history`

## Implementation Order

```
1. Chart Interactivity      [Frontend, M]
2. Event Log Filtering      [Frontend, S]
3. Data Export              [Frontend, S]   — needs 1 for canvas ref
4. Speed Control Fix        [Full-stack, M]
5. Breakpoint Management    [Full-stack, L]
6. Timeline/Scrubber        [Full-stack, M] — benefits from 5
7. Entity State History     [Full-stack, L]
8. Graph Edge Animation     [Full-stack, L]
```
