# SQLite-Backed Storage for Visual Debugger

## Context

The visual debugger stores all event history, sim logs, and time-series data in-memory. For large simulations, this creates three problems:

1. **Unbounded memory growth**: `Data._samples` (used by every Probe, LatencyTracker, ThroughputTracker) is an unbounded `list[tuple[float, Any]]` that grows linearly with simulation events. 10M samples = ~600MB.
2. **Lost history**: Event log and sim logs are capped at 5000 entries (deque maxlen). Older data is permanently discarded -- you can't scroll back to see what happened at t=0 after 5000+ events.
3. **No server-side query optimization**: Time-range filtering (`Data.between()`) is O(n) linear scan. Bucketing (`Data.bucket()`) is O(n log n) and recomputed from scratch on every API request. No downsampling for chart rendering.

The frontend compounds this: event/log ring buffers hold only 2000 entries, all filtering is client-side, and charts render every returned data point to canvas with no point-count limits.

## Current Architecture & Bottlenecks

### Backend Memory Model

| Component | Storage | Capacity | Memory Risk |
|-----------|---------|----------|-------------|
| `Data._samples` | `list[tuple[float, Any]]` | **Unbounded** | **HIGH** -- 10M samples = ~600MB |
| `_event_log` | `deque[RecordedEvent]` | 5,000 (maxlen) | Low -- but loses all older history |
| `_log_buffer` | `deque[RecordedLog]` | 5,000 (maxlen) | Low -- but loses all older history |
| `EventHeap._heap` | `list[Event]` | Unbounded (transient) | Medium -- proportional to pending work |
| `FIFOQueue._queue` | `deque[Event]` | `float('inf')` default | Medium -- user-controllable |

### API Bottlenecks

| Endpoint | Issue |
|----------|-------|
| `GET /api/events` | No pagination, returns full slice from bounded deque |
| `GET /api/timeseries` | Returns ALL raw data points, no downsampling |
| `GET /api/chart_data` | Bucketing recomputed from scratch every request (O(n log n)) |
| WebSocket `state_update` | Full entity state serialization on every message (no diffing) |
| No `/api/logs` endpoint | Logs only available via WebSocket drain, no historical browsing |

### Frontend Bottlenecks

| Component | Issue |
|-----------|-------|
| Event log | Ring buffer of 2000 entries, displays last 200 as DOM nodes (no virtualization) |
| Sim logs | Ring buffer of 2000 entries, displays last 200 as DOM nodes |
| Charts | Fetch all data points from backend, render every point to canvas |
| Filtering | All client-side (event type, entity, text search, time range) |

## Approach: SQLite via `VisualStore`

Add a `VisualStore` class in `happysimulator/visual/store.py` that owns a SQLite database. The bridge drains in-memory buffers into SQLite periodically. API endpoints read from SQLite for paginated/filtered queries. The core `Data` class is NOT modified -- it remains pure in-memory for all non-visual consumers.

**Why SQLite**: Zero-config (stdlib `sqlite3`), no external service, WAL mode supports concurrent read/write, efficient indexed range queries, handles millions of rows easily.

## Database Schema

```sql
-- Full event history (replaces bounded deque)
CREATE TABLE events (
    event_id    INTEGER PRIMARY KEY,
    time_s      REAL NOT NULL,
    event_type  TEXT NOT NULL,
    target_name TEXT NOT NULL,
    source_name TEXT,
    is_internal INTEGER NOT NULL DEFAULT 0,
    context     TEXT  -- JSON string
);
CREATE INDEX idx_events_time ON events (time_s);
CREATE INDEX idx_events_type ON events (event_type);

-- Full sim log history (replaces bounded deque)
CREATE TABLE logs (
    log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    time_s      REAL,
    wall_time   TEXT NOT NULL,
    level       TEXT NOT NULL,
    logger_name TEXT NOT NULL,
    message     TEXT NOT NULL
);
CREATE INDEX idx_logs_time ON logs (time_s);
CREATE INDEX idx_logs_level ON logs (level);

-- Time-series samples (mirrors Data._samples, queryable)
CREATE TABLE timeseries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_name TEXT NOT NULL,
    time_s      REAL NOT NULL,
    value       REAL NOT NULL
);
CREATE INDEX idx_ts_series_time ON timeseries (series_name, time_s);
```

## Implementation Plan

### Phase 1: `VisualStore` + drain thread (backend)

**New file: `happysimulator/visual/store.py`** (~300 lines)

`VisualStore` class with:
- SQLite connection with WAL mode, `synchronous=NORMAL`, thread-local connections
- `insert_events(batch)`, `insert_logs(batch)`, `insert_timeseries(series, batch)` -- all use `executemany()`
- `query_events(cursor, limit, direction, time_min, time_max, event_types, entity_filter, search, exclude_internal)` -- returns `(results, next_cursor)`
- `query_logs(cursor, limit, direction, time_min, time_max, min_level, search)` -- returns `(results, next_cursor)`
- `query_timeseries(series, start_s, end_s, max_points)` -- with nth-point downsampling
- `query_timeseries_bucketed(series, window_s, start_s, end_s, aggregate)` -- SQL `GROUP BY FLOOR(time_s/window)` for mean/max/count/rate; Python fallback for p50/p99
- `count_events(**filters)`, `clear()`, `close()`

**Modified: `happysimulator/visual/bridge.py`** (~60 lines added)

- Add `_store: VisualStore` to `__init__`, created with temp file path by default
- Add `_pending_events`, `_pending_logs`, `_pending_timeseries` drain buffers
- Add drain thread (daemon, 500ms interval) that flushes pending buffers to SQLite via `executemany()`
- Also drain from `Data._samples` objects (probes + charts) by tracking a cursor position per `Data` instance
- Modify `_on_event()` to append to `_pending_events` alongside existing deque
- Modify `_BridgeLogHandler.emit()` to append to `_pending_logs`
- Modify `reset()` to call `_store.clear()`
- Modify `close()` to stop drain thread, do final flush, close store

**Modified: `happysimulator/visual/__init__.py`** (~3 lines)

- Add `db_path: str | Path | None = None` parameter to `serve()`, pass through to `SimulationBridge`

### Phase 2: Paginated API endpoints (backend)

**Modified: `happysimulator/visual/server.py`** (~50 lines changed)

- `/api/events` -- add `cursor`, `limit`, `direction`, `time_min`, `time_max`, `event_type`, `entity`, `search`, `exclude_internal` query params. When store is available, return `{events, next_cursor, total}`. Old `last_n` param still works as fallback.
- New `/api/logs` -- paginated log endpoint with `cursor`, `limit`, `min_level`, `search`, `time_min`, `time_max`
- `/api/timeseries` -- add `max_points` param for server-side downsampling
- `/api/chart_data` -- add `max_points` param; use `query_timeseries_bucketed()` when store is available
- New `/api/events/count` -- return total count + time range for UI status display

### Phase 3: Frontend pagination + virtual scrolling

**New dependency: `@tanstack/react-virtual`**

**Modified: `visual-frontend/src/hooks/useSimState.ts`** (~25 lines added)
- Keep existing `eventLog` / `simLogs` ring buffers for real-time WebSocket streaming (unchanged)
- Add `historicalEvents`, `eventsCursor`, `eventsTotal`, `isLoadingEvents` state for paginated browsing
- Add `historicalLogs`, `logsCursor`, `logsTotal`, `isLoadingLogs` similarly
- Add `loadMoreEvents()`, `loadMoreLogs()`, `searchEvents()`, `searchLogs()` actions

**Modified: `visual-frontend/src/components/EventLog.tsx`** (~80 lines changed)
- Dual-mode: "live tail" (existing behavior, WebSocket-fed ring buffer) vs "browse history" (API-paginated)
- Move filtering to server-side: debounce filter inputs, call `/api/events?...` with query params
- Add "Load More" button or infinite scroll for older events
- Add `@tanstack/react-virtual` for virtualizing the event list (currently renders 200 DOM nodes, replace with virtual window)
- Show total event count from `/api/events/count`

**Modified: `visual-frontend/src/components/SimulationLog.tsx`** (~40 lines changed)
- Same dual-mode: live tail vs paginated history browse via `/api/logs`

**Modified: `visual-frontend/src/components/DashboardPanel.tsx`** (~5 lines)
- Pass `max_points={Math.min(containerWidth * 2, 2000)}` to chart data requests

**Modified: `visual-frontend/src/components/InspectorPanel.tsx`** (~5 lines)
- Pass `max_points` to timeseries requests

**Modified: `visual-frontend/src/types.ts`** (~15 lines)
- Add `PaginatedEventsResponse`, `PaginatedLogsResponse` types

### Phase 4: Optional in-memory truncation

After draining to SQLite, optionally truncate `Data._samples` to keep only the most recent N samples (e.g., 10,000). This is behind a flag (`max_in_memory_samples` on `serve()`), defaulting to `None` (no truncation) for backward compatibility. Enables truly bounded memory for very long-running simulations.

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Leave `Data` class unchanged | Yes | Used by all non-visual consumers; modifying would break core library semantics |
| SQLite over alternatives | SQLite | stdlib, zero-config, WAL for concurrent R/W, handles millions of rows |
| Drain thread vs sync writes | Drain thread | Avoids per-event disk I/O on the hot simulation path |
| Drain interval | 500ms | Balances write throughput (batch ~50K rows) with data freshness |
| Cursor-based pagination | Yes | Efficient for large result sets, no offset-scanning |
| Server-side filtering | Yes | Moves O(n) filtering from JS to indexed SQL queries |
| Keep WebSocket incremental | Yes | Real-time tail uses existing ring buffer; history browsing is separate REST concern |
| Frontend virtual scrolling | `@tanstack/react-virtual` | Lightweight, well-maintained, replaces 200-DOM-node rendering |
| Default DB location | Temp file | Auto-cleaned on close; users can pass explicit `db_path=` for persistence |

## Files Summary

**New files:**
- `happysimulator/visual/store.py` -- VisualStore class
- `tests/unit/test_visual_store.py` -- Unit tests

**Modified files:**
- `happysimulator/visual/__init__.py` -- `db_path` param on `serve()`
- `happysimulator/visual/bridge.py` -- drain thread, pending buffers, store integration
- `happysimulator/visual/server.py` -- paginated endpoints, downsampling params
- `visual-frontend/src/hooks/useSimState.ts` -- historical browsing state
- `visual-frontend/src/components/EventLog.tsx` -- server-side pagination, virtual scrolling
- `visual-frontend/src/components/SimulationLog.tsx` -- server-side pagination
- `visual-frontend/src/components/DashboardPanel.tsx` -- max_points param
- `visual-frontend/src/components/InspectorPanel.tsx` -- max_points param
- `visual-frontend/src/types.ts` -- paginated response types
- `visual-frontend/package.json` -- add `@tanstack/react-virtual`

**Intentionally NOT modified:**
- `happysimulator/instrumentation/data.py` -- core Data class stays pure in-memory
- `happysimulator/instrumentation/probe.py` -- probes continue writing to Data
- `happysimulator/visual/dashboard.py` -- Chart class unchanged
- `happysimulator/visual/serializers.py` -- serialization unchanged

## Verification

1. Run existing tests: `pytest tests/ -q` -- all ~2755 should pass (no core changes)
2. Run new store tests: `pytest tests/unit/test_visual_store.py -v`
3. Manual test with visual debugger:
   - `python examples/visual_debugger.py` -- open browser, step through events
   - Verify event log shows total count, "Load More" works for history
   - Verify charts render correctly with downsampled data
   - Verify sim logs show full history with level filtering
   - Run at max speed to 100s+ sim time, confirm memory stays bounded
   - Reset simulation, confirm SQLite clears and UI resets correctly
4. Build frontend: `cd visual-frontend && npm run build`
