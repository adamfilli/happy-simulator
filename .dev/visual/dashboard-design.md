# Predefined Dashboard Charts from Python Code

## Context

Users define simulations in Python and launch the visual debugger with `serve(sim)`. Currently, the Dashboard tab starts empty — users must manually add charts via the UI. Users want to **predefine charts in Python code** so the Dashboard tab is pre-populated when the debugger opens, with matplotlib-level configurability (axis labels, scales, transforms like mean/p99/p999, colors). Predefined charts should still be draggable/resizable/closeable like manually-added ones.

## Python API Design

```python
from happysimulator.visual import serve, Chart

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1)

serve(sim, charts=[
    # Raw time series — just the sampled values
    Chart(depth_data, title="Queue Depth", y_label="items"),

    # Windowed P99 — bucketed percentile over 1s windows
    Chart(depth_data, title="P99 Queue Depth",
          transform="p99", window_s=1.0, y_label="items", color="#f59e0b"),

    # Rate — events per second from a counter's Data
    Chart(latency_data, title="Throughput",
          transform="rate", window_s=1.0, y_label="events/s"),

    # Convenience: create from a Probe (auto-derives title/label)
    Chart.from_probe(depth_probe),
    Chart.from_probe(depth_probe, transform="p99", window_s=1.0),
])
```

### `Chart` class

```python
@dataclass
class Chart:
    data: Data                          # the Data object to read from
    title: str = ""                     # chart title (shown in title bar)
    y_label: str = ""                   # Y-axis label
    x_label: str = "Time (s)"          # X-axis label
    color: str = "#3b82f6"             # line color (CSS hex)
    transform: str = "raw"             # "raw" | "mean" | "p50" | "p99" | "p999" | "max" | "rate"
    window_s: float = 1.0              # bucket window for aggregated transforms
    y_min: float | None = None         # fixed Y-axis minimum (None = auto)
    y_max: float | None = None         # fixed Y-axis maximum (None = auto)

    @classmethod
    def from_probe(cls, probe: Probe, **kwargs) -> Chart:
        """Convenience: derive title and data from a Probe."""
        defaults = {
            "title": f"{probe._target_name}.{probe._metric}",
            "data": probe.data_sink,
        }
        defaults.update(kwargs)
        return cls(**defaults)
```

### Transforms

All transforms leverage the existing `Data` class methods — no new computation code needed:

| Transform | Backend computation | Output |
|-----------|-------------------|--------|
| `"raw"` | `data.times()`, `data.raw_values()` | Raw samples |
| `"mean"` | `data.bucket(window_s).means()` | Windowed mean |
| `"p50"` | `data.bucket(window_s).p50s()` | Windowed p50 |
| `"p99"` | `data.bucket(window_s).p99s()` | Windowed p99 |
| `"p999"` | Manual: `data.bucket(window_s)` then `percentile(0.999)` per bucket | Windowed p99.9 |
| `"max"` | `data.bucket(window_s).maxes()` | Windowed max |
| `"rate"` | `data.rate(window_s).times()/.raw_values()` | Count per window |

## Backend Changes

### 1. New file: `happysimulator/visual/dashboard.py`

The `Chart` dataclass. Assigns a unique `chart_id` on creation. Has a `to_config()` method that serializes display config (everything except the `data` reference) to a JSON-safe dict, and a `get_data()` method that reads from the `Data` object and applies the transform.

```python
def to_config(self) -> dict:
    return {
        "chart_id": self.chart_id,
        "title": self.title,
        "y_label": self.y_label,
        "x_label": self.x_label,
        "color": self.color,
        "transform": self.transform,
        "window_s": self.window_s,
        "y_min": self.y_min,
        "y_max": self.y_max,
    }

def get_data(self) -> dict:
    if self.transform == "raw":
        return {"times": self.data.times(), "values": self.data.raw_values()}
    elif self.transform == "rate":
        rate_data = self.data.rate(self.window_s)
        return {"times": rate_data.times(), "values": rate_data.raw_values()}
    else:
        bucketed = self.data.bucket(self.window_s)
        times = bucketed.times()
        values = getattr(bucketed, TRANSFORM_MAP[self.transform])()
        return {"times": times, "values": values}
```

### 2. `happysimulator/visual/__init__.py`

Extend `serve()` signature:

```python
def serve(sim, *, charts: list[Chart] | None = None, host="127.0.0.1", port=8765):
```

Pass `charts` to `SimulationBridge`.

### 3. `happysimulator/visual/bridge.py`

- Constructor accepts `charts: list[Chart]`
- `get_chart_configs() -> list[dict]` — returns `[chart.to_config() for chart in charts]`
- `get_chart_data(chart_id: str) -> dict` — finds chart by ID, calls `chart.get_data()`, returns times + values + config

### 4. `happysimulator/visual/server.py`

Two new endpoints:

| Endpoint | Method | Returns |
|----------|--------|---------|
| `GET /api/charts` | GET | `[{chart_id, title, y_label, x_label, color, transform, ...}, ...]` |
| `GET /api/chart_data?chart_id=<id>` | GET | `{chart_id, times, values, config}` |

## Frontend Changes

### 5. `visual-frontend/src/types.ts`

```typescript
export interface ChartConfig {
  chart_id: string;
  title: string;
  y_label: string;
  x_label: string;
  color: string;
  transform: string;
  window_s: number;
  y_min: number | null;
  y_max: number | null;
}

// Extend DashboardPanelConfig to support both probe-based and chart-based panels
export interface DashboardPanelConfig {
  id: string;
  label: string;
  x: number;
  y: number;
  // One of these two — probe-based (user-added) or chart-based (predefined)
  probeName?: string;
  chartConfig?: ChartConfig;
}
```

### 6. `visual-frontend/src/components/DashboardPanel.tsx`

- If `chartConfig` is present, fetch from `/api/chart_data?chart_id=<id>` instead of `/api/timeseries`
- Pass `chartConfig.color`, `chartConfig.y_label`, `chartConfig.x_label` to `TimeSeriesChart`
- Pass `chartConfig.y_min`, `chartConfig.y_max` for fixed axis bounds

### 7. `visual-frontend/src/components/TimeSeriesChart.tsx`

Extend props to accept chart configuration:

```typescript
interface Props {
  times: number[];
  values: number[];
  label: string;
  color?: string;
  yLabel?: string;      // NEW: Y-axis label text
  xLabel?: string;      // NEW: X-axis label text
  yMin?: number | null;  // NEW: fixed Y-axis min
  yMax?: number | null;  // NEW: fixed Y-axis max
}
```

Rendering changes:
- Draw `yLabel` rotated 90deg on the left edge
- Draw `xLabel` centered below the X-axis
- Use `yMin`/`yMax` when set instead of auto-scaling from data

### 8. `visual-frontend/src/App.tsx`

On initial load (alongside topology/state fetch), fetch `/api/charts`. If charts exist, populate `dashboardPanels` in the store and auto-switch to Dashboard view.

### 9. `visual-frontend/src/components/DashboardView.tsx`

On mount, if `dashboardPanels` is empty, fetch `/api/charts` and populate. Predefined charts get auto-staggered positions like manually-added ones.

### 10. Rebuild static assets

`npm run build`

## Files Modified

| File | Change |
|------|--------|
| `happysimulator/visual/dashboard.py` | **New** — `Chart` dataclass with transforms |
| `happysimulator/visual/__init__.py` | Add `charts` param to `serve()`, export `Chart` |
| `happysimulator/visual/bridge.py` | Accept charts, add `get_chart_configs()` / `get_chart_data()` |
| `happysimulator/visual/server.py` | Add `GET /api/charts` and `GET /api/chart_data` endpoints |
| `visual-frontend/src/types.ts` | Add `ChartConfig`, extend `DashboardPanelConfig` |
| `visual-frontend/src/components/TimeSeriesChart.tsx` | Add axis labels, fixed bounds |
| `visual-frontend/src/components/DashboardPanel.tsx` | Support chart-based data fetching |
| `visual-frontend/src/components/DashboardView.tsx` | Auto-load predefined charts |
| `visual-frontend/src/App.tsx` | Fetch `/api/charts` on init |
| `examples/visual_debugger.py` | Add predefined charts to demo |

## Example Usage

```python
from happysimulator.visual import serve, Chart

sink = Sink("Sink")
server = Server("Server", downstream=sink, service_time=0.08)
source = Source.constant(rate=10, target=server, event_type="Request")

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1)

sim = Simulation(sources=[source], entities=[server, sink], probes=[depth_probe])

serve(sim, charts=[
    Chart(depth_data, title="Queue Depth", y_label="items"),
    Chart(depth_data, title="P99 Queue Depth",
          transform="p99", window_s=1.0, y_label="items", color="#f59e0b"),
])
```

When the browser opens, the Dashboard tab shows two pre-populated charts — "Queue Depth" (raw) and "P99 Queue Depth" (windowed p99 in amber). Both are draggable, resizable, and live-update as the simulation runs. Users can still add more charts via "Add Chart".

## Verification

1. Update `examples/visual_debugger.py` with predefined charts
2. `python examples/visual_debugger.py` — debugger opens
3. Dashboard tab is auto-selected with predefined charts visible
4. Charts show "No data yet" initially (sim hasn't started)
5. Step/Play — charts populate with data, transforms applied correctly
6. Drag/close predefined charts — works same as user-added
7. "Add Chart" button still works alongside predefined ones
8. Y-axis labels visible on charts that specify them
9. Fixed axis bounds respected when y_min/y_max set
10. Reset — charts clear and re-populate on next run
