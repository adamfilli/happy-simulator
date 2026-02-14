# Visual Debugger

Browser-based simulation debugger with entity graphs, charts, event logs, and playback controls.

Requires optional dependencies:

```bash
pip install happysim[visual]
```

## Quick Start

```python
from happysimulator.visual import serve

serve(sim)  # opens browser at http://127.0.0.1:8765
```

## Charts

Add predefined charts to the dashboard:

```python
from happysimulator.visual import serve, Chart
from happysimulator import Data, Probe

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1)

serve(sim, charts=[
    Chart(depth_data, title="Queue Depth", y_label="items"),
    Chart(depth_data, title="P99 Queue Depth",
          transform="p99", window_s=1.0, y_label="items", color="#f59e0b"),
    Chart.from_probe(depth_probe, transform="mean", window_s=0.5),
])
```

### Chart Options

| Field | Type | Description |
|-------|------|-------------|
| `data` | `Data` | Time-series data source |
| `title` | `str` | Chart title |
| `y_label` | `str` | Y-axis label |
| `x_label` | `str` | X-axis label |
| `color` | `str` | Hex color (e.g. `"#f59e0b"`) |
| `transform` | `str` | Aggregation transform |
| `window_s` | `float` | Bucket window size for transforms |
| `y_min` / `y_max` | `float` | Y-axis range |

### Transforms

`"raw"` | `"mean"` | `"p50"` | `"p99"` | `"p999"` | `"max"` | `"rate"` — all backed by `Data.bucket()`.

## UI Features

- **Controls**: Step, Play, Pause, Debug (play with breakpoint awareness), Run To time/event, Reset
- **Graph View**: Entity topology with ELK.js layout — click to inspect
- **Dashboard View**: Draggable/resizable chart grid with time range filtering
- **Inspector**: Entity metrics, probe time series, source load profile visualization
- **Event Log**: Expandable rows showing full context — `stack` (entity journey), `trace` spans, `request_id`, `created_at`
- **Sim Logs**: Streamed `happysimulator.*` logger output with level filtering

## Configuration

```python
serve(sim, charts=[...], port=8765)
```

## Next Steps

- [Observability](observability.md) — Data, Probe, and analysis without the browser
- [Simulation Control](simulation-control.md) — programmatic pause/step/breakpoints
