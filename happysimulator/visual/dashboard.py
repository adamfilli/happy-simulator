"""Predefined dashboard charts for the visual debugger.

Users define Chart objects in Python and pass them to serve().
Each Chart references a Data object and specifies how to transform
and display it (raw, bucketed percentiles, rate, etc.).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from happysimulator.instrumentation.data import Data
    from happysimulator.instrumentation.probe import Probe

TRANSFORM_MAP: dict[str, str] = {
    "mean": "means",
    "p50": "p50s",
    "p99": "p99s",
    "max": "maxes",
}


@dataclass
class Chart:
    """A predefined dashboard chart backed by a Data object.

    Args:
        data: The Data object to read samples from.
        title: Chart title shown in the title bar.
        y_label: Y-axis label text.
        x_label: X-axis label text.
        color: Line color as CSS hex string.
        transform: One of "raw", "mean", "p50", "p99", "p999", "max", "rate".
        window_s: Bucket window size for aggregated transforms.
        y_min: Fixed Y-axis minimum (None for auto-scale).
        y_max: Fixed Y-axis maximum (None for auto-scale).
    """

    data: Data
    title: str = ""
    y_label: str = ""
    x_label: str = "Time (s)"
    color: str = "#3b82f6"
    transform: str = "raw"
    window_s: float = 1.0
    y_min: float | None = None
    y_max: float | None = None
    chart_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8], init=False)

    @classmethod
    def from_probe(cls, probe: Probe, **kwargs: Any) -> Chart:
        """Create a Chart from a Probe, deriving title and data automatically."""
        defaults: dict[str, Any] = {
            "title": f"{probe.target.name}.{probe.metric}",
            "data": probe.data_sink,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    def to_config(self) -> dict[str, Any]:
        """Serialize display config (everything except the data reference)."""
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

    def get_data(self, start_s: float | None = None, end_s: float | None = None) -> dict[str, Any]:
        """Read from the Data object and apply the transform.

        Args:
            start_s: Optional start time filter (inclusive).
            end_s: Optional end time filter (exclusive).
        """
        data = self.data
        if start_s is not None or end_s is not None:
            s = start_s if start_s is not None else 0.0
            e = end_s if end_s is not None else float("inf")
            data = data.between(s, e)

        if self.transform == "raw":
            return {"times": data.times(), "values": data.raw_values()}

        if self.transform == "rate":
            rate_data = data.rate(self.window_s)
            return {"times": rate_data.times(), "values": rate_data.raw_values()}

        if self.transform == "p999":
            bucketed = data.bucket(self.window_s)
            times = bucketed.times()
            from happysimulator.instrumentation.data import _percentile_sorted
            from collections import defaultdict
            import math

            buckets_map: dict[int, list[float]] = defaultdict(list)
            for t, v in data.values:
                idx = int(math.floor(t / self.window_s))
                buckets_map[idx].append(float(v))

            values = []
            for key in sorted(buckets_map.keys()):
                vals = sorted(buckets_map[key])
                values.append(_percentile_sorted(vals, 0.999))
            return {"times": times, "values": values}

        # mean, p50, p99, max — use BucketedData methods
        if self.transform not in TRANSFORM_MAP:
            return {"times": [], "values": []}

        bucketed = data.bucket(self.window_s)
        times = bucketed.times()
        method_name = TRANSFORM_MAP[self.transform]
        values = getattr(bucketed, method_name)()
        return {"times": times, "values": values}
