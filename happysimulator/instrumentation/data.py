"""Time-series data storage for simulation metrics.

Data provides a container for collecting timestamped samples during
simulation. Used by entities to record statistics (latency, throughput,
queue depth) and by Probes for periodic measurements.

BucketedData provides time-windowed aggregation over Data.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, List, Tuple

from happysimulator.core.temporal import Instant


class Data:
    """Container for timestamped metric samples with analysis utilities.

    Stores (time, value) pairs for post-simulation analysis. Values should
    be numeric (int or float) for aggregation methods to work.

    Samples are stored in append order. For time-ordered data, ensure
    add_stat is called with non-decreasing times.
    """

    def __init__(self) -> None:
        self._samples: List[Tuple[float, Any]] = []

    def add_stat(self, value: Any, time: Instant) -> None:
        """Record a data point at the given simulation time.

        Args:
            value: The metric value to record.
            time: The simulation time of this sample.
        """
        self._samples.append((time.to_seconds(), value))

    def clear(self) -> None:
        """Remove all recorded samples."""
        self._samples.clear()

    @property
    def values(self) -> List[Tuple[float, Any]]:
        """All recorded samples as (time_seconds, value) tuples."""
        return self._samples

    # === Slicing ===

    def between(self, start_s: float, end_s: float) -> Data:
        """Return a new Data with samples in [start, end).

        Args:
            start_s: Start time in seconds (inclusive).
            end_s: End time in seconds (exclusive).
        """
        result = Data()
        result._samples = [(t, v) for t, v in self._samples if start_s <= t < end_s]
        return result

    # === Aggregations ===

    def mean(self) -> float:
        """Mean of sample values. Returns 0.0 if empty."""
        vals = [v for _, v in self._samples]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def min(self) -> float:
        """Minimum sample value. Returns 0.0 if empty."""
        vals = [v for _, v in self._samples]
        if not vals:
            return 0.0
        return builtins_min(vals)

    def max(self) -> float:
        """Maximum sample value. Returns 0.0 if empty."""
        vals = [v for _, v in self._samples]
        if not vals:
            return 0.0
        return builtins_max(vals)

    def percentile(self, p: float) -> float:
        """Calculate percentile from sample values.

        Args:
            p: Percentile in [0, 1]. E.g., 0.99 for p99.

        Returns:
            Interpolated percentile value, or 0.0 if empty.
        """
        vals = sorted(v for _, v in self._samples)
        if not vals:
            return 0.0
        if p <= 0:
            return float(vals[0])
        if p >= 1:
            return float(vals[-1])
        n = len(vals)
        pos = p * (n - 1)
        lo = int(pos)
        hi = builtins_min(lo + 1, n - 1)
        frac = pos - lo
        return float(vals[lo] * (1.0 - frac) + vals[hi] * frac)

    def count(self) -> int:
        """Number of recorded samples."""
        return len(self._samples)

    def sum(self) -> float:
        """Sum of sample values. Returns 0.0 if empty."""
        return builtins_sum(v for _, v in self._samples)

    def std(self) -> float:
        """Population standard deviation of sample values. Returns 0.0 if fewer than 2 samples."""
        vals = [v for _, v in self._samples]
        if len(vals) < 2:
            return 0.0
        return statistics.pstdev(vals)

    # === Time-windowed aggregation ===

    def bucket(self, window_s: float = 1.0) -> BucketedData:
        """Group samples into fixed-width time windows.

        Args:
            window_s: Width of each bucket in seconds.

        Returns:
            BucketedData with per-bucket aggregations.
        """
        from collections import defaultdict
        buckets: dict[int, list[float]] = defaultdict(list)
        for t, v in self._samples:
            bucket_idx = int(math.floor(t / window_s))
            buckets[bucket_idx].append(float(v))

        sorted_keys = sorted(buckets.keys())
        result = BucketedData()
        for key in sorted_keys:
            vals = buckets[key]
            vals_sorted = sorted(vals)
            bucket_start = key * window_s

            result._times.append(bucket_start)
            result._means.append(sum(vals) / len(vals))
            result._counts.append(len(vals))
            result._maxes.append(builtins_max(vals))
            result._sums.append(sum(vals))
            result._p50s.append(_percentile_sorted(vals_sorted, 0.50))
            result._p99s.append(_percentile_sorted(vals_sorted, 0.99))

        return result

    # === Convenience ===

    def times(self) -> list[float]:
        """Just the timestamps from all samples."""
        return [t for t, _ in self._samples]

    def raw_values(self) -> list[float]:
        """Just the values from all samples."""
        return [v for _, v in self._samples]

    # === Rate of change ===

    def rate(self, window_s: float = 1.0) -> Data:
        """Compute rate of change (count per window) over time windows.

        Useful for throughput data where each sample represents one event.

        Args:
            window_s: Width of each time window in seconds.

        Returns:
            New Data with (bucket_start, count/window_s) pairs.
        """
        bucketed = self.bucket(window_s)
        result = Data()
        for t, c in zip(bucketed.times(), bucketed.counts()):
            # Build Instant from seconds for add_stat
            result._samples.append((t, c / window_s))
        return result

    def __len__(self) -> int:
        return len(self._samples)

    def __bool__(self) -> bool:
        return len(self._samples) > 0


def _percentile_sorted(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from pre-sorted values (p in [0, 1])."""
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])
    n = len(sorted_values)
    pos = p * (n - 1)
    lo = int(pos)
    hi = builtins_min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


class BucketedData:
    """Time-windowed aggregation result from Data.bucket()."""

    def __init__(self) -> None:
        self._times: list[float] = []
        self._means: list[float] = []
        self._counts: list[int] = []
        self._maxes: list[float] = []
        self._sums: list[float] = []
        self._p50s: list[float] = []
        self._p99s: list[float] = []

    def times(self) -> list[float]:
        """Bucket start times in seconds."""
        return self._times

    def means(self) -> list[float]:
        """Mean value per bucket."""
        return self._means

    def counts(self) -> list[int]:
        """Number of samples per bucket."""
        return self._counts

    def maxes(self) -> list[float]:
        """Maximum value per bucket."""
        return self._maxes

    def sums(self) -> list[float]:
        """Sum of values per bucket."""
        return self._sums

    def p50s(self) -> list[float]:
        """50th percentile (median) per bucket."""
        return self._p50s

    def p99s(self) -> list[float]:
        """99th percentile per bucket."""
        return self._p99s

    def to_dict(self) -> dict[str, list]:
        """Return dict with keys: time_s, mean, p50, p99, max, count, sum."""
        return {
            "time_s": list(self._times),
            "mean": list(self._means),
            "p50": list(self._p50s),
            "p99": list(self._p99s),
            "max": list(self._maxes),
            "count": list(self._counts),
            "sum": list(self._sums),
        }

    def __len__(self) -> int:
        return len(self._times)

    def __bool__(self) -> bool:
        return len(self._times) > 0


# Avoid shadowing builtins
import builtins as _builtins

builtins_min = _builtins.min
builtins_max = _builtins.max
builtins_sum = _builtins.sum
