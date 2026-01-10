"""Time-series data storage for simulation metrics.

Data provides a simple container for collecting timestamped samples
during simulation. Used by entities to record statistics (latency,
throughput, queue depth) and by Probes for periodic measurements.
"""

from typing import Any, List, Tuple
from happysimulator.core.instant import Instant


class Data:
    """Container for timestamped metric samples.

    Stores (time, value) pairs for post-simulation analysis. Values can
    be any type (floats for measurements, ints for counts, etc.).

    Samples are stored in append order. For time-ordered data, ensure
    add_stat is called with non-decreasing times.
    """

    def __init__(self):
        self._samples: List[Tuple[float, Any]] = []

    def add_stat(self, value: Any, time: Instant) -> None:
        """Record a data point at the given simulation time.

        Args:
            value: The metric value to record.
            time: The simulation time of this sample.
        """
        self._samples.append((time.to_seconds(), value))

    @property
    def values(self) -> List[Tuple[float, Any]]:
        """All recorded samples as (time_seconds, value) tuples."""
        return self._samples