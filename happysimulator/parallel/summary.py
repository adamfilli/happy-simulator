"""Summary for parallel simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from happysimulator.instrumentation.summary import EntitySummary, SimulationSummary


@dataclass
class ParallelSimulationSummary:
    """Aggregate summary of a parallel simulation run.

    Attributes:
        duration_s: Simulated duration in seconds.
        total_events_processed: Sum of events across all partitions.
        events_per_second: Aggregate sim events / sim duration.
        wall_clock_seconds: Wall-clock time for the entire run.
        partitions: Per-partition SimulationSummary instances.
        entities: Merged entity summaries from all partitions.
        partition_wall_times: Per-partition wall-clock seconds.
        speedup: Sequential wall time / parallel wall time.
        parallelism_efficiency: speedup / number_of_partitions.
        total_windows: Number of barrier windows executed (0 if no links).
        total_cross_partition_events: Events exchanged across partitions.
        window_size_s: Barrier window size in seconds (0 if no links).
        barrier_overhead_seconds: Cumulative wall time spent in barriers.
        coordination_efficiency: 1 - (barrier_overhead / wall_clock).
    """

    duration_s: float
    total_events_processed: int
    events_per_second: float
    wall_clock_seconds: float
    partitions: dict[str, SimulationSummary] = field(default_factory=dict)
    entities: dict[str, EntitySummary] = field(default_factory=dict)
    partition_wall_times: dict[str, float] = field(default_factory=dict)
    speedup: float = 1.0
    parallelism_efficiency: float = 1.0
    total_windows: int = 0
    total_cross_partition_events: int = 0
    window_size_s: float = 0.0
    barrier_overhead_seconds: float = 0.0
    coordination_efficiency: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "total_events_processed": self.total_events_processed,
            "events_per_second": self.events_per_second,
            "wall_clock_seconds": self.wall_clock_seconds,
            "partitions": {
                name: s.to_dict() for name, s in self.partitions.items()
            },
            "entities": {
                name: e.to_dict() for name, e in self.entities.items()
            },
            "partition_wall_times": dict(self.partition_wall_times),
            "speedup": self.speedup,
            "parallelism_efficiency": self.parallelism_efficiency,
            "total_windows": self.total_windows,
            "total_cross_partition_events": self.total_cross_partition_events,
            "window_size_s": self.window_size_s,
            "barrier_overhead_seconds": self.barrier_overhead_seconds,
            "coordination_efficiency": self.coordination_efficiency,
        }

    def __str__(self) -> str:
        lines = [
            "Parallel Simulation Summary",
            f"  Duration: {self.duration_s:.2f}s (sim) / {self.wall_clock_seconds:.3f}s (wall)",
            f"  Events processed: {self.total_events_processed}",
            f"  Events/sec (sim): {self.events_per_second:.1f}",
            f"  Partitions: {len(self.partitions)}",
            f"  Speedup: {self.speedup:.2f}x",
            f"  Efficiency: {self.parallelism_efficiency:.1%}",
        ]
        if self.total_windows > 0:
            lines.extend([
                f"  Windows: {self.total_windows} (size={self.window_size_s:.4f}s)",
                f"  Cross-partition events: {self.total_cross_partition_events}",
                f"  Barrier overhead: {self.barrier_overhead_seconds:.3f}s",
                f"  Coordination efficiency: {self.coordination_efficiency:.1%}",
            ])
        return "\n".join(lines)
