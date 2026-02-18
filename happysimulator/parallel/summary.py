"""Summary types for parallel simulation results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from happysimulator.instrumentation.summary import EntitySummary, SimulationSummary


@dataclass
class ParallelSimulationSummary:
    """Merged summary from all partitions of a parallel simulation.

    Aggregate metrics combine per-partition results. Speedup and efficiency
    measure the benefit of parallel execution versus sequential.

    Attributes:
        duration_s: Simulation duration (max across partitions).
        total_events_processed: Sum of events across all partitions.
        events_cancelled: Sum of cancelled events across all partitions.
        events_per_second: Aggregate throughput (sum).
        wall_clock_seconds: Wall time of the slowest partition.
        partitions: Per-partition simulation summaries.
        entities: Merged entity summaries (disjoint sets).
        partition_wall_times: Wall seconds per partition.
        speedup: Estimated sequential time / actual wall time.
        parallelism_efficiency: Speedup / number of partitions.
    """

    duration_s: float
    total_events_processed: int
    events_cancelled: int
    events_per_second: float
    wall_clock_seconds: float
    partitions: dict[str, SimulationSummary]
    entities: dict[str, EntitySummary]
    partition_wall_times: dict[str, float]
    speedup: float
    parallelism_efficiency: float

    def __str__(self) -> str:
        lines = [
            "Parallel Simulation Summary",
            f"  Partitions: {len(self.partitions)}",
            f"  Duration: {self.duration_s:.2f}s (sim) / {self.wall_clock_seconds:.3f}s (wall)",
            f"  Events processed: {self.total_events_processed:,}",
            f"  Events cancelled: {self.events_cancelled:,}",
            f"  Events/sec (sim): {self.events_per_second:,.1f}",
            f"  Speedup: {self.speedup:.2f}x",
            f"  Efficiency: {self.parallelism_efficiency:.0%}",
            "  Per-partition wall times:",
        ]
        for name, wt in sorted(self.partition_wall_times.items()):
            events = self.partitions[name].total_events_processed
            lines.append(f"    {name}: {wt:.3f}s ({events:,} events)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "total_events_processed": self.total_events_processed,
            "events_cancelled": self.events_cancelled,
            "events_per_second": self.events_per_second,
            "wall_clock_seconds": self.wall_clock_seconds,
            "speedup": self.speedup,
            "parallelism_efficiency": self.parallelism_efficiency,
            "partitions": {
                name: s.to_dict() for name, s in self.partitions.items()
            },
            "entities": {
                name: e.to_dict() for name, e in self.entities.items()
            },
            "partition_wall_times": dict(self.partition_wall_times),
        }
