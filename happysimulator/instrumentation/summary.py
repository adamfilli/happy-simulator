"""Simulation summary generated after a run completes.

SimulationSummary provides a structured overview of what happened during
a simulation, including per-entity statistics. It's returned by
Simulation.run() and also accessible via Simulation.summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueueStats:
    """Queue-specific statistics for QueuedResource entities."""
    peak_depth: int
    total_accepted: int
    total_dropped: int


@dataclass
class EntitySummary:
    """Per-entity statistics from a simulation run."""
    name: str
    entity_type: str
    events_handled: int
    queue_stats: QueueStats | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "type": self.entity_type,
            "events_handled": self.events_handled,
        }
        if self.queue_stats is not None:
            result["queue"] = {
                "peak_depth": self.queue_stats.peak_depth,
                "total_accepted": self.queue_stats.total_accepted,
                "total_dropped": self.queue_stats.total_dropped,
            }
        return result


@dataclass
class SimulationSummary:
    """Auto-generated summary of a simulation run.

    Returned by Simulation.run() and also accessible via Simulation.summary.
    """
    duration_s: float
    total_events_processed: int
    events_per_second: float
    wall_clock_seconds: float
    entities: dict[str, EntitySummary] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "Simulation Summary",
            f"  Duration: {self.duration_s:.2f}s (sim) / {self.wall_clock_seconds:.3f}s (wall)",
            f"  Events processed: {self.total_events_processed}",
            f"  Events/sec (sim): {self.events_per_second:.1f}",
        ]
        if self.entities:
            lines.append("  Entities:")
            for name, es in self.entities.items():
                line = f"    {name} ({es.entity_type}): {es.events_handled} events"
                if es.queue_stats is not None:
                    qs = es.queue_stats
                    line += f" | queue: peak={qs.peak_depth}, accepted={qs.total_accepted}, dropped={qs.total_dropped}"
                lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "total_events_processed": self.total_events_processed,
            "events_per_second": self.events_per_second,
            "wall_clock_seconds": self.wall_clock_seconds,
            "entities": {
                name: es.to_dict() for name, es in self.entities.items()
            },
        }
