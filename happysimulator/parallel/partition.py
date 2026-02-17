"""Partition declaration for parallel simulation.

A SimulationPartition groups entities, sources, and probes that execute
together on a single thread. Partitions run in parallel with each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from happysimulator.core.protocols import Simulatable
    from happysimulator.core.temporal import Instant
    from happysimulator.faults.schedule import FaultSchedule
    from happysimulator.instrumentation.recorder import TraceRecorder
    from happysimulator.load.source import Source


@dataclass
class SimulationPartition:
    """Declaration of an independent entity group for parallel execution.

    All entities, sources, and probes within a partition execute sequentially
    on a single thread, exactly as in a normal Simulation. Partitions execute
    in parallel with each other.

    Constraints:
        - No entity in this partition may reference an entity in another
          partition (validated at init, enforced at runtime).
        - No Source may target an entity outside this partition.
        - No shared Resource, Mutex, Network, or other synchronization
          primitive may span partitions.

    Args:
        name: Unique identifier for this partition.
        entities: Simulation actors that respond to events.
        sources: Load generators that produce events.
        probes: Measurement sources that run as daemons.
        fault_schedule: Optional fault injection schedule.
        trace_recorder: Optional recorder for debugging/visualization.
        start_time: Override start time (defaults to ParallelSimulation's).
        end_time: Override end time (defaults to ParallelSimulation's).
    """

    name: str
    entities: list[Simulatable] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    probes: list[Source] = field(default_factory=list)
    fault_schedule: FaultSchedule | None = None
    trace_recorder: TraceRecorder | None = None
    start_time: Instant | None = None
    end_time: Instant | None = None
