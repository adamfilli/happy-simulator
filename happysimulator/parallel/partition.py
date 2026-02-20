"""Partition definition for parallel simulation.

A SimulationPartition groups entities and sources that run together on a
single thread.  Partitions that need to communicate declare PartitionLink
objects between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from happysimulator.core.protocols import Simulatable
    from happysimulator.faults.schedule import FaultSchedule
    from happysimulator.instrumentation.recorder import TraceRecorder
    from happysimulator.load.source import Source


@dataclass
class SimulationPartition:
    """Declaration of one partition in a parallel simulation.

    Attributes:
        name: Unique human-readable label.
        entities: Simulation actors belonging to this partition.
        sources: Load generators for this partition.
        probes: Measurement sources (daemon) for this partition.
        fault_schedule: Optional fault injection for this partition.
        trace_recorder: Optional trace recorder for this partition.
    """

    name: str
    entities: list[Simulatable] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    probes: list[Source] = field(default_factory=list)
    fault_schedule: FaultSchedule | None = None
    trace_recorder: TraceRecorder | None = None
