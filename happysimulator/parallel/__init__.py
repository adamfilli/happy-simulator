"""Parallel execution for happy-simulator.

This package provides two styles of parallel execution:

1. **ParallelRunner** — process-based parallelism for parameter sweeps and
   Monte Carlo replicas (each run is fully independent).

2. **ParallelSimulation** — thread-based partitioned execution where a single
   model is split into partitions that run concurrently.  Optional
   ``PartitionLink`` declarations enable cross-partition communication
   coordinated through barrier-based time windows.
"""

from happysimulator.parallel.link import PartitionLink
from happysimulator.parallel.partition import SimulationPartition
from happysimulator.parallel.runner import ParallelResult, ParallelRunner, RunConfig
from happysimulator.parallel.simulation import ParallelSimulation
from happysimulator.parallel.summary import ParallelSimulationSummary

__all__ = [
    "ParallelResult",
    "ParallelRunner",
    "ParallelSimulation",
    "ParallelSimulationSummary",
    "PartitionLink",
    "RunConfig",
    "SimulationPartition",
]
