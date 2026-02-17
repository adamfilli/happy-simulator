"""Parallel execution utilities for happysimulator.

Two parallel execution modes:

1. **Process-based** (``ParallelRunner``): Run independent simulations
   (parameter sweeps, Monte Carlo replicas) across processes.

2. **Thread-based** (``ParallelSimulation``): Run a *single* simulation
   with partitioned entity groups on separate threads. Requires
   free-threaded Python 3.13t+ for true parallelism.

Example — Partitioned simulation::

    from happysimulator.parallel import ParallelSimulation, SimulationPartition

    sim = ParallelSimulation(
        partitions=[
            SimulationPartition("us", entities=[us_srv], sources=[us_src]),
            SimulationPartition("eu", entities=[eu_srv], sources=[eu_src]),
        ],
        duration=60.0,
    )
    summary = sim.run()
    print(f"Speedup: {summary.speedup:.1f}x")

Example — Monte Carlo replicas::

    from happysimulator.parallel import ParallelRunner

    runner = ParallelRunner(max_workers=4)
    results = runner.run_replicas(build_sim, n_replicas=20, base_seed=42)
"""

from happysimulator.parallel.partition import SimulationPartition
from happysimulator.parallel.runner import ParallelResult, ParallelRunner, RunConfig
from happysimulator.parallel.simulation import ParallelSimulation
from happysimulator.parallel.summary import ParallelSimulationSummary

__all__ = [
    "ParallelResult",
    "ParallelRunner",
    "ParallelSimulation",
    "ParallelSimulationSummary",
    "RunConfig",
    "SimulationPartition",
]
