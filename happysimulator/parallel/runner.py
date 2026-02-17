"""Parallel execution utilities for running multiple simulations concurrently.

Provides process-based parallelism for parameter sweeps and Monte Carlo replicas.
Each simulation runs in its own process to avoid GIL contention.

Example — Monte Carlo replicas::

    from happysimulator.parallel import ParallelRunner

    def build_sim():
        sink = Sink("Sink")
        server = MyServer("Server", downstream=sink)
        source = Source.poisson(rate=10, target=server)
        return Simulation(duration=100.0, sources=[source], entities=[server, sink])

    runner = ParallelRunner(max_workers=4)
    results = runner.run_replicas(build_sim, n_replicas=20, base_seed=42)
    for r in results:
        print(f"{r.name}: {r.summary.total_events_processed} events")

Example — Parameter sweep::

    from happysimulator.parallel import ParallelRunner, RunConfig

    configs = [
        RunConfig(name=f"rate_{r}", build_fn=lambda r=r: build_sim(rate=r), seed=42)
        for r in [1, 5, 10, 50, 100]
    ]
    results = runner.run_sweep(configs)
"""

from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from happysimulator.instrumentation.summary import SimulationSummary


@dataclass
class RunConfig:
    """Configuration for one simulation run.

    Attributes:
        name: Human-readable label for this run.
        build_fn: Factory that constructs and returns a Simulation.
            Must be picklable (top-level function or callable class).
        seed: Optional random seed set before building the simulation.
    """

    name: str
    build_fn: Callable
    seed: int | None = None


@dataclass
class ParallelResult:
    """Result from a single parallel simulation run.

    Attributes:
        name: Label from the RunConfig.
        summary: SimulationSummary from the completed run.
        artifacts: User-defined data extracted via the extract_fn.
    """

    name: str
    summary: SimulationSummary
    artifacts: dict[str, Any] = field(default_factory=dict)


def _run_one(config: RunConfig) -> ParallelResult:
    """Worker function executed in a subprocess."""
    if config.seed is not None:
        random.seed(config.seed)
    sim = config.build_fn()
    summary = sim.run()
    return ParallelResult(name=config.name, summary=summary)


class ParallelRunner:
    """Run multiple independent simulations in parallel using processes.

    Uses ``concurrent.futures.ProcessPoolExecutor`` to distribute simulation
    runs across CPU cores. Each simulation is constructed inside its
    subprocess via a ``build_fn`` factory, avoiding pickling of complex
    simulation objects.

    Args:
        max_workers: Maximum number of worker processes. Defaults to
            the number of CPU cores.
    """

    def __init__(self, max_workers: int | None = None):
        self._max_workers = max_workers

    def run_sweep(self, configs: list[RunConfig]) -> list[ParallelResult]:
        """Run multiple simulation configurations in parallel.

        Args:
            configs: List of RunConfig objects, each defining a simulation
                to build and run.

        Returns:
            List of ParallelResult in the same order as configs.
        """
        if not configs:
            return []

        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(_run_one, cfg) for cfg in configs]
            return [f.result() for f in futures]

    def run_replicas(
        self,
        build_fn: Callable,
        n_replicas: int,
        base_seed: int = 42,
    ) -> list[ParallelResult]:
        """Run N replicas of the same simulation with different random seeds.

        Each replica gets a seed of ``base_seed + i`` where ``i`` is the
        replica index (0-based).

        Args:
            build_fn: Factory that constructs a Simulation. Must be picklable.
            n_replicas: Number of replicas to run.
            base_seed: Starting seed value.

        Returns:
            List of ParallelResult, one per replica.
        """
        configs = [
            RunConfig(
                name=f"replica_{i}",
                build_fn=build_fn,
                seed=base_seed + i,
            )
            for i in range(n_replicas)
        ]
        return self.run_sweep(configs)
