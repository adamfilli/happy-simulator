"""Parallel partitioned simulation vs single-threaded benchmark.

Builds N identical server→sink chains and runs them two ways:
1. All chains in a single Simulation (sequential baseline)
2. Each chain in its own SimulationPartition via ParallelSimulation

Reports wall clock for each and computes speedup.
"""

from __future__ import annotations

import random
import sys
import time
import tracemalloc

from happysimulator import (
    Event,
    Instant,
    QueuedResource,
    Simulation,
    Sink,
    Source,
)
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.parallel import ParallelSimulation, SimulationPartition
from tests.perf.runner import BenchmarkResult

BASE_EVENT_COUNT = 200_000
WARMUP_EVENTS = 500
NUM_PARTITIONS = 4


class _MinimalServer(QueuedResource):
    """Server with near-zero service time for throughput testing."""

    def __init__(self, name: str, downstream):
        super().__init__(name, policy=FIFOQueue())
        self._downstream = downstream

    def handle_queued_event(self, event: Event):
        yield 0.0
        return [
            Event(
                time=self.now,
                event_type="Done",
                target=self._downstream,
                context=event.context,
            )
        ]


def _build_chains(n: int, events_per_chain: int):
    """Build n independent server→sink chains with sources.

    Returns:
        List of (server, sink, source) tuples.
    """
    rate = events_per_chain * 10  # high rate → short duration
    duration_s = events_per_chain / rate

    chains = []
    for i in range(n):
        sink = Sink(f"Sink_{i}")
        server = _MinimalServer(f"Server_{i}", downstream=sink)
        source = Source.constant(
            rate=rate,
            target=server,
            name=f"Source_{i}",
            stop_after=duration_s,
        )
        chains.append((server, sink, source, duration_s))
    return chains


def _run_sequential(n: int, events_per_chain: int) -> tuple[int, float]:
    """Run all chains in a single Simulation."""
    random.seed(42)
    chains = _build_chains(n, events_per_chain)

    servers = [c[0] for c in chains]
    sinks = [c[1] for c in chains]
    sources = [c[2] for c in chains]
    duration_s = chains[0][3]

    sim = Simulation(
        end_time=Instant.from_seconds(duration_s + 0.001),
        sources=sources,
        entities=servers + sinks,
    )

    start = time.perf_counter()
    summary = sim.run()
    wall = time.perf_counter() - start
    return summary.total_events_processed, wall


def _run_parallel(n: int, events_per_chain: int) -> tuple[int, float, float]:
    """Run each chain in its own partition via ParallelSimulation.

    Returns:
        (total_events, wall_clock, speedup_reported_by_summary)
    """
    random.seed(42)
    chains = _build_chains(n, events_per_chain)
    duration_s = chains[0][3]

    partitions = []
    for i, (server, sink, source, _) in enumerate(chains):
        partitions.append(
            SimulationPartition(
                name=f"P{i}",
                entities=[server, sink],
                sources=[source],
            )
        )

    ps = ParallelSimulation(
        partitions=partitions,
        duration=duration_s + 0.001,
        max_workers=n,
    )

    start = time.perf_counter()
    summary = ps.run()
    wall = time.perf_counter() - start
    return summary.total_events_processed, wall, summary.speedup


def _tracemalloc_safe() -> bool:
    """Return True if tracemalloc can be used safely.

    tracemalloc deadlocks under free-threaded Python when multiple threads
    allocate memory concurrently, so we skip memory tracking in that case.
    """
    try:
        return sys._is_gil_enabled()
    except AttributeError:
        return True  # older Python — GIL is always on


def run(*, scale: float = 1.0) -> BenchmarkResult:
    """Run the parallel partition benchmark."""
    events_per_chain = int(BASE_EVENT_COUNT * scale) // NUM_PARTITIONS
    use_tracemalloc = _tracemalloc_safe()

    # Warm-up (small sequential run)
    _run_sequential(NUM_PARTITIONS, WARMUP_EVENTS)

    # Measured sequential run
    if use_tracemalloc:
        tracemalloc.reset_peak()
    seq_events, seq_wall = _run_sequential(NUM_PARTITIONS, events_per_chain)
    seq_peak = tracemalloc.get_traced_memory()[1] if use_tracemalloc else 0

    # Measured parallel run
    if use_tracemalloc:
        tracemalloc.reset_peak()
    par_events, par_wall, par_internal_speedup = _run_parallel(
        NUM_PARTITIONS, events_per_chain,
    )
    par_peak = tracemalloc.get_traced_memory()[1] if use_tracemalloc else 0

    # Use the parallel run as the primary result (wall clock, events/sec)
    # and report sequential as the comparison baseline in extras.
    eps = par_events / par_wall if par_wall > 0 else 0.0
    measured_speedup = seq_wall / par_wall if par_wall > 0 else 0.0

    return BenchmarkResult(
        name="parallel_partition",
        events_processed=par_events,
        wall_clock_s=par_wall,
        events_per_second=eps,
        peak_memory_mb=par_peak / (1024 * 1024),
        extra={
            "num_partitions": NUM_PARTITIONS,
            "sequential_wall_s": round(seq_wall, 4),
            "sequential_events_per_sec": round(seq_events / seq_wall, 1) if seq_wall > 0 else 0.0,
            "sequential_peak_mem_mb": round(seq_peak / (1024 * 1024), 2),
            "measured_speedup": round(measured_speedup, 3),
            "internal_speedup": round(par_internal_speedup, 3),
            "gil_enabled": 1.0 if use_tracemalloc else 0.0,
        },
    )
