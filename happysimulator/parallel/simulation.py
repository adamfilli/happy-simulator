"""Parallel simulation orchestrator.

ParallelSimulation partitions a model into independently executing
sub-simulations that run on threads via ``ThreadPoolExecutor``.  When
``PartitionLink`` objects are provided, a ``WindowedCoordinator`` synchronizes
partitions at barrier boundaries sized to the minimum cross-partition latency.
"""

from __future__ import annotations

import logging
import sys
import time as _time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.parallel.summary import ParallelSimulationSummary
from happysimulator.parallel.validation import build_entity_sets, validate_partitions

if TYPE_CHECKING:
    from happysimulator.parallel.link import PartitionLink
    from happysimulator.parallel.partition import SimulationPartition

logger = logging.getLogger(__name__)


class ParallelSimulation:
    """Run partitioned simulations in parallel.

    When no ``links`` are provided the partitions are fully independent and
    execute concurrently without synchronization.  When links exist, a
    ``WindowedCoordinator`` manages barrier-based time windows.

    Args:
        partitions: List of partition declarations.
        start_time: Simulation start (default: Epoch).
        end_time: Simulation end (default: Infinity for auto-terminate).
        duration: Alternative to end_time — simulated seconds to run.
        max_workers: Thread pool size (default: number of partitions).
        links: Cross-partition link declarations.
        window_size: Override barrier window size (default: min link latency).
        seed: Random seed for coordinator (packet loss, latency sampling).
    """

    def __init__(
        self,
        partitions: list[SimulationPartition],
        *,
        start_time: Instant | None = None,
        end_time: Instant | None = None,
        duration: float | None = None,
        max_workers: int | None = None,
        links: list[PartitionLink] | None = None,
        window_size: float | None = None,
        seed: int = 42,
    ) -> None:
        if not partitions:
            raise ValueError("At least one partition is required")

        if duration is not None and end_time is not None:
            raise ValueError("Cannot specify both 'duration' and 'end_time'")

        self._start_time = start_time or Instant.Epoch
        if duration is not None:
            self._end_time = self._start_time + duration
        elif end_time is not None:
            self._end_time = end_time
        else:
            self._end_time = Instant.Infinity

        self._links: list[PartitionLink] = links or []
        self._max_workers = max_workers or len(partitions)
        self._seed = seed

        # Validate partitions and links
        validate_partitions(partitions, self._links, window_size)

        # Compute window size
        if self._links:
            min_link_latency = min(link.min_latency for link in self._links)
            self._window_size = window_size if window_size is not None else min_link_latency
        else:
            self._window_size = 0.0

        # Build per-partition Simulation instances
        self._partitions = partitions
        self._simulations: dict[str, Simulation] = {}
        self._entity_sets = build_entity_sets(partitions)

        for p in partitions:
            sim = Simulation(
                start_time=self._start_time,
                end_time=self._end_time,
                sources=p.sources if p.sources else None,
                entities=p.entities if p.entities else None,
                probes=p.probes if p.probes else None,
                trace_recorder=p.trace_recorder,
                fault_schedule=p.fault_schedule,
            )
            self._simulations[p.name] = sim

        # Install event routers if links exist
        self._outboxes: dict[str, list[tuple[Event, Instant]]] = {}
        if self._links:
            self._install_routers()

        # GIL warning
        try:
            if hasattr(sys, "_is_gil_enabled") and sys._is_gil_enabled():
                warnings.warn(
                    "GIL is enabled.  ParallelSimulation uses threads and "
                    "benefits from free-threaded Python (python -X gil=0).",
                    stacklevel=2,
                )
        except Exception:
            pass

    def _install_routers(self) -> None:
        """Install event routers on each partition's Simulation."""
        from happysimulator.parallel.routing import make_event_router

        # Build entity-to-partition map (include sources/probes — they self-schedule)
        entity_to_partition: dict[int, str] = {}
        for p in self._partitions:
            for entity in p.entities:
                entity_to_partition[id(entity)] = p.name
            for source in p.sources:
                entity_to_partition[id(source)] = p.name
            for probe in p.probes:
                entity_to_partition[id(probe)] = p.name

        # Build linked entity sets per partition
        linked_from: dict[str, set[int]] = {p.name: set() for p in self._partitions}
        for link in self._links:
            dest_eids = self._entity_sets[link.dest_partition]
            linked_from[link.source_partition].update(dest_eids)

        for p in self._partitions:
            outbox: list[tuple[Event, Instant]] = []
            self._outboxes[p.name] = outbox
            router = make_event_router(
                partition_name=p.name,
                local_entity_ids=self._entity_sets[p.name],
                linked_entity_ids=frozenset(linked_from[p.name]),
                outbox=outbox,
            )
            self._simulations[p.name]._event_router = router

    @property
    def simulations(self) -> dict[str, Simulation]:
        """Access per-partition Simulation instances."""
        return dict(self._simulations)

    def schedule(self, events: Event | list[Event], *, partition: str) -> None:
        """Inject events into a specific partition's simulation."""
        if partition not in self._simulations:
            raise ValueError(f"Unknown partition: '{partition}'")
        self._simulations[partition].schedule(events)

    def run(self) -> ParallelSimulationSummary:
        """Execute all partitions and return an aggregate summary."""
        if self._links:
            return self._run_coordinated()
        return self._run_independent()

    def _run_independent(self) -> ParallelSimulationSummary:
        """Fire-and-forget: each partition runs its full simulation independently."""
        wall_start = _time.monotonic()
        partition_summaries = {}
        partition_wall_times: dict[str, float] = {}

        def run_one(name: str) -> tuple[str, float]:
            t0 = _time.monotonic()
            summary = self._simulations[name].run()
            elapsed = _time.monotonic() - t0
            partition_summaries[name] = summary
            return name, elapsed

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {
                pool.submit(run_one, name): name
                for name in self._simulations
            }
            for future in as_completed(futures):
                name, elapsed = future.result()
                partition_wall_times[name] = elapsed

        wall_elapsed = _time.monotonic() - wall_start
        return self._build_summary(
            partition_summaries, partition_wall_times, wall_elapsed,
        )

    def _run_coordinated(self) -> ParallelSimulationSummary:
        """Delegate to WindowedCoordinator for barrier-based execution."""
        from happysimulator.parallel.coordinator import WindowedCoordinator

        # Build entity-to-partition map for the coordinator
        # Include sources/probes since they self-schedule
        entity_to_partition: dict[int, str] = {}
        for p in self._partitions:
            for entity in p.entities:
                entity_to_partition[id(entity)] = p.name
            for source in p.sources:
                entity_to_partition[id(source)] = p.name
            for probe in p.probes:
                entity_to_partition[id(probe)] = p.name

        coordinator = WindowedCoordinator(
            simulations=self._simulations,
            links=self._links,
            outboxes=self._outboxes,
            entity_to_partition=entity_to_partition,
            window_size=self._window_size,
            start_time=self._start_time,
            end_time=self._end_time,
            max_workers=self._max_workers,
            seed=self._seed,
        )
        return coordinator.run()

    def _build_summary(
        self,
        partition_summaries: dict[str, object],
        partition_wall_times: dict[str, float],
        wall_elapsed: float,
        total_windows: int = 0,
        total_cross_partition_events: int = 0,
        barrier_overhead: float = 0.0,
    ) -> ParallelSimulationSummary:
        """Merge per-partition summaries into an aggregate."""
        from happysimulator.instrumentation.summary import SimulationSummary

        total_events = sum(
            s.total_events_processed
            for s in partition_summaries.values()
            if isinstance(s, SimulationSummary)
        )
        duration_s = max(
            (s.duration_s for s in partition_summaries.values()
             if isinstance(s, SimulationSummary)),
            default=0.0,
        )
        eps = total_events / duration_s if duration_s > 0 else 0.0

        # Merge entity summaries
        merged_entities = {}
        for s in partition_summaries.values():
            if isinstance(s, SimulationSummary):
                merged_entities.update(s.entities)

        # Sequential estimate = sum of partition wall times
        sequential_estimate = sum(partition_wall_times.values())
        speedup = sequential_estimate / wall_elapsed if wall_elapsed > 0 else 1.0
        n_partitions = len(partition_summaries)
        efficiency = speedup / n_partitions if n_partitions > 0 else 1.0

        coord_efficiency = (
            1.0 - (barrier_overhead / wall_elapsed)
            if wall_elapsed > 0 else 1.0
        )

        return ParallelSimulationSummary(
            duration_s=duration_s,
            total_events_processed=total_events,
            events_per_second=eps,
            wall_clock_seconds=wall_elapsed,
            partitions={
                name: s for name, s in partition_summaries.items()
                if isinstance(s, SimulationSummary)
            },
            entities=merged_entities,
            partition_wall_times=partition_wall_times,
            speedup=speedup,
            parallelism_efficiency=efficiency,
            total_windows=total_windows,
            total_cross_partition_events=total_cross_partition_events,
            window_size_s=self._window_size,
            barrier_overhead_seconds=barrier_overhead,
            coordination_efficiency=coord_efficiency,
        )
