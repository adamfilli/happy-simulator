"""Parallel simulation executor.

ParallelSimulation runs independent entity groups (partitions) on separate
threads for true parallel execution. Each partition runs a full, independent
Simulation instance internally. No cross-partition entity references or
events are permitted.

Requires free-threaded Python (3.13t+) for actual parallelism. With the
standard GIL-enabled interpreter, partitions run concurrently but not in
parallel.
"""

from __future__ import annotations

import logging
import sys
import concurrent.futures
from typing import TYPE_CHECKING

from happysimulator.core.simulation import Simulation
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.summary import SimulationSummary
from happysimulator.parallel.partition import SimulationPartition
from happysimulator.parallel.summary import ParallelSimulationSummary
from happysimulator.parallel.validation import validate_partitions, build_entity_id_set

if TYPE_CHECKING:
    from happysimulator.load.source import Source

logger = logging.getLogger(__name__)


class ParallelSimulation:
    """Run independent entity groups in parallel.

    Each partition gets its own Simulation instance running on a separate
    thread. Partitions must be truly independent — no shared entities,
    resources, or cross-partition event targets.

    Args:
        partitions: Independent entity groups to run in parallel.
        start_time: When all partitions begin. Defaults to Instant.Epoch.
        end_time: When all partitions stop. Defaults to Instant.Infinity.
        duration: Simulation duration in seconds (alternative to end_time).
        max_workers: Maximum threads. Defaults to number of partitions.

    Raises:
        ValueError: If partitions have cross-references or other violations.

    Example::

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("us", entities=[us_srv], sources=[us_src]),
                SimulationPartition("eu", entities=[eu_srv], sources=[eu_src]),
            ],
            duration=60.0,
        )
        summary = sim.run()
        print(f"Speedup: {summary.speedup:.1f}x")
    """

    def __init__(
        self,
        partitions: list[SimulationPartition],
        start_time: Instant | None = None,
        end_time: Instant | None = None,
        duration: float | None = None,
        max_workers: int | None = None,
    ):
        if duration is not None and end_time is not None:
            raise ValueError("Cannot specify both 'duration' and 'end_time'")

        self._partitions = list(partitions)
        self._start_time = start_time or Instant.Epoch
        self._duration = duration
        self._end_time = end_time
        self._max_workers = max_workers

        # Validate partition independence
        validate_partitions(self._partitions)

        # Build per-partition simulations
        self._simulations: dict[str, Simulation] = {}
        self._entity_id_sets: dict[str, frozenset[int]] = {}
        self._build_simulations()

        # Scheduled events waiting for run()
        self._pending_schedules: dict[str, list[Event]] = {
            p.name: [] for p in self._partitions
        }

        self._summary: ParallelSimulationSummary | None = None

        logger.info(
            "ParallelSimulation initialized: %d partition(s)",
            len(self._partitions),
        )

    @property
    def partitions(self) -> dict[str, Simulation]:
        """Access individual partition simulations by name."""
        return dict(self._simulations)

    @property
    def summary(self) -> ParallelSimulationSummary | None:
        """Summary from the most recent run(), or None if not yet run."""
        return self._summary

    def schedule(
        self,
        events: Event | list[Event],
        partition: str,
    ) -> None:
        """Schedule events into a specific partition.

        Args:
            events: Event(s) to schedule.
            partition: Name of the target partition.

        Raises:
            KeyError: If partition name is not found.
        """
        if partition not in self._simulations:
            raise KeyError(
                f"Unknown partition '{partition}'. "
                f"Available: {list(self._simulations.keys())}"
            )
        self._simulations[partition].schedule(events)

    def schedule_all(self, events_by_partition: dict[str, list[Event]]) -> None:
        """Schedule events into multiple partitions at once.

        Args:
            events_by_partition: Mapping of partition name to events.

        Raises:
            KeyError: If any partition name is not found.
        """
        for partition, events in events_by_partition.items():
            self.schedule(events, partition)

    @staticmethod
    def from_groups(
        groups: dict[str, tuple[list, list]],
        start_time: Instant | None = None,
        end_time: Instant | None = None,
        duration: float | None = None,
        max_workers: int | None = None,
    ) -> ParallelSimulation:
        """Create a ParallelSimulation from a dict of (entities, sources) tuples.

        Convenience factory for the common case of grouping entities with
        their corresponding sources.

        Args:
            groups: Mapping of partition name to (entities, sources) tuples.
            start_time: When all partitions begin.
            end_time: When all partitions stop.
            duration: Simulation duration in seconds.
            max_workers: Maximum threads.

        Returns:
            A configured ParallelSimulation.
        """
        partitions = []
        for name, (entities, sources) in groups.items():
            partitions.append(SimulationPartition(
                name=name,
                entities=entities,
                sources=sources,
            ))
        return ParallelSimulation(
            partitions=partitions,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            max_workers=max_workers,
        )

    def run(self) -> ParallelSimulationSummary:
        """Execute all partitions in parallel and return merged results.

        Each partition runs as a standard Simulation on a separate thread.
        The method blocks until all partitions complete.

        Returns:
            ParallelSimulationSummary with aggregate and per-partition metrics.
        """
        num_partitions = len(self._simulations)

        if num_partitions == 0:
            return self._empty_summary()

        # Warn if GIL will prevent true parallelism
        if num_partitions > 1:
            self._warn_if_gil_enabled()

        workers = self._max_workers or num_partitions

        logger.info(
            "Starting parallel simulation: %d partition(s), %d worker(s)",
            num_partitions, workers,
        )

        if num_partitions == 1:
            # Single partition — skip thread pool overhead
            name = next(iter(self._simulations))
            sim = self._simulations[name]
            result = sim.run()
            results = {name: result}
        else:
            results = self._run_parallel(workers)

        self._summary = self._merge_summaries(results)

        logger.info(
            "Parallel simulation complete: %d events, %.2fx speedup, %.0f%% efficiency",
            self._summary.total_events_processed,
            self._summary.speedup,
            self._summary.parallelism_efficiency * 100,
        )

        return self._summary

    def _build_simulations(self) -> None:
        """Create a Simulation instance for each partition."""
        for part in self._partitions:
            start = part.start_time or self._start_time
            end = part.end_time or self._end_time

            kwargs: dict = {
                "start_time": start,
                "sources": list(part.sources),
                "entities": list(part.entities),
                "probes": list(part.probes),
                "trace_recorder": part.trace_recorder,
                "fault_schedule": part.fault_schedule,
            }

            if self._duration is not None and end is None:
                kwargs["duration"] = self._duration
            else:
                kwargs["end_time"] = end

            sim = Simulation(**kwargs)

            # Install runtime cross-partition guard
            entity_ids = build_entity_id_set(part)
            self._entity_id_sets[part.name] = entity_ids
            sim._event_validator = _make_partition_guard(part.name, entity_ids)

            self._simulations[part.name] = sim

    def _run_parallel(self, workers: int) -> dict[str, SimulationSummary]:
        """Execute partitions on a thread pool."""
        results: dict[str, SimulationSummary] = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(sim.run): name
                for name, sim in self._simulations.items()
            }

            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    logger.error(
                        "Partition '%s' failed with exception", name,
                        exc_info=True,
                    )
                    raise

        return results

    def _merge_summaries(
        self,
        results: dict[str, SimulationSummary],
    ) -> ParallelSimulationSummary:
        """Aggregate per-partition summaries into a parallel summary."""
        wall_times = {
            name: summary.wall_clock_seconds
            for name, summary in results.items()
        }
        total_wall = max(wall_times.values()) if wall_times else 0.0
        sequential_estimate = sum(wall_times.values())

        # Merge entity summaries (disjoint by construction)
        merged_entities = {}
        for summary in results.values():
            merged_entities.update(summary.entities)

        total_events = sum(s.total_events_processed for s in results.values())
        total_cancelled = sum(s.events_cancelled for s in results.values())

        return ParallelSimulationSummary(
            duration_s=max(
                (s.duration_s for s in results.values()), default=0.0
            ),
            total_events_processed=total_events,
            events_cancelled=total_cancelled,
            events_per_second=sum(
                s.events_per_second for s in results.values()
            ),
            wall_clock_seconds=total_wall,
            partitions=results,
            entities=merged_entities,
            partition_wall_times=wall_times,
            speedup=(
                sequential_estimate / total_wall
                if total_wall > 0 else 1.0
            ),
            parallelism_efficiency=(
                (sequential_estimate / total_wall) / len(results)
                if total_wall > 0 and len(results) > 0 else 1.0
            ),
        )

    def _empty_summary(self) -> ParallelSimulationSummary:
        """Return an empty summary when there are no partitions."""
        return ParallelSimulationSummary(
            duration_s=0.0,
            total_events_processed=0,
            events_cancelled=0,
            events_per_second=0.0,
            wall_clock_seconds=0.0,
            partitions={},
            entities={},
            partition_wall_times={},
            speedup=1.0,
            parallelism_efficiency=1.0,
        )

    @staticmethod
    def _warn_if_gil_enabled() -> None:
        """Log a warning if the GIL is enabled (no true parallelism)."""
        gil_check = getattr(sys, "_is_gil_enabled", None)
        if gil_check is not None:
            # Python 3.13+ exposes this
            if gil_check():
                logger.warning(
                    "ParallelSimulation: GIL is enabled. Partitions will run "
                    "concurrently but not in parallel. For true parallelism, "
                    "use free-threaded Python: python3.13t"
                )
        else:
            # Pre-3.13 — GIL is always enabled
            logger.warning(
                "ParallelSimulation: Python %s does not support free-threaded "
                "mode. Partitions will run concurrently but not in parallel.",
                sys.version.split()[0],
            )


def _make_partition_guard(
    partition_name: str,
    entity_ids: frozenset[int],
) -> callable:
    """Create an event validator that rejects cross-partition targets.

    The validator is installed on each partition's Simulation instance.
    It checks that every dispatched event targets an entity within the
    partition. CallbackEntity targets (from Event.once()) are exempt
    since they are created inline and don't belong to any partition.
    """
    from happysimulator.core.callback_entity import CallbackEntity

    def validator(event: Event) -> None:
        target = event.target
        # CallbackEntity instances are ephemeral (Event.once) — exempt
        if isinstance(target, CallbackEntity):
            return
        target_id = id(target)
        if target_id not in entity_ids:
            target_name = getattr(target, "name", repr(target))
            raise RuntimeError(
                f"Partition '{partition_name}': event '{event.event_type}' "
                f"targets entity '{target_name}' which is not in this "
                f"partition. Cross-partition events are not supported."
            )

    return validator
