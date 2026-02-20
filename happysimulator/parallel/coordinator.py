"""Windowed barrier-based coordinator for cross-partition simulation.

The coordinator advances all partitions through time windows of size
``window_size``.  After each window, cross-partition events are exchanged
via outboxes, applying latency and packet-loss from ``PartitionLink``
declarations.
"""

from __future__ import annotations

import logging
import random
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.parallel.summary import ParallelSimulationSummary

if TYPE_CHECKING:
    from happysimulator.core.simulation import Simulation
    from happysimulator.parallel.link import PartitionLink

logger = logging.getLogger(__name__)


class WindowedCoordinator:
    """Barrier-based coordinator for cross-partition event exchange.

    Each iteration:
      1. EXECUTE — run all partitions in parallel up to ``T + window_size``
      2. EXCHANGE — deliver cross-partition events from outboxes
      3. ADVANCE — move T forward by ``window_size``

    Args:
        simulations: Partition name → Simulation mapping.
        links: Cross-partition link declarations.
        outboxes: Partition name → mutable outbox list (populated by routers).
        entity_to_partition: ``id(entity)`` → partition name mapping.
        window_size: Barrier window size in seconds.
        start_time: Simulation start time.
        end_time: Simulation end time.
        max_workers: Thread pool size.
        seed: Random seed for packet loss / latency sampling.
    """

    def __init__(
        self,
        simulations: dict[str, Simulation],
        links: list[PartitionLink],
        outboxes: dict[str, list[tuple[Event, Instant]]],
        entity_to_partition: dict[int, str],
        window_size: float,
        start_time: Instant,
        end_time: Instant,
        max_workers: int | None,
        seed: int = 42,
    ) -> None:
        self._simulations = simulations
        self._links = links
        self._outboxes = outboxes
        self._entity_to_partition = entity_to_partition
        self._window_size = window_size
        self._start_time = start_time
        self._end_time = end_time
        self._max_workers = max_workers or len(simulations)
        self._rng = random.Random(seed)

        # Build link lookup: (source_partition, dest_partition) → link
        self._link_map: dict[tuple[str, str], PartitionLink] = {}
        for link in links:
            self._link_map[(link.source_partition, link.dest_partition)] = link

    def run(self) -> ParallelSimulationSummary:
        """Execute the windowed barrier loop and return aggregate summary."""
        wall_start = _time.monotonic()
        current_time = self._start_time
        total_windows = 0
        total_cross_events = 0
        barrier_overhead = 0.0
        partition_wall_times: dict[str, float] = {
            name: 0.0 for name in self._simulations
        }

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            while current_time < self._end_time:
                window_end_s = current_time.to_seconds() + self._window_size
                # Clamp to end_time
                if self._end_time != Instant.Infinity:
                    end_s = self._end_time.to_seconds()
                    if window_end_s > end_s:
                        window_end_s = end_s
                window_end = Instant.from_seconds(window_end_s)

                # 1. EXECUTE (parallel)
                futures = {}
                for name, sim in self._simulations.items():
                    futures[pool.submit(self._run_partition_window, name, window_end)] = name

                for future in as_completed(futures):
                    name, elapsed = future.result()
                    partition_wall_times[name] += elapsed

                # 2. EXCHANGE (main thread, sequential)
                barrier_start = _time.monotonic()
                cross_events = self._exchange_events(window_end)
                total_cross_events += cross_events
                barrier_overhead += _time.monotonic() - barrier_start

                # 3. ADVANCE
                current_time = window_end
                total_windows += 1

                # Check if all partitions have exhausted their heaps
                if all(
                    not sim._event_heap.has_events()
                    for sim in self._simulations.values()
                ):
                    logger.info(
                        "All partition heaps exhausted at window %d (T=%r)",
                        total_windows, current_time,
                    )
                    break

        # Finalize each partition
        partition_summaries = {}
        for name, sim in self._simulations.items():
            sim._is_running = False
            sim._is_paused = False
            sim._summary = sim._build_summary()
            partition_summaries[name] = sim._summary

        wall_elapsed = _time.monotonic() - wall_start

        # Build aggregate summary via ParallelSimulation._build_summary pattern
        from happysimulator.instrumentation.summary import SimulationSummary

        total_events = sum(
            s.total_events_processed for s in partition_summaries.values()
        )
        duration_s = max(
            (s.duration_s for s in partition_summaries.values()), default=0.0
        )
        eps = total_events / duration_s if duration_s > 0 else 0.0

        merged_entities = {}
        for s in partition_summaries.values():
            merged_entities.update(s.entities)

        sequential_estimate = sum(partition_wall_times.values())
        speedup = sequential_estimate / wall_elapsed if wall_elapsed > 0 else 1.0
        n = len(partition_summaries)
        efficiency = speedup / n if n > 0 else 1.0
        coord_eff = 1.0 - (barrier_overhead / wall_elapsed) if wall_elapsed > 0 else 1.0

        return ParallelSimulationSummary(
            duration_s=duration_s,
            total_events_processed=total_events,
            events_per_second=eps,
            wall_clock_seconds=wall_elapsed,
            partitions=partition_summaries,
            entities=merged_entities,
            partition_wall_times=partition_wall_times,
            speedup=speedup,
            parallelism_efficiency=efficiency,
            total_windows=total_windows,
            total_cross_partition_events=total_cross_events,
            window_size_s=self._window_size,
            barrier_overhead_seconds=barrier_overhead,
            coordination_efficiency=coord_eff,
        )

    def _run_partition_window(
        self, name: str, window_end: Instant
    ) -> tuple[str, float]:
        """Run one partition up to ``window_end`` and return (name, elapsed)."""
        t0 = _time.monotonic()
        self._simulations[name]._run_window(window_end)
        return name, _time.monotonic() - t0

    def _exchange_events(self, window_end: Instant) -> int:
        """Deliver cross-partition events from outboxes.  Returns count delivered."""
        delivered = 0
        for source_name, outbox in self._outboxes.items():
            for event, send_time in outbox:
                target_id = id(event.target)
                dest_name = self._entity_to_partition.get(target_id)
                if dest_name is None:
                    logger.warning(
                        "Cross-partition event targets unknown entity: %r",
                        event,
                    )
                    continue

                link = self._link_map.get((source_name, dest_name))
                if link is None:
                    raise RuntimeError(
                        f"No PartitionLink from '{source_name}' to "
                        f"'{dest_name}' for cross-partition event {event!r}"
                    )

                # Packet loss
                if link.packet_loss > 0 and self._rng.random() < link.packet_loss:
                    continue

                # Latency override
                if link.latency is not None:
                    sampled = link.latency.sample()
                    event.time = send_time + sampled
                else:
                    # Validate min_latency
                    delay = (event.time - send_time).to_seconds()
                    if delay < link.min_latency - 1e-12:
                        raise RuntimeError(
                            f"Cross-partition event violates min_latency: "
                            f"delay={delay:.6f}s < min_latency={link.min_latency}s "
                            f"(event={event!r}, link={source_name}→{dest_name})"
                        )

                # Inject into destination partition
                self._simulations[dest_name].schedule(event)
                delivered += 1

            outbox.clear()

        return delivered
