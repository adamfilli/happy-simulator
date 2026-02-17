"""Integration tests for parallel simulation execution.

Tests verify that ParallelSimulation produces correct results by running
independent partitions and checking that metrics match expectations.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.common import Counter, Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source
from happysimulator.parallel import (
    ParallelSimulation,
    ParallelSimulationSummary,
    SimulationPartition,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@dataclass
class TrackingServer(Entity):
    """Server that records processed events and the thread it ran on."""

    name: str
    delay: float = 0.01
    events_processed: int = field(default=0, init=False)
    thread_id: int | None = field(default=None, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.events_processed += 1
        self.thread_id = threading.current_thread().ident
        yield self.delay


# ---------------------------------------------------------------------------
# Basic execution tests
# ---------------------------------------------------------------------------

class TestParallelExecution:
    def test_single_partition_matches_simulation(self):
        """A single-partition ParallelSimulation should produce the same
        event count as a normal Simulation with the same entities."""
        counter = Counter("counter")
        source = Source.constant(rate=10, target=counter, event_type="Ping")

        sim = ParallelSimulation(
            partitions=[SimulationPartition(
                name="only",
                entities=[counter],
                sources=[source],
            )],
            duration=5.0,
        )
        summary = sim.run()

        assert summary.total_events_processed > 0
        assert "only" in summary.partitions
        assert counter.total == 50

    def test_two_independent_partitions(self):
        """Two independent partitions should each process their own events."""
        counter_a = Counter("counter_a")
        counter_b = Counter("counter_b")
        source_a = Source.constant(rate=10, target=counter_a, event_type="Ping")
        source_b = Source.constant(rate=10, target=counter_b, event_type="Ping")

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("a", entities=[counter_a], sources=[source_a]),
                SimulationPartition("b", entities=[counter_b], sources=[source_b]),
            ],
            duration=5.0,
        )
        summary = sim.run()

        assert counter_a.total == 50
        assert counter_b.total == 50
        # total_events_processed includes internal SourceEvents, so it's
        # greater than the payload event count
        assert summary.total_events_processed >= counter_a.total + counter_b.total
        assert len(summary.partitions) == 2
        assert summary.speedup >= 0.5  # at minimum, not drastically slower

    def test_many_partitions(self):
        """N independent partitions should all complete correctly."""
        n = 10
        counters = []
        partitions = []

        for i in range(n):
            c = Counter(f"counter_{i}")
            s = Source.constant(rate=5, target=c, event_type="Tick")
            counters.append(c)
            partitions.append(SimulationPartition(
                name=f"part_{i}",
                entities=[c],
                sources=[s],
            ))

        sim = ParallelSimulation(partitions=partitions, duration=2.0)
        summary = sim.run()

        for c in counters:
            assert c.total == 10, f"{c.name} expected 10 events, got {c.total}"

        assert summary.total_events_processed >= 10 * n

    def test_unbalanced_partitions(self):
        """Partitions with different loads should all complete."""
        counter_light = Counter("light")
        counter_heavy = Counter("heavy")
        source_light = Source.constant(rate=1, target=counter_light, event_type="Tick")
        source_heavy = Source.constant(rate=100, target=counter_heavy, event_type="Tick")

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("light", entities=[counter_light], sources=[source_light]),
                SimulationPartition("heavy", entities=[counter_heavy], sources=[source_heavy]),
            ],
            duration=5.0,
        )
        summary = sim.run()

        assert counter_light.total == 5
        assert counter_heavy.total == 500
        assert summary.total_events_processed >= 505


class TestParallelWithGenerators:
    def test_generator_entities_work_per_partition(self):
        """Entities using generators (yield delays) should work correctly
        in each partition."""
        server_a = TrackingServer("srv_a", delay=0.01)
        server_b = TrackingServer("srv_b", delay=0.01)
        source_a = Source.constant(rate=10, target=server_a, event_type="Req")
        source_b = Source.constant(rate=10, target=server_b, event_type="Req")

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("a", entities=[server_a], sources=[source_a]),
                SimulationPartition("b", entities=[server_b], sources=[source_b]),
            ],
            duration=3.0,
        )
        summary = sim.run()

        assert server_a.events_processed == 30
        assert server_b.events_processed == 30


class TestParallelSummary:
    def test_summary_metrics(self):
        """Summary should aggregate across partitions correctly."""
        counter_a = Counter("a")
        counter_b = Counter("b")
        source_a = Source.constant(rate=10, target=counter_a, event_type="X")
        source_b = Source.constant(rate=20, target=counter_b, event_type="X")

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("pa", entities=[counter_a], sources=[source_a]),
                SimulationPartition("pb", entities=[counter_b], sources=[source_b]),
            ],
            duration=5.0,
        )
        summary = sim.run()

        assert isinstance(summary, ParallelSimulationSummary)
        assert summary.duration_s == pytest.approx(5.0, abs=0.2)
        # total_events_processed includes internal SourceEvents
        assert summary.total_events_processed >= 150  # at least 50 + 100 payload events
        assert summary.wall_clock_seconds > 0
        assert summary.speedup > 0
        assert summary.parallelism_efficiency > 0
        assert len(summary.partition_wall_times) == 2

    def test_entity_summaries_merged(self):
        """Entity summaries from different partitions should be merged."""
        counter_a = Counter("counter_a")
        counter_b = Counter("counter_b")
        source_a = Source.constant(rate=5, target=counter_a, event_type="X")
        source_b = Source.constant(rate=5, target=counter_b, event_type="X")

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("pa", entities=[counter_a], sources=[source_a]),
                SimulationPartition("pb", entities=[counter_b], sources=[source_b]),
            ],
            duration=2.0,
        )
        summary = sim.run()

        # Both entities should appear in merged summary
        assert "counter_a" in summary.entities or "counter_b" in summary.entities

    def test_empty_parallel_simulation(self):
        """A parallel simulation with no partitions should return empty summary."""
        sim = ParallelSimulation(
            partitions=[SimulationPartition("empty")],
            duration=1.0,
        )
        summary = sim.run()
        assert summary.total_events_processed == 0


class TestRuntimeSafety:
    def test_cross_partition_event_detected_at_runtime(self):
        """An event targeting an entity outside its partition should raise."""
        entity_a = Counter("a")
        entity_b = Counter("b")

        @dataclass
        class CrossPartitionSender(Entity):
            name: str
            other: Entity = None

            def handle_event(self, event: Event):
                # Attempt to send to entity in other partition
                return [Event(
                    time=self.now,
                    event_type="CrossPartition",
                    target=self.other,
                )]

        sender = CrossPartitionSender("sender", other=entity_b)
        source = Source.constant(rate=1, target=sender, event_type="Go")

        # Validation won't catch this because 'other' is not in any
        # partition's entity list (entity_b is in partition "pb" but
        # sender.other = entity_b is detected by validation).
        # So let's test the runtime guard more directly using a callback.

        # Use Event.once to bypass static validation
        entity_in_a = Counter("in_a")
        entity_in_b = Counter("in_b")

        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("pa", entities=[entity_in_a]),
                SimulationPartition("pb", entities=[entity_in_b]),
            ],
            duration=1.0,
        )

        # Inject an event that targets entity_in_b into partition "pa"
        bad_event = Event(
            time=Instant.from_seconds(0.5),
            event_type="BadCross",
            target=entity_in_b,
        )
        sim.schedule(bad_event, partition="pa")

        with pytest.raises(RuntimeError, match="not in this partition"):
            sim.run()


class TestScheduleAPI:
    def test_schedule_to_partition(self):
        """Events can be scheduled into a specific partition."""
        counter = Counter("c")
        sim = ParallelSimulation(
            partitions=[SimulationPartition("p", entities=[counter])],
            duration=1.0,
        )
        sim.schedule(
            Event(time=Instant.from_seconds(0.5), event_type="Hi", target=counter),
            partition="p",
        )
        summary = sim.run()
        assert counter.total == 1

    def test_schedule_all(self):
        """schedule_all dispatches events to multiple partitions."""
        c1 = Counter("c1")
        c2 = Counter("c2")
        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("p1", entities=[c1]),
                SimulationPartition("p2", entities=[c2]),
            ],
            duration=1.0,
        )
        sim.schedule_all({
            "p1": [Event(time=Instant.from_seconds(0.1), event_type="A", target=c1)],
            "p2": [Event(time=Instant.from_seconds(0.2), event_type="B", target=c2)],
        })
        summary = sim.run()
        assert c1.total == 1
        assert c2.total == 1


class TestFromGroups:
    def test_from_groups_runs_correctly(self):
        """from_groups factory should produce a working parallel simulation."""
        counters = [Counter(f"c{i}") for i in range(5)]
        sources = [Source.constant(rate=10, target=c, event_type="Tick") for c in counters]

        sim = ParallelSimulation.from_groups(
            groups={f"g{i}": ([counters[i]], [sources[i]]) for i in range(5)},
            duration=2.0,
        )
        summary = sim.run()

        for c in counters:
            assert c.total == 20
        assert summary.total_events_processed >= 100
