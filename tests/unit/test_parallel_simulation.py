"""Unit tests for parallel simulation types and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.parallel import (
    ParallelSimulation,
    ParallelSimulationSummary,
    SimulationPartition,
)
from happysimulator.parallel.validation import (
    build_entity_id_set,
    validate_partitions,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@dataclass
class SimpleEntity(Entity):
    """Minimal entity for testing."""

    name: str
    downstream: Entity | None = None
    events_received: int = field(default=0, init=False)

    def handle_event(self, event: Event):
        self.events_received += 1
        if self.downstream is not None:
            return [Event(time=self.now, event_type="Forward", target=self.downstream)]
        return None


@dataclass
class ListRefEntity(Entity):
    """Entity that holds a list of other entities."""

    name: str
    targets: list[Entity] = field(default_factory=list)

    def handle_event(self, event: Event):
        return None


@dataclass
class DictRefEntity(Entity):
    """Entity that holds a dict of other entities."""

    name: str
    peers: dict[str, Entity] = field(default_factory=dict)

    def handle_event(self, event: Event):
        return None


# ---------------------------------------------------------------------------
# SimulationPartition dataclass tests
# ---------------------------------------------------------------------------

class TestSimulationPartition:
    def test_basic_construction(self):
        e = SimpleEntity("a")
        p = SimulationPartition(name="p1", entities=[e])
        assert p.name == "p1"
        assert p.entities == [e]
        assert p.sources == []
        assert p.probes == []

    def test_defaults(self):
        p = SimulationPartition(name="empty")
        assert p.entities == []
        assert p.start_time is None
        assert p.end_time is None
        assert p.fault_schedule is None


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestValidatePartitions:
    def test_empty_partitions_raises(self):
        with pytest.raises(ValueError, match="At least one partition"):
            validate_partitions([])

    def test_duplicate_partition_names_raises(self):
        e1 = SimpleEntity("a")
        e2 = SimpleEntity("b")
        with pytest.raises(ValueError, match="Duplicate partition names"):
            validate_partitions([
                SimulationPartition(name="same", entities=[e1]),
                SimulationPartition(name="same", entities=[e2]),
            ])

    def test_entity_in_two_partitions_raises(self):
        shared = SimpleEntity("shared")
        with pytest.raises(ValueError, match="appears in both"):
            validate_partitions([
                SimulationPartition(name="p1", entities=[shared]),
                SimulationPartition(name="p2", entities=[shared]),
            ])

    def test_cross_partition_downstream_raises(self):
        other = SimpleEntity("other")
        linked = SimpleEntity("linked", downstream=other)
        with pytest.raises(ValueError, match="references entity 'other'.*via attribute"):
            validate_partitions([
                SimulationPartition(name="p1", entities=[linked]),
                SimulationPartition(name="p2", entities=[other]),
            ])

    def test_cross_partition_list_ref_raises(self):
        target = SimpleEntity("target")
        holder = ListRefEntity("holder", targets=[target])
        with pytest.raises(ValueError, match="references entity 'target'"):
            validate_partitions([
                SimulationPartition(name="p1", entities=[holder]),
                SimulationPartition(name="p2", entities=[target]),
            ])

    def test_cross_partition_dict_ref_raises(self):
        target = SimpleEntity("target")
        holder = DictRefEntity("holder", peers={"peer": target})
        with pytest.raises(ValueError, match="references entity 'target'"):
            validate_partitions([
                SimulationPartition(name="p1", entities=[holder]),
                SimulationPartition(name="p2", entities=[target]),
            ])

    def test_independent_partitions_pass(self):
        e1 = SimpleEntity("a")
        e2 = SimpleEntity("b")
        # Should not raise
        validate_partitions([
            SimulationPartition(name="p1", entities=[e1]),
            SimulationPartition(name="p2", entities=[e2]),
        ])

    def test_same_partition_references_pass(self):
        downstream = SimpleEntity("sink")
        server = SimpleEntity("server", downstream=downstream)
        # Both in the same partition — should pass
        validate_partitions([
            SimulationPartition(name="p1", entities=[server, downstream]),
        ])

    def test_entity_not_in_any_partition_downstream_ignored(self):
        """Entity referencing an unregistered entity is not flagged."""
        external = SimpleEntity("external")
        server = SimpleEntity("server", downstream=external)
        # external is not in any partition, so no cross-partition error
        validate_partitions([
            SimulationPartition(name="p1", entities=[server]),
        ])


# ---------------------------------------------------------------------------
# build_entity_id_set tests
# ---------------------------------------------------------------------------

class TestBuildEntityIdSet:
    def test_includes_top_level_entities(self):
        e1 = SimpleEntity("a")
        e2 = SimpleEntity("b")
        part = SimulationPartition(name="p", entities=[e1, e2])
        ids = build_entity_id_set(part)
        assert id(e1) in ids
        assert id(e2) in ids

    def test_includes_sub_entities(self):
        """Sub-entities discovered via attribute walking should be included."""
        inner = SimpleEntity("inner")
        outer = SimpleEntity("outer", downstream=inner)
        part = SimulationPartition(name="p", entities=[outer, inner])
        ids = build_entity_id_set(part)
        assert id(inner) in ids
        assert id(outer) in ids


# ---------------------------------------------------------------------------
# ParallelSimulationSummary tests
# ---------------------------------------------------------------------------

class TestParallelSimulationSummary:
    def test_str_representation(self):
        from happysimulator.instrumentation.summary import SimulationSummary
        ps = ParallelSimulationSummary(
            duration_s=10.0,
            total_events_processed=1000,
            events_cancelled=5,
            events_per_second=100.0,
            wall_clock_seconds=2.5,
            partitions={
                "p1": SimulationSummary(duration_s=10.0, total_events_processed=500),
                "p2": SimulationSummary(duration_s=10.0, total_events_processed=500),
            },
            entities={},
            partition_wall_times={"p1": 2.5, "p2": 2.0},
            speedup=1.8,
            parallelism_efficiency=0.9,
        )
        s = str(ps)
        assert "Parallel Simulation Summary" in s
        assert "Partitions: 2" in s
        assert "1.80x" in s

    def test_to_dict(self):
        from happysimulator.instrumentation.summary import SimulationSummary
        ps = ParallelSimulationSummary(
            duration_s=5.0,
            total_events_processed=100,
            events_cancelled=0,
            events_per_second=20.0,
            wall_clock_seconds=1.0,
            partitions={
                "p1": SimulationSummary(duration_s=5.0, total_events_processed=100),
            },
            entities={},
            partition_wall_times={"p1": 1.0},
            speedup=1.0,
            parallelism_efficiency=1.0,
        )
        d = ps.to_dict()
        assert d["total_events_processed"] == 100
        assert d["speedup"] == 1.0
        assert "p1" in d["partitions"]


# ---------------------------------------------------------------------------
# ParallelSimulation construction tests
# ---------------------------------------------------------------------------

class TestParallelSimulationConstruction:
    def test_duration_and_end_time_raises(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            ParallelSimulation(
                partitions=[SimulationPartition("p", entities=[SimpleEntity("a")])],
                duration=10.0,
                end_time=Instant.from_seconds(10.0),
            )

    def test_cross_partition_raises_at_init(self):
        shared = SimpleEntity("shared")
        with pytest.raises(ValueError, match="appears in both"):
            ParallelSimulation(
                partitions=[
                    SimulationPartition("p1", entities=[shared]),
                    SimulationPartition("p2", entities=[shared]),
                ],
                duration=1.0,
            )

    def test_schedule_unknown_partition_raises(self):
        e = SimpleEntity("a")
        sim = ParallelSimulation(
            partitions=[SimulationPartition("p1", entities=[e])],
            duration=1.0,
        )
        with pytest.raises(KeyError, match="Unknown partition 'nope'"):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="X", target=e),
                partition="nope",
            )

    def test_partitions_property(self):
        e1 = SimpleEntity("a")
        e2 = SimpleEntity("b")
        sim = ParallelSimulation(
            partitions=[
                SimulationPartition("p1", entities=[e1]),
                SimulationPartition("p2", entities=[e2]),
            ],
            duration=1.0,
        )
        parts = sim.partitions
        assert "p1" in parts
        assert "p2" in parts

    def test_from_groups_factory(self):
        e1 = SimpleEntity("a")
        e2 = SimpleEntity("b")
        sim = ParallelSimulation.from_groups(
            groups={
                "g1": ([e1], []),
                "g2": ([e2], []),
            },
            duration=1.0,
        )
        assert "g1" in sim.partitions
        assert "g2" in sim.partitions
