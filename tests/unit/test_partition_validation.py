"""Unit tests for partition validation logic."""

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source
from happysimulator.parallel.link import PartitionLink
from happysimulator.parallel.partition import SimulationPartition
from happysimulator.parallel.validation import build_entity_sets, validate_partitions


class SimpleEntity(Entity):
    """Minimal entity for testing."""

    def __init__(self, name, downstream=None):
        super().__init__(name)
        self.downstream = downstream

    def handle_event(self, event):
        if self.downstream is not None:
            return [Event(time=self.now, event_type="Fwd", target=self.downstream)]
        return None


class TestValidatePartitions:
    def test_valid_single_partition(self):
        e = SimpleEntity("e1")
        p = SimulationPartition(name="P1", entities=[e])
        validate_partitions([p])  # should not raise

    def test_duplicate_partition_name_raises(self):
        e1 = SimpleEntity("e1")
        e2 = SimpleEntity("e2")
        p1 = SimulationPartition(name="A", entities=[e1])
        p2 = SimulationPartition(name="A", entities=[e2])
        with pytest.raises(ValueError, match="Duplicate partition name"):
            validate_partitions([p1, p2])

    def test_entity_in_two_partitions_raises(self):
        shared = SimpleEntity("shared")
        p1 = SimulationPartition(name="P1", entities=[shared])
        p2 = SimulationPartition(name="P2", entities=[shared])
        with pytest.raises(ValueError, match="is in partitions"):
            validate_partitions([p1, p2])

    def test_cross_reference_without_link_raises(self):
        e2 = SimpleEntity("e2")
        e1 = SimpleEntity("e1", downstream=e2)
        p1 = SimulationPartition(name="P1", entities=[e1])
        p2 = SimulationPartition(name="P2", entities=[e2])
        with pytest.raises(ValueError, match="no PartitionLink exists"):
            validate_partitions([p1, p2])

    def test_cross_reference_with_link_allowed(self):
        e2 = SimpleEntity("e2")
        e1 = SimpleEntity("e1", downstream=e2)
        p1 = SimulationPartition(name="P1", entities=[e1])
        p2 = SimulationPartition(name="P2", entities=[e2])
        link = PartitionLink(source_partition="P1", dest_partition="P2", min_latency=0.01)
        validate_partitions([p1, p2], links=[link])  # should not raise

    def test_link_unknown_partition_raises(self):
        e1 = SimpleEntity("e1")
        p1 = SimulationPartition(name="P1", entities=[e1])
        link = PartitionLink(source_partition="P1", dest_partition="MISSING", min_latency=0.01)
        with pytest.raises(ValueError, match="unknown dest partition"):
            validate_partitions([p1], links=[link])

    def test_link_unknown_source_partition_raises(self):
        e1 = SimpleEntity("e1")
        p1 = SimulationPartition(name="P1", entities=[e1])
        link = PartitionLink(source_partition="MISSING", dest_partition="P1", min_latency=0.01)
        with pytest.raises(ValueError, match="unknown source partition"):
            validate_partitions([p1], links=[link])

    def test_window_size_too_large_raises(self):
        e1 = SimpleEntity("e1")
        e2 = SimpleEntity("e2")
        p1 = SimulationPartition(name="P1", entities=[e1])
        p2 = SimulationPartition(name="P2", entities=[e2])
        link = PartitionLink(source_partition="P1", dest_partition="P2", min_latency=0.05)
        with pytest.raises(ValueError, match="window_size"):
            validate_partitions([p1, p2], links=[link], window_size=0.1)

    def test_window_size_valid(self):
        e1 = SimpleEntity("e1")
        e2 = SimpleEntity("e2")
        p1 = SimulationPartition(name="P1", entities=[e1])
        p2 = SimulationPartition(name="P2", entities=[e2])
        link = PartitionLink(source_partition="P1", dest_partition="P2", min_latency=0.1)
        validate_partitions([p1, p2], links=[link], window_size=0.05)  # should not raise


class TestBuildEntitySets:
    def test_builds_correct_sets(self):
        e1 = SimpleEntity("e1")
        e2 = SimpleEntity("e2")
        e3 = SimpleEntity("e3")
        p1 = SimulationPartition(name="A", entities=[e1, e2])
        p2 = SimulationPartition(name="B", entities=[e3])
        sets = build_entity_sets([p1, p2])
        assert id(e1) in sets["A"]
        assert id(e2) in sets["A"]
        assert id(e3) in sets["B"]
        assert id(e3) not in sets["A"]
