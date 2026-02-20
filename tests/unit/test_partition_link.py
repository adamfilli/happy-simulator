"""Unit tests for PartitionLink dataclass."""

import pytest

from happysimulator.parallel.link import PartitionLink


class TestPartitionLink:
    def test_valid_link(self):
        link = PartitionLink(
            source_partition="A", dest_partition="B", min_latency=0.01
        )
        assert link.source_partition == "A"
        assert link.dest_partition == "B"
        assert link.min_latency == 0.01
        assert link.latency is None
        assert link.packet_loss == 0.0

    def test_min_latency_zero_raises(self):
        with pytest.raises(ValueError, match="min_latency must be > 0"):
            PartitionLink(source_partition="A", dest_partition="B", min_latency=0)

    def test_min_latency_negative_raises(self):
        with pytest.raises(ValueError, match="min_latency must be > 0"):
            PartitionLink(source_partition="A", dest_partition="B", min_latency=-1.0)

    def test_packet_loss_out_of_range_raises(self):
        with pytest.raises(ValueError, match="packet_loss must be in"):
            PartitionLink(
                source_partition="A", dest_partition="B",
                min_latency=0.01, packet_loss=1.0,
            )

    def test_packet_loss_negative_raises(self):
        with pytest.raises(ValueError, match="packet_loss must be in"):
            PartitionLink(
                source_partition="A", dest_partition="B",
                min_latency=0.01, packet_loss=-0.1,
            )

    def test_same_partition_raises(self):
        with pytest.raises(ValueError, match="source and dest must differ"):
            PartitionLink(
                source_partition="A", dest_partition="A", min_latency=0.01
            )

    def test_frozen(self):
        link = PartitionLink(
            source_partition="A", dest_partition="B", min_latency=0.01
        )
        with pytest.raises(AttributeError):
            link.min_latency = 0.02  # type: ignore[misc]

    def test_bidirectional(self):
        a_to_b, b_to_a = PartitionLink.bidirectional("A", "B", min_latency=0.05)
        assert a_to_b.source_partition == "A"
        assert a_to_b.dest_partition == "B"
        assert b_to_a.source_partition == "B"
        assert b_to_a.dest_partition == "A"
        assert a_to_b.min_latency == b_to_a.min_latency == 0.05

    def test_bidirectional_with_packet_loss(self):
        a_to_b, b_to_a = PartitionLink.bidirectional(
            "X", "Y", min_latency=0.1, packet_loss=0.05
        )
        assert a_to_b.packet_loss == 0.05
        assert b_to_a.packet_loss == 0.05

    def test_valid_packet_loss(self):
        link = PartitionLink(
            source_partition="A", dest_partition="B",
            min_latency=0.01, packet_loss=0.5,
        )
        assert link.packet_loss == 0.5
