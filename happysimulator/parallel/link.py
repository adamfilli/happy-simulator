"""Cross-partition link declarations for coordinated parallel simulation.

A PartitionLink allows events to flow between two partitions.  The
``min_latency`` field sizes the barrier window — it must be > 0 so that
the coordinator can safely advance both partitions by ``min_latency``
seconds before exchanging events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from happysimulator.distributions.latency_distribution import LatencyDistribution


@dataclass(frozen=True)
class PartitionLink:
    """Unidirectional link between two partitions.

    Attributes:
        source_partition: Name of the sending partition.
        dest_partition: Name of the receiving partition.
        min_latency: Minimum event propagation time in seconds (must be > 0).
        latency: Optional distribution for sampling arrival times.  When set,
            the coordinator overrides each event's timestamp with
            ``send_time + latency.sample()``.  When ``None``, the event's
            existing timestamp is used (and must satisfy
            ``event.time - send_time >= min_latency``).
        packet_loss: Probability in [0, 1) that a cross-partition event
            is silently dropped.
    """

    source_partition: str
    dest_partition: str
    min_latency: float
    latency: LatencyDistribution | None = None
    packet_loss: float = 0.0

    def __post_init__(self) -> None:
        if self.min_latency <= 0:
            raise ValueError(
                f"PartitionLink min_latency must be > 0, got {self.min_latency}"
            )
        if not (0.0 <= self.packet_loss < 1.0):
            raise ValueError(
                f"PartitionLink packet_loss must be in [0, 1), got {self.packet_loss}"
            )
        if self.source_partition == self.dest_partition:
            raise ValueError(
                f"PartitionLink source and dest must differ, got '{self.source_partition}'"
            )

    @staticmethod
    def bidirectional(
        partition_a: str,
        partition_b: str,
        min_latency: float,
        latency: LatencyDistribution | None = None,
        packet_loss: float = 0.0,
    ) -> tuple[PartitionLink, PartitionLink]:
        """Create a pair of links (A→B and B→A) with identical parameters."""
        return (
            PartitionLink(
                source_partition=partition_a,
                dest_partition=partition_b,
                min_latency=min_latency,
                latency=latency,
                packet_loss=packet_loss,
            ),
            PartitionLink(
                source_partition=partition_b,
                dest_partition=partition_a,
                min_latency=min_latency,
                latency=latency,
                packet_loss=packet_loss,
            ),
        )
