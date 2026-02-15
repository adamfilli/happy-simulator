"""Point-to-point network link with configurable characteristics.

NetworkLink models a network connection between two entities, simulating
latency, bandwidth constraints, packet loss, and jitter. Events passing
through the link are delayed and may be dropped based on configuration.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.clock import Clock
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NetworkLinkStats:
    """Frozen snapshot of NetworkLink statistics.

    Attributes:
        bytes_transmitted: Total bytes successfully transmitted.
        packets_sent: Number of packets successfully transmitted.
        packets_dropped: Number of packets dropped (loss).
    """

    bytes_transmitted: int = 0
    packets_sent: int = 0
    packets_dropped: int = 0


@dataclass
class NetworkLink(Entity):
    """Simulates network transmission delay and bandwidth constraints.

    Models a point-to-point network connection. Events sent through the link
    experience configurable latency, optional jitter, and may be dropped
    based on a packet loss rate. Bandwidth limits add transmission delay
    based on payload size.

    The link forwards events to its configured egress after applying delays.
    If no egress is set, events are dropped with a warning.

    Attributes:
        name: Identifier for logging and debugging.
        latency: Base one-way delay distribution.
        bandwidth_bps: Link capacity in bits per second. None means infinite.
        packet_loss_rate: Probability [0, 1] of dropping each packet.
        jitter: Optional additional random delay distribution.
        egress: Destination entity for forwarded events.
    """

    name: str
    latency: LatencyDistribution
    bandwidth_bps: float | None = None
    packet_loss_rate: float = 0.0
    jitter: LatencyDistribution | None = None
    egress: Entity | None = None

    # Statistics
    bytes_transmitted: int = field(default=0, init=False)
    packets_sent: int = field(default=0, init=False)
    packets_dropped: int = field(default=0, init=False)
    _bytes_in_flight: int = field(default=0, init=False)

    def __post_init__(self):
        if self.packet_loss_rate < 0.0 or self.packet_loss_rate > 1.0:
            raise ValueError(
                f"packet_loss_rate must be in [0, 1], got {self.packet_loss_rate}"
            )
        logger.debug(
            "[%s] NetworkLink created: latency=%s, bandwidth=%s bps, loss=%.2f%%",
            self.name,
            self.latency,
            self.bandwidth_bps or "infinite",
            self.packet_loss_rate * 100,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject the simulation clock."""
        super().set_clock(clock)
        # Propagate clock to egress if it's an entity that needs it
        if self.egress is not None and hasattr(self.egress, "set_clock"):
            self.egress.set_clock(clock)

    @property
    def current_utilization(self) -> float:
        """Current link utilization as fraction of bandwidth.

        Returns 0.0 if bandwidth is infinite (None).
        """
        if self.bandwidth_bps is None or self.bandwidth_bps == 0:
            return 0.0
        # Rough estimate based on bytes currently in flight
        # More accurate tracking would require maintaining transmission windows
        return min(1.0, (self._bytes_in_flight * 8) / self.bandwidth_bps)

    @property
    def link_stats(self) -> NetworkLinkStats:
        """Frozen snapshot of current link statistics."""
        return NetworkLinkStats(
            bytes_transmitted=self.bytes_transmitted,
            packets_sent=self.packets_sent,
            packets_dropped=self.packets_dropped,
        )

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Process an event through the network link.

        Applies packet loss, calculates total delay (latency + jitter +
        transmission time), and forwards the event to the egress.

        Args:
            event: The event to transmit through the link.

        Yields:
            Total transmission delay in seconds.

        Returns:
            Forwarded event targeting the egress, or None if dropped.
        """
        # Check for packet loss
        if self.packet_loss_rate > 0 and random.random() < self.packet_loss_rate:
            self.packets_dropped += 1
            logger.debug(
                "[%s] Packet dropped (loss): type=%s",
                self.name,
                event.event_type,
            )
            return None

        # Calculate total delay
        total_delay = self._calculate_delay(event)

        # Track bytes in flight for utilization calculation
        payload_size = self._get_payload_size(event)
        self._bytes_in_flight += payload_size

        logger.debug(
            "[%s] Transmitting: type=%s delay=%.6fs size=%d bytes",
            self.name,
            event.event_type,
            total_delay,
            payload_size,
        )

        # Wait for transmission
        yield total_delay

        # Transmission complete
        self._bytes_in_flight = max(0, self._bytes_in_flight - payload_size)
        self.bytes_transmitted += payload_size
        self.packets_sent += 1

        # Forward to egress
        if self.egress is None:
            logger.warning(
                "[%s] No egress configured, event lost: type=%s",
                self.name,
                event.event_type,
            )
            return None

        # Create forwarded event targeting the egress
        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=self.egress,
            daemon=event.daemon,
            context=event.context.copy(),
        )

        # Preserve completion hooks from original event
        forwarded.on_complete = list(event.on_complete)

        logger.debug(
            "[%s] Delivered to egress: type=%s",
            self.name,
            event.event_type,
        )

        return forwarded

    def _calculate_delay(self, event: Event) -> float:
        """Calculate total transmission delay for an event.

        Components:
        - Base latency from the latency distribution
        - Jitter (if configured)
        - Transmission time based on payload size and bandwidth

        Args:
            event: The event being transmitted.

        Returns:
            Total delay in seconds.
        """
        # Base latency
        delay = self.latency.get_latency(self.now).to_seconds()

        # Add jitter if configured
        if self.jitter is not None:
            delay += self.jitter.get_latency(self.now).to_seconds()

        # Add transmission time based on bandwidth
        if self.bandwidth_bps is not None and self.bandwidth_bps > 0:
            payload_size = self._get_payload_size(event)
            transmission_time = (payload_size * 8) / self.bandwidth_bps
            delay += transmission_time

        return max(0.0, delay)

    def _get_payload_size(self, event: Event) -> int:
        """Extract payload size from event context.

        Looks for 'payload_size' or 'size' in the event's metadata.
        Defaults to 0 if not specified.

        Args:
            event: The event to check.

        Returns:
            Payload size in bytes.
        """
        metadata = event.context.get("metadata", {})
        size = metadata.get("payload_size") or metadata.get("size") or 0
        return int(size)
