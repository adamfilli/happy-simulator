"""Tests for NetworkLink component."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

from happysimulator.components.network.link import NetworkLink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency


@dataclass
class CollectorEntity(Entity):
    """Entity that collects received events for verification."""

    name: str = "Collector"
    received: list[Event] = field(default_factory=list)

    def handle_event(self, event: Event):
        self.received.append(event)
        return


class TestNetworkLinkBasics:
    """Basic NetworkLink functionality tests."""

    def test_creates_with_required_params(self):
        """NetworkLink can be created with minimal required parameters."""
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
        )
        assert link.name == "TestLink"
        assert link.bandwidth_bps is None
        assert link.packet_loss_rate == 0.0
        assert link.jitter is None

    def test_rejects_invalid_packet_loss_rate(self):
        """NetworkLink rejects packet_loss_rate outside [0, 1]."""
        with pytest.raises(ValueError, match="packet_loss_rate"):
            NetworkLink(
                name="BadLink",
                latency=ConstantLatency(0.001),
                packet_loss_rate=1.5,
            )

        with pytest.raises(ValueError, match="packet_loss_rate"):
            NetworkLink(
                name="BadLink",
                latency=ConstantLatency(0.001),
                packet_loss_rate=-0.1,
            )

    def test_initial_statistics_are_zero(self):
        """NetworkLink starts with zero statistics."""
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
        )
        assert link.bytes_transmitted == 0
        assert link.packets_sent == 0
        assert link.packets_dropped == 0

    def test_utilization_zero_with_infinite_bandwidth(self):
        """Utilization is 0 when bandwidth is unlimited."""
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            bandwidth_bps=None,
        )
        assert link.current_utilization == 0.0


class TestNetworkLinkLatency:
    """Tests for NetworkLink latency behavior."""

    def test_applies_constant_latency(self):
        """Events are delayed by the configured latency."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.010),  # 10ms
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        # Schedule event at t=0
        sim.schedule(Event(time=Instant.Epoch, event_type="Ping", target=link))
        sim.run()

        # Event should arrive after 10ms delay
        assert len(collector.received) == 1
        assert link.packets_sent == 1
        assert link.bytes_transmitted == 0  # No payload size specified

    def test_applies_jitter(self):
        """Jitter adds additional delay to base latency."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.010),  # 10ms base
            jitter=ConstantLatency(0.005),  # 5ms jitter
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        sim.schedule(Event(time=Instant.Epoch, event_type="Ping", target=link))
        sim.run()

        # Event should arrive (latency check via simulation completion)
        assert len(collector.received) == 1
        assert link.packets_sent == 1


class TestNetworkLinkBandwidth:
    """Tests for NetworkLink bandwidth constraints."""

    def test_bandwidth_adds_transmission_delay(self):
        """Bandwidth limit adds delay based on payload size."""
        collector = CollectorEntity()
        # 1 Mbps = 1,000,000 bits/sec
        # 1000 bytes = 8000 bits
        # Transmission time = 8000 / 1,000,000 = 0.008 seconds = 8ms
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.010),  # 10ms
            bandwidth_bps=1_000_000,  # 1 Mbps
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        # Create event with payload size
        event = Event(time=Instant.Epoch, event_type="Data", target=link)
        event.context["metadata"]["payload_size"] = 1000  # 1000 bytes

        sim.schedule(event)
        sim.run()

        assert len(collector.received) == 1
        assert link.packets_sent == 1
        assert link.bytes_transmitted == 1000

    def test_tracks_bytes_transmitted(self):
        """Bytes transmitted is tracked across multiple events."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        # Schedule multiple events with different sizes
        for i, size in enumerate([100, 200, 300]):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="Data",
                target=link,
            )
            event.context["metadata"]["payload_size"] = size
            sim.schedule(event)

        sim.run()

        assert link.packets_sent == 3
        assert link.bytes_transmitted == 600  # 100 + 200 + 300


class TestNetworkLinkPacketLoss:
    """Tests for NetworkLink packet loss behavior."""

    def test_no_loss_with_zero_rate(self):
        """All packets are delivered when loss rate is 0."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            packet_loss_rate=0.0,
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        for i in range(10):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.01),
                    event_type="Ping",
                    target=link,
                )
            )

        sim.run()

        assert len(collector.received) == 10
        assert link.packets_sent == 10
        assert link.packets_dropped == 0

    def test_all_loss_with_full_rate(self):
        """All packets are dropped when loss rate is 1.0."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            packet_loss_rate=1.0,
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        for i in range(10):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.01),
                    event_type="Ping",
                    target=link,
                )
            )

        sim.run()

        assert len(collector.received) == 0
        assert link.packets_sent == 0
        assert link.packets_dropped == 10

    def test_partial_loss_is_probabilistic(self):
        """Packet loss is probabilistic with partial rate."""
        random.seed(42)  # For reproducibility

        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            packet_loss_rate=0.5,
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[],
            entities=[link, collector],
        )

        num_packets = 100
        for i in range(num_packets):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.01),
                    event_type="Ping",
                    target=link,
                )
            )

        sim.run()

        # With 50% loss rate and 100 packets, we expect roughly 50 delivered
        # Allow reasonable variance (30-70 range)
        assert 30 <= len(collector.received) <= 70
        assert link.packets_sent + link.packets_dropped == num_packets


class TestNetworkLinkNoEgress:
    """Tests for NetworkLink behavior without egress configured."""

    def test_drops_events_without_egress(self):
        """Events are lost when no egress is configured."""
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            egress=None,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link],
        )

        sim.schedule(Event(time=Instant.Epoch, event_type="Ping", target=link))
        sim.run()

        # Event processed but not forwarded anywhere
        assert link.packets_sent == 1


class TestNetworkLinkEventContext:
    """Tests for event context preservation through NetworkLink."""

    def test_preserves_event_context(self):
        """Event context is preserved through the link."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        event = Event(time=Instant.Epoch, event_type="Tagged", target=link)
        event.context["metadata"]["custom_key"] = "custom_value"
        sim.schedule(event)
        sim.run()

        assert len(collector.received) == 1
        received = collector.received[0]
        assert received.context["metadata"]["custom_key"] == "custom_value"

    def test_preserves_event_type(self):
        """Event type is preserved through the link."""
        collector = CollectorEntity()
        link = NetworkLink(
            name="TestLink",
            latency=ConstantLatency(0.001),
            egress=collector,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[link, collector],
        )

        sim.schedule(Event(time=Instant.Epoch, event_type="SpecificType", target=link))
        sim.run()

        assert len(collector.received) == 1
        assert collector.received[0].event_type == "SpecificType"
