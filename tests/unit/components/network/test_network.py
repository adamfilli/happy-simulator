"""Tests for Network topology manager."""

from __future__ import annotations

from dataclasses import dataclass, field

from happysimulator.components.network.link import NetworkLink
from happysimulator.components.network.network import LinkStats, Network, Partition
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


@dataclass
class SenderEntity(Entity):
    """Entity that can send events through a network."""

    name: str = "Sender"

    def handle_event(self, event: Event):
        return None


class TestNetworkBasics:
    """Basic Network functionality tests."""

    def test_creates_with_name(self):
        """Network can be created with just a name."""
        network = Network(name="TestNetwork")
        assert network.name == "TestNetwork"
        assert network.default_link is None

    def test_creates_with_default_link(self):
        """Network can be created with a default link."""
        default_link = NetworkLink(
            name="DefaultLink",
            latency=ConstantLatency(0.010),
        )
        network = Network(name="TestNetwork", default_link=default_link)
        assert network.default_link is default_link

    def test_initial_statistics_are_zero(self):
        """Network starts with zero statistics."""
        network = Network(name="TestNetwork")
        assert network.events_routed == 0
        assert network.events_dropped_no_route == 0
        assert network.events_dropped_partition == 0


class TestNetworkRouting:
    """Tests for Network routing functionality."""

    def test_routes_through_configured_link(self):
        """Events are routed through the configured link."""
        sender = SenderEntity(name="Sender")
        receiver = CollectorEntity(name="Receiver")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.010),
        )
        network = Network(name="TestNetwork")
        network.add_link(sender, receiver, link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver],
        )

        # Create event with routing metadata
        event = Event(time=Instant.Epoch, event_type="Message", target=network)
        event.context["metadata"]["source"] = "Sender"
        event.context["metadata"]["destination"] = "Receiver"
        sim.schedule(event)

        sim.run()

        assert len(receiver.received) == 1
        assert network.events_routed == 1
        assert link.packets_sent == 1

    def test_routes_through_default_link(self):
        """Events use default link when no specific route exists."""
        sender = SenderEntity(name="Sender")
        receiver = CollectorEntity(name="Receiver")

        default_link = NetworkLink(
            name="DefaultLink",
            latency=ConstantLatency(0.010),
            egress=receiver,
        )
        network = Network(name="TestNetwork", default_link=default_link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver],
        )

        event = Event(time=Instant.Epoch, event_type="Message", target=network)
        event.context["metadata"]["source"] = "Sender"
        event.context["metadata"]["destination"] = "Receiver"
        sim.schedule(event)

        sim.run()

        assert len(receiver.received) == 1
        assert network.events_routed == 1

    def test_drops_event_without_routing_metadata(self):
        """Events without source/destination metadata are dropped."""
        network = Network(name="TestNetwork")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network],
        )

        # Event without routing metadata
        event = Event(time=Instant.Epoch, event_type="BadMessage", target=network)
        sim.schedule(event)

        sim.run()

        assert network.events_dropped_no_route == 1

    def test_drops_event_without_route_or_default(self):
        """Events are dropped when no route and no default link exists."""
        network = Network(name="TestNetwork")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network],
        )

        event = Event(time=Instant.Epoch, event_type="Message", target=network)
        event.context["metadata"]["source"] = "Unknown1"
        event.context["metadata"]["destination"] = "Unknown2"
        sim.schedule(event)

        sim.run()

        assert network.events_dropped_no_route == 1


class TestNetworkBidirectional:
    """Tests for bidirectional link functionality."""

    def test_bidirectional_link_works_both_ways(self):
        """Bidirectional links route in both directions."""
        entity_a = CollectorEntity(name="EntityA")
        entity_b = CollectorEntity(name="EntityB")

        link = NetworkLink(
            name="BiLink",
            latency=ConstantLatency(0.005),
        )
        network = Network(name="TestNetwork")
        network.add_bidirectional_link(entity_a, entity_b, link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, entity_a, entity_b],
        )

        # Send A -> B
        event1 = Event(
            time=Instant.Epoch,
            event_type="MessageAtoB",
            target=network,
        )
        event1.context["metadata"]["source"] = "EntityA"
        event1.context["metadata"]["destination"] = "EntityB"
        sim.schedule(event1)

        # Send B -> A
        event2 = Event(
            time=Instant.from_seconds(0.1),
            event_type="MessageBtoA",
            target=network,
        )
        event2.context["metadata"]["source"] = "EntityB"
        event2.context["metadata"]["destination"] = "EntityA"
        sim.schedule(event2)

        sim.run()

        assert len(entity_a.received) == 1
        assert len(entity_b.received) == 1
        assert entity_a.received[0].event_type == "MessageBtoA"
        assert entity_b.received[0].event_type == "MessageAtoB"
        assert network.events_routed == 2


class TestNetworkPartitions:
    """Tests for network partition functionality."""

    def test_partition_drops_events(self):
        """Events between partitioned groups are dropped."""
        entity_a = CollectorEntity(name="EntityA")
        entity_b = CollectorEntity(name="EntityB")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.005),
        )
        network = Network(name="TestNetwork")
        network.add_bidirectional_link(entity_a, entity_b, link)

        # Create partition
        network.partition([entity_a], [entity_b])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, entity_a, entity_b],
        )

        # Try to send A -> B (should be dropped)
        event = Event(time=Instant.Epoch, event_type="Message", target=network)
        event.context["metadata"]["source"] = "EntityA"
        event.context["metadata"]["destination"] = "EntityB"
        sim.schedule(event)

        sim.run()

        assert len(entity_b.received) == 0
        assert network.events_dropped_partition == 1
        assert network.events_routed == 0

    def test_partition_is_bidirectional(self):
        """Partitions block traffic in both directions."""
        entity_a = CollectorEntity(name="EntityA")
        entity_b = CollectorEntity(name="EntityB")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.005),
        )
        network = Network(name="TestNetwork")
        network.add_bidirectional_link(entity_a, entity_b, link)
        network.partition([entity_a], [entity_b])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, entity_a, entity_b],
        )

        # Try A -> B
        event1 = Event(time=Instant.Epoch, event_type="Message", target=network)
        event1.context["metadata"]["source"] = "EntityA"
        event1.context["metadata"]["destination"] = "EntityB"
        sim.schedule(event1)

        # Try B -> A
        event2 = Event(
            time=Instant.from_seconds(0.1),
            event_type="Message",
            target=network,
        )
        event2.context["metadata"]["source"] = "EntityB"
        event2.context["metadata"]["destination"] = "EntityA"
        sim.schedule(event2)

        sim.run()

        assert len(entity_a.received) == 0
        assert len(entity_b.received) == 0
        assert network.events_dropped_partition == 2

    def test_heal_partition_restores_connectivity(self):
        """Healing a partition restores communication."""
        entity_a = CollectorEntity(name="EntityA")
        entity_b = CollectorEntity(name="EntityB")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.005),
        )
        network = Network(name="TestNetwork")
        network.add_bidirectional_link(entity_a, entity_b, link)

        # Create and then heal partition
        network.partition([entity_a], [entity_b])
        network.heal_partition()

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, entity_a, entity_b],
        )

        event = Event(time=Instant.Epoch, event_type="Message", target=network)
        event.context["metadata"]["source"] = "EntityA"
        event.context["metadata"]["destination"] = "EntityB"
        sim.schedule(event)

        sim.run()

        assert len(entity_b.received) == 1
        assert network.events_routed == 1
        assert network.events_dropped_partition == 0

    def test_partition_allows_intra_group_traffic(self):
        """Partitions don't affect traffic within the same group."""
        entity_a1 = CollectorEntity(name="EntityA1")
        entity_a2 = CollectorEntity(name="EntityA2")
        entity_b = CollectorEntity(name="EntityB")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.005),
        )
        network = Network(name="TestNetwork")
        network.add_link(entity_a1, entity_a2, link)

        # Partition: [A1, A2] vs [B]
        network.partition([entity_a1, entity_a2], [entity_b])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, entity_a1, entity_a2, entity_b],
        )

        # A1 -> A2 should work (same group)
        event = Event(time=Instant.Epoch, event_type="Message", target=network)
        event.context["metadata"]["source"] = "EntityA1"
        event.context["metadata"]["destination"] = "EntityA2"
        sim.schedule(event)

        sim.run()

        assert len(entity_a2.received) == 1
        assert network.events_routed == 1
        assert network.events_dropped_partition == 0


class TestNetworkSendHelper:
    """Tests for the send() convenience method."""

    def test_send_creates_proper_event(self):
        """send() creates an event with correct routing metadata."""
        sender = SenderEntity(name="Sender")
        receiver = CollectorEntity(name="Receiver")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.010),
        )
        network = Network(name="TestNetwork")
        network.add_link(sender, receiver, link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver],
        )

        # Use send() helper
        event = network.send(sender, receiver, "TestMessage")
        sim.schedule(event)

        sim.run()

        assert len(receiver.received) == 1
        assert receiver.received[0].event_type == "TestMessage"

    def test_send_includes_payload(self):
        """send() includes additional payload in metadata."""
        sender = SenderEntity(name="Sender")
        receiver = CollectorEntity(name="Receiver")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.010),
        )
        network = Network(name="TestNetwork")
        network.add_link(sender, receiver, link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver],
        )

        event = network.send(
            sender,
            receiver,
            "DataMessage",
            payload={"data": "test_value", "size": 100},
        )
        sim.schedule(event)

        sim.run()

        assert len(receiver.received) == 1
        received = receiver.received[0]
        assert received.context["metadata"]["data"] == "test_value"
        assert received.context["metadata"]["size"] == 100


class TestNetworkMultipleLinks:
    """Tests for networks with multiple configured links."""

    def test_different_links_for_different_routes(self):
        """Different routes can use different links with different characteristics."""
        sender = SenderEntity(name="Sender")
        receiver1 = CollectorEntity(name="Receiver1")
        receiver2 = CollectorEntity(name="Receiver2")

        # Fast link to receiver1
        fast_link = NetworkLink(
            name="FastLink",
            latency=ConstantLatency(0.001),
        )
        # Slow link to receiver2
        slow_link = NetworkLink(
            name="SlowLink",
            latency=ConstantLatency(0.100),
        )

        network = Network(name="TestNetwork")
        network.add_link(sender, receiver1, fast_link)
        network.add_link(sender, receiver2, slow_link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver1, receiver2],
        )

        # Send to both receivers at the same time
        event1 = Event(time=Instant.Epoch, event_type="Fast", target=network)
        event1.context["metadata"]["source"] = "Sender"
        event1.context["metadata"]["destination"] = "Receiver1"
        sim.schedule(event1)

        event2 = Event(time=Instant.Epoch, event_type="Slow", target=network)
        event2.context["metadata"]["source"] = "Sender"
        event2.context["metadata"]["destination"] = "Receiver2"
        sim.schedule(event2)

        sim.run()

        assert len(receiver1.received) == 1
        assert len(receiver2.received) == 1
        assert network.events_routed == 2
        assert fast_link.packets_sent == 1
        assert slow_link.packets_sent == 1


class TestPartitionHandle:
    """Tests for Partition handle and selective healing."""

    def test_partition_returns_handle(self):
        """partition() returns a Partition handle."""
        entity_a = CollectorEntity(name="EntityA")
        entity_b = CollectorEntity(name="EntityB")

        network = Network(name="TestNetwork")
        link = NetworkLink(name="Link", latency=ConstantLatency(0.005))
        network.add_bidirectional_link(entity_a, entity_b, link)

        handle = network.partition([entity_a], [entity_b])

        assert isinstance(handle, Partition)
        assert handle.is_active

    def test_selective_heal_removes_only_target_partition(self):
        """Healing a partition handle only removes that partition's pairs."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")
        c = CollectorEntity(name="C")

        network = Network(name="TestNetwork")
        link_ab = NetworkLink(name="LinkAB", latency=ConstantLatency(0.005))
        link_ac = NetworkLink(name="LinkAC", latency=ConstantLatency(0.005))
        link_bc = NetworkLink(name="LinkBC", latency=ConstantLatency(0.005))
        network.add_bidirectional_link(a, b, link_ab)
        network.add_bidirectional_link(a, c, link_ac)
        network.add_bidirectional_link(b, c, link_bc)

        # Create two separate partitions
        p1 = network.partition([a], [b])
        p2 = network.partition([a], [c])

        assert p1.is_active
        assert p2.is_active
        assert network.is_partitioned("A", "B")
        assert network.is_partitioned("A", "C")

        # Heal only partition 1 (A <-> B)
        p1.heal()

        assert not p1.is_active
        assert p2.is_active
        assert not network.is_partitioned("A", "B")
        assert network.is_partitioned("A", "C")

    def test_heal_partition_clears_all(self):
        """heal_partition() still clears all partitions for backward compat."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")
        c = CollectorEntity(name="C")

        network = Network(name="TestNetwork")
        p1 = network.partition([a], [b])
        p2 = network.partition([a], [c])

        network.heal_partition()

        assert not p1.is_active
        assert not p2.is_active
        assert not network.is_partitioned("A", "B")
        assert not network.is_partitioned("A", "C")

    def test_partition_is_active_after_heal_all(self):
        """is_active returns False after heal_partition()."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")

        network = Network(name="TestNetwork")
        handle = network.partition([a], [b])

        assert handle.is_active
        network.heal_partition()
        assert not handle.is_active

    def test_selective_heal_works_in_simulation(self):
        """Selective heal restores connectivity during simulation."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")
        c = CollectorEntity(name="C")

        network = Network(name="TestNetwork")
        link_ab = NetworkLink(name="LinkAB", latency=ConstantLatency(0.005))
        link_ac = NetworkLink(name="LinkAC", latency=ConstantLatency(0.005))
        network.add_bidirectional_link(a, b, link_ab)
        network.add_bidirectional_link(a, c, link_ac)

        p_ab = network.partition([a], [b])
        network.partition([a], [c])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, a, b, c],
        )

        # Heal only A <-> B
        p_ab.heal()

        # A -> B should work now
        event1 = Event(time=Instant.Epoch, event_type="Msg", target=network)
        event1.context["metadata"]["source"] = "A"
        event1.context["metadata"]["destination"] = "B"
        sim.schedule(event1)

        # A -> C should still be blocked
        event2 = Event(time=Instant.from_seconds(0.1), event_type="Msg", target=network)
        event2.context["metadata"]["source"] = "A"
        event2.context["metadata"]["destination"] = "C"
        sim.schedule(event2)

        sim.run()

        assert len(b.received) == 1
        assert len(c.received) == 0
        assert network.events_routed == 1
        assert network.events_dropped_partition == 1


class TestAsymmetricPartitions:
    """Tests for asymmetric (directed) partition functionality."""

    def test_asymmetric_partition_blocks_one_direction(self):
        """Asymmetric partition blocks A->B but allows B->A."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")

        network = Network(name="TestNetwork")
        link = NetworkLink(name="Link", latency=ConstantLatency(0.005))
        network.add_bidirectional_link(a, b, link)

        network.partition([a], [b], asymmetric=True)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, a, b],
        )

        # A -> B should be blocked
        event1 = Event(time=Instant.Epoch, event_type="Msg", target=network)
        event1.context["metadata"]["source"] = "A"
        event1.context["metadata"]["destination"] = "B"
        sim.schedule(event1)

        # B -> A should work
        event2 = Event(time=Instant.from_seconds(0.1), event_type="Msg", target=network)
        event2.context["metadata"]["source"] = "B"
        event2.context["metadata"]["destination"] = "A"
        sim.schedule(event2)

        sim.run()

        assert len(b.received) == 0  # A -> B blocked
        assert len(a.received) == 1  # B -> A allowed
        assert network.events_dropped_partition == 1
        assert network.events_routed == 1

    def test_asymmetric_partition_handle_heal(self):
        """Asymmetric partition handle heal removes only directed pairs."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")

        network = Network(name="TestNetwork")
        handle = network.partition([a], [b], asymmetric=True)

        assert handle.is_active
        assert network.is_partitioned("A", "B")
        assert not network.is_partitioned("B", "A")

        handle.heal()

        assert not handle.is_active
        assert not network.is_partitioned("A", "B")

    def test_heal_partition_clears_asymmetric(self):
        """heal_partition() clears asymmetric partitions too."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")

        network = Network(name="TestNetwork")
        network.partition([a], [b], asymmetric=True)

        assert network.is_partitioned("A", "B")

        network.heal_partition()

        assert not network.is_partitioned("A", "B")

    def test_mixed_symmetric_asymmetric_partitions(self):
        """Symmetric and asymmetric partitions can coexist."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")
        c = CollectorEntity(name="C")

        network = Network(name="TestNetwork")
        link_ab = NetworkLink(name="LinkAB", latency=ConstantLatency(0.005))
        link_ac = NetworkLink(name="LinkAC", latency=ConstantLatency(0.005))
        network.add_bidirectional_link(a, b, link_ab)
        network.add_bidirectional_link(a, c, link_ac)

        # Symmetric partition between A and B
        network.partition([a], [b])
        # Asymmetric partition: A -X-> C (A cannot send to C)
        p_asym = network.partition([a], [c], asymmetric=True)

        # A <-> B: both directions blocked
        assert network.is_partitioned("A", "B")
        assert network.is_partitioned("B", "A")

        # A -> C: blocked (asymmetric)
        assert network.is_partitioned("A", "C")
        # C -> A: NOT blocked (asymmetric, only A->C is blocked)
        assert not network.is_partitioned("C", "A")

        # Heal only asymmetric
        p_asym.heal()
        assert not network.is_partitioned("A", "C")
        assert network.is_partitioned("A", "B")  # symmetric still active


class TestTrafficMatrix:
    """Tests for traffic_matrix() method."""

    def test_empty_network_returns_empty_matrix(self):
        """Empty network returns empty traffic matrix."""
        network = Network(name="TestNetwork")
        assert network.traffic_matrix() == []

    def test_traffic_matrix_reflects_link_stats(self):
        """traffic_matrix() reflects actual link statistics."""
        sender = SenderEntity(name="Sender")
        receiver = CollectorEntity(name="Receiver")

        link = NetworkLink(name="Link", latency=ConstantLatency(0.005))
        network = Network(name="TestNetwork")
        network.add_link(sender, receiver, link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver],
        )

        # Send 3 events
        for i in range(3):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="Msg",
                target=network,
            )
            event.context["metadata"]["source"] = "Sender"
            event.context["metadata"]["destination"] = "Receiver"
            sim.schedule(event)

        sim.run()

        matrix = network.traffic_matrix()
        assert len(matrix) == 1

        stats = matrix[0]
        assert isinstance(stats, LinkStats)
        assert stats.source == "Sender"
        assert stats.destination == "Receiver"
        assert stats.packets_sent == 3
        assert stats.packets_dropped == 0

    def test_bidirectional_link_has_two_entries(self):
        """Bidirectional links produce two traffic matrix entries."""
        a = CollectorEntity(name="A")
        b = CollectorEntity(name="B")

        link = NetworkLink(name="Link", latency=ConstantLatency(0.005))
        network = Network(name="TestNetwork")
        network.add_bidirectional_link(a, b, link)

        matrix = network.traffic_matrix()
        assert len(matrix) == 2

        sources = {s.source for s in matrix}
        dests = {s.destination for s in matrix}
        assert sources == {"A", "B"}
        assert dests == {"A", "B"}

    def test_traffic_matrix_tracks_bytes(self):
        """traffic_matrix() includes bytes_transmitted from link stats."""
        sender = SenderEntity(name="Sender")
        receiver = CollectorEntity(name="Receiver")

        link = NetworkLink(
            name="Link",
            latency=ConstantLatency(0.005),
            bandwidth_bps=1_000_000_000,
        )
        network = Network(name="TestNetwork")
        network.add_link(sender, receiver, link)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[network, sender, receiver],
        )

        event = Event(time=Instant.Epoch, event_type="Msg", target=network)
        event.context["metadata"]["source"] = "Sender"
        event.context["metadata"]["destination"] = "Receiver"
        event.context["metadata"]["payload_size"] = 1024
        sim.schedule(event)

        sim.run()

        matrix = network.traffic_matrix()
        assert len(matrix) == 1
        assert matrix[0].bytes_transmitted == 1024
