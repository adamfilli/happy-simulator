"""Tests for CRDTStore entity."""

from happysimulator import (
    Event,
    Instant,
    Network,
    SimFuture,
    Simulation,
    datacenter_network,
)
from happysimulator.components.crdt.crdt_store import CRDTStore, CRDTStoreStats
from happysimulator.components.crdt.g_counter import GCounter


class TestCRDTStoreCreation:
    """Tests for CRDTStore construction."""

    def test_creates_with_defaults(self):
        network = Network(name="net")
        store = CRDTStore("node-a", network=network)
        assert store.name == "node-a"
        assert store.stats.writes == 0
        assert store.stats.reads == 0

    def test_stats_initialized_to_zero(self):
        stats = CRDTStoreStats()
        assert stats.writes == 0
        assert stats.reads == 0
        assert stats.gossip_sent == 0
        assert stats.gossip_received == 0
        assert stats.keys_merged == 0
        assert stats.convergence_checks == 0

    def test_crdts_starts_empty(self):
        network = Network(name="net")
        store = CRDTStore("node-a", network=network)
        assert store.crdts == {}

    def test_add_peers(self):
        network = Network(name="net")
        a = CRDTStore("node-a", network=network)
        b = CRDTStore("node-b", network=network)
        a.add_peers([b])
        assert len(a._peers) == 1

    def test_get_gossip_event_returns_none_without_peers(self):
        network = Network(name="net")
        store = CRDTStore("node-a", network=network)
        Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[store, network],
        )
        assert store.get_gossip_event() is None

    def test_get_gossip_event_disabled(self):
        network = Network(name="net")
        a = CRDTStore("node-a", network=network, gossip_interval=0)
        b = CRDTStore("node-b", network=network)
        a.add_peers([b])
        Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[a, b, network],
        )
        assert a.get_gossip_event() is None


class TestCRDTStoreReadWrite:
    """Tests for local read/write operations."""

    def test_write_creates_crdt(self):
        """Write to a key creates a CRDT and applies the operation."""
        network = Network(name="net")
        store = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
        )

        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store,
            context={
                "metadata": {
                    "key": "hits",
                    "operation": "increment",
                    "value": 5,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[store, network],
        )
        sim.schedule(write_event)
        sim.run()

        assert store.stats.writes == 1
        assert "hits" in store.crdts
        assert store.crdts["hits"].value == 5

    def test_read_returns_value(self):
        """Read returns the current CRDT value."""
        network = Network(name="net")
        store = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
        )

        # Write then read
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store,
            context={
                "metadata": {
                    "key": "hits",
                    "operation": "increment",
                    "value": 3,
                }
            },
        )

        reply = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.2),
            event_type="Read",
            target=store,
            context={"metadata": {"key": "hits", "reply_future": reply}},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[store, network],
        )
        sim.schedule([write_event, read_event])
        sim.run()

        assert reply.is_resolved
        assert reply.value["value"] == 3
        assert store.stats.reads == 1

    def test_read_nonexistent_key_returns_none(self):
        """Reading a key that doesn't exist returns None."""
        network = Network(name="net")
        store = CRDTStore("node-a", network=network)

        reply = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=store,
            context={"metadata": {"key": "missing", "reply_future": reply}},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[store, network],
        )
        sim.schedule(read_event)
        sim.run()

        assert reply.is_resolved
        assert reply.value["value"] is None

    def test_write_with_reply_future(self):
        """Write resolves reply future with result."""
        network = Network(name="net")
        store = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
        )

        reply = SimFuture()
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store,
            context={
                "metadata": {
                    "key": "counter",
                    "operation": "increment",
                    "value": 1,
                    "reply_future": reply,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[store, network],
        )
        sim.schedule(write_event)
        sim.run()

        assert reply.is_resolved
        assert reply.value["status"] == "ok"


class TestCRDTStoreGossip:
    """Tests for gossip-based replication."""

    def test_gossip_converges_two_nodes(self):
        """Two nodes with different writes converge after gossip."""
        network = Network(name="net")
        store_a = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_b = CRDTStore(
            "node-b",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_a.add_peers([store_b])
        store_b.add_peers([store_a])
        network.add_bidirectional_link(
            store_a,
            store_b,
            datacenter_network("link"),
        )

        # Write to node-a
        write_a = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_a,
            context={
                "metadata": {
                    "key": "hits",
                    "operation": "increment",
                    "value": 5,
                }
            },
        )
        # Write to node-b
        write_b = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_b,
            context={
                "metadata": {
                    "key": "hits",
                    "operation": "increment",
                    "value": 3,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[store_a, store_b, network],
        )

        # Schedule writes and gossip events
        sim.schedule([write_a, write_b])

        # Schedule gossip ticks (non-daemon so they are processed)
        gossip_a = Event(
            time=Instant.from_seconds(1.0),
            event_type="GossipTick",
            target=store_a,
            daemon=False,
        )
        sim.schedule(gossip_a)
        sim.run()

        # Both nodes should have converged: value = 5 + 3 = 8
        assert store_a.crdts["hits"].value == 8
        assert store_b.crdts["hits"].value == 8
        assert store_a.stats.gossip_sent >= 1

    def test_gossip_propagates_new_keys(self):
        """Gossip propagates keys that only exist on one node."""
        network = Network(name="net")
        store_a = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_b = CRDTStore(
            "node-b",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_a.add_peers([store_b])
        store_b.add_peers([store_a])
        network.add_bidirectional_link(
            store_a,
            store_b,
            datacenter_network("link"),
        )

        # Write different keys to each node
        write_a = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_a,
            context={
                "metadata": {
                    "key": "key-a",
                    "operation": "increment",
                    "value": 1,
                }
            },
        )
        write_b = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_b,
            context={
                "metadata": {
                    "key": "key-b",
                    "operation": "increment",
                    "value": 1,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[store_a, store_b, network],
        )
        sim.schedule([write_a, write_b])

        # Gossip from a->b and then b->a (via response)
        gossip_a = Event(
            time=Instant.from_seconds(1.0),
            event_type="GossipTick",
            target=store_a,
            daemon=False,
        )
        sim.schedule(gossip_a)
        sim.run()

        # Both nodes should have both keys
        assert "key-a" in store_a.crdts
        assert "key-b" in store_a.crdts
        assert "key-a" in store_b.crdts
        assert "key-b" in store_b.crdts


class TestCRDTStorePartition:
    """Tests for behavior during network partitions."""

    def test_writes_during_partition(self):
        """Both nodes accept writes independently during partition."""
        network = Network(name="net")
        store_a = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_b = CRDTStore(
            "node-b",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_a.add_peers([store_b])
        store_b.add_peers([store_a])
        network.add_bidirectional_link(
            store_a,
            store_b,
            datacenter_network("link"),
        )

        # Create partition
        network.partition([store_a], [store_b])

        write_a = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_a,
            context={
                "metadata": {
                    "key": "counter",
                    "operation": "increment",
                    "value": 10,
                }
            },
        )
        write_b = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_b,
            context={
                "metadata": {
                    "key": "counter",
                    "operation": "increment",
                    "value": 7,
                }
            },
        )

        # Gossip during partition (will be dropped by network)
        gossip_a = Event(
            time=Instant.from_seconds(1.0),
            event_type="GossipTick",
            target=store_a,
            daemon=False,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[store_a, store_b, network],
        )
        sim.schedule([write_a, write_b, gossip_a])
        sim.run()

        # Each node has its own value (diverged)
        assert store_a.crdts["counter"].value == 10
        assert store_b.crdts["counter"].value == 7

    def test_convergence_after_heal(self):
        """Nodes converge after partition heals and gossip resumes."""
        network = Network(name="net")
        store_a = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_b = CRDTStore(
            "node-b",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
            gossip_interval=1.0,
        )
        store_a.add_peers([store_b])
        store_b.add_peers([store_a])
        network.add_bidirectional_link(
            store_a,
            store_b,
            datacenter_network("link"),
        )

        # Write during partition
        partition = network.partition([store_a], [store_b])

        write_a = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_a,
            context={
                "metadata": {
                    "key": "counter",
                    "operation": "increment",
                    "value": 10,
                }
            },
        )
        write_b = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=store_b,
            context={
                "metadata": {
                    "key": "counter",
                    "operation": "increment",
                    "value": 7,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[],
            entities=[store_a, store_b, network],
        )
        sim.schedule([write_a, write_b])

        # Heal partition at t=2.0, then gossip at t=3.0
        heal_event = Event.once(
            time=Instant.from_seconds(2.0),
            event_type="Heal",
            fn=lambda e: partition.heal(),
        )
        gossip_after_heal = Event(
            time=Instant.from_seconds(3.0),
            event_type="GossipTick",
            target=store_a,
            daemon=False,
        )

        sim.schedule([heal_event, gossip_after_heal])
        sim.run()

        # Both should converge to 10 + 7 = 17
        assert store_a.crdts["counter"].value == 17
        assert store_b.crdts["counter"].value == 17

    def test_get_or_create(self):
        """get_or_create creates CRDT on first access."""
        network = Network(name="net")
        store = CRDTStore(
            "node-a",
            network=network,
            crdt_factory=lambda nid: GCounter(nid),
        )

        Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[store, network],
        )

        crdt = store.get_or_create("new-key")
        assert isinstance(crdt, GCounter)
        assert crdt.value == 0

        # Same key returns same instance
        assert store.get_or_create("new-key") is crdt
