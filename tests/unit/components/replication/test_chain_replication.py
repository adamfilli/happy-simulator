"""Tests for ChainReplication."""

import pytest

from happysimulator import (
    Event,
    Instant,
    Network,
    Simulation,
    SimFuture,
    datacenter_network,
)
from happysimulator.components.datastore import KVStore
from happysimulator.components.replication.chain_replication import (
    ChainNode,
    ChainNodeRole,
    ChainReplicationStats,
    build_chain,
)


def _make_store(name: str) -> KVStore:
    return KVStore(name, write_latency=0.001, read_latency=0.001)


class TestBuildChain:
    """Tests for build_chain factory."""

    def test_builds_two_node_chain(self):
        """Two-node chain has HEAD and TAIL."""
        network = Network(name="net")
        nodes = build_chain(["head", "tail"], network, _make_store)

        assert len(nodes) == 2
        assert nodes[0].role == ChainNodeRole.HEAD
        assert nodes[1].role == ChainNodeRole.TAIL

    def test_builds_three_node_chain(self):
        """Three-node chain has HEAD, MIDDLE, TAIL."""
        network = Network(name="net")
        nodes = build_chain(["head", "mid", "tail"], network, _make_store)

        assert len(nodes) == 3
        assert nodes[0].role == ChainNodeRole.HEAD
        assert nodes[1].role == ChainNodeRole.MIDDLE
        assert nodes[2].role == ChainNodeRole.TAIL

    def test_wires_next_and_prev(self):
        """Nodes are linked via next_node and prev_node."""
        network = Network(name="net")
        nodes = build_chain(["a", "b", "c"], network, _make_store)

        assert nodes[0].next_node is nodes[1]
        assert nodes[0].prev_node is None
        assert nodes[1].next_node is nodes[2]
        assert nodes[1].prev_node is nodes[0]
        assert nodes[2].next_node is None
        assert nodes[2].prev_node is nodes[1]

    def test_wires_head_node(self):
        """All nodes have head_node pointing to the HEAD."""
        network = Network(name="net")
        nodes = build_chain(["a", "b", "c"], network, _make_store)

        for node in nodes:
            assert node.head_node is nodes[0]

    def test_rejects_single_node(self):
        """Chain requires at least 2 nodes."""
        network = Network(name="net")
        with pytest.raises(ValueError, match="at least 2"):
            build_chain(["alone"], network, _make_store)

    def test_craq_enabled(self):
        """CRAQ flag propagates to all nodes."""
        network = Network(name="net")
        nodes = build_chain(["a", "b"], network, _make_store, craq_enabled=True)

        for node in nodes:
            assert node._craq_enabled


class TestChainNodeCreation:
    """Tests for ChainNode construction."""

    def test_creates_with_defaults(self):
        """ChainNode creates with MIDDLE role by default."""
        store = _make_store("store")
        network = Network(name="net")
        node = ChainNode("node", store=store, network=network)

        assert node.name == "node"
        assert node.role == ChainNodeRole.MIDDLE
        assert node.stats.writes_received == 0


class TestChainWritePropagation:
    """Tests for write propagation through the chain."""

    def test_two_node_write(self):
        """Write propagates from HEAD to TAIL and acks."""
        network = Network(name="net")
        nodes = build_chain(["head", "tail"], network, _make_store)
        network.add_bidirectional_link(nodes[0], nodes[1], datacenter_network("link"))

        reply_future = SimFuture()
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=nodes[0],
            context={"metadata": {
                "key": "x", "value": 42, "reply_future": reply_future,
            }},
        )

        all_entities = [*nodes, network]
        for n in nodes:
            all_entities.append(n.store)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(write_event)
        sim.run()

        # Both nodes should have the value
        assert nodes[0].store.get_sync("x") == 42
        assert nodes[1].store.get_sync("x") == 42

        # Write should be acknowledged
        assert reply_future.is_resolved
        assert reply_future.value["status"] == "ok"

    def test_three_node_write(self):
        """Write propagates through HEAD -> MIDDLE -> TAIL."""
        network = Network(name="net")
        nodes = build_chain(["head", "mid", "tail"], network, _make_store)
        network.add_bidirectional_link(nodes[0], nodes[1], datacenter_network("link01"))
        network.add_bidirectional_link(nodes[1], nodes[2], datacenter_network("link12"))
        # HEAD needs route to TAIL for ack
        network.add_bidirectional_link(nodes[0], nodes[2], datacenter_network("link02"))

        reply_future = SimFuture()
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=nodes[0],
            context={"metadata": {
                "key": "y", "value": 99, "reply_future": reply_future,
            }},
        )

        all_entities = [*nodes, network]
        for n in nodes:
            all_entities.append(n.store)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(write_event)
        sim.run()

        # All nodes should have the value
        for node in nodes:
            assert node.store.get_sync("y") == 99

        assert reply_future.is_resolved

    def test_stats_tracked(self):
        """Chain replication stats are tracked correctly."""
        network = Network(name="net")
        nodes = build_chain(["head", "tail"], network, _make_store)
        network.add_bidirectional_link(nodes[0], nodes[1], datacenter_network("link"))

        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=nodes[0],
            context={"metadata": {"key": "x", "value": 1}},
        )

        all_entities = [*nodes, network]
        for n in nodes:
            all_entities.append(n.store)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(write_event)
        sim.run()

        assert nodes[0].stats.writes_received == 1
        assert nodes[0].stats.propagations_sent == 1
        assert nodes[1].stats.propagations_received == 1
        assert nodes[1].stats.acks_sent == 1


class TestChainReads:
    """Tests for read operations."""

    def test_tail_read(self):
        """TAIL serves reads with strongly consistent data."""
        network = Network(name="net")
        nodes = build_chain(["head", "tail"], network, _make_store)
        # Pre-populate tail store
        nodes[1].store.put_sync("z", 77)

        reply_future = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=nodes[1],
            context={"metadata": {"key": "z", "reply_future": reply_future}},
        )

        all_entities = [*nodes, network]
        for n in nodes:
            all_entities.append(n.store)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(read_event)
        sim.run()

        assert reply_future.is_resolved
        assert reply_future.value["value"] == 77


class TestCRAQ:
    """Tests for CRAQ read optimization."""

    def test_clean_key_read_from_any_node(self):
        """CRAQ: clean key can be read from any node."""
        network = Network(name="net")
        nodes = build_chain(["head", "tail"], network, _make_store, craq_enabled=True)
        # Pre-populate head store (simulating committed key)
        nodes[0].store.put_sync("committed", 42)

        reply_future = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=nodes[0],
            context={"metadata": {"key": "committed", "reply_future": reply_future}},
        )

        all_entities = [*nodes, network]
        for n in nodes:
            all_entities.append(n.store)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(read_event)
        sim.run()

        assert reply_future.is_resolved
        assert reply_future.value["value"] == 42

    def test_dirty_key_forwarded_to_tail(self):
        """CRAQ: dirty key read is forwarded to tail."""
        network = Network(name="net")
        nodes = build_chain(["head", "tail"], network, _make_store, craq_enabled=True)
        network.add_bidirectional_link(nodes[0], nodes[1], datacenter_network("link"))

        # Mark key as dirty on head, put value on tail
        nodes[0]._dirty_keys.add("inflight")
        nodes[0].store.put_sync("inflight", "stale")
        nodes[1].store.put_sync("inflight", "committed")

        reply_future = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=nodes[0],
            context={"metadata": {"key": "inflight", "reply_future": reply_future}},
        )

        all_entities = [*nodes, network]
        for n in nodes:
            all_entities.append(n.store)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(read_event)
        sim.run()

        assert reply_future.is_resolved
        assert reply_future.value["value"] == "committed"
