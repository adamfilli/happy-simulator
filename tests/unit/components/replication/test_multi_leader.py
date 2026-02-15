"""Tests for MultiLeaderReplication."""

from happysimulator import (
    Event,
    Instant,
    Network,
    SimFuture,
    Simulation,
    datacenter_network,
)
from happysimulator.components.datastore import KVStore
from happysimulator.components.replication.conflict_resolver import (
    LastWriterWins,
    VersionedValue,
)
from happysimulator.components.replication.multi_leader import (
    LeaderNode,
    MultiLeaderStats,
)


def _make_store(name: str) -> KVStore:
    return KVStore(name, write_latency=0.001, read_latency=0.001)


def _make_leaders(n: int, network: Network, **kwargs) -> list[LeaderNode]:
    """Create n LeaderNodes wired as peers."""
    leaders = [
        LeaderNode(
            f"leader-{i}",
            store=_make_store(f"store-{i}"),
            network=network,
            **kwargs,
        )
        for i in range(n)
    ]
    for leader in leaders:
        leader.add_peers([l for l in leaders if l is not leader])
    return leaders


class TestLeaderNodeCreation:
    """Tests for LeaderNode construction."""

    def test_creates_with_defaults(self):
        """LeaderNode creates with LWW resolver by default."""
        store = _make_store("store")
        network = Network(name="net")
        leader = LeaderNode("leader", store=store, network=network)

        assert leader.name == "leader"
        assert leader.stats.writes == 0
        assert leader.peers == []

    def test_add_peers(self):
        """add_peers wires peer list and initializes VectorClock."""
        network = Network(name="net")
        leaders = _make_leaders(3, network)

        assert len(leaders[0].peers) == 2
        assert leaders[0]._vclock is not None

    def test_merkle_tree_starts_empty(self):
        """MerkleTree is empty on creation."""
        store = _make_store("store")
        network = Network(name="net")
        leader = LeaderNode("leader", store=store, network=network)

        assert leader.merkle_tree.size == 0


class TestMultiLeaderWrite:
    """Tests for write operations."""

    def test_local_write(self):
        """Write applies to local store and updates versions."""
        network = Network(name="net")
        leaders = _make_leaders(2, network)
        # Don't add network links — just test local apply
        # We'll use a single write without replication reaching peers

        reply_future = SimFuture()
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=leaders[0],
            context={
                "metadata": {
                    "key": "x",
                    "value": 42,
                    "reply_future": reply_future,
                }
            },
        )

        all_entities = [*leaders, network]
        all_entities.extend(l.store for l in leaders)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(write_event)
        sim.run()

        assert leaders[0].store.get_sync("x") == 42
        assert leaders[0].stats.writes == 1
        assert "x" in leaders[0].versions
        assert leaders[0].merkle_tree.size == 1

    def test_write_replicates_to_peer(self):
        """Write on one leader replicates to peer."""
        network = Network(name="net")
        leaders = _make_leaders(2, network)
        network.add_bidirectional_link(
            leaders[0],
            leaders[1],
            datacenter_network("link"),
        )

        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=leaders[0],
            context={"metadata": {"key": "x", "value": 42}},
        )

        all_entities = [*leaders, network]
        all_entities.extend(l.store for l in leaders)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(write_event)
        sim.run()

        # Both leaders should have the value
        assert leaders[0].store.get_sync("x") == 42
        assert leaders[1].store.get_sync("x") == 42
        assert leaders[0].stats.replications_sent == 1
        assert leaders[1].stats.replications_received == 1

    def test_write_replicates_to_three_peers(self):
        """Write replicates to all peers in a 3-node cluster."""
        network = Network(name="net")
        leaders = _make_leaders(3, network)
        for i in range(3):
            for j in range(i + 1, 3):
                network.add_bidirectional_link(
                    leaders[i],
                    leaders[j],
                    datacenter_network(f"link-{i}-{j}"),
                )

        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=leaders[0],
            context={"metadata": {"key": "y", "value": 99}},
        )

        all_entities = [*leaders, network]
        all_entities.extend(l.store for l in leaders)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(write_event)
        sim.run()

        for leader in leaders:
            assert leader.store.get_sync("y") == 99


class TestMultiLeaderRead:
    """Tests for read operations."""

    def test_read_from_local_store(self):
        """Read returns value from local store."""
        network = Network(name="net")
        leader = LeaderNode("leader", store=_make_store("s"), network=network)
        leader.store.put_sync("z", 77)

        reply_future = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=leader,
            context={"metadata": {"key": "z", "reply_future": reply_future}},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[leader, network, leader.store],
        )
        sim.schedule(read_event)
        sim.run()

        assert reply_future.is_resolved
        assert reply_future.value["value"] == 77


class TestConflictResolution:
    """Tests for concurrent write conflict resolution."""

    def test_lww_resolves_concurrent_writes(self):
        """LWW picks the version with the higher timestamp on conflict."""
        network = Network(name="net")
        leaders = _make_leaders(2, network, conflict_resolver=LastWriterWins())
        network.add_bidirectional_link(
            leaders[0],
            leaders[1],
            datacenter_network("link"),
        )

        # Write different values to the same key on each leader at different times
        write0 = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=leaders[0],
            context={"metadata": {"key": "conflict", "value": "from-0"}},
        )
        write1 = Event(
            time=Instant.from_seconds(0.2),
            event_type="Write",
            target=leaders[1],
            context={"metadata": {"key": "conflict", "value": "from-1"}},
        )

        all_entities = [*leaders, network]
        all_entities.extend(l.store for l in leaders)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule([write0, write1])
        sim.run()

        # Both leaders should converge to the same value
        val0 = leaders[0].store.get_sync("conflict")
        val1 = leaders[1].store.get_sync("conflict")
        assert val0 == val1

    def test_custom_resolver(self):
        """Custom resolver is used for concurrent writes."""
        resolved_keys = []

        def track_resolver(key, versions):
            resolved_keys.append(key)
            return versions[-1]  # Pick last

        from happysimulator.components.replication.conflict_resolver import CustomResolver

        resolver = CustomResolver(track_resolver)

        network = Network(name="net")
        leaders = _make_leaders(2, network, conflict_resolver=resolver)
        network.add_bidirectional_link(
            leaders[0],
            leaders[1],
            datacenter_network("link"),
        )

        # Concurrent writes (same time = concurrent vector clocks)
        write0 = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=leaders[0],
            context={"metadata": {"key": "k", "value": "a"}},
        )
        write1 = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=leaders[1],
            context={"metadata": {"key": "k", "value": "b"}},
        )

        all_entities = [*leaders, network]
        all_entities.extend(l.store for l in leaders)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule([write0, write1])
        sim.run()

        # Conflict should have been detected and resolved
        total_conflicts = sum(l.stats.conflicts_detected for l in leaders)
        assert total_conflicts >= 1


class TestAntiEntropy:
    """Tests for anti-entropy synchronization."""

    def test_anti_entropy_repairs_missing_key(self):
        """Anti-entropy detects and repairs a key missing from one replica."""
        network = Network(name="net")
        leaders = _make_leaders(2, network, anti_entropy_interval=1.0)
        network.add_bidirectional_link(
            leaders[0],
            leaders[1],
            datacenter_network("link"),
        )

        # Directly insert into leader-0 (bypassing replication to simulate divergence)
        leaders[0].store.put_sync("divergent", "value-from-0")
        leaders[0]._versions["divergent"] = VersionedValue(
            value="value-from-0",
            timestamp=1.0,
            writer_id="leader-0",
            vector_clock={"leader-0": 1, "leader-1": 0},
        )
        leaders[0]._merkle.update("divergent", "value-from-0")

        # Schedule anti-entropy event (non-daemon so sim processes it)
        ae0 = Event(
            time=Instant.from_seconds(1.0),
            event_type="AntiEntropy",
            target=leaders[0],
            daemon=False,
        )

        all_entities = [*leaders, network]
        all_entities.extend(l.store for l in leaders)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=all_entities,
        )
        sim.schedule(ae0)
        sim.run()

        # Leader-1 should have received the key via anti-entropy
        assert leaders[1].store.get_sync("divergent") == "value-from-0"
        assert leaders[0].stats.anti_entropy_syncs >= 1


class TestMultiLeaderStats:
    """Tests for statistics tracking."""

    def test_stats_initialized(self):
        """Stats start at zero."""
        stats = MultiLeaderStats()
        assert stats.writes == 0
        assert stats.reads == 0
        assert stats.conflicts_detected == 0
        assert stats.anti_entropy_syncs == 0
