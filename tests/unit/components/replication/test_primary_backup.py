"""Tests for PrimaryBackupReplication."""

from happysimulator import (
    CallbackEntity,
    Event,
    Instant,
    Network,
    SimFuture,
    Simulation,
    datacenter_network,
)
from happysimulator.components.datastore import KVStore
from happysimulator.components.replication.primary_backup import (
    BackupNode,
    PrimaryNode,
    ReplicationMode,
)


class TestPrimaryNodeCreation:
    """Tests for PrimaryNode construction."""

    def test_creates_with_defaults(self):
        """PrimaryNode creates with ASYNC mode by default."""
        store = KVStore("primary_store")
        network = Network(name="net")
        primary = PrimaryNode("primary", store=store, backups=[], network=network)

        assert primary.name == "primary"
        assert primary.mode == ReplicationMode.ASYNC
        assert primary.stats.writes == 0

    def test_creates_with_sync_mode(self):
        """PrimaryNode accepts explicit mode."""
        store = KVStore("store")
        network = Network(name="net")
        primary = PrimaryNode(
            "primary",
            store=store,
            backups=[],
            network=network,
            mode=ReplicationMode.SYNC,
        )

        assert primary.mode == ReplicationMode.SYNC

    def test_tracks_backup_lag(self):
        """PrimaryNode initializes backup lag tracking."""
        store = KVStore("store")
        network = Network(name="net")
        placeholder = CallbackEntity("placeholder", fn=lambda e: None)
        b1 = BackupNode("backup1", store=KVStore("b1"), network=network, primary=placeholder)
        b2 = BackupNode("backup2", store=KVStore("b2"), network=network, primary=placeholder)
        primary = PrimaryNode("primary", store=store, backups=[b1, b2], network=network)

        lag = primary.backup_lag
        assert "backup1" in lag
        assert "backup2" in lag


class TestBackupNodeCreation:
    """Tests for BackupNode construction."""

    def test_creates_with_defaults(self):
        """BackupNode creates with read serving enabled."""
        store = KVStore("backup_store")
        network = Network(name="net")
        primary = CallbackEntity("primary", fn=lambda e: None)
        backup = BackupNode("backup", store=store, network=network, primary=primary)

        assert backup.name == "backup"
        assert backup.stats.replications_applied == 0
        assert backup.last_applied_seq == 0


class TestAsyncReplication:
    """Tests for ASYNC replication mode using a simulation."""

    def test_async_write_replicates(self):
        """ASYNC write applies to primary and replicates to backup."""
        network = Network(name="net")
        primary_store = KVStore("ps", write_latency=0.001, read_latency=0.001)
        backup_store = KVStore("bs", write_latency=0.001, read_latency=0.001)

        primary = PrimaryNode(
            "primary",
            store=primary_store,
            backups=[],
            network=network,
            mode=ReplicationMode.ASYNC,
        )
        backup = BackupNode(
            "backup",
            store=backup_store,
            network=network,
            primary=primary,
        )
        primary._backups = [backup]
        primary._backup_lag = {backup.name: 0}

        network.add_bidirectional_link(primary, backup, datacenter_network("link"))

        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=primary,
            context={"metadata": {"key": "x", "value": 42}},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[primary, backup, network, primary_store, backup_store],
        )
        sim.schedule(write_event)
        sim.run()

        # Primary should have the value
        assert primary_store.get_sync("x") == 42
        assert primary.stats.writes == 1

        # Backup should have received the replication
        assert backup_store.get_sync("x") == 42
        assert backup.stats.replications_applied == 1

    def test_async_read(self):
        """Read returns value from primary store."""
        network = Network(name="net")
        store = KVStore("ps", write_latency=0.001, read_latency=0.001)
        store.put_sync("y", 99)

        primary = PrimaryNode(
            "primary",
            store=store,
            backups=[],
            network=network,
        )

        reply_future = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=primary,
            context={"metadata": {"key": "y", "reply_future": reply_future}},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[primary, network, store],
        )
        sim.schedule(read_event)
        sim.run()

        assert reply_future.is_resolved
        assert reply_future.value["value"] == 99


class TestSyncReplication:
    """Tests for SYNC replication mode using a simulation."""

    def test_sync_write_waits_for_all_acks(self):
        """SYNC write waits for all backup acks before completing."""
        network = Network(name="net")
        primary_store = KVStore("ps", write_latency=0.001, read_latency=0.001)
        backup_store_1 = KVStore("bs1", write_latency=0.001, read_latency=0.001)
        backup_store_2 = KVStore("bs2", write_latency=0.001, read_latency=0.001)

        primary = PrimaryNode(
            "primary",
            store=primary_store,
            backups=[],
            network=network,
            mode=ReplicationMode.SYNC,
        )
        backup1 = BackupNode(
            "backup1",
            store=backup_store_1,
            network=network,
            primary=primary,
        )
        backup2 = BackupNode(
            "backup2",
            store=backup_store_2,
            network=network,
            primary=primary,
        )
        primary._backups = [backup1, backup2]
        primary._backup_lag = {backup1.name: 0, backup2.name: 0}

        network.add_bidirectional_link(primary, backup1, datacenter_network("link1"))
        network.add_bidirectional_link(primary, backup2, datacenter_network("link2"))

        reply_future = SimFuture()
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=primary,
            context={
                "metadata": {
                    "key": "x",
                    "value": 42,
                    "reply_future": reply_future,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[
                primary,
                backup1,
                backup2,
                network,
                primary_store,
                backup_store_1,
                backup_store_2,
            ],
        )
        sim.schedule(write_event)
        sim.run()

        # Both backups should have the value
        assert backup_store_1.get_sync("x") == 42
        assert backup_store_2.get_sync("x") == 42

        # Reply should be resolved
        assert reply_future.is_resolved
        assert reply_future.value["status"] == "ok"


class TestSemiSyncReplication:
    """Tests for SEMI_SYNC replication mode."""

    def test_semi_sync_write_waits_for_one_ack(self):
        """SEMI_SYNC write waits for at least one backup ack."""
        network = Network(name="net")
        primary_store = KVStore("ps", write_latency=0.001, read_latency=0.001)
        backup_store_1 = KVStore("bs1", write_latency=0.001, read_latency=0.001)
        backup_store_2 = KVStore("bs2", write_latency=0.001, read_latency=0.001)

        primary = PrimaryNode(
            "primary",
            store=primary_store,
            backups=[],
            network=network,
            mode=ReplicationMode.SEMI_SYNC,
        )
        backup1 = BackupNode(
            "backup1",
            store=backup_store_1,
            network=network,
            primary=primary,
        )
        backup2 = BackupNode(
            "backup2",
            store=backup_store_2,
            network=network,
            primary=primary,
        )
        primary._backups = [backup1, backup2]
        primary._backup_lag = {backup1.name: 0, backup2.name: 0}

        network.add_bidirectional_link(primary, backup1, datacenter_network("link1"))
        network.add_bidirectional_link(primary, backup2, datacenter_network("link2"))

        reply_future = SimFuture()
        write_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Write",
            target=primary,
            context={
                "metadata": {
                    "key": "x",
                    "value": 42,
                    "reply_future": reply_future,
                }
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[
                primary,
                backup1,
                backup2,
                network,
                primary_store,
                backup_store_1,
                backup_store_2,
            ],
        )
        sim.schedule(write_event)
        sim.run()

        # At least one backup should have the value
        has_value = backup_store_1.get_sync("x") == 42 or backup_store_2.get_sync("x") == 42
        assert has_value

        # Reply should be resolved
        assert reply_future.is_resolved


class TestBackupReads:
    """Tests for stale reads from backup nodes."""

    def test_backup_serves_read(self):
        """BackupNode serves reads from its local store."""
        network = Network(name="net")
        backup_store = KVStore("bs", write_latency=0.001, read_latency=0.001)
        backup_store.put_sync("y", 99)

        primary = CallbackEntity("primary", fn=lambda e: None)
        backup = BackupNode(
            "backup",
            store=backup_store,
            network=network,
            primary=primary,
        )

        reply_future = SimFuture()
        read_event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Read",
            target=backup,
            context={"metadata": {"key": "y", "reply_future": reply_future}},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[backup, network, backup_store],
        )
        sim.schedule(read_event)
        sim.run()

        assert reply_future.is_resolved
        assert reply_future.value["value"] == 99
        assert reply_future.value["stale"] is True
