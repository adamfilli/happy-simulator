"""Primary-backup (master-slave) replication.

Models a primary node that accepts writes and replicates to backup nodes,
with configurable consistency modes:

- **ASYNC**: Write returns immediately after local apply; backups replicate
  in background. Lowest latency, risk of data loss on primary failure.
- **SEMI_SYNC**: Write waits for at least one backup acknowledgment.
  Balances durability and latency.
- **SYNC**: Write waits for ALL backup acknowledgments. Highest durability,
  highest latency.

Backups can optionally serve stale reads.

Example::

    from happysimulator.components.replication import (
        PrimaryNode, BackupNode, ReplicationMode,
    )

    primary = PrimaryNode("primary", store=kv, backups=[backup1, backup2],
                          network=net, mode=ReplicationMode.SEMI_SYNC)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture, all_of
from happysimulator.components.datastore.kv_store import KVStore

logger = logging.getLogger(__name__)


class ReplicationMode(Enum):
    """Write durability mode for primary-backup replication."""

    ASYNC = "async"
    SEMI_SYNC = "semi_sync"
    SYNC = "sync"


@dataclass
class PrimaryBackupStats:
    """Statistics for PrimaryNode.

    Attributes:
        writes: Total write requests received.
        reads: Total read requests received.
        replications_sent: Total replication messages sent to backups.
        acks_received: Total acknowledgments received from backups.
        write_latency_sum: Sum of write latencies for averaging.
    """

    writes: int = 0
    reads: int = 0
    replications_sent: int = 0
    acks_received: int = 0
    write_latency_sum: float = 0.0


@dataclass
class BackupStats:
    """Statistics for BackupNode.

    Attributes:
        replications_applied: Total replication events applied.
        reads: Total read requests served.
        last_applied_seq: Last sequence number applied from primary.
    """

    replications_applied: int = 0
    reads: int = 0
    last_applied_seq: int = 0


class PrimaryNode(Entity):
    """Primary node in a primary-backup replication scheme.

    Accepts Write and Read events. On Write, applies locally and replicates
    to backups according to the configured mode. Tracks per-backup replication
    lag via monotonic sequence numbers.

    Args:
        name: Entity name.
        store: KVStore for local data.
        backups: List of BackupNode entities.
        network: Network for sending replication messages.
        mode: Replication consistency mode.
    """

    def __init__(
        self,
        name: str,
        store: KVStore,
        backups: list[Entity],
        network: Entity,
        mode: ReplicationMode = ReplicationMode.ASYNC,
    ):
        super().__init__(name)
        self._store = store
        self._backups = backups
        self._network = network
        self._mode = mode
        self._seq: int = 0
        self._backup_lag: dict[str, int] = {b.name: 0 for b in backups}
        self.stats = PrimaryBackupStats()

    @property
    def mode(self) -> ReplicationMode:
        """Current replication mode."""
        return self._mode

    @property
    def backup_lag(self) -> dict[str, int]:
        """Per-backup replication lag in sequence numbers."""
        return dict(self._backup_lag)

    @property
    def store(self) -> KVStore:
        """The underlying KVStore."""
        return self._store

    def handle_event(
        self, event: Event
    ) -> Generator[float | SimFuture | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Handle Write, Read, and ReplicationAck events."""
        if event.event_type == "Write":
            return (yield from self._handle_write(event))
        elif event.event_type == "Read":
            return (yield from self._handle_read(event))
        elif event.event_type == "ReplicationAck":
            self._handle_ack(event)
            return None
        return None

    def _handle_write(
        self, event: Event
    ) -> Generator[float | SimFuture | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Process a write: apply locally, replicate to backups."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self.stats.writes += 1
        self._seq += 1
        seq = self._seq

        # Apply locally
        yield from self._store.put(key, value)

        # Update lag tracking
        for b in self._backups:
            self._backup_lag[b.name] = seq - self._backup_lag.get(f"_acked_{b.name}", 0)

        # Send replication to backups
        if self._mode == ReplicationMode.ASYNC:
            # Fire and forget
            events = []
            for backup in self._backups:
                events.append(self._network.send(
                    self, backup, "Replicate",
                    payload={"key": key, "value": value, "seq": seq},
                ))
                self.stats.replications_sent += 1
            if events:
                yield 0.0, events
            if reply_future is not None:
                reply_future.resolve({"status": "ok", "seq": seq})
            return None

        elif self._mode == ReplicationMode.SEMI_SYNC:
            # Wait for at least one ack
            ack_futures = []
            events = []
            for backup in self._backups:
                ack_future = SimFuture()
                events.append(self._network.send(
                    self, backup, "Replicate",
                    payload={
                        "key": key, "value": value, "seq": seq,
                        "ack_future": ack_future,
                    },
                ))
                ack_futures.append(ack_future)
                self.stats.replications_sent += 1
            yield 0.0, events

            # Wait for first ack (any_of requires 2+, handle edge case)
            if len(ack_futures) >= 2:
                from happysimulator.core.sim_future import any_of
                _idx, _val = yield any_of(*ack_futures)
            elif ack_futures:
                yield ack_futures[0]

            if reply_future is not None:
                reply_future.resolve({"status": "ok", "seq": seq})
            return None

        else:  # SYNC
            # Wait for all acks
            ack_futures = []
            events = []
            for backup in self._backups:
                ack_future = SimFuture()
                events.append(self._network.send(
                    self, backup, "Replicate",
                    payload={
                        "key": key, "value": value, "seq": seq,
                        "ack_future": ack_future,
                    },
                ))
                ack_futures.append(ack_future)
                self.stats.replications_sent += 1
            yield 0.0, events

            if len(ack_futures) >= 2:
                yield all_of(*ack_futures)
            elif ack_futures:
                yield ack_futures[0]

            if reply_future is not None:
                reply_future.resolve({"status": "ok", "seq": seq})
            return None

    def _handle_read(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Process a read: return value from local store."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self.stats.reads += 1
        value = yield from self._store.get(key)

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "value": value})
        return None

    def _handle_ack(self, event: Event) -> None:
        """Process a replication acknowledgment from a backup."""
        metadata = event.context.get("metadata", {})
        backup_name = metadata.get("source")
        seq = metadata.get("seq", 0)

        self.stats.acks_received += 1
        acked_key = f"_acked_{backup_name}"
        prev = self._backup_lag.get(acked_key, 0)
        if seq > prev:
            self._backup_lag[acked_key] = seq
            self._backup_lag[backup_name] = self._seq - seq


class BackupNode(Entity):
    """Backup node in a primary-backup replication scheme.

    Receives Replicate events from the primary, applies them locally,
    and sends ReplicationAck. Can optionally serve Read events (stale data).

    Args:
        name: Entity name.
        store: KVStore for local data.
        network: Network for sending ack messages.
        primary: The primary node entity (for ack routing).
        serve_reads: Whether to serve Read events.
    """

    def __init__(
        self,
        name: str,
        store: KVStore,
        network: Entity,
        primary: Entity,
        serve_reads: bool = True,
    ):
        super().__init__(name)
        self._store = store
        self._network = network
        self._primary = primary
        self._serve_reads = serve_reads
        self.stats = BackupStats()

    @property
    def store(self) -> KVStore:
        """The underlying KVStore."""
        return self._store

    @property
    def last_applied_seq(self) -> int:
        """Last applied sequence number."""
        return self.stats.last_applied_seq

    def handle_event(
        self, event: Event
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Handle Replicate and Read events."""
        if event.event_type == "Replicate":
            return (yield from self._handle_replicate(event))
        elif event.event_type == "Read" and self._serve_reads:
            return (yield from self._handle_read(event))
        return None

    def _handle_replicate(
        self, event: Event
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Apply a replicated write and send ack."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        seq = metadata.get("seq", 0)
        ack_future: SimFuture | None = metadata.get("ack_future")

        # Apply locally
        yield from self._store.put(key, value)

        self.stats.replications_applied += 1
        self.stats.last_applied_seq = seq

        # Resolve ack future if present (for SEMI_SYNC/SYNC)
        if ack_future is not None:
            ack_future.resolve({"backup": self.name, "seq": seq})

        # Also send ack event for lag tracking (ASYNC mode)
        ack_event = self._network.send(
            self, self._primary, "ReplicationAck",
            payload={"seq": seq},
        )
        yield 0.0, [ack_event]
        return None

    def _handle_read(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Serve a read from the local (possibly stale) store."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self.stats.reads += 1
        value = yield from self._store.get(key)

        if reply_future is not None:
            reply_future.resolve({
                "status": "ok",
                "value": value,
                "stale": True,
                "seq": self.stats.last_applied_seq,
            })
        return None
