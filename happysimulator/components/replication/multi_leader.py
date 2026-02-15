"""Multi-leader (multi-master) replication.

Models a set of leader nodes where any node can accept writes. Each write
is stamped with a VectorClock and replicated to all peers. Concurrent
writes are detected via vector clock comparison and resolved using a
pluggable ConflictResolver.

Periodic anti-entropy synchronization uses MerkleTree comparison to
detect and repair divergent keys between replicas.

Based on patterns from CouchDB, Dynamo, and Riak.

Example::

    from happysimulator.components.replication import (
        LeaderNode, LastWriterWins,
    )

    leaders = [LeaderNode(f"dc-{i}", ...) for i in range(3)]
    for leader in leaders:
        leader.add_peers([l for l in leaders if l is not leader])
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.logical_clocks import VectorClock
from happysimulator.core.sim_future import SimFuture
from happysimulator.components.datastore.kv_store import KVStore
from happysimulator.components.replication.conflict_resolver import (
    ConflictResolver,
    LastWriterWins,
    VersionedValue,
)
from happysimulator.sketching.merkle_tree import MerkleTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MultiLeaderStats:
    """Statistics for a LeaderNode.

    Attributes:
        writes: Local write requests processed.
        reads: Local read requests processed.
        replications_sent: Replication messages sent to peers.
        replications_received: Replication messages received from peers.
        conflicts_detected: Number of concurrent write conflicts detected.
        conflicts_resolved: Number of conflicts resolved.
        anti_entropy_syncs: Number of anti-entropy synchronization rounds.
        anti_entropy_keys_repaired: Keys repaired via anti-entropy.
    """

    writes: int = 0
    reads: int = 0
    replications_sent: int = 0
    replications_received: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    anti_entropy_syncs: int = 0
    anti_entropy_keys_repaired: int = 0


class LeaderNode(Entity):
    """A leader node in a multi-leader replication scheme.

    Any node can accept writes. Writes are stamped with a VectorClock
    for conflict detection. Replication is asynchronous — writes are sent
    to all peers after local apply. Concurrent writes (detected via vector
    clock comparison) are resolved using the configured ConflictResolver.

    Periodic anti-entropy compares MerkleTree root hashes with a random
    peer and repairs divergent keys.

    Args:
        name: Entity name (also used as VectorClock node ID).
        store: KVStore for local data.
        network: Network for inter-node communication.
        conflict_resolver: Strategy for resolving concurrent writes.
        anti_entropy_interval: Seconds between anti-entropy rounds (0 to disable).
    """

    def __init__(
        self,
        name: str,
        store: KVStore,
        network: Entity,
        conflict_resolver: ConflictResolver | None = None,
        anti_entropy_interval: float = 0.0,
    ):
        super().__init__(name)
        self._store = store
        self._network = network
        self._resolver = conflict_resolver or LastWriterWins()
        self._anti_entropy_interval = anti_entropy_interval

        # Peers (set via add_peers)
        self._peers: list[Entity] = []

        # Per-key versioned values
        self._versions: dict[str, VersionedValue] = {}

        # MerkleTree for anti-entropy
        self._merkle = MerkleTree.build({})

        # VectorClock (initialized lazily when peers are known)
        self._vclock: VectorClock | None = None

        self._writes = 0
        self._reads = 0
        self._replications_sent = 0
        self._replications_received = 0
        self._conflicts_detected = 0
        self._conflicts_resolved = 0
        self._anti_entropy_syncs = 0
        self._anti_entropy_keys_repaired = 0

    @property
    def stats(self) -> MultiLeaderStats:
        """Frozen snapshot of leader node statistics."""
        return MultiLeaderStats(
            writes=self._writes,
            reads=self._reads,
            replications_sent=self._replications_sent,
            replications_received=self._replications_received,
            conflicts_detected=self._conflicts_detected,
            conflicts_resolved=self._conflicts_resolved,
            anti_entropy_syncs=self._anti_entropy_syncs,
            anti_entropy_keys_repaired=self._anti_entropy_keys_repaired,
        )

    @property
    def store(self) -> KVStore:
        """The underlying KVStore."""
        return self._store

    @property
    def peers(self) -> list[Entity]:
        """Current peer nodes."""
        return list(self._peers)

    @property
    def merkle_tree(self) -> MerkleTree:
        """The MerkleTree for anti-entropy synchronization."""
        return self._merkle

    @property
    def versions(self) -> dict[str, VersionedValue]:
        """Per-key versioned values."""
        return dict(self._versions)

    def add_peers(self, peers: list[Entity]) -> None:
        """Set peer nodes for replication.

        Also initializes the VectorClock with all node IDs.

        Args:
            peers: List of peer LeaderNode entities.
        """
        self._peers = list(peers)
        all_ids = [self.name] + [p.name for p in peers]
        self._vclock = VectorClock(self.name, all_ids)

    def get_anti_entropy_event(self) -> Event | None:
        """Create the first anti-entropy daemon event.

        Returns:
            A daemon event scheduled for the first anti-entropy interval,
            or None if anti-entropy is disabled.
        """
        if self._anti_entropy_interval <= 0 or not self._peers:
            return None
        return Event(
            time=self.now if self.now.to_seconds() > 0 else self.now.__class__.from_seconds(self._anti_entropy_interval),
            event_type="AntiEntropy",
            target=self,
            daemon=True,
        )

    def handle_event(
        self, event: Event,
    ) -> Generator[float | SimFuture | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Route events by type."""
        if event.event_type == "Write":
            return (yield from self._handle_write(event))
        elif event.event_type == "Read":
            return (yield from self._handle_read(event))
        elif event.event_type == "Replicate":
            return (yield from self._handle_replicate(event))
        elif event.event_type == "AntiEntropy":
            return (yield from self._handle_anti_entropy(event))
        elif event.event_type == "AntiEntropyRequest":
            return (yield from self._handle_anti_entropy_request(event))
        elif event.event_type == "AntiEntropyResponse":
            return (yield from self._handle_anti_entropy_response(event))
        return None

    def _handle_write(
        self, event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Process a local write: stamp with VectorClock, apply, replicate."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self._writes += 1

        # Tick vector clock
        if self._vclock is not None:
            vc_snapshot = self._vclock.send()
        else:
            vc_snapshot = {self.name: 1}

        timestamp = self.now.to_seconds()

        versioned = VersionedValue(
            value=value,
            timestamp=timestamp,
            writer_id=self.name,
            vector_clock=vc_snapshot,
        )

        # Apply locally
        yield from self._store.put(key, value)
        self._versions[key] = versioned
        self._merkle.update(key, value)

        # Replicate to all peers
        events = []
        for peer in self._peers:
            events.append(self._network.send(
                self, peer, "Replicate",
                payload={
                    "key": key,
                    "value": value,
                    "timestamp": timestamp,
                    "writer_id": self.name,
                    "vector_clock": vc_snapshot,
                },
            ))
            self._replications_sent += 1

        if events:
            yield 0.0, events

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "key": key})
        return None

    def _handle_read(
        self, event: Event,
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Serve a read from the local store."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self._reads += 1
        value = yield from self._store.get(key)

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "value": value})
        return None

    def _handle_replicate(
        self, event: Event,
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Process a replication message from a peer."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        timestamp = metadata.get("timestamp", 0.0)
        writer_id = metadata.get("writer_id", "unknown")
        remote_vc = metadata.get("vector_clock", {})

        self._replications_received += 1

        # Update local vector clock
        if self._vclock is not None and remote_vc:
            self._vclock.receive(remote_vc)

        incoming = VersionedValue(
            value=value,
            timestamp=timestamp,
            writer_id=writer_id,
            vector_clock=remote_vc,
        )

        existing = self._versions.get(key)

        if existing is None:
            # No local version — apply
            yield from self._store.put(key, value)
            self._versions[key] = incoming
            self._merkle.update(key, value)
        else:
            # Compare vector clocks
            existing_vc = existing.vector_clock or {}
            incoming_vc = incoming.vector_clock or {}

            if _vc_dominates(incoming_vc, existing_vc):
                # Incoming is newer — apply
                yield from self._store.put(key, value)
                self._versions[key] = incoming
                self._merkle.update(key, value)
            elif _vc_dominates(existing_vc, incoming_vc):
                # Existing is newer — discard
                pass
            else:
                # Concurrent — conflict!
                self._conflicts_detected += 1
                winner = self._resolver.resolve(key, [existing, incoming])
                self._conflicts_resolved += 1

                if winner is not existing:
                    yield from self._store.put(key, winner.value)
                    self._versions[key] = winner
                    self._merkle.update(key, winner.value)

        return None

    def _handle_anti_entropy(
        self, event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Periodic anti-entropy: pick a random peer and exchange data."""
        if not self._peers:
            return None

        self._anti_entropy_syncs += 1
        peer = random.choice(self._peers)

        # Send our root hash AND our versioned data so the peer can reconcile
        data_to_send: dict[str, dict] = {}
        for key, vv in self._versions.items():
            data_to_send[key] = {
                "value": vv.value,
                "timestamp": vv.timestamp,
                "writer_id": vv.writer_id,
                "vector_clock": vv.vector_clock,
            }

        ae_event = self._network.send(
            self, peer, "AntiEntropyRequest",
            payload={
                "root_hash": self._merkle.root_hash,
                "versions": data_to_send,
            },
        )

        # Schedule next anti-entropy round
        next_ae = Event(
            time=self.now.__class__.from_seconds(
                self.now.to_seconds() + self._anti_entropy_interval
            ),
            event_type="AntiEntropy",
            target=self,
            daemon=True,
        )

        yield 0.0, [ae_event, next_ae]
        return None

    def _handle_anti_entropy_request(
        self, event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Respond to an anti-entropy request: reconcile their data, send ours."""
        metadata = event.context.get("metadata", {})
        remote_hash = metadata.get("root_hash", "")
        remote_versions: dict[str, dict] = metadata.get("versions", {})
        source_name = metadata.get("source", "")

        # Apply their data locally (reconcile)
        for key, vdata in remote_versions.items():
            remote_vv = VersionedValue(
                value=vdata["value"],
                timestamp=vdata["timestamp"],
                writer_id=vdata["writer_id"],
                vector_clock=vdata.get("vector_clock"),
            )
            existing = self._versions.get(key)
            if existing is None:
                yield from self._store.put(key, remote_vv.value)
                self._versions[key] = remote_vv
                self._merkle.update(key, remote_vv.value)
                self._anti_entropy_keys_repaired += 1
            else:
                existing_vc = existing.vector_clock or {}
                remote_vc = remote_vv.vector_clock or {}
                if _vc_dominates(remote_vc, existing_vc):
                    yield from self._store.put(key, remote_vv.value)
                    self._versions[key] = remote_vv
                    self._merkle.update(key, remote_vv.value)
                    self._anti_entropy_keys_repaired += 1
                elif not _vc_dominates(existing_vc, remote_vc):
                    self._conflicts_detected += 1
                    winner = self._resolver.resolve(key, [existing, remote_vv])
                    self._conflicts_resolved += 1
                    if winner is not existing:
                        yield from self._store.put(key, winner.value)
                        self._versions[key] = winner
                        self._merkle.update(key, winner.value)
                        self._anti_entropy_keys_repaired += 1

        if remote_hash == self._merkle.root_hash:
            # After reconciliation, if hashes match, no need to respond
            return None

        # Find the requester peer
        requester = None
        for p in self._peers:
            if p.name == source_name:
                requester = p
                break

        if requester is None:
            return None

        # Send our versioned data back for the requester to reconcile
        data_to_send: dict[str, dict] = {}
        for key, vv in self._versions.items():
            data_to_send[key] = {
                "value": vv.value,
                "timestamp": vv.timestamp,
                "writer_id": vv.writer_id,
                "vector_clock": vv.vector_clock,
            }

        resp = self._network.send(
            self, requester, "AntiEntropyResponse",
            payload={"versions": data_to_send},
        )
        yield 0.0, [resp]
        return None

    def _handle_anti_entropy_response(
        self, event: Event,
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Reconcile data from a peer's anti-entropy response."""
        metadata = event.context.get("metadata", {})
        remote_versions: dict[str, dict] = metadata.get("versions", {})

        for key, vdata in remote_versions.items():
            remote_vv = VersionedValue(
                value=vdata["value"],
                timestamp=vdata["timestamp"],
                writer_id=vdata["writer_id"],
                vector_clock=vdata.get("vector_clock"),
            )

            existing = self._versions.get(key)
            if existing is None:
                # New key — apply
                yield from self._store.put(key, remote_vv.value)
                self._versions[key] = remote_vv
                self._merkle.update(key, remote_vv.value)
                self._anti_entropy_keys_repaired += 1
            else:
                existing_vc = existing.vector_clock or {}
                remote_vc = remote_vv.vector_clock or {}

                if _vc_dominates(remote_vc, existing_vc):
                    yield from self._store.put(key, remote_vv.value)
                    self._versions[key] = remote_vv
                    self._merkle.update(key, remote_vv.value)
                    self._anti_entropy_keys_repaired += 1
                elif not _vc_dominates(existing_vc, remote_vc):
                    # Concurrent — resolve
                    self._conflicts_detected += 1
                    winner = self._resolver.resolve(key, [existing, remote_vv])
                    self._conflicts_resolved += 1
                    if winner is not existing:
                        yield from self._store.put(key, winner.value)
                        self._versions[key] = winner
                        self._merkle.update(key, winner.value)
                        self._anti_entropy_keys_repaired += 1

        return None


def _vc_dominates(a: dict[str, int], b: dict[str, int]) -> bool:
    """Check if vector clock ``a`` causally dominates ``b``."""
    all_keys = set(a) | set(b)
    all_geq = True
    any_gt = False
    for k in all_keys:
        val_a = a.get(k, 0)
        val_b = b.get(k, 0)
        if val_a < val_b:
            all_geq = False
            break
        if val_a > val_b:
            any_gt = True
    return all_geq and any_gt
