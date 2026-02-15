"""CRDT-backed key-value store with gossip-based replication.

CRDTStore is an Entity that maintains a local key-value store where
each key maps to a CRDT instance. Replication between nodes uses a
gossip protocol: periodically, each node pushes its full state to a
random peer, which merges and responds with its own state (push-pull).

Unlike consensus-based replication, CRDTs allow writes during network
partitions. Convergence is guaranteed once connectivity is restored
and gossip resumes.

Example::

    from happysimulator.components.crdt import CRDTStore, GCounter

    store_a = CRDTStore(
        "node-a", network=net, crdt_factory=lambda nid: GCounter(nid), gossip_interval=1.0
    )
    store_b = CRDTStore(
        "node-b", network=net, crdt_factory=lambda nid: GCounter(nid), gossip_interval=1.0
    )
    store_a.add_peers([store_b])
    store_b.add_peers([store_a])
"""

from __future__ import annotations

import hashlib
import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from happysimulator.components.crdt.lww_register import LWWRegister
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from happysimulator.components.crdt.protocol import CRDT
    from happysimulator.core.sim_future import SimFuture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CRDTStoreStats:
    """Statistics for a CRDTStore node.

    Attributes:
        writes: Local write operations processed.
        reads: Local read operations processed.
        gossip_sent: Gossip push messages sent.
        gossip_received: Gossip messages received (push or response).
        keys_merged: Total key-level merge operations performed.
        convergence_checks: Number of convergence hash comparisons.
    """

    writes: int = 0
    reads: int = 0
    gossip_sent: int = 0
    gossip_received: int = 0
    keys_merged: int = 0
    convergence_checks: int = 0


class CRDTStore(Entity):
    """Key-value store backed by CRDTs with gossip replication.

    Each key maps to a CRDT instance created by ``crdt_factory``.
    Writes apply directly to the local CRDT (no coordination).
    Gossip periodically synchronizes state with a random peer using
    full-state push-pull.

    Args:
        name: Entity name (also used as the node_id for CRDTs).
        network: Network entity for inter-node communication.
        crdt_factory: Callable that creates a new CRDT given a node_id.
            Defaults to creating LWWRegister instances.
        gossip_interval: Seconds between gossip rounds (0 to disable).
    """

    def __init__(
        self,
        name: str,
        network: Entity,
        crdt_factory: Callable[[str], CRDT] = lambda node_id: LWWRegister(node_id),
        gossip_interval: float = 1.0,
    ):
        super().__init__(name)
        self._network = network
        self._crdt_factory = crdt_factory
        self._gossip_interval = gossip_interval
        self._peers: list[Entity] = []
        self._crdts: dict[str, CRDT] = {}
        self._last_peer_hash: str = ""
        self._writes = 0
        self._reads = 0
        self._gossip_sent = 0
        self._gossip_received = 0
        self._keys_merged = 0
        self._convergence_checks = 0

    @property
    def stats(self) -> CRDTStoreStats:
        """Return a frozen snapshot of store statistics."""
        return CRDTStoreStats(
            writes=self._writes,
            reads=self._reads,
            gossip_sent=self._gossip_sent,
            gossip_received=self._gossip_received,
            keys_merged=self._keys_merged,
            convergence_checks=self._convergence_checks,
        )

    @property
    def crdts(self) -> dict[str, CRDT]:
        """The local CRDT instances keyed by name."""
        return dict(self._crdts)

    @property
    def convergence_lag(self) -> bool:
        """True if local state hash differs from last-seen peer hash.

        A rough indicator — not authoritative across all peers.
        """
        self._convergence_checks += 1
        return self._state_hash() != self._last_peer_hash

    def add_peers(self, peers: list[Entity]) -> None:
        """Set peer nodes for gossip replication.

        Args:
            peers: List of peer CRDTStore entities.
        """
        self._peers = list(peers)

    def get_gossip_event(self) -> Event | None:
        """Create the first gossip daemon event.

        Returns:
            A daemon event for the first gossip tick, or None if
            gossip is disabled or there are no peers.
        """
        if self._gossip_interval <= 0 or not self._peers:
            return None
        from happysimulator.core.temporal import Instant

        t = self.now if self.now.to_seconds() > 0 else Instant.from_seconds(self._gossip_interval)
        return Event(
            time=t,
            event_type="GossipTick",
            target=self,
            daemon=True,
        )

    def get_or_create(self, key: str) -> CRDT:
        """Get or create a CRDT for the given key.

        Args:
            key: The key to look up or create.

        Returns:
            The CRDT instance for this key.
        """
        if key not in self._crdts:
            self._crdts[key] = self._crdt_factory(self.name)
        return self._crdts[key]

    def handle_event(
        self,
        event: Event,
    ) -> Generator[
        float | SimFuture | tuple[float, list[Event] | Event], None, list[Event] | Event | None
    ]:
        """Route events by type."""
        if event.event_type == "Write":
            return (yield from self._handle_write(event))
        elif event.event_type == "Read":
            return (yield from self._handle_read(event))
        elif event.event_type == "GossipTick":
            return (yield from self._handle_gossip_tick(event))
        elif event.event_type == "GossipPush":
            return (yield from self._handle_gossip_push(event))
        elif event.event_type == "GossipResponse":
            return (yield from self._handle_gossip_response(event))
        return None

    def _handle_write(
        self,
        event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Apply a write to the local CRDT.

        Context metadata keys:
        - ``key``: The key to write to.
        - ``value``: The value (interpretation depends on CRDT type).
        - ``operation``: Operation name (e.g., "increment", "set", "add").
            Defaults to "set".
        - ``reply_future``: Optional SimFuture to resolve with result.
        """
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        operation = metadata.get("operation", "set")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self._writes += 1

        crdt = self.get_or_create(key)
        self._apply_operation(crdt, operation, value)

        yield 0.0  # minimal processing time

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "key": key, "value": crdt.value})
        return None

    def _handle_read(
        self,
        event: Event,
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Read a value from the local CRDT store.

        Context metadata keys:
        - ``key``: The key to read.
        - ``reply_future``: Optional SimFuture to resolve with result.
        """
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self._reads += 1

        crdt = self._crdts.get(key)
        value = crdt.value if crdt is not None else None

        yield 0.0

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "value": value})
        return None

    def _handle_gossip_tick(
        self,
        event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Periodic gossip: push full state to a random peer."""
        if not self._peers:
            return None

        peer = random.choice(self._peers)
        state = self._serialize_state()
        state_hash = self._state_hash()

        push_event = self._network.send(
            self,
            peer,
            "GossipPush",
            payload={
                "state": state,
                "state_hash": state_hash,
            },
        )
        self._gossip_sent += 1

        # Schedule next gossip tick
        from happysimulator.core.temporal import Instant

        next_tick = Event(
            time=Instant.from_seconds(self.now.to_seconds() + self._gossip_interval),
            event_type="GossipTick",
            target=self,
            daemon=True,
        )

        yield 0.0, [push_event, next_tick]
        return None

    def _handle_gossip_push(
        self,
        event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Receive a gossip push: merge remote state, respond with ours."""
        metadata = event.context.get("metadata", {})
        remote_state: dict = metadata.get("state", {})
        remote_hash: str = metadata.get("state_hash", "")
        source_name: str = metadata.get("source", "")

        self._gossip_received += 1
        self._last_peer_hash = remote_hash

        # Merge remote state into local
        self._merge_remote_state(remote_state)

        # Find the requester to send response
        requester = None
        for p in self._peers:
            if p.name == source_name:
                requester = p
                break

        if requester is None:
            return None

        # Respond with our state
        state = self._serialize_state()
        resp = self._network.send(
            self,
            requester,
            "GossipResponse",
            payload={
                "state": state,
                "state_hash": self._state_hash(),
            },
        )
        self._gossip_sent += 1

        yield 0.0, [resp]
        return None

    def _handle_gossip_response(
        self,
        event: Event,
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Receive a gossip response: merge remote state."""
        metadata = event.context.get("metadata", {})
        remote_state: dict = metadata.get("state", {})
        remote_hash: str = metadata.get("state_hash", "")

        self._gossip_received += 1
        self._last_peer_hash = remote_hash

        self._merge_remote_state(remote_state)

        yield 0.0
        return None

    def _apply_operation(self, crdt: CRDT, operation: str, value: Any) -> None:
        """Apply an operation to a CRDT based on its type.

        Dispatches to the appropriate method based on operation name.
        Falls back to common method names if the operation is generic.
        """
        if hasattr(crdt, operation):
            method = getattr(crdt, operation)
            if value is not None:
                method(value)
            else:
                method()
        elif operation == "set" and hasattr(crdt, "set"):
            crdt.set(value)
        else:
            logger.warning(
                "[%s] Unknown operation %r on CRDT type %s",
                self.name,
                operation,
                type(crdt).__name__,
            )

    def _serialize_state(self) -> dict:
        """Serialize all local CRDTs for gossip."""
        return {key: crdt.to_dict() for key, crdt in self._crdts.items()}

    def _state_hash(self) -> str:
        """Compute a hash of the current state for convergence checking."""
        # Deterministic hash: sort keys, hash serialized state
        parts = [f"{key}:{self._crdts[key].to_dict()}" for key in sorted(self._crdts.keys())]
        content = "|".join(parts)
        return hashlib.md5(content.encode()).hexdigest()

    def _merge_remote_state(self, remote_state: dict) -> None:
        """Merge serialized remote state into local CRDTs."""
        for key, remote_dict in remote_state.items():
            remote_dict.get("type", "")

            if key in self._crdts:
                # Merge into existing local CRDT
                local_crdt = self._crdts[key]
                remote_crdt = local_crdt.__class__.from_dict(remote_dict)
                local_crdt.merge(remote_crdt)
                self._keys_merged += 1
            else:
                # Create from remote state
                remote_crdt = self._reconstruct_crdt(remote_dict)
                if remote_crdt is not None:
                    self._crdts[key] = remote_crdt
                    self._keys_merged += 1

    def _reconstruct_crdt(self, data: dict) -> CRDT | None:
        """Reconstruct a CRDT from serialized data.

        Uses the ``type`` field to dispatch to the correct class.
        """
        from happysimulator.components.crdt.g_counter import GCounter
        from happysimulator.components.crdt.or_set import ORSet
        from happysimulator.components.crdt.pn_counter import PNCounter

        type_map: dict[str, type] = {
            "GCounter": GCounter,
            "PNCounter": PNCounter,
            "LWWRegister": LWWRegister,
            "ORSet": ORSet,
        }

        crdt_type = data.get("type", "")
        cls = type_map.get(crdt_type)
        if cls is None:
            logger.warning("[%s] Unknown CRDT type: %s", self.name, crdt_type)
            return None
        return cls.from_dict(data)
