"""Chain replication protocol.

Models a chain of nodes where writes enter at the HEAD, propagate through
MIDDLE nodes, and commit at the TAIL. The TAIL sends a WriteAck back to
the HEAD, which resolves the client's future.

Reads are served by the TAIL (strongly consistent) or by any node in CRAQ
mode if the key is "clean" (fully committed).

Based on:
- van Renesse & Schneider, "Chain Replication for Supporting High
  Throughput and Availability" (OSDI 2004)
- Terrace & Freedman, "Object Storage on CRAQ" (USENIX ATC 2009)

Example::

    from happysimulator.components.replication import build_chain, ChainNodeRole

    nodes = build_chain(
        names=["head", "mid", "tail"],
        network=network,
        store_factory=lambda name: KVStore(name),
    )
    # nodes[0] is HEAD, nodes[-1] is TAIL
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture
from happysimulator.components.datastore.kv_store import KVStore

logger = logging.getLogger(__name__)


class ChainNodeRole(Enum):
    """Role of a node in the replication chain."""

    HEAD = "head"
    MIDDLE = "middle"
    TAIL = "tail"


@dataclass(frozen=True)
class ChainReplicationStats:
    """Statistics for a ChainNode.

    Attributes:
        writes_received: Write events received (HEAD only).
        propagations_sent: Propagate events forwarded to next node.
        propagations_received: Propagate events received from prev node.
        acks_sent: WriteAck events sent (TAIL only).
        reads_served: Read events served.
    """

    writes_received: int = 0
    propagations_sent: int = 0
    propagations_received: int = 0
    acks_sent: int = 0
    reads_served: int = 0


class ChainNode(Entity):
    """A node in a chain replication topology.

    Depending on its role:
    - HEAD: Accepts Write events, applies locally, sends Propagate to next.
      Parks on a SimFuture until TAIL sends WriteAck.
    - MIDDLE: Receives Propagate, applies, forwards to next.
    - TAIL: Receives Propagate, applies, sends WriteAck back to HEAD.

    With CRAQ enabled, any node can serve Read events for "clean" keys
    (keys that have been fully committed through the chain). Dirty keys
    are forwarded to the TAIL.

    Args:
        name: Entity name.
        store: KVStore for local data.
        network: Network for inter-node communication.
        role: This node's role in the chain.
        craq_enabled: Enable CRAQ read optimization.
    """

    def __init__(
        self,
        name: str,
        store: KVStore,
        network: Entity,
        role: ChainNodeRole = ChainNodeRole.MIDDLE,
        craq_enabled: bool = False,
    ):
        super().__init__(name)
        self._store = store
        self._network = network
        self._role = role
        self._craq_enabled = craq_enabled

        # Chain topology (set by build_chain or manually)
        self.next_node: ChainNode | None = None
        self.prev_node: ChainNode | None = None
        self.head_node: ChainNode | None = None

        # CRAQ: track keys with uncommitted writes
        self._dirty_keys: set[str] = set()

        # Pending write futures (HEAD: seq -> SimFuture)
        self._pending_writes: dict[int, SimFuture] = {}
        self._next_seq: int = 0

        self._writes_received = 0
        self._propagations_sent = 0
        self._propagations_received = 0
        self._acks_sent = 0
        self._reads_served = 0

    @property
    def stats(self) -> ChainReplicationStats:
        """Frozen snapshot of chain node statistics."""
        return ChainReplicationStats(
            writes_received=self._writes_received,
            propagations_sent=self._propagations_sent,
            propagations_received=self._propagations_received,
            acks_sent=self._acks_sent,
            reads_served=self._reads_served,
        )

    @property
    def role(self) -> ChainNodeRole:
        """This node's role in the chain."""
        return self._role

    @property
    def store(self) -> KVStore:
        """The underlying KVStore."""
        return self._store

    @property
    def dirty_keys(self) -> set[str]:
        """Keys with uncommitted writes (CRAQ)."""
        return set(self._dirty_keys)

    def handle_event(
        self, event: Event,
    ) -> Generator[float | SimFuture | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Route events by type."""
        if event.event_type == "Write":
            return (yield from self._handle_write(event))
        elif event.event_type == "Propagate":
            return (yield from self._handle_propagate(event))
        elif event.event_type == "WriteAck":
            self._handle_write_ack(event)
            return None
        elif event.event_type == "Read":
            return (yield from self._handle_read(event))
        elif event.event_type == "CommitNotify":
            self._handle_commit_notify(event)
            return None
        return None

    def _handle_write(
        self, event: Event,
    ) -> Generator[float | SimFuture | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """HEAD: accept write, apply locally, propagate to next."""
        if self._role != ChainNodeRole.HEAD:
            logger.warning("[%s] Write received by non-HEAD node", self.name)
            metadata = event.context.get("metadata", {})
            reply_future: SimFuture | None = metadata.get("reply_future")
            if reply_future is not None:
                reply_future.resolve({"status": "error", "reason": "not_head"})
            return None

        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        reply_future: SimFuture | None = metadata.get("reply_future")

        self._writes_received += 1
        self._next_seq += 1
        seq = self._next_seq

        # Apply locally
        yield from self._store.put(key, value)

        # Mark dirty for CRAQ
        if self._craq_enabled:
            self._dirty_keys.add(key)

        if self.next_node is not None:
            # Create ack future
            ack_future = SimFuture()
            self._pending_writes[seq] = ack_future

            # Propagate to next
            prop_event = self._network.send(
                self, self.next_node, "Propagate",
                payload={"key": key, "value": value, "seq": seq},
            )
            self._propagations_sent += 1
            yield 0.0, [prop_event]

            # Wait for ack from tail
            yield ack_future

            # Clean up
            self._pending_writes.pop(seq, None)
            if self._craq_enabled:
                self._dirty_keys.discard(key)
        else:
            # Single-node chain (HEAD is also TAIL)
            if self._craq_enabled:
                self._dirty_keys.discard(key)

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "seq": seq})
        return None

    def _handle_propagate(
        self, event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """MIDDLE/TAIL: receive propagation, apply, forward or ack."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        value = metadata.get("value")
        seq = metadata.get("seq", 0)

        self._propagations_received += 1

        # Apply locally
        yield from self._store.put(key, value)

        if self._craq_enabled:
            self._dirty_keys.add(key)

        if self._role == ChainNodeRole.TAIL:
            # Send ack back to head
            head = self.head_node or self.prev_node
            if head is not None:
                ack_event = self._network.send(
                    self, head, "WriteAck",
                    payload={"key": key, "seq": seq},
                )
                self._acks_sent += 1
                yield 0.0, [ack_event]

            # CRAQ: key is now clean, notify chain
            if self._craq_enabled:
                self._dirty_keys.discard(key)
                # Notify upstream nodes that key is committed
                events = self._build_commit_notifications(key, seq)
                if events:
                    yield 0.0, events

        elif self.next_node is not None:
            # Forward to next
            prop_event = self._network.send(
                self, self.next_node, "Propagate",
                payload={"key": key, "value": value, "seq": seq},
            )
            self._propagations_sent += 1
            yield 0.0, [prop_event]

        return None

    def _handle_write_ack(self, event: Event) -> None:
        """HEAD: receive ack from tail, resolve pending future."""
        metadata = event.context.get("metadata", {})
        seq = metadata.get("seq", 0)

        future = self._pending_writes.get(seq)
        if future is not None:
            future.resolve({"status": "ok", "seq": seq})

    def _handle_commit_notify(self, event: Event) -> None:
        """CRAQ: mark key as clean (committed)."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        if key and self._craq_enabled:
            self._dirty_keys.discard(key)

    def _handle_read(
        self, event: Event,
    ) -> Generator[float | tuple[float, list[Event] | Event], None, list[Event] | Event | None]:
        """Serve a read request."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")
        reply_future: SimFuture | None = metadata.get("reply_future")

        # CRAQ: if not tail and key is dirty, forward to tail
        if (
            self._craq_enabled
            and self._role != ChainNodeRole.TAIL
            and key in self._dirty_keys
        ):
            # Forward read to tail
            if self.head_node is not None:
                # Find tail (last in chain)
                tail = self._find_tail()
                if tail is not None and tail is not self:
                    fwd_event = self._network.send(
                        self, tail, "Read",
                        payload={"key": key, "reply_future": reply_future},
                    )
                    yield 0.0, [fwd_event]
                    return None

        # Serve locally
        self._reads_served += 1
        value = yield from self._store.get(key)

        if reply_future is not None:
            reply_future.resolve({"status": "ok", "value": value})
        return None

    def _find_tail(self) -> ChainNode | None:
        """Walk the chain to find the tail node."""
        node = self
        while node.next_node is not None:
            node = node.next_node
        return node if node._role == ChainNodeRole.TAIL else None

    def _build_commit_notifications(self, key: str, seq: int) -> list[Event]:
        """Build CommitNotify events for upstream nodes."""
        events = []
        node = self.prev_node
        while node is not None:
            events.append(self._network.send(
                self, node, "CommitNotify",
                payload={"key": key, "seq": seq},
            ))
            node = node.prev_node
        return events


def build_chain(
    names: list[str],
    network: Entity,
    store_factory: Callable[[str], KVStore],
    craq_enabled: bool = False,
) -> list[ChainNode]:
    """Build a chain of ChainNodes with topology wired.

    Args:
        names: Names for each node. First is HEAD, last is TAIL.
        network: Network entity for inter-node communication.
        store_factory: Factory function creating a KVStore for each node.
        craq_enabled: Enable CRAQ read optimization on all nodes.

    Returns:
        List of ChainNode instances with topology wired.

    Raises:
        ValueError: If fewer than 2 names are provided.
    """
    if len(names) < 2:
        raise ValueError("Chain requires at least 2 nodes")

    nodes: list[ChainNode] = []
    for i, name in enumerate(names):
        if i == 0:
            role = ChainNodeRole.HEAD
        elif i == len(names) - 1:
            role = ChainNodeRole.TAIL
        else:
            role = ChainNodeRole.MIDDLE

        node = ChainNode(
            name=name,
            store=store_factory(f"{name}_store"),
            network=network,
            role=role,
            craq_enabled=craq_enabled,
        )
        nodes.append(node)

    # Wire topology
    head = nodes[0]
    for i, node in enumerate(nodes):
        node.head_node = head
        if i > 0:
            node.prev_node = nodes[i - 1]
        if i < len(nodes) - 1:
            node.next_node = nodes[i + 1]

    return nodes
