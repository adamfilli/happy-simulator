"""Logical clocks for distributed systems simulation.

Provides three classic logical clock algorithms used in distributed systems:

- **LamportClock**: Monotonic counter for total ordering (Lamport 1978).
- **VectorClock**: Per-node counters for causal ordering (Fidge/Mattern 1988).
- **HybridLogicalClock**: Physical + logical components (Kulkarni et al. 2014).

These are pure algorithm classes — not Entities. Entities store them as fields
and call their methods during event handling, exactly like ``NodeClock`` and
``FixedSkew``/``LinearDrift``.

Usage::

    from happysimulator import LamportClock, VectorClock, HybridLogicalClock

    # Lamport: simple monotonic counter
    clock = LamportClock()
    clock.tick()           # Local event
    ts = clock.send()      # Returns timestamp to embed in message
    clock.receive(ts)      # max(local, remote) + 1

    # Vector clock: per-node counters for causal ordering
    vc = VectorClock("node-1", ["node-1", "node-2", "node-3"])
    vc.tick()
    snapshot = vc.send()   # dict[str, int] to embed in message
    vc.receive(snapshot)   # Element-wise max + increment own

    # HLC: physical + logical (CockroachDB/Spanner-style)
    hlc = HybridLogicalClock("node-1", physical_clock=node_clock)
    ts = hlc.now()         # HLCTimestamp(physical_ns, logical, node_id)
    hlc.receive(remote_ts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from happysimulator.core.node_clock import NodeClock
from happysimulator.core.temporal import Instant


# =============================================================================
# Lamport Clock
# =============================================================================


class LamportClock:
    """Monotonic counter for establishing total ordering of events.

    Each local event increments the counter. When sending a message, the
    current counter is included. On receive, the counter advances to
    ``max(local, remote) + 1``, ensuring causal ordering.

    Args:
        initial: Starting counter value (default 0).
    """

    __slots__ = ("_time",)

    def __init__(self, initial: int = 0):
        self._time = initial

    @property
    def time(self) -> int:
        """Current counter value."""
        return self._time

    def tick(self) -> None:
        """Record a local event (increment counter)."""
        self._time += 1

    def send(self) -> int:
        """Increment counter and return timestamp to embed in a message."""
        self._time += 1
        return self._time

    def receive(self, remote_ts: int) -> None:
        """Update counter on receiving a message.

        Sets counter to ``max(local, remote) + 1``.

        Args:
            remote_ts: The Lamport timestamp from the received message.
        """
        self._time = max(self._time, remote_ts) + 1


# =============================================================================
# Vector Clock
# =============================================================================


class VectorClock:
    """Per-node counters for determining causal ordering between events.

    Each node maintains a vector of counters (one per node in the system).
    The ``happened_before`` relation enables detecting causally related vs.
    concurrent events — fundamental to conflict detection in replicated
    datastores (Dynamo, Riak) and consistency verification.

    Args:
        node_id: This node's identifier.
        node_ids: All node identifiers in the system (including this node).
    """

    __slots__ = ("_node_id", "_vector")

    def __init__(self, node_id: str, node_ids: list[str]):
        self._node_id = node_id
        self._vector: dict[str, int] = {nid: 0 for nid in node_ids}
        if node_id not in self._vector:
            self._vector[node_id] = 0

    @property
    def node_id(self) -> str:
        """This node's identifier."""
        return self._node_id

    def tick(self) -> None:
        """Record a local event (increment own counter)."""
        self._vector[self._node_id] += 1

    def send(self) -> dict[str, int]:
        """Increment own counter and return a snapshot for embedding in a message."""
        self._vector[self._node_id] += 1
        return dict(self._vector)

    def receive(self, remote: dict[str, int]) -> None:
        """Update vector on receiving a message.

        Takes the element-wise max of local and remote vectors, then
        increments own counter.

        Args:
            remote: The vector clock snapshot from the received message.
        """
        for nid, ts in remote.items():
            if nid in self._vector:
                self._vector[nid] = max(self._vector[nid], ts)
            else:
                self._vector[nid] = ts
        self._vector[self._node_id] += 1

    def snapshot(self) -> dict[str, int]:
        """Return a frozen copy of the current vector."""
        return dict(self._vector)

    def happened_before(self, other: VectorClock) -> bool:
        """Test if this clock causally precedes ``other``.

        Returns True if all components of this vector are <= the
        corresponding components of ``other``, and at least one is
        strictly less.

        Args:
            other: The vector clock to compare against.
        """
        all_keys = set(self._vector) | set(other._vector)
        all_leq = True
        any_lt = False
        for k in all_keys:
            local_val = self._vector.get(k, 0)
            other_val = other._vector.get(k, 0)
            if local_val > other_val:
                all_leq = False
                break
            if local_val < other_val:
                any_lt = True
        return all_leq and any_lt

    def is_concurrent(self, other: VectorClock) -> bool:
        """Test if this clock and ``other`` are concurrent (neither happened-before).

        Args:
            other: The vector clock to compare against.
        """
        return not self.happened_before(other) and not other.happened_before(self)

    def merge(self, other: VectorClock) -> VectorClock:
        """Create a new VectorClock with element-wise max (no increment).

        Unlike ``receive()``, this does NOT increment the local counter.
        Useful for read-only merges (e.g., computing a combined view).

        Args:
            other: The vector clock to merge with.

        Returns:
            A new VectorClock with the merged vector.
        """
        all_keys = sorted(set(self._vector) | set(other._vector))
        merged = VectorClock(self._node_id, all_keys)
        for k in all_keys:
            merged._vector[k] = max(
                self._vector.get(k, 0),
                other._vector.get(k, 0),
            )
        return merged


# =============================================================================
# HLC Timestamp
# =============================================================================


@dataclass(frozen=True, order=False)
class HLCTimestamp:
    """Immutable timestamp from a Hybrid Logical Clock.

    Total ordering is defined by ``(physical_ns, logical, node_id)``.

    Attributes:
        physical_ns: Nanosecond wall-clock component.
        logical: Logical counter within the same physical tick.
        node_id: Originating node identifier.
    """

    physical_ns: int
    logical: int
    node_id: str

    def __lt__(self, other: HLCTimestamp) -> bool:
        return (self.physical_ns, self.logical, self.node_id) < (
            other.physical_ns,
            other.logical,
            other.node_id,
        )

    def __le__(self, other: HLCTimestamp) -> bool:
        return (self.physical_ns, self.logical, self.node_id) <= (
            other.physical_ns,
            other.logical,
            other.node_id,
        )

    def __gt__(self, other: HLCTimestamp) -> bool:
        return (self.physical_ns, self.logical, self.node_id) > (
            other.physical_ns,
            other.logical,
            other.node_id,
        )

    def __ge__(self, other: HLCTimestamp) -> bool:
        return (self.physical_ns, self.logical, self.node_id) >= (
            other.physical_ns,
            other.logical,
            other.node_id,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HLCTimestamp):
            return NotImplemented
        return (self.physical_ns, self.logical, self.node_id) == (
            other.physical_ns,
            other.logical,
            other.node_id,
        )

    def __hash__(self) -> int:
        return hash((self.physical_ns, self.logical, self.node_id))

    def to_dict(self) -> dict:
        """Serialize to a plain dict for embedding in event contexts."""
        return {
            "physical_ns": self.physical_ns,
            "logical": self.logical,
            "node_id": self.node_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HLCTimestamp:
        """Deserialize from a plain dict.

        Args:
            d: Dict with keys ``physical_ns``, ``logical``, ``node_id``.
        """
        return cls(
            physical_ns=d["physical_ns"],
            logical=d["logical"],
            node_id=d["node_id"],
        )


# =============================================================================
# Hybrid Logical Clock
# =============================================================================


class HybridLogicalClock:
    """Hybrid Logical Clock combining physical and logical components.

    HLC provides the best of both worlds: timestamps that respect causality
    (like Lamport clocks) while staying close to physical time (enabling
    bounded-staleness reads). Used in CockroachDB, Spanner, and similar
    systems.

    The physical component comes from either a ``NodeClock`` (for skew
    modeling) or a plain callable returning an ``Instant``.

    Algorithm (Kulkarni et al. 2014):

    - **now/send**: Read physical time ``pt``. If ``pt > last.physical_ns``,
      reset logical to 0. Otherwise keep physical from last and increment
      logical. Return ``(pt, logical, node_id)``.
    - **receive**: Take max of physical time, last physical, and remote
      physical. Adjust logical based on which components tied.

    Args:
        node_id: This node's identifier.
        physical_clock: A ``NodeClock`` to read perceived time from.
            Mutually exclusive with ``wall_time``.
        wall_time: A callable returning the current ``Instant``.
            Mutually exclusive with ``physical_clock``.

    Raises:
        ValueError: If both or neither time source is provided.
    """

    __slots__ = ("_node_id", "_get_physical_ns", "_last")

    def __init__(
        self,
        node_id: str,
        physical_clock: NodeClock | None = None,
        wall_time: Callable[[], Instant] | None = None,
    ):
        if physical_clock is not None and wall_time is not None:
            raise ValueError(
                "Provide either physical_clock or wall_time, not both."
            )
        if physical_clock is None and wall_time is None:
            raise ValueError(
                "Provide either physical_clock or wall_time."
            )

        self._node_id = node_id

        if physical_clock is not None:
            self._get_physical_ns: Callable[[], int] = (
                lambda: physical_clock.now.nanoseconds
            )
        else:
            assert wall_time is not None
            self._get_physical_ns = lambda: wall_time().nanoseconds

        self._last = HLCTimestamp(physical_ns=0, logical=0, node_id=node_id)

    @property
    def node_id(self) -> str:
        """This node's identifier."""
        return self._node_id

    def now(self) -> HLCTimestamp:
        """Generate a timestamp for the current instant.

        Reads the physical clock and advances the logical component
        if the physical time hasn't changed since the last call.
        """
        pt = self._get_physical_ns()
        if pt > self._last.physical_ns:
            self._last = HLCTimestamp(
                physical_ns=pt, logical=0, node_id=self._node_id
            )
        else:
            self._last = HLCTimestamp(
                physical_ns=self._last.physical_ns,
                logical=self._last.logical + 1,
                node_id=self._node_id,
            )
        return self._last

    def send(self) -> HLCTimestamp:
        """Generate a timestamp to embed in an outgoing message.

        Equivalent to ``now()`` — included for API symmetry with
        ``LamportClock`` and ``VectorClock``.
        """
        return self.now()

    def receive(self, remote: HLCTimestamp) -> None:
        """Update clock state on receiving a remote timestamp.

        Implements the HLC receive algorithm: takes the max of physical
        time, last local physical, and remote physical, then adjusts the
        logical component based on which values tied.

        Args:
            remote: The HLC timestamp from the received message.
        """
        pt = self._get_physical_ns()
        max_pt = max(pt, self._last.physical_ns, remote.physical_ns)

        if max_pt == self._last.physical_ns == remote.physical_ns:
            # All three tied — advance logical past both
            logical = max(self._last.logical, remote.logical) + 1
        elif max_pt == self._last.physical_ns:
            # Local physical wins or ties with pt (but not remote)
            logical = self._last.logical + 1
        elif max_pt == remote.physical_ns:
            # Remote physical wins (but not local)
            logical = remote.logical + 1
        else:
            # Physical time advanced past both — reset logical
            logical = 0

        self._last = HLCTimestamp(
            physical_ns=max_pt, logical=logical, node_id=self._node_id
        )
