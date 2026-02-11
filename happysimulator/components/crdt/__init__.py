"""Conflict-free Replicated Data Types (CRDTs) for distributed simulation.

CRDTs are data structures that converge automatically after network
partitions heal, without requiring consensus. They guarantee eventual
consistency by ensuring merge operations are commutative, associative,
and idempotent.

Provided CRDTs:

- **GCounter**: Grow-only counter (increment only)
- **PNCounter**: Positive-negative counter (increment and decrement)
- **LWWRegister**: Last-writer-wins register (HLC timestamp ordering)
- **ORSet**: Observed-remove set (add-wins semantics)

The **CRDTStore** entity wraps CRDTs in a key-value store with
gossip-based replication for use in simulations.
"""

from happysimulator.components.crdt.protocol import CRDT
from happysimulator.components.crdt.g_counter import GCounter
from happysimulator.components.crdt.pn_counter import PNCounter
from happysimulator.components.crdt.lww_register import LWWRegister
from happysimulator.components.crdt.or_set import ORSet
from happysimulator.components.crdt.crdt_store import CRDTStore, CRDTStoreStats

__all__ = [
    "CRDT",
    "GCounter",
    "PNCounter",
    "LWWRegister",
    "ORSet",
    "CRDTStore",
    "CRDTStoreStats",
]
