"""Protocol definition for Conflict-free Replicated Data Types (CRDTs).

CRDTs are distributed data structures that converge automatically
after network partitions heal, without requiring consensus. Any CRDT
implementation must satisfy:

- **Commutativity**: ``merge(a, b) == merge(b, a)``
- **Associativity**: ``merge(a, merge(b, c)) == merge(merge(a, b), c)``
- **Idempotency**: ``merge(a, a) == a``

These properties guarantee that replicas converge to the same state
regardless of the order or number of merge operations.
"""

from __future__ import annotations

from typing import Any, Protocol, Self, runtime_checkable


@runtime_checkable
class CRDT(Protocol):
    """Protocol for all CRDT types.

    All CRDTs must support:
    - ``value``: Read the current resolved value.
    - ``merge(other)``: Merge another replica's state (in-place).
    - ``to_dict()`` / ``from_dict()``: Serialization for gossip.
    """

    @property
    def value(self) -> Any:
        """The current resolved value of this CRDT."""
        ...

    def merge(self, other: Self) -> None:
        """Merge another replica's state into this one (in-place).

        Must be commutative, associative, and idempotent.

        Args:
            other: Another instance of the same CRDT type.
        """
        ...

    def to_dict(self) -> dict:
        """Serialize this CRDT's state to a plain dict for gossip."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize a CRDT from a plain dict.

        Args:
            data: Dict produced by ``to_dict()``.
        """
        ...
