"""Grow-only counter (G-Counter) CRDT.

A G-Counter is a replicated counter that can only be incremented.
Each node maintains its own count, and the total value is the sum
of all node counts. Merge takes the element-wise maximum.

This is the foundational CRDT — PNCounter builds on two G-Counters.

Example::

    a = GCounter("node-a")
    b = GCounter("node-b")

    a.increment(5)
    b.increment(3)

    a.merge(b)
    assert a.value == 8  # 5 + 3
"""

from __future__ import annotations

from typing import Self


class GCounter:
    """Grow-only counter CRDT.

    Each node has its own monotonically increasing count. The total
    value is the sum across all nodes. Merge uses element-wise max,
    which is commutative, associative, and idempotent.

    Args:
        node_id: Identifier for this replica.
    """

    __slots__ = ("_node_id", "_counts")

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._counts: dict[str, int] = {}

    @property
    def node_id(self) -> str:
        """This replica's identifier."""
        return self._node_id

    @property
    def value(self) -> int:
        """Total count across all nodes."""
        return sum(self._counts.values())

    def increment(self, n: int = 1) -> None:
        """Increment this node's count.

        Args:
            n: Amount to increment (must be positive).

        Raises:
            ValueError: If n is not positive.
        """
        if n < 1:
            raise ValueError(f"Increment must be positive, got {n}")
        self._counts[self._node_id] = self._counts.get(self._node_id, 0) + n

    def node_value(self, node_id: str) -> int:
        """Get a specific node's count.

        Args:
            node_id: The node to query.

        Returns:
            The node's count, or 0 if unknown.
        """
        return self._counts.get(node_id, 0)

    def merge(self, other: GCounter) -> None:
        """Merge another G-Counter into this one (element-wise max).

        Args:
            other: Another GCounter to merge from.
        """
        for node_id, count in other._counts.items():
            self._counts[node_id] = max(self._counts.get(node_id, 0), count)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "type": "GCounter",
            "node_id": self._node_id,
            "counts": dict(self._counts),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from a plain dict.

        Args:
            data: Dict produced by ``to_dict()``.
        """
        counter = cls(data["node_id"])
        counter._counts = dict(data["counts"])
        return counter

    def __repr__(self) -> str:
        return f"GCounter(node_id={self._node_id!r}, value={self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GCounter):
            return NotImplemented
        return self._counts == other._counts
