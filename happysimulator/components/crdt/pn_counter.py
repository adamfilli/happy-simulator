"""Positive-Negative counter (PN-Counter) CRDT.

A PN-Counter supports both increment and decrement by combining two
G-Counters: one for increments (P) and one for decrements (N). The
value is ``P.value - N.value``.

Example::

    c = PNCounter("node-a")
    c.increment(10)
    c.decrement(3)
    assert c.value == 7
"""

from __future__ import annotations

from typing import Self

from happysimulator.components.crdt.g_counter import GCounter


class PNCounter:
    """Positive-Negative counter CRDT.

    Wraps two G-Counters: ``_p`` for positive increments and ``_n``
    for negative decrements. The value is ``_p.value - _n.value``.

    Args:
        node_id: Identifier for this replica.
    """

    __slots__ = ("_node_id", "_p", "_n")

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._p = GCounter(node_id)
        self._n = GCounter(node_id)

    @property
    def node_id(self) -> str:
        """This replica's identifier."""
        return self._node_id

    @property
    def value(self) -> int:
        """Net count (increments - decrements)."""
        return self._p.value - self._n.value

    @property
    def increments(self) -> int:
        """Total increments across all nodes."""
        return self._p.value

    @property
    def decrements(self) -> int:
        """Total decrements across all nodes."""
        return self._n.value

    def increment(self, n: int = 1) -> None:
        """Increment the counter.

        Args:
            n: Amount to increment (must be positive).
        """
        self._p.increment(n)

    def decrement(self, n: int = 1) -> None:
        """Decrement the counter.

        Args:
            n: Amount to decrement (must be positive).
        """
        self._n.increment(n)

    def merge(self, other: PNCounter) -> None:
        """Merge another PN-Counter into this one.

        Merges both the P and N G-Counters independently.

        Args:
            other: Another PNCounter to merge from.
        """
        self._p.merge(other._p)
        self._n.merge(other._n)

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "type": "PNCounter",
            "node_id": self._node_id,
            "p": self._p.to_dict(),
            "n": self._n.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from a plain dict.

        Args:
            data: Dict produced by ``to_dict()``.
        """
        counter = cls(data["node_id"])
        counter._p = GCounter.from_dict(data["p"])
        counter._n = GCounter.from_dict(data["n"])
        return counter

    def __repr__(self) -> str:
        return f"PNCounter(node_id={self._node_id!r}, value={self.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PNCounter):
            return NotImplemented
        return self._p == other._p and self._n == other._n
