"""Last-Writer-Wins Register (LWW-Register) CRDT.

A register that resolves concurrent writes by keeping the value with
the highest ``HLCTimestamp``. Ties are broken by ``node_id`` (part of
HLCTimestamp's total order).

Example::

    from happysimulator.core.logical_clocks import HLCTimestamp

    r = LWWRegister("node-a")
    r.set("hello", HLCTimestamp(physical_ns=1000, logical=0, node_id="node-a"))
    assert r.value == "hello"
"""

from __future__ import annotations

from typing import Any, Self

from happysimulator.core.logical_clocks import HLCTimestamp


class LWWRegister:
    """Last-Writer-Wins register CRDT.

    Stores a single value tagged with an ``HLCTimestamp``. On merge,
    the value with the higher timestamp wins. HLCTimestamp provides
    total ordering via ``(physical_ns, logical, node_id)``.

    Args:
        node_id: Identifier for this replica.
        value: Initial value (default None).
        timestamp: Initial timestamp (default None = never written).
    """

    __slots__ = ("_node_id", "_timestamp", "_value")

    def __init__(
        self,
        node_id: str,
        value: Any = None,
        timestamp: HLCTimestamp | None = None,
    ):
        self._node_id = node_id
        self._value = value
        self._timestamp = timestamp

    @property
    def node_id(self) -> str:
        """This replica's identifier."""
        return self._node_id

    @property
    def value(self) -> Any:
        """Current value of the register."""
        return self._value

    @property
    def timestamp(self) -> HLCTimestamp | None:
        """Timestamp of the current value."""
        return self._timestamp

    def get(self) -> Any:
        """Return the current value (alias for ``value`` property)."""
        return self._value

    def set(self, value: Any, timestamp: HLCTimestamp) -> None:
        """Set the value if the timestamp is newer than the current one.

        If the register has never been written (timestamp is None),
        the value is always accepted.

        Args:
            value: The new value.
            timestamp: The HLC timestamp for this write.
        """
        if self._timestamp is None or timestamp > self._timestamp:
            self._value = value
            self._timestamp = timestamp

    def merge(self, other: LWWRegister) -> None:
        """Merge another register into this one (highest timestamp wins).

        If the other register has never been written, this is a no-op.

        Args:
            other: Another LWWRegister to merge from.
        """
        if other._timestamp is None:
            return
        if self._timestamp is None or other._timestamp > self._timestamp:
            self._value = other._value
            self._timestamp = other._timestamp

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "type": "LWWRegister",
            "node_id": self._node_id,
            "value": self._value,
            "timestamp": self._timestamp.to_dict() if self._timestamp else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from a plain dict.

        Args:
            data: Dict produced by ``to_dict()``.
        """
        ts = HLCTimestamp.from_dict(data["timestamp"]) if data["timestamp"] else None
        return cls(
            node_id=data["node_id"],
            value=data["value"],
            timestamp=ts,
        )

    def __repr__(self) -> str:
        return f"LWWRegister(node_id={self._node_id!r}, value={self._value!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LWWRegister):
            return NotImplemented
        return self._value == other._value and self._timestamp == other._timestamp
