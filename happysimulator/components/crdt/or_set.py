"""Observed-Remove Set (OR-Set) CRDT.

An OR-Set supports both add and remove operations with add-wins
semantics on concurrent operations. Each addition generates a unique
tag ``(node_id, sequence_number)``. Remove deletes all *observed* tags
for an element. If a concurrent add creates a new tag, it survives
the remove (add-wins).

Example::

    a = ORSet("node-a")
    a.add("apple")
    a.add("banana")
    a.remove("apple")
    assert a.elements == frozenset({"banana"})
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import Iterator


class ORSet:
    """Observed-Remove Set CRDT.

    Maintains a dict mapping elements to sets of tags. Each tag is a
    ``(node_id, sequence_number)`` tuple, generated deterministically
    (no UUIDs) for reproducible tests.

    Add-wins semantics: a concurrent add and remove of the same
    element results in the element being present (the new tag from
    the add survives the remove of old tags).

    Args:
        node_id: Identifier for this replica.
    """

    __slots__ = ("_entries", "_node_id", "_seq")

    def __init__(self, node_id: str):
        self._node_id = node_id
        self._entries: dict[Any, set[tuple[str, int]]] = {}
        self._seq: int = 0

    @property
    def node_id(self) -> str:
        """This replica's identifier."""
        return self._node_id

    @property
    def value(self) -> frozenset:
        """Current elements in the set (alias for ``elements``)."""
        return self.elements

    @property
    def elements(self) -> frozenset:
        """Frozenset of elements that have at least one tag."""
        return frozenset(e for e, tags in self._entries.items() if tags)

    def add(self, element: Any) -> None:
        """Add an element with a new unique tag.

        Args:
            element: The element to add.
        """
        tag = (self._node_id, self._seq)
        self._seq += 1
        if element not in self._entries:
            self._entries[element] = set()
        self._entries[element].add(tag)

    def remove(self, element: Any) -> None:
        """Remove an element by clearing all its observed tags.

        If the element is not present, this is a no-op.

        Args:
            element: The element to remove.
        """
        if element in self._entries:
            self._entries[element].clear()

    def contains(self, element: Any) -> bool:
        """Check if an element is in the set.

        Args:
            element: The element to check.

        Returns:
            True if the element has at least one tag.
        """
        return bool(self._entries.get(element))

    def merge(self, other: ORSet) -> None:
        """Merge another OR-Set into this one.

        For each element, the resulting tag set is the union of both
        replicas' tags. This means:
        - Elements added on either side are present.
        - An element removed on one side but concurrently added on
          the other survives (add-wins).

        Args:
            other: Another ORSet to merge from.
        """
        for element, other_tags in other._entries.items():
            if element not in self._entries:
                self._entries[element] = set(other_tags)
            else:
                self._entries[element] |= other_tags

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        entries = {}
        for element, tags in self._entries.items():
            entries[str(element)] = [list(tag) for tag in sorted(tags)]
        return {
            "type": "ORSet",
            "node_id": self._node_id,
            "seq": self._seq,
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Deserialize from a plain dict.

        Args:
            data: Dict produced by ``to_dict()``.
        """
        s = cls(data["node_id"])
        s._seq = data["seq"]
        for element, tags in data["entries"].items():
            s._entries[element] = {tuple(tag) for tag in tags}
        return s

    def __contains__(self, element: Any) -> bool:
        return self.contains(element)

    def __len__(self) -> int:
        return sum(1 for tags in self._entries.values() if tags)

    def __iter__(self) -> Iterator[Any]:
        return iter(e for e, tags in self._entries.items() if tags)

    def __repr__(self) -> str:
        return f"ORSet(node_id={self._node_id!r}, elements={self.elements!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ORSet):
            return NotImplemented
        # Compare only non-empty tag sets
        self_active = {e: tags for e, tags in self._entries.items() if tags}
        other_active = {e: tags for e, tags in other._entries.items() if tags}
        return self_active == other_active
