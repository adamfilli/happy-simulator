"""Merkle tree for anti-entropy synchronization.

A hash tree over sorted key-value data that enables efficient detection of
divergent key ranges between two replicas. Two nodes can compare root hashes
to detect any difference, then walk down the tree to find exactly which key
ranges have diverged — transferring only O(log N) hashes instead of the
full dataset.

Used in Dynamo, Cassandra, and Riak for replica synchronization.

Example::

    from happysimulator.sketching import MerkleTree

    # Build trees on two replicas
    tree_a = MerkleTree.build({"x": 1, "y": 2, "z": 3})
    tree_b = MerkleTree.build({"x": 1, "y": 999, "z": 3})

    # Compare root hashes — O(1) to detect any difference
    assert tree_a.root_hash != tree_b.root_hash

    # Find divergent ranges — O(log N) hashes exchanged
    diffs = tree_a.diff(tree_b)
    # diffs contains KeyRange covering "y"
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class KeyRange:
    """An inclusive range of keys [start, end].

    Attributes:
        start: First key in the range (inclusive).
        end: Last key in the range (inclusive).
    """

    start: str
    end: str

    def contains(self, key: str) -> bool:
        """Check if a key falls within this range."""
        return self.start <= key <= self.end

    def __repr__(self) -> str:
        return f"KeyRange({self.start!r}, {self.end!r})"


@dataclass(frozen=True)
class MerkleNode:
    """A node in the Merkle tree.

    Leaf nodes cover a single key; internal nodes cover the union of
    their children's key ranges.

    Attributes:
        hash: SHA-256 hex digest of the subtree contents.
        key_range: The range of keys covered by this subtree.
        left: Left child (None for leaf nodes).
        right: Right child (None for leaf nodes).
    """

    hash: str
    key_range: KeyRange
    left: MerkleNode | None = None
    right: MerkleNode | None = None

    @property
    def is_leaf(self) -> bool:
        """True if this is a leaf node."""
        return self.left is None and self.right is None


def _hash_leaf(key: str, value: Any) -> str:
    """Hash a single key-value pair."""
    data = f"{key}:{value!r}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _hash_children(left_hash: str, right_hash: str) -> str:
    """Hash two child hashes together."""
    data = f"{left_hash}|{right_hash}".encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _build_tree(sorted_items: list[tuple[str, Any]]) -> MerkleNode:
    """Recursively build a balanced binary Merkle tree from sorted items."""
    if len(sorted_items) == 1:
        key, value = sorted_items[0]
        return MerkleNode(
            hash=_hash_leaf(key, value),
            key_range=KeyRange(start=key, end=key),
        )

    mid = len(sorted_items) // 2
    left = _build_tree(sorted_items[:mid])
    right = _build_tree(sorted_items[mid:])

    return MerkleNode(
        hash=_hash_children(left.hash, right.hash),
        key_range=KeyRange(start=left.key_range.start, end=right.key_range.end),
        left=left,
        right=right,
    )


class MerkleTree:
    """Hash tree for detecting divergent key ranges between replicas.

    Binary tree over sorted keys where each internal node's hash is derived
    from its children's hashes. Comparing two trees top-down lets you skip
    matching subtrees and only transfer data for divergent ranges.

    Attributes:
        root_hash: SHA-256 hex digest of the root node (empty string if empty).
        size: Number of key-value pairs in the tree.
    """

    def __init__(self) -> None:
        self._root: MerkleNode | None = None
        self._data: dict[str, Any] = {}

    @classmethod
    def build(cls, data: dict[str, Any]) -> MerkleTree:
        """Build a Merkle tree from a dictionary of key-value pairs.

        Args:
            data: Key-value pairs to build the tree from. Keys are sorted
                lexicographically to produce a deterministic tree structure.

        Returns:
            A new MerkleTree covering all provided keys.
        """
        tree = cls()
        if data:
            tree._data = dict(data)
            sorted_items = sorted(data.items(), key=lambda kv: kv[0])
            tree._root = _build_tree(sorted_items)
        return tree

    @property
    def root_hash(self) -> str:
        """SHA-256 hex digest of the root node, or empty string if empty."""
        if self._root is None:
            return ""
        return self._root.hash

    @property
    def root(self) -> MerkleNode | None:
        """The root node of the tree, or None if empty."""
        return self._root

    @property
    def size(self) -> int:
        """Number of key-value pairs in the tree."""
        return len(self._data)

    def update(self, key: str, value: Any) -> None:
        """Update a single key and rebuild the tree.

        Args:
            key: The key to update or insert.
            value: The new value.
        """
        self._data[key] = value
        if self._data:
            sorted_items = sorted(self._data.items(), key=lambda kv: kv[0])
            self._root = _build_tree(sorted_items)
        else:
            self._root = None

    def remove(self, key: str) -> bool:
        """Remove a key and rebuild the tree.

        Args:
            key: The key to remove.

        Returns:
            True if the key existed, False otherwise.
        """
        if key not in self._data:
            return False
        del self._data[key]
        if self._data:
            sorted_items = sorted(self._data.items(), key=lambda kv: kv[0])
            self._root = _build_tree(sorted_items)
        else:
            self._root = None
        return True

    def get(self, key: str) -> Any | None:
        """Get a value by key (for convenience during anti-entropy sync).

        Args:
            key: The key to look up.

        Returns:
            The value, or None if not present.
        """
        return self._data.get(key)

    def keys(self) -> list[str]:
        """Return all keys in sorted order."""
        return sorted(self._data.keys())

    def items(self) -> list[tuple[str, Any]]:
        """Return all key-value pairs in sorted order."""
        return sorted(self._data.items(), key=lambda kv: kv[0])

    def diff(self, other: MerkleTree) -> list[KeyRange]:
        """Find key ranges that differ between this tree and another.

        Walks both trees top-down, skipping subtrees whose hashes match.
        Returns the key ranges of leaf-level divergences.

        Args:
            other: The other MerkleTree to compare against.

        Returns:
            List of KeyRange objects covering keys that differ.
            Empty list if the trees are identical.
        """
        if self._root is None and other._root is None:
            return []
        if self._root is None:
            return [other._root.key_range]
        if other._root is None:
            return [self._root.key_range]
        if self._root.hash == other._root.hash:
            return []

        return _diff_nodes(self._root, other._root)

    def __repr__(self) -> str:
        h = self.root_hash[:8] if self.root_hash else "empty"
        return f"MerkleTree(size={self.size}, root_hash={h}...)"


def _diff_nodes(a: MerkleNode, b: MerkleNode) -> list[KeyRange]:
    """Recursively find divergent key ranges between two nodes."""
    # If hashes match, subtrees are identical
    if a.hash == b.hash:
        return []

    # If either is a leaf, the whole range differs
    if a.is_leaf or b.is_leaf:
        # Return the union of both ranges
        start = min(a.key_range.start, b.key_range.start)
        end = max(a.key_range.end, b.key_range.end)
        return [KeyRange(start=start, end=end)]

    # Both are internal nodes — recurse into children
    result: list[KeyRange] = []

    assert a.left is not None and a.right is not None
    assert b.left is not None and b.right is not None

    result.extend(_diff_nodes(a.left, b.left))
    result.extend(_diff_nodes(a.right, b.right))

    return result
