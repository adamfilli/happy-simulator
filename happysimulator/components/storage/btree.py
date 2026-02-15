"""B-tree index with page-level I/O simulation.

Models a B-tree index as used in traditional RDBMS storage engines. Each
node represents a disk page, so tree operations have I/O cost proportional
to tree depth. Supports the same get/put/delete/scan interface as LSMTree
for direct comparison.

Key properties:
- Point lookups: O(depth) page reads
- Inserts: O(depth) page reads + O(1) page writes (amortized)
- Range scans: O(depth + result_size / page_size) page reads
- Node splits add write amplification
"""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from typing import Any, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BTreeStats:
    """Frozen snapshot of B-tree statistics.

    Attributes:
        reads: Total get operations.
        writes: Total put/delete operations.
        page_reads: Total disk page reads.
        page_writes: Total disk page writes.
        node_splits: Total node split operations.
        tree_depth: Current depth of the tree.
        total_keys: Total number of keys stored.
    """

    reads: int = 0
    writes: int = 0
    page_reads: int = 0
    page_writes: int = 0
    node_splits: int = 0
    tree_depth: int = 0
    total_keys: int = 0


class _BTreeNode:
    """Internal B-tree node (not exported).

    Each node represents a disk page. Leaf nodes store key-value pairs;
    internal nodes store keys and child pointers.
    """

    __slots__ = ("leaf", "keys", "values", "children")

    def __init__(self, leaf: bool = True) -> None:
        self.leaf = leaf
        self.keys: list[str] = []
        self.values: list[Any] = []      # Only used in leaf nodes
        self.children: list[_BTreeNode] = []  # Only used in internal nodes


class BTree(Entity):
    """B-tree index with page-level I/O cost simulation.

    Each tree traversal costs depth * page_read_latency in simulated
    I/O time. Writes additionally pay page_write_latency per modified
    page. Node splits create extra write overhead.

    Args:
        name: Entity name.
        order: Maximum number of children per internal node.
            Leaf nodes hold at most (order - 1) keys.
        disk: Optional Resource for disk I/O contention.
        page_read_latency: Seconds per page read.
        page_write_latency: Seconds per page write.

    Example::

        btree = BTree("index", order=128)
        sim = Simulation(entities=[btree], ...)
    """

    def __init__(
        self,
        name: str,
        *,
        order: int = 128,
        disk: Any | None = None,
        page_read_latency: float = 0.001,
        page_write_latency: float = 0.002,
    ) -> None:
        if order < 3:
            raise ValueError(f"order must be >= 3, got {order}")

        super().__init__(name)
        self._order = order
        self._disk = disk
        self._page_read_latency = page_read_latency
        self._page_write_latency = page_write_latency

        self._root = _BTreeNode(leaf=True)
        self._depth = 1
        self._total_keys = 0

        # Stats
        self._total_reads = 0
        self._total_writes = 0
        self._total_page_reads = 0
        self._total_page_writes = 0
        self._total_splits = 0

    @property
    def depth(self) -> int:
        """Current depth of the B-tree."""
        return self._depth

    @property
    def size(self) -> int:
        """Total number of keys in the tree."""
        return self._total_keys

    @property
    def stats(self) -> BTreeStats:
        """Frozen snapshot of B-tree statistics."""
        return BTreeStats(
            reads=self._total_reads,
            writes=self._total_writes,
            page_reads=self._total_page_reads,
            page_writes=self._total_page_writes,
            node_splits=self._total_splits,
            tree_depth=self._depth,
            total_keys=self._total_keys,
        )

    def get(self, key: str) -> Generator[float, None, Any | None]:
        """Look up a key, yielding page read latency for each tree level."""
        self._total_reads += 1

        node = self._root
        for _ in range(self._depth):
            self._total_page_reads += 1
            yield self._page_read_latency

            if node.leaf:
                idx = bisect.bisect_left(node.keys, key)
                if idx < len(node.keys) and node.keys[idx] == key:
                    return node.values[idx]
                return None

            # Internal node: find child
            idx = bisect.bisect_right(node.keys, key)
            node = node.children[idx]

        # Should not reach here, but handle edge case
        return None

    def get_sync(self, key: str) -> Any | None:
        """Look up without yielding latency."""
        self._total_reads += 1

        node = self._root
        while not node.leaf:
            self._total_page_reads += 1
            idx = bisect.bisect_right(node.keys, key)
            node = node.children[idx]

        self._total_page_reads += 1
        idx = bisect.bisect_left(node.keys, key)
        if idx < len(node.keys) and node.keys[idx] == key:
            return node.values[idx]
        return None

    def put(self, key: str, value: Any) -> Generator[float, None, None]:
        """Insert or update a key-value pair, yielding I/O latency.

        Traverses from root to leaf (page reads), then writes the leaf
        page. May split nodes, adding extra page writes.
        """
        self._total_writes += 1

        # Read latency for traversal
        yield self._depth * self._page_read_latency
        self._total_page_reads += self._depth

        # Perform the actual insert
        is_update = self._insert(key, value)

        # Write latency for the modified leaf
        self._total_page_writes += 1
        yield self._page_write_latency

    def put_sync(self, key: str, value: Any) -> None:
        """Insert without yielding latency."""
        self._total_writes += 1
        self._total_page_reads += self._depth
        self._insert(key, value)
        self._total_page_writes += 1

    def delete(self, key: str) -> Generator[float, None, bool]:
        """Delete a key, yielding I/O latency.

        Returns True if the key was found and deleted, False otherwise.
        Uses lazy deletion (mark as None) for simplicity.
        """
        self._total_writes += 1

        # Read latency for traversal
        yield self._depth * self._page_read_latency
        self._total_page_reads += self._depth

        deleted = self._delete(key)

        if deleted:
            self._total_page_writes += 1
            yield self._page_write_latency

        return deleted

    def scan(self, start_key: str, end_key: str) -> Generator[float, None, list[tuple[str, Any]]]:
        """Range scan, yielding I/O latency.

        Traverses to the start leaf, then scans sequentially.
        """
        self._total_reads += 1

        # Traverse to start leaf
        yield self._depth * self._page_read_latency
        self._total_page_reads += self._depth

        # Collect results by walking the tree
        results: list[tuple[str, Any]] = []
        self._scan_node(self._root, start_key, end_key, results)

        # Additional page reads for data pages beyond the first
        extra_pages = max(0, len(results) // (self._order - 1))
        if extra_pages > 0:
            yield extra_pages * self._page_read_latency
            self._total_page_reads += extra_pages

        return results

    def _insert(self, key: str, value: Any) -> bool:
        """Insert key-value pair, splitting nodes as needed.

        Returns True if this was an update (key existed), False if new.
        """
        root = self._root

        # If root is full, split it first
        if len(root.keys) >= self._order - 1:
            new_root = _BTreeNode(leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self._root = new_root
            self._depth += 1
            root = new_root

        return self._insert_non_full(root, key, value)

    def _insert_non_full(self, node: _BTreeNode, key: str, value: Any) -> bool:
        """Insert into a node that is guaranteed not to be full."""
        if node.leaf:
            idx = bisect.bisect_left(node.keys, key)
            if idx < len(node.keys) and node.keys[idx] == key:
                # Update existing key
                node.values[idx] = value
                return True
            # Insert new key
            node.keys.insert(idx, key)
            node.values.insert(idx, value)
            self._total_keys += 1
            return False

        # Internal node: find the right child
        idx = bisect.bisect_right(node.keys, key)
        child = node.children[idx]

        # Split child if full before descending
        if len(child.keys) >= self._order - 1:
            self._split_child(node, idx)
            # After split, determine which child to descend into
            # With B+ tree leaf semantics, key >= separator goes right
            if key >= node.keys[idx]:
                idx += 1
            child = node.children[idx]

        return self._insert_non_full(child, key, value)

    def _split_child(self, parent: _BTreeNode, child_idx: int) -> None:
        """Split a full child node, promoting a separator key to parent."""
        self._total_splits += 1
        self._total_page_writes += 2  # write both halves

        child = parent.children[child_idx]
        mid = len(child.keys) // 2

        # Create new right sibling
        new_node = _BTreeNode(leaf=child.leaf)

        if child.leaf:
            # Leaf split: right sibling gets keys[mid:], left keeps keys[:mid]
            # Parent gets a copy of the right sibling's first key as separator
            new_node.keys = child.keys[mid:]
            new_node.values = child.values[mid:]
            child.keys = child.keys[:mid]
            child.values = child.values[:mid]
            separator = new_node.keys[0]
        else:
            # Internal node split: median key is promoted (removed from both)
            separator = child.keys[mid]
            new_node.keys = child.keys[mid + 1:]
            new_node.children = child.children[mid + 1:]
            child.keys = child.keys[:mid]
            child.children = child.children[:mid + 1]

        # Insert separator into parent
        parent.keys.insert(child_idx, separator)
        parent.children.insert(child_idx + 1, new_node)

    def _delete(self, key: str) -> bool:
        """Delete a key from the tree. Returns True if found."""
        # Simple approach: find and remove from leaf
        node = self._root
        while not node.leaf:
            idx = bisect.bisect_right(node.keys, key)
            node = node.children[idx]

        idx = bisect.bisect_left(node.keys, key)
        if idx < len(node.keys) and node.keys[idx] == key:
            node.keys.pop(idx)
            node.values.pop(idx)
            self._total_keys -= 1
            return True
        return False

    def _scan_node(
        self,
        node: _BTreeNode,
        start_key: str,
        end_key: str,
        results: list[tuple[str, Any]],
    ) -> None:
        """Recursively collect key-value pairs in [start_key, end_key)."""
        if node.leaf:
            for i, key in enumerate(node.keys):
                if key >= end_key:
                    break
                if key >= start_key:
                    results.append((key, node.values[i]))
            return

        for i, child in enumerate(node.children):
            # Determine if this subtree could contain keys in range
            low = node.keys[i - 1] if i > 0 else None
            high = node.keys[i] if i < len(node.keys) else None

            if high is not None and high <= start_key:
                continue
            if low is not None and low >= end_key:
                break

            self._scan_node(child, start_key, end_key, results)

    def handle_event(self, event: Event) -> None:
        """BTree does not process events directly."""
        pass

    def __repr__(self) -> str:
        return (
            f"BTree('{self.name}', order={self._order}, depth={self._depth}, "
            f"keys={self._total_keys})"
        )
