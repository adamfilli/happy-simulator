"""Causal event graph for post-simulation analysis.

Builds a directed acyclic graph of "event A caused event B" relationships
from trace recorder data. Events created during another event's invoke()
automatically have their ``parent_id`` set, forming parent-child edges.

Usage::

    from happysimulator import Simulation, InMemoryTraceRecorder
    from happysimulator.analysis import build_causal_graph

    recorder = InMemoryTraceRecorder()
    sim = Simulation(..., trace_recorder=recorder)
    sim.run()

    graph = build_causal_graph(recorder)
    # or: graph = sim.causal_graph()

    for root in graph.roots():
        print(f"Root: {root.event_type} at {root.time}")
    print(f"Critical path length: {len(graph.critical_path())}")
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from happysimulator.instrumentation.recorder import InMemoryTraceRecorder


@dataclass(frozen=True)
class CausalNode:
    """A node in the causal event graph.

    Attributes:
        event_id: Unique event identifier (from context["id"]).
        event_type: Human-readable event type label.
        time: Simulation time when the event was scheduled.
        parent_id: ID of the event that caused this one, or None for roots.
    """

    event_id: str
    event_type: str
    time: Instant
    parent_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "time_s": self.time.to_seconds(),
            "parent_id": self.parent_id,
        }


class CausalGraph:
    """Directed acyclic graph of causal event relationships.

    Nodes are events; edges point from parent (cause) to child (effect).
    Built from trace recorder ``simulation.schedule`` spans that carry
    ``parent_id`` metadata.

    Args:
        nodes: Mapping of event_id to CausalNode.
    """

    def __init__(self, nodes: dict[str, CausalNode]) -> None:
        self._nodes = nodes

        # Build adjacency indexes
        self._children: dict[str, list[str]] = defaultdict(list)
        self._root_ids: list[str] = []

        for node in nodes.values():
            if node.parent_id is not None and node.parent_id in nodes:
                self._children[node.parent_id].append(node.event_id)
            else:
                self._root_ids.append(node.event_id)

        # Sort roots by time for deterministic ordering
        self._root_ids.sort(key=lambda eid: self._nodes[eid].time)

    @property
    def nodes(self) -> dict[str, CausalNode]:
        """All nodes keyed by event_id."""
        return self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._nodes

    def parent(self, event_id: str) -> CausalNode | None:
        """Return the parent node, or None if this is a root."""
        node = self._nodes[event_id]
        if node.parent_id is not None and node.parent_id in self._nodes:
            return self._nodes[node.parent_id]
        return None

    def children(self, event_id: str) -> list[CausalNode]:
        """Return direct children of the given event."""
        return [self._nodes[cid] for cid in self._children.get(event_id, [])]

    def ancestors(self, event_id: str) -> list[CausalNode]:
        """Return the chain from parent to root (nearest first)."""
        result: list[CausalNode] = []
        current = self._nodes[event_id]
        while current.parent_id is not None and current.parent_id in self._nodes:
            current = self._nodes[current.parent_id]
            result.append(current)
        return result

    def descendants(self, event_id: str) -> list[CausalNode]:
        """Return all descendants via BFS (breadth-first)."""
        result: list[CausalNode] = []
        queue: deque[str] = deque(self._children.get(event_id, []))
        while queue:
            cid = queue.popleft()
            result.append(self._nodes[cid])
            queue.extend(self._children.get(cid, []))
        return result

    def roots(self) -> list[CausalNode]:
        """Return all root nodes (events with no parent in the graph)."""
        return [self._nodes[rid] for rid in self._root_ids]

    def leaves(self) -> list[CausalNode]:
        """Return all leaf nodes (events with no children)."""
        return [
            node for node in self._nodes.values()
            if not self._children.get(node.event_id)
        ]

    def depth(self, event_id: str) -> int:
        """Return the distance from the root (root = 0)."""
        d = 0
        current = self._nodes[event_id]
        while current.parent_id is not None and current.parent_id in self._nodes:
            current = self._nodes[current.parent_id]
            d += 1
        return d

    def critical_path(self) -> list[CausalNode]:
        """Return the longest causal chain in the graph.

        Uses topological ordering + dynamic programming on the DAG.
        Returns the full path from root to leaf of maximum length.
        """
        if not self._nodes:
            return []

        # Compute in-degree for topological sort
        in_degree: dict[str, int] = {eid: 0 for eid in self._nodes}
        for eid, children_ids in self._children.items():
            for cid in children_ids:
                if cid in in_degree:
                    in_degree[cid] += 1

        # Kahn's algorithm for topological order
        topo_order: list[str] = []
        queue: deque[str] = deque(
            eid for eid, deg in in_degree.items() if deg == 0
        )
        while queue:
            eid = queue.popleft()
            topo_order.append(eid)
            for cid in self._children.get(eid, []):
                if cid in in_degree:
                    in_degree[cid] -= 1
                    if in_degree[cid] == 0:
                        queue.append(cid)

        # DP: longest path from any root to each node
        dist: dict[str, int] = {eid: 0 for eid in self._nodes}
        pred: dict[str, str | None] = {eid: None for eid in self._nodes}

        for eid in topo_order:
            for cid in self._children.get(eid, []):
                if cid in dist and dist[eid] + 1 > dist[cid]:
                    dist[cid] = dist[eid] + 1
                    pred[cid] = eid

        # Find the endpoint of the longest path
        end_id = max(dist, key=lambda eid: dist[eid])

        # Reconstruct path
        path_ids: list[str] = []
        current: str | None = end_id
        while current is not None:
            path_ids.append(current)
            current = pred[current]
        path_ids.reverse()

        return [self._nodes[eid] for eid in path_ids]

    def filter(self, predicate: Callable[[CausalNode], bool]) -> CausalGraph:
        """Return a new graph containing only nodes matching the predicate.

        Parent links are re-linked: if a node's parent is excluded, the
        node searches up the ancestor chain for the nearest included
        ancestor, preserving causal connectivity.
        """
        kept_ids = {eid for eid, node in self._nodes.items() if predicate(node)}

        new_nodes: dict[str, CausalNode] = {}
        for eid in kept_ids:
            node = self._nodes[eid]
            # Find nearest ancestor that is also kept
            new_parent_id: str | None = None
            cursor = node.parent_id
            while cursor is not None and cursor in self._nodes:
                if cursor in kept_ids:
                    new_parent_id = cursor
                    break
                cursor = self._nodes[cursor].parent_id

            new_nodes[eid] = CausalNode(
                event_id=node.event_id,
                event_type=node.event_type,
                time=node.time,
                parent_id=new_parent_id,
            )

        return CausalGraph(new_nodes)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with summary stats."""
        return {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "stats": {
                "total_nodes": len(self._nodes),
                "roots": len(self._root_ids),
                "leaves": len(self.leaves()),
                "max_depth": max(
                    (self.depth(eid) for eid in self._nodes), default=0
                ),
                "critical_path_length": len(self.critical_path()),
            },
        }


def build_causal_graph(
    recorder: InMemoryTraceRecorder,
    *,
    exclude_event_types: set[str] | None = None,
) -> CausalGraph:
    """Build a CausalGraph from an InMemoryTraceRecorder.

    Reads ``simulation.schedule`` spans which contain ``event_id``,
    ``event_type``, ``scheduled_time``, and ``parent_id``.
    Deduplicates by event_id (ProcessContinuation shares ID with parent
    event since they share the same context dict).

    Args:
        recorder: The trace recorder from a completed simulation.
        exclude_event_types: Event types to omit from the graph.

    Returns:
        A CausalGraph with one node per unique scheduled event.
    """
    exclude = exclude_event_types or set()
    nodes: dict[str, CausalNode] = {}

    for span in recorder.spans:
        if span["kind"] != "simulation.schedule":
            continue

        event_id = span.get("event_id")
        event_type = span.get("event_type")
        if event_id is None or event_type is None:
            continue

        if event_type in exclude:
            continue

        # Deduplicate: first occurrence wins (ProcessContinuation reuses ID)
        if event_id in nodes:
            continue

        data = span.get("data", {})
        scheduled_time = data.get("scheduled_time") or span.get("time")
        parent_id = data.get("parent_id")

        nodes[event_id] = CausalNode(
            event_id=event_id,
            event_type=event_type,
            time=scheduled_time,
            parent_id=parent_id,
        )

    return CausalGraph(nodes)
