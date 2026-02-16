"""Entity graph discovery via attribute introspection.

Discovers which entities are connected by scanning known downstream
attribute patterns. Used at serve-time to build the initial graph.
"""

from __future__ import annotations

from collections import Counter as _Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity

if TYPE_CHECKING:
    from happysimulator.core.simulation import Simulation

GROUP_THRESHOLD = 20


@dataclass
class Node:
    id: str
    type: str
    category: str
    profile: dict | None = None  # {times: [...], values: [...]} for sources
    is_group: bool = False
    member_count: int = 0
    member_ids: list[str] = field(default_factory=list)


@dataclass
class Edge:
    source: str
    target: str
    kind: str = "data"  # "data" or "probe"


@dataclass
class Topology:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    member_to_group: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id,
                    "type": n.type,
                    "category": n.category,
                    **({"profile": n.profile} if n.profile else {}),
                    **(
                        {
                            "is_group": True,
                            "member_count": n.member_count,
                            "member_ids": n.member_ids[:50],
                        }
                        if n.is_group
                        else {}
                    ),
                }
                for n in self.nodes
            ],
            "edges": [{"source": e.source, "target": e.target, "kind": e.kind} for e in self.edges],
        }

    def add_edge_if_new(self, source: str, target: str, kind: str = "data") -> bool:
        """Add an edge if it doesn't already exist. Returns True if added."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return False
        self.edges.append(Edge(source=source, target=target, kind=kind))
        return True


def classify(entity: object) -> str:
    """Classify an entity into a visual category."""
    from happysimulator.components.common import Counter, Sink
    from happysimulator.components.queued_resource import QueuedResource
    from happysimulator.components.random_router import RandomRouter
    from happysimulator.components.resource import Resource
    from happysimulator.load.source import Source

    try:
        from happysimulator.components.rate_limiter.inductor import Inductor
        from happysimulator.components.rate_limiter.rate_limited_entity import RateLimitedEntity
    except ImportError:
        RateLimitedEntity = type(None)
        Inductor = type(None)

    try:
        from happysimulator.instrumentation.collectors import LatencyTracker, ThroughputTracker
    except ImportError:
        LatencyTracker = type(None)
        ThroughputTracker = type(None)

    try:
        from happysimulator.instrumentation.probe import Probe
    except ImportError:
        Probe = type(None)

    if isinstance(entity, Probe):
        return "probe"
    if isinstance(entity, Source):
        return "source"
    if isinstance(entity, (Sink, Counter, LatencyTracker, ThroughputTracker)):
        return "sink"
    if isinstance(entity, QueuedResource):
        return "queued_resource"
    if isinstance(entity, (RateLimitedEntity, Inductor)):
        return "rate_limiter"
    try:
        from happysimulator.components.load_balancer.load_balancer import LoadBalancer
    except ImportError:
        LoadBalancer = type(None)
    if isinstance(entity, (RandomRouter, LoadBalancer)):
        return "router"
    if isinstance(entity, Resource):
        return "resource"
    return "other"


_DOWNSTREAM_ATTRS = ["downstream", "targets", "target", "_downstream", "_target"]


def _find_downstream(entity: object) -> list[Entity]:
    """Find downstream entities by scanning known attribute patterns.

    Probes are excluded --- their ``target`` is a monitoring relationship,
    not a data-flow edge.
    """
    try:
        from happysimulator.instrumentation.probe import Probe
    except ImportError:
        Probe = type(None)

    if isinstance(entity, Probe):
        return []  # handled separately with kind="probe"

    # Prefer explicit declaration via downstream_entities()
    declared = getattr(entity, "downstream_entities", None)
    if callable(declared):
        result = declared()
        if result:
            return result

    # Fallback: scan common attribute names
    found: list[Entity] = []
    for attr_name in _DOWNSTREAM_ATTRS:
        val = getattr(entity, attr_name, None)
        if val is None:
            continue
        if isinstance(val, Entity):
            found.append(val)
        elif isinstance(val, list):
            found.extend(v for v in val if isinstance(v, Entity))

    return found


def _sample_profile(source: object, end_s: float, num_points: int = 200) -> dict | None:
    """Sample a Source's rate profile over [0, end_s]."""
    from happysimulator.core.temporal import Instant

    provider = getattr(source, "_time_provider", None)
    if provider is None:
        return None
    profile = getattr(provider, "profile", None)
    if profile is None:
        return None

    step = end_s / num_points
    times: list[float] = []
    values: list[float] = []
    for i in range(num_points + 1):
        t = i * step
        times.append(round(t, 6))
        values.append(profile.get_rate(Instant.from_seconds(t)))
    return {"times": times, "values": values}


def _group_topology(topology: Topology) -> None:
    """Collapse same-type nodes that exceed GROUP_THRESHOLD into group nodes."""
    # Count nodes by type
    type_counts: _Counter[str] = _Counter()
    for node in topology.nodes:
        type_counts[node.type] += 1

    types_to_group = {t for t, count in type_counts.items() if count > GROUP_THRESHOLD}
    if not types_to_group:
        return

    # Partition nodes into kept vs grouped
    kept_nodes: list[Node] = []
    grouped_by_type: dict[str, list[Node]] = {}
    for node in topology.nodes:
        if node.type in types_to_group:
            grouped_by_type.setdefault(node.type, []).append(node)
        else:
            kept_nodes.append(node)

    # Create group nodes and populate member_to_group
    group_nodes: list[Node] = []
    for entity_type, members in grouped_by_type.items():
        group_id = f"group:{entity_type}"
        member_ids = [m.id for m in members]
        # Use the category of the first member (all same type = same category)
        category = members[0].category if members else "other"
        group_node = Node(
            id=group_id,
            type=entity_type,
            category=category,
            is_group=True,
            member_count=len(members),
            member_ids=member_ids,
        )
        group_nodes.append(group_node)
        for mid in member_ids:
            topology.member_to_group[mid] = group_id

    # Rewrite edges: replace member node IDs with group IDs, deduplicate
    member_set = set(topology.member_to_group.keys())
    seen_edges: set[tuple[str, str, str]] = set()
    new_edges: list[Edge] = []
    for edge in topology.edges:
        source = topology.member_to_group.get(edge.source, edge.source)
        target = topology.member_to_group.get(edge.target, edge.target)
        edge_key = (source, target, edge.kind)
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            new_edges.append(Edge(source=source, target=target, kind=edge.kind))

    topology.nodes = kept_nodes + group_nodes
    topology.edges = new_edges


def discover(sim: Simulation) -> Topology:
    """Build the initial topology graph from a simulation's entities and sources."""
    from happysimulator.load.source import Source as SourceCls

    topology = Topology()
    seen_names: set[str] = set()

    end_s = sim._end_time.to_seconds()
    if end_s == float("inf"):
        end_s = 60.0

    all_entities = list(sim._sources) + list(sim._entities) + list(sim._probes)

    for entity in all_entities:
        name = getattr(entity, "name", type(entity).__name__)
        if name in seen_names:
            continue
        seen_names.add(name)

        profile = None
        if isinstance(entity, SourceCls):
            profile = _sample_profile(entity, end_s)

        topology.nodes.append(
            Node(
                id=name,
                type=type(entity).__name__,
                category=classify(entity),
                profile=profile,
            )
        )

        for downstream in _find_downstream(entity):
            ds_name = getattr(downstream, "name", type(downstream).__name__)
            # Ensure downstream node exists
            if ds_name not in seen_names:
                seen_names.add(ds_name)
                topology.nodes.append(
                    Node(
                        id=ds_name,
                        type=type(downstream).__name__,
                        category=classify(downstream),
                    )
                )
            topology.add_edge_if_new(name, ds_name)

    # Source -> target via _event_provider._target
    for source in sim._sources:
        ep = getattr(source, "_event_provider", None)
        target = getattr(ep, "_target", None)
        if target is not None and isinstance(target, Entity):
            t_name = getattr(target, "name", type(target).__name__)
            if t_name not in seen_names:
                seen_names.add(t_name)
                topology.nodes.append(
                    Node(
                        id=t_name,
                        type=type(target).__name__,
                        category=classify(target),
                    )
                )
            topology.add_edge_if_new(source.name, t_name)

    # Probe -> target (monitoring edges)
    try:
        from happysimulator.instrumentation.probe import Probe
    except ImportError:
        Probe = type(None)

    for probe in sim._probes:
        if isinstance(probe, Probe):
            t_name = getattr(probe.target, "name", type(probe.target).__name__)
            topology.add_edge_if_new(probe.name, t_name, kind="probe")

    # Collapse large same-type groups
    _group_topology(topology)

    return topology
