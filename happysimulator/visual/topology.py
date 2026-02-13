"""Entity graph discovery via attribute introspection.

Discovers which entities are connected by scanning known downstream
attribute patterns. Used at serve-time to build the initial graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity

if TYPE_CHECKING:
    from happysimulator.core.simulation import Simulation


@dataclass
class Node:
    id: str
    type: str
    category: str
    profile: dict | None = None  # {times: [...], values: [...]} for sources


@dataclass
class Edge:
    source: str
    target: str
    kind: str = "data"  # "data" or "probe"


@dataclass
class Topology:
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "id": n.id, "type": n.type, "category": n.category,
                    **({"profile": n.profile} if n.profile else {}),
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
    from happysimulator.load.source import Source
    from happysimulator.components.common import Sink, Counter
    from happysimulator.components.queued_resource import QueuedResource
    from happysimulator.components.random_router import RandomRouter
    from happysimulator.components.resource import Resource

    try:
        from happysimulator.components.rate_limiter.rate_limited_entity import RateLimitedEntity
        from happysimulator.components.rate_limiter.inductor import Inductor
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
    if isinstance(entity, RandomRouter):
        return "router"
    if isinstance(entity, Resource):
        return "resource"
    return "other"


_DOWNSTREAM_ATTRS = ["downstream", "targets", "target", "_downstream", "_target"]


def _find_downstream(entity: object) -> list[Entity]:
    """Find downstream entities by scanning known attribute patterns.

    Probes are excluded — their ``target`` is a monitoring relationship,
    not a data-flow edge.
    """
    try:
        from happysimulator.instrumentation.probe import Probe
    except ImportError:
        Probe = type(None)

    if isinstance(entity, Probe):
        return []  # handled separately with kind="probe"

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


def discover(sim: "Simulation") -> Topology:
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

        topology.nodes.append(Node(
            id=name,
            type=type(entity).__name__,
            category=classify(entity),
            profile=profile,
        ))

        for downstream in _find_downstream(entity):
            ds_name = getattr(downstream, "name", type(downstream).__name__)
            # Ensure downstream node exists
            if ds_name not in seen_names:
                seen_names.add(ds_name)
                topology.nodes.append(Node(
                    id=ds_name,
                    type=type(downstream).__name__,
                    category=classify(downstream),
                ))
            topology.add_edge_if_new(name, ds_name)

    # Source -> target via _event_provider._target
    for source in sim._sources:
        ep = getattr(source, "_event_provider", None)
        target = getattr(ep, "_target", None)
        if target is not None and isinstance(target, Entity):
            t_name = getattr(target, "name", type(target).__name__)
            if t_name not in seen_names:
                seen_names.add(t_name)
                topology.nodes.append(Node(
                    id=t_name,
                    type=type(target).__name__,
                    category=classify(target),
                ))
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

    return topology
