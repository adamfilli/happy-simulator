"""Tests for visual topology discovery via downstream_entities()."""

from __future__ import annotations

from collections.abc import Generator

from happysimulator.components.common import Sink
from happysimulator.components.server.server import Server
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source
from happysimulator.visual.topology import Topology, _find_downstream, classify, discover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleEntity(Entity):
    """Entity with a single downstream target."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self._downstream = downstream

    def downstream_entities(self) -> list[Entity]:
        return [self._downstream]

    def handle_event(self, event: Event) -> list[Event]:
        return [self.forward(event, self._downstream)]


class MultiTargetEntity(Entity):
    """Entity with multiple downstream targets."""

    def __init__(self, name: str, targets: list[Entity]):
        super().__init__(name)
        self._targets = targets

    def downstream_entities(self) -> list[Entity]:
        return list(self._targets)

    def handle_event(self, event: Event) -> list[Event]:
        return []


class LeafEntity(Entity):
    """Entity with no downstream — uses default downstream_entities()."""

    def __init__(self, name: str):
        super().__init__(name)

    def handle_event(self, event: Event) -> list[Event]:
        return []


# ---------------------------------------------------------------------------
# _find_downstream tests
# ---------------------------------------------------------------------------


class TestFindDownstream:
    def test_returns_empty_for_entity_with_no_override(self):
        leaf = LeafEntity("leaf")
        assert _find_downstream(leaf) == []

    def test_returns_downstream_from_override(self):
        sink = Sink()
        entity = SimpleEntity("a", sink)
        result = _find_downstream(entity)
        assert result == [sink]

    def test_returns_multiple_targets(self):
        s1, s2 = Sink("s1"), Sink("s2")
        entity = MultiTargetEntity("router", [s1, s2])
        result = _find_downstream(entity)
        assert result == [s1, s2]

    def test_no_fallback_to_attribute_scan(self):
        """Entities with downstream attrs but no override should return []."""

        class OldStyleEntity(Entity):
            def __init__(self):
                super().__init__("old")
                self.downstream = Sink("hidden")

            def handle_event(self, event):
                return []

        entity = OldStyleEntity()
        # With _DOWNSTREAM_ATTRS removed, this should NOT find 'downstream'
        assert _find_downstream(entity) == []


# ---------------------------------------------------------------------------
# discover() integration tests
# ---------------------------------------------------------------------------


class TestDiscover:
    def test_simple_pipeline(self):
        """Source -> Server -> Sink topology is fully discovered."""
        sink = Sink("sink")
        server = Server("server", downstream=sink)
        source = Source.constant(rate=10, target=server, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[server, sink],
        )

        topo = discover(sim)
        topo_dict = topo.to_dict()

        node_ids = {n["id"] for n in topo_dict["nodes"]}
        assert "src" in node_ids
        assert "server" in node_ids
        assert "sink" in node_ids

        edge_pairs = {(e["source"], e["target"]) for e in topo_dict["edges"]}
        assert ("src", "server") in edge_pairs
        assert ("server", "sink") in edge_pairs

    def test_multi_target_entity(self):
        """Entity with multiple targets produces multiple edges."""
        s1, s2 = Sink("s1"), Sink("s2")
        router = MultiTargetEntity("router", [s1, s2])

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, s1, s2],
        )

        topo = discover(sim)
        edge_pairs = {(e.source, e.target) for e in topo.edges}
        assert ("router", "s1") in edge_pairs
        assert ("router", "s2") in edge_pairs

    def test_source_target_via_downstream_entities(self):
        """Source discovers its target via downstream_entities(), not attr scan."""
        sink = Sink("sink")
        source = Source.constant(rate=10, target=sink, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[sink],
        )

        topo = discover(sim)
        edge_pairs = {(e.source, e.target) for e in topo.edges}
        assert ("src", "sink") in edge_pairs

    def test_leaf_entity_has_no_edges(self):
        """Entity with no downstream_entities() override produces no edges."""
        leaf = LeafEntity("leaf")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[leaf],
        )

        topo = discover(sim)
        assert len(topo.edges) == 0
        assert len(topo.nodes) == 1
        assert topo.nodes[0].id == "leaf"

    def test_downstream_node_auto_created(self):
        """Downstream entities not in sim.entities are still added as nodes."""
        hidden_sink = Sink("hidden")
        entity = SimpleEntity("fwd", hidden_sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[entity],  # hidden_sink intentionally not registered
        )

        topo = discover(sim)
        node_ids = {n.id for n in topo.nodes}
        assert "hidden" in node_ids
        edge_pairs = {(e.source, e.target) for e in topo.edges}
        assert ("fwd", "hidden") in edge_pairs


# ---------------------------------------------------------------------------
# classify tests
# ---------------------------------------------------------------------------


class TestClassify:
    def test_source_classified(self):
        sink = Sink()
        source = Source.constant(rate=1, target=sink)
        assert classify(source) == "source"

    def test_sink_classified(self):
        assert classify(Sink()) == "sink"

    def test_server_classified(self):
        assert classify(Server("s")) == "queued_resource"

    def test_unknown_classified(self):
        assert classify(LeafEntity("x")) == "other"
