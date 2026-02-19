"""Tests for visual_widget() protocol and topology integration."""

from __future__ import annotations

from happysimulator.components.common import Sink
from happysimulator.components.server.server import Server
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source
from happysimulator.visual.topology import discover


class WidgetEntity(Entity):
    """Entity with a custom visual widget."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self._downstream = downstream

    def handle_event(self, event: Event) -> list[Event]:
        return [self.forward(event, self._downstream)]

    def downstream_entities(self) -> list[Entity]:
        return [self._downstream]

    def visual_widget(self) -> dict | None:
        return {"type": "queue", "depth": "depth"}


class TestVisualWidget:
    def test_base_entity_returns_none(self):
        """Entity.visual_widget() returns None by default."""
        sink = Sink("s")
        assert sink.visual_widget() is None

    def test_custom_override_returns_dict(self):
        """Subclass can return a widget definition dict."""
        sink = Sink("s")
        entity = WidgetEntity("w", sink)
        widget = entity.visual_widget()
        assert widget is not None
        assert widget["type"] == "queue"
        assert widget["depth"] == "depth"

    def test_server_visual_widget(self):
        """Server returns a slots widget definition."""
        server = Server("srv")
        widget = server.visual_widget()
        assert widget is not None
        assert widget["type"] == "slots"
        assert widget["total"] == "concurrency"
        assert widget["active"] == "active_requests"

    def test_discover_populates_node_widget(self):
        """discover() populates Node.widget from entity's visual_widget()."""
        sink = Sink("sink")
        entity = WidgetEntity("w", sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[entity, sink],
        )

        topo = discover(sim)
        widget_node = next(n for n in topo.nodes if n.id == "w")
        assert widget_node.widget is not None
        assert widget_node.widget["type"] == "queue"

        sink_node = next(n for n in topo.nodes if n.id == "sink")
        assert sink_node.widget is None

    def test_topology_to_dict_includes_widget(self):
        """Topology.to_dict() includes widget when present, omits when None."""
        sink = Sink("sink")
        entity = WidgetEntity("w", sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[entity, sink],
        )

        topo = discover(sim)
        topo_dict = topo.to_dict()

        widget_node_dict = next(n for n in topo_dict["nodes"] if n["id"] == "w")
        assert "widget" in widget_node_dict
        assert widget_node_dict["widget"]["type"] == "queue"

        sink_node_dict = next(n for n in topo_dict["nodes"] if n["id"] == "sink")
        assert "widget" not in sink_node_dict

    def test_server_widget_in_topology(self):
        """Server's widget appears in discovered topology."""
        sink = Sink("sink")
        server = Server("srv", downstream=sink)
        source = Source.constant(rate=10, target=server, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[server, sink],
        )

        topo = discover(sim)
        server_node = next(n for n in topo.nodes if n.id == "srv")
        assert server_node.widget is not None
        assert server_node.widget["type"] == "slots"
