"""Tests for ConditionalRouter component."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.conditional_router import ConditionalRouter
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestConditionalRouterBasics:

    def test_creates_with_parameters(self):
        sink = Sink()
        router = ConditionalRouter(
            "router",
            routes=[(lambda e: True, sink)],
        )
        assert router.name == "router"
        assert router.total_routed == 0
        assert router.dropped == 0

    def test_routes_first_match(self):
        target_a = Sink("a")
        target_b = Sink("b")
        router = ConditionalRouter(
            "router",
            routes=[
                (lambda e: e.context.get("x") == 1, target_a),
                (lambda e: True, target_b),
            ],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, target_a, target_b],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router, context={"x": 1}))
        sim.run()

        assert target_a.events_received == 1
        assert target_b.events_received == 0
        assert router.total_routed == 1

    def test_uses_default_when_no_match(self):
        target_a = Sink("a")
        default_sink = Sink("default")
        router = ConditionalRouter(
            "router",
            routes=[(lambda e: False, target_a)],
            default=default_sink,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, target_a, default_sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router))
        sim.run()

        assert default_sink.events_received == 1
        assert target_a.events_received == 0

    def test_drops_when_no_match_and_no_default(self):
        target_a = Sink("a")
        router = ConditionalRouter(
            "router",
            routes=[(lambda e: False, target_a)],
            drop_unmatched=True,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, target_a],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router))
        sim.run()

        assert router.dropped == 1
        assert target_a.events_received == 0


class TestConditionalRouterByContextField:

    def test_routes_by_field_value(self):
        sink_a = Sink("a")
        sink_b = Sink("b")
        router = ConditionalRouter.by_context_field(
            "router", "color", {"red": sink_a, "blue": sink_b},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, sink_a, sink_b],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router, context={"color": "red"}))
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router, context={"color": "blue"}))
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router, context={"color": "blue"}))
        sim.run()

        assert sink_a.events_received == 1
        assert sink_b.events_received == 2

    def test_stats_snapshot(self):
        sink = Sink()
        router = ConditionalRouter(
            "router", routes=[(lambda e: True, sink)],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, sink],
        )
        for i in range(3):
            sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router))
        sim.run()

        stats = router.stats
        assert stats.total_routed == 3
        assert stats.dropped == 0

    def test_routed_counts_by_target(self):
        sink_a = Sink("a")
        sink_b = Sink("b")
        router = ConditionalRouter.by_context_field(
            "router", "t", {"x": sink_a, "y": sink_b},
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[router, sink_a, sink_b],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router, context={"t": "x"}))
        sim.schedule(Event(time=Instant.Epoch, event_type="T", target=router, context={"t": "y"}))
        sim.run()

        counts = router.routed_counts
        assert counts["a"] == 1
        assert counts["b"] == 1
