"""Tests for ConveyorBelt component."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.conveyor import ConveyorBelt
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestConveyorBasics:

    def test_creates_with_parameters(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=1.0)
        assert belt.name == "belt"
        assert belt.transit_time == 1.0
        assert belt.items_in_transit == 0
        assert belt.items_transported == 0

    def test_transports_single_item(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=0.5)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[belt, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=belt))
        sim.run()

        assert belt.items_transported == 1
        assert belt.items_in_transit == 0
        assert sink.events_received == 1

    def test_transit_time_delay(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=1.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[belt, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=belt))
        sim.run()

        # Item should arrive at t=1.0
        assert sink.events_received == 1
        assert sink.completion_times[0] == Instant.from_seconds(1.0)

    def test_multiple_items(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=0.5)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[belt, sink],
        )
        for i in range(5):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.2),
                    event_type="Item",
                    target=belt,
                )
            )
        sim.run()

        assert belt.items_transported == 5
        assert sink.events_received == 5


class TestConveyorCapacity:

    def test_unlimited_capacity_by_default(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=1.0)
        assert belt.has_capacity() is True

    def test_rejects_when_at_capacity(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=1.0, capacity=2)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[belt, sink],
        )
        # Schedule 4 items at same time, only 2 should fit
        for i in range(4):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Item", target=belt)
            )
        sim.run()

        assert belt.items_transported == 2
        assert belt.items_rejected == 2

    def test_stats_snapshot(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=0.5, capacity=3)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[belt, sink],
        )
        for i in range(5):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Item", target=belt)
            )
        sim.run()

        stats = belt.stats
        assert stats.items_transported == 3
        assert stats.items_rejected == 2

    def test_preserves_event_context(self):
        sink = Sink()
        belt = ConveyorBelt("belt", downstream=sink, transit_time=0.1)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[belt, sink],
        )
        sim.schedule(
            Event(
                time=Instant.Epoch,
                event_type="Item",
                target=belt,
                context={"created_at": Instant.Epoch, "payload": "test"},
            )
        )
        sim.run()

        assert sink.events_received == 1
