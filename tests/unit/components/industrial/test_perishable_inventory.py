"""Tests for PerishableInventory component."""

from __future__ import annotations

import pytest

from happysimulator.components.common import Sink
from happysimulator.components.industrial.perishable_inventory import PerishableInventory
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestPerishableInventoryBasics:
    def test_creates_with_defaults(self):
        inv = PerishableInventory("inv")
        assert inv.stock == 100
        assert inv.shelf_life_s == 3600.0

    def test_consume_decrements_stock(self):
        inv = PerishableInventory("inv", initial_stock=10, reorder_point=0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[inv],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Consume", target=inv))
        sim.run()

        assert inv.stock == 9

    def test_stockout_when_empty(self):
        inv = PerishableInventory("inv", initial_stock=0, reorder_point=0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[inv],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Consume", target=inv))
        sim.run()

        assert inv.stats.stockouts == 1

    def test_spoilage_removes_expired_items(self):
        inv = PerishableInventory(
            "inv",
            initial_stock=10,
            shelf_life_s=1.0,
            spoilage_check_interval_s=0.5,
            reorder_point=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[inv],
        )
        sim.schedule(inv.start_event())
        sim.run()

        assert inv.stats.total_spoiled == 10
        assert inv.stock == 0

    def test_waste_target_receives_spoilage(self):
        waste = Sink("waste")
        inv = PerishableInventory(
            "inv",
            initial_stock=5,
            shelf_life_s=1.0,
            spoilage_check_interval_s=0.5,
            reorder_point=0,
            waste_target=waste,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[inv, waste],
        )
        sim.schedule(inv.start_event())
        sim.run()

        assert waste.events_received >= 1

    def test_reorder_triggered_after_consumption(self):
        inv = PerishableInventory(
            "inv",
            initial_stock=25,
            shelf_life_s=10000.0,
            spoilage_check_interval_s=10000.0,
            reorder_point=20,
            order_quantity=50,
            lead_time=2.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            entities=[inv],
        )
        for i in range(6):
            sim.schedule(
                Event(time=Instant.from_seconds(i * 0.1), event_type="Consume", target=inv)
            )
        sim.run()

        assert inv.stats.reorders == 1
        assert inv.stock == 25 - 6 + 50

    def test_downstream_gets_fulfilled(self):
        downstream = Sink("ds")
        inv = PerishableInventory(
            "inv",
            initial_stock=5,
            downstream=downstream,
            reorder_point=0,
            shelf_life_s=10000.0,
            spoilage_check_interval_s=10000.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[inv, downstream],
        )
        for _i in range(3):
            sim.schedule(Event(time=Instant.Epoch, event_type="Consume", target=inv))
        sim.run()

        assert downstream.events_received == 3

    def test_waste_rate(self):
        inv = PerishableInventory(
            "inv",
            initial_stock=10,
            shelf_life_s=0.5,
            spoilage_check_interval_s=0.3,
            reorder_point=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[inv],
        )
        # Consume 5 before spoilage
        for i in range(5):
            sim.schedule(
                Event(time=Instant.from_seconds(i * 0.05), event_type="Consume", target=inv)
            )
        sim.schedule(inv.start_event())
        sim.run()

        stats = inv.stats
        assert stats.total_consumed == 5
        assert stats.total_spoiled == 5
        assert stats.waste_rate == pytest.approx(0.5)
