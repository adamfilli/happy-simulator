"""Integration tests for inventory and supply chain components.

Tests InventoryBuffer and PerishableInventory in multi-component pipelines
with sources driving demand.
"""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.inventory import InventoryBuffer
from happysimulator.components.industrial.perishable_inventory import PerishableInventory
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class TestBasicReordering:
    """Source → InventoryBuffer → Sink with steady demand."""

    def test_reorders_prevent_stockout(self):
        downstream = Sink("fulfilled")
        inv = InventoryBuffer(
            "warehouse",
            initial_stock=30,
            reorder_point=10,
            order_quantity=25,
            lead_time=1.0,
            downstream=downstream,
        )

        source = Source.constant(rate=5.0, target=inv, stop_after=10.0, event_type="Consume")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=15.0,
            sources=[source],
            entities=[inv, downstream],
        )
        sim.run()

        stats = inv.stats
        assert stats.reorders >= 1
        assert downstream.events_received > 0
        # With adequate reordering, fill rate should be high
        assert stats.fill_rate > 0.5


class TestStockoutScenario:
    """High demand + long lead time causes stockouts."""

    def test_stockout_with_high_demand(self):
        downstream = Sink("fulfilled")
        stockout_sink = Sink("stockouts")
        inv = InventoryBuffer(
            "warehouse",
            initial_stock=5,
            reorder_point=3,
            order_quantity=10,
            lead_time=10.0,  # Very long lead time
            downstream=downstream,
            stockout_target=stockout_sink,
        )

        # High demand rate exhausts stock before reorder arrives
        source = Source.constant(rate=5.0, target=inv, stop_after=5.0, event_type="Consume")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=20.0,
            sources=[source],
            entities=[inv, downstream, stockout_sink],
        )
        sim.run()

        stats = inv.stats
        assert stats.stockouts > 0
        assert stockout_sink.events_received > 0
        assert stats.fill_rate < 1.0


class TestPerishableInventoryWithSpoilage:
    """Source → PerishableInventory → Sink where items spoil after shelf life."""

    def test_items_spoil_after_shelf_life(self):
        downstream = Sink("fulfilled")
        waste_sink = Sink("waste")
        inv = PerishableInventory(
            "produce",
            initial_stock=20,
            shelf_life_s=2.0,
            spoilage_check_interval_s=0.5,
            reorder_point=0,  # No reorders to simplify
            downstream=downstream,
            waste_target=waste_sink,
        )

        # Low demand so some items expire
        source = Source.constant(rate=2.0, target=inv, stop_after=3.0, event_type="Consume")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[source],
            entities=[inv, downstream, waste_sink],
        )
        sim.schedule(inv.start_event())
        sim.run()

        stats = inv.stats
        assert stats.total_spoiled > 0
        assert waste_sink.events_received >= 1
        assert stats.waste_rate > 0


class TestPerishableUnderLowDemand:
    """Low demand means most items expire, yielding high waste rate."""

    def test_high_waste_rate_under_low_demand(self):
        downstream = Sink("fulfilled")
        waste_sink = Sink("waste")
        inv = PerishableInventory(
            "dairy",
            initial_stock=50,
            shelf_life_s=1.0,
            spoilage_check_interval_s=0.3,
            reorder_point=0,
            downstream=downstream,
            waste_target=waste_sink,
        )

        # Very low demand: only consume 5 items before all expire
        source = Source.constant(rate=2.0, target=inv, stop_after=1.0, event_type="Consume")

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[source],
            entities=[inv, downstream, waste_sink],
        )
        sim.schedule(inv.start_event())
        sim.run()

        stats = inv.stats
        assert stats.total_spoiled > stats.total_consumed
        assert stats.waste_rate > 0.5
