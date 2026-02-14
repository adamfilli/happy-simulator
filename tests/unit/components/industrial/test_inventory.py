"""Tests for InventoryBuffer component."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.inventory import InventoryBuffer
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestInventoryBasics:

    def test_creates_with_defaults(self):
        inv = InventoryBuffer("inv")
        assert inv.stock == 100
        assert inv.reorder_point == 20
        assert inv.order_quantity == 50
        assert inv.lead_time == 5.0

    def test_consume_decrements_stock(self):
        inv = InventoryBuffer("inv", initial_stock=10, reorder_point=0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[inv],
        )
        sim.schedule(
            Event(time=Instant.Epoch, event_type="Consume", target=inv)
        )
        sim.run()

        assert inv.stock == 9

    def test_stockout_when_empty(self):
        stockout_sink = Sink("stockouts")
        inv = InventoryBuffer(
            "inv", initial_stock=0,
            stockout_target=stockout_sink,
            reorder_point=10,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(100.0),
            entities=[inv, stockout_sink],
        )
        sim.schedule(
            Event(time=Instant.Epoch, event_type="Consume", target=inv)
        )
        sim.run()

        assert inv.stats.stockouts == 1
        assert stockout_sink.events_received == 1

    def test_reorder_triggered(self):
        inv = InventoryBuffer(
            "inv",
            initial_stock=25,
            reorder_point=20,
            order_quantity=50,
            lead_time=2.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            entities=[inv],
        )
        # Consume 6 items to bring stock from 25 to 19 (below reorder point)
        for i in range(6):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Consume",
                    target=inv,
                )
            )
        sim.run()

        assert inv.stats.reorders == 1
        # After lead_time=2.0, stock should be replenished
        assert inv.stock == 25 - 6 + 50  # 69

    def test_downstream_gets_fulfilled_events(self):
        downstream = Sink("downstream")
        inv = InventoryBuffer(
            "inv", initial_stock=5, downstream=downstream,
            reorder_point=0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[inv, downstream],
        )
        for i in range(3):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Consume", target=inv)
            )
        sim.run()

        assert downstream.events_received == 3
        assert inv.stock == 2

    def test_fill_rate(self):
        inv = InventoryBuffer("inv", initial_stock=3, reorder_point=0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[inv],
        )
        # Consume 5 items, only 3 available
        for i in range(5):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Consume",
                    target=inv,
                )
            )
        sim.run()

        stats = inv.stats
        assert stats.items_consumed == 3
        assert stats.stockouts == 2
        assert stats.fill_rate == pytest.approx(0.6)

    def test_stats_snapshot(self):
        inv = InventoryBuffer("inv", initial_stock=50)
        stats = inv.stats
        assert stats.current_stock == 50
        assert stats.stockouts == 0
        assert stats.fill_rate == 1.0
