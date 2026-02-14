"""Tests for PooledCycleResource component."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.pooled_cycle import PooledCycleResource
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestPooledCycleBasics:

    def test_creates_with_parameters(self):
        pool = PooledCycleResource("pool", pool_size=4, cycle_time=1.0)
        assert pool.pool_size == 4
        assert pool.available == 4
        assert pool.active == 0
        assert pool.completed == 0

    def test_single_cycle(self):
        sink = Sink()
        pool = PooledCycleResource("pool", pool_size=2, cycle_time=0.5, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[pool, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=pool))
        sim.run()

        assert pool.completed == 1
        assert sink.events_received == 1

    def test_pool_exhaustion_queues(self):
        sink = Sink()
        pool = PooledCycleResource("pool", pool_size=1, cycle_time=1.0, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[pool, sink],
        )
        # Two items at same time, only 1 pool unit
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=pool))
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=pool))
        sim.run()

        assert pool.completed == 2
        assert sink.events_received == 2

    def test_rejects_when_queue_full(self):
        sink = Sink()
        pool = PooledCycleResource(
            "pool", pool_size=1, cycle_time=1.0, downstream=sink, queue_capacity=1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[pool, sink],
        )
        # Three items at same time: 1 starts, 1 queues, 1 rejected
        for _ in range(3):
            sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=pool))
        sim.run()

        assert pool.rejected == 1
        assert pool.completed == 2

    def test_utilization(self):
        pool = PooledCycleResource("pool", pool_size=4, cycle_time=1.0)
        assert pool.utilization == 0.0

    def test_stats_snapshot(self):
        sink = Sink()
        pool = PooledCycleResource("pool", pool_size=2, cycle_time=0.5, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[pool, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=pool))
        sim.run()

        stats = pool.stats
        assert stats.pool_size == 2
        assert stats.completed == 1
        assert stats.rejected == 0

    def test_preserves_event_context(self):
        sink = Sink()
        pool = PooledCycleResource("pool", pool_size=2, cycle_time=0.1, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[pool, sink],
        )
        sim.schedule(
            Event(
                time=Instant.Epoch, event_type="Item", target=pool,
                context={"created_at": Instant.Epoch, "payload": "test"},
            )
        )
        sim.run()

        assert sink.events_received == 1
