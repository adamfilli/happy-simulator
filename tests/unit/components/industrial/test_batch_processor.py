"""Tests for BatchProcessor component."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.batch_processor import BatchProcessor
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestBatchProcessorBasics:

    def test_creates_with_defaults(self):
        sink = Sink()
        bp = BatchProcessor("batch", downstream=sink)
        assert bp.batch_size == 10
        assert bp.process_time == 1.0
        assert bp.timeout_s == 0.0
        assert bp.batches_processed == 0
        assert bp.items_processed == 0

    def test_processes_full_batch(self):
        sink = Sink()
        bp = BatchProcessor("batch", downstream=sink, batch_size=3, process_time=0.1)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[bp, sink],
        )
        for i in range(3):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Item", target=bp)
            )
        sim.run()

        assert bp.batches_processed == 1
        assert bp.items_processed == 3
        assert sink.events_received == 3

    def test_does_not_process_partial_without_timeout(self):
        sink = Sink()
        bp = BatchProcessor("batch", downstream=sink, batch_size=5, process_time=0.1)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[bp, sink],
        )
        # Only send 3 items, batch_size is 5, no timeout
        for i in range(3):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Item", target=bp)
            )
        sim.run()

        assert bp.batches_processed == 0
        assert bp.buffer_depth == 3
        assert sink.events_received == 0

    def test_timeout_flushes_partial_batch(self):
        sink = Sink()
        bp = BatchProcessor(
            "batch", downstream=sink,
            batch_size=10, process_time=0.1, timeout_s=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[bp, sink],
        )
        # Send 3 items, timeout should flush after 1s
        for i in range(3):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Item", target=bp)
            )
        sim.run()

        assert bp.batches_processed == 1
        assert bp.items_processed == 3
        assert bp.timeouts == 1
        assert sink.events_received == 3

    def test_multiple_batches(self):
        sink = Sink()
        bp = BatchProcessor("batch", downstream=sink, batch_size=3, process_time=0.01)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[bp, sink],
        )
        for i in range(9):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Item",
                    target=bp,
                )
            )
        sim.run()

        assert bp.batches_processed == 3
        assert bp.items_processed == 9
        assert sink.events_received == 9

    def test_stats_snapshot(self):
        sink = Sink()
        bp = BatchProcessor(
            "batch", downstream=sink,
            batch_size=2, process_time=0.01, timeout_s=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[bp, sink],
        )
        for i in range(3):
            sim.schedule(
                Event(time=Instant.Epoch, event_type="Item", target=bp)
            )
        sim.run()

        stats = bp.stats
        # 2 items in first batch, 1 in timeout batch
        assert stats.batches_processed == 2
        assert stats.items_processed == 3
