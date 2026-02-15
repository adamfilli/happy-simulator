"""Integration tests for manufacturing pipeline components.

Tests multi-component pipelines using ConveyorBelt, InspectionStation,
BatchProcessor, and BreakdownScheduler wired together in realistic
configurations.
"""

from __future__ import annotations

import random

from happysimulator.components.common import Sink
from happysimulator.components.industrial.batch_processor import BatchProcessor
from happysimulator.components.industrial.breakdown import BreakdownScheduler
from happysimulator.components.industrial.conveyor import ConveyorBelt
from happysimulator.components.industrial.inspection import InspectionStation
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class TestProductionLineWithInspection:
    """Source → ConveyorBelt → InspectionStation → (pass: Sink, fail: Sink)."""

    def test_pass_fail_split_accounts_for_all_items(self):
        random.seed(42)
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")

        station = InspectionStation(
            "inspect",
            pass_sink,
            fail_sink,
            pass_rate=0.8,
            inspection_time=0.01,
        )
        belt = ConveyorBelt("belt", downstream=station, transit_time=0.1)

        source = Source.constant(rate=20.0, target=belt, stop_after=5.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=7.0,
            sources=[source],
            entities=[belt, station, pass_sink, fail_sink],
        )
        sim.run()

        total_items = pass_sink.events_received + fail_sink.events_received
        assert total_items == belt.items_transported
        assert total_items == station.inspected
        assert station.passed == pass_sink.events_received
        assert station.failed == fail_sink.events_received
        assert 0 < station.failed < station.inspected

    def test_all_pass_with_perfect_quality(self):
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")

        station = InspectionStation(
            "inspect",
            pass_sink,
            fail_sink,
            pass_rate=1.0,
            inspection_time=0.01,
        )
        belt = ConveyorBelt("belt", downstream=station, transit_time=0.05)

        source = Source.constant(rate=10.0, target=belt, stop_after=2.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=4.0,
            sources=[source],
            entities=[belt, station, pass_sink, fail_sink],
        )
        sim.run()

        assert pass_sink.events_received == station.inspected
        assert fail_sink.events_received == 0


class TestMultiStageWithBatchProcessing:
    """Source → ConveyorBelt → InspectionStation → BatchProcessor → Sink."""

    def test_batches_form_from_passed_items(self):
        random.seed(42)
        sink = Sink("output")
        fail_sink = Sink("reject")

        batch = BatchProcessor(
            "batch",
            downstream=sink,
            batch_size=5,
            process_time=0.05,
        )
        station = InspectionStation(
            "inspect",
            batch,
            fail_sink,
            pass_rate=1.0,
            inspection_time=0.01,
        )
        belt = ConveyorBelt("belt", downstream=station, transit_time=0.05)

        source = Source.constant(rate=20.0, target=belt, stop_after=3.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[source],
            entities=[belt, station, batch, sink, fail_sink],
        )
        sim.run()

        # All items that passed inspection should have been processed
        assert batch.items_processed == station.passed
        assert batch.batches_processed >= 1
        assert sink.events_received == batch.items_processed

    def test_partial_batch_flushed_by_timeout(self):
        sink = Sink("output")
        fail_sink = Sink("reject")

        batch = BatchProcessor(
            "batch",
            downstream=sink,
            batch_size=100,
            process_time=0.01,
            timeout_s=1.0,
        )
        station = InspectionStation(
            "inspect",
            batch,
            fail_sink,
            pass_rate=1.0,
            inspection_time=0.01,
        )
        belt = ConveyorBelt("belt", downstream=station, transit_time=0.05)

        # Send only 5 items — not enough for a full batch of 100
        source = Source.constant(rate=10.0, target=belt, stop_after=0.5)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[source],
            entities=[belt, station, batch, sink, fail_sink],
        )
        sim.run()

        assert batch.timeouts >= 1
        assert batch.items_processed > 0
        assert sink.events_received == batch.items_processed


class TestProductionLineWithBreakdowns:
    """Source → InspectionStation (+ BreakdownScheduler) → Sink.

    Breakdowns reduce throughput by taking the machine offline periodically.
    """

    def test_breakdowns_reduce_throughput(self):
        random.seed(42)
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")

        station = InspectionStation(
            "inspect",
            pass_sink,
            fail_sink,
            pass_rate=1.0,
            inspection_time=0.01,
        )
        breakdown = BreakdownScheduler(
            "bd",
            target=station,
            mean_time_to_failure=5.0,
            mean_repair_time=2.0,
        )

        source = Source.constant(rate=10.0, target=station, stop_after=50.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=55.0,
            sources=[source],
            entities=[station, breakdown, pass_sink, fail_sink],
        )
        sim.schedule(breakdown.start_event())
        sim.run()

        stats = breakdown.stats
        assert stats.breakdown_count > 0
        assert stats.total_downtime_s > 0
        assert 0.0 < stats.availability < 1.0
        assert pass_sink.events_received > 0
