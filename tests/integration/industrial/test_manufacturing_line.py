"""Integration test for manufacturing line simulation."""

from __future__ import annotations

import pytest

from examples.manufacturing_line import run_manufacturing_simulation, ManufacturingConfig


class TestManufacturingLineSimulation:

    def test_runs_to_completion(self):
        """Manufacturing simulation runs without errors."""
        config = ManufacturingConfig(duration_s=300.0, seed=42)
        result = run_manufacturing_simulation(config)

        assert result.summary.total_events_processed > 0

    def test_parts_flow_through_pipeline(self):
        """Parts are processed by all stations."""
        config = ManufacturingConfig(duration_s=300.0, seed=42)
        result = run_manufacturing_simulation(config)

        assert result.stations["Cut"].parts_processed > 0
        assert result.stations["Assemble"].parts_processed > 0
        assert result.inspector.inspected > 0

    def test_inspection_produces_pass_and_fail(self):
        """Inspection routes items to pass/fail paths."""
        config = ManufacturingConfig(duration_s=600.0, defect_rate=0.2, seed=42)
        result = run_manufacturing_simulation(config)

        stats = result.inspector.stats
        assert stats.passed > 0
        assert stats.failed > 0

    def test_batch_packaging(self):
        """Batch processor accumulates and processes items."""
        config = ManufacturingConfig(duration_s=600.0, seed=42)
        result = run_manufacturing_simulation(config)

        pkg = result.packager.stats
        assert pkg.batches_processed > 0
        assert pkg.items_processed > 0

    def test_breakdowns_occur(self):
        """Breakdown scheduler produces breakdowns."""
        config = ManufacturingConfig(
            duration_s=600.0, mttf=50.0, mttr=5.0, seed=42,
        )
        result = run_manufacturing_simulation(config)

        bd = result.breakdown.stats
        assert bd.breakdown_count > 0
        assert bd.total_downtime_s > 0

    def test_reasonable_throughput(self):
        """Throughput is reasonable for the configuration."""
        config = ManufacturingConfig(duration_s=600.0, seed=99)
        result = run_manufacturing_simulation(config)

        # With parts arriving and flowing through pipeline, expect some output
        assert result.sink.count > 0
