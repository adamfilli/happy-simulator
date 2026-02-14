"""Integration test for drive-through simulation."""

from __future__ import annotations

from examples.drive_through import run_drive_through_simulation, DriveThruConfig


class TestDriveThroughSimulation:

    def test_runs_to_completion(self):
        config = DriveThruConfig(duration_s=300.0, seed=42)
        result = run_drive_through_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_cars_served(self):
        config = DriveThruConfig(duration_s=300.0, seed=42)
        result = run_drive_through_simulation(config)
        assert result.sink.count > 0

    def test_kitchen_routing(self):
        config = DriveThruConfig(duration_s=600.0, seed=88)
        result = run_drive_through_simulation(config)
        assert result.fast_kitchen.processed > 0
        assert result.slow_kitchen.processed > 0
