"""Integration test for laundromat simulation."""

from __future__ import annotations

from examples.laundromat import run_laundromat_simulation, LaundromatConfig


class TestLaundromatSimulation:

    def test_runs_to_completion(self):
        config = LaundromatConfig(duration_s=3600.0, seed=42)
        result = run_laundromat_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_customers_processed(self):
        config = LaundromatConfig(duration_s=3600.0, seed=42)
        result = run_laundromat_simulation(config)
        assert result.washers.processed > 0

    def test_dryers_used(self):
        config = LaundromatConfig(duration_s=7200.0, seed=88)
        result = run_laundromat_simulation(config)
        assert result.dryers.completed > 0
