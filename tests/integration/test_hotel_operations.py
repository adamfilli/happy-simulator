"""Integration test for hotel operations simulation."""

from __future__ import annotations

from examples.hotel_operations import run_hotel_simulation, HotelConfig


class TestHotelSimulation:

    def test_runs_to_completion(self):
        config = HotelConfig(duration_s=7200.0, seed=42)
        result = run_hotel_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_guests_checked_in(self):
        config = HotelConfig(duration_s=14400.0, seed=42)
        result = run_hotel_simulation(config)
        assert result.front_desk._checked_in > 0

    def test_gate_blocks_during_turnover(self):
        config = HotelConfig(duration_s=86400.0, seed=88)
        result = run_hotel_simulation(config)
        assert result.gate.stats.passed_through > 0
