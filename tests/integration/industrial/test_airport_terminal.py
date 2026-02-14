"""Integration test for airport terminal simulation."""

from __future__ import annotations

from examples.airport_terminal import run_airport_simulation, AirportConfig


class TestAirportSimulation:

    def test_runs_to_completion(self):
        config = AirportConfig(duration_s=3600.0, seed=42)
        result = run_airport_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_passengers_boarded(self):
        config = AirportConfig(duration_s=3600.0, seed=42)
        result = run_airport_simulation(config)
        assert result.sink.count > 0

    def test_all_classes_routed(self):
        config = AirportConfig(duration_s=7200.0, seed=88)
        result = run_airport_simulation(config)
        assert result.router.total_routed > 0
