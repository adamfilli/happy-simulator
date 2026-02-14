"""Integration test for theme park simulation."""

from __future__ import annotations

from examples.theme_park import run_theme_park_simulation, ThemeParkConfig


class TestThemeParkSimulation:

    def test_runs_to_completion(self):
        config = ThemeParkConfig(duration_s=3600.0, seed=42)
        result = run_theme_park_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_guests_ride(self):
        config = ThemeParkConfig(duration_s=3600.0, seed=42)
        result = run_theme_park_simulation(config)
        assert result.sink.count > 0

    def test_all_rides_used(self):
        config = ThemeParkConfig(duration_s=7200.0, seed=88)
        result = run_theme_park_simulation(config)
        for name, ride in result.rides.items():
            assert ride.completed > 0, f"{name} had no riders"
