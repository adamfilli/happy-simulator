"""Integration test for car wash pipeline simulation."""

from __future__ import annotations

import random

import pytest

from examples.car_wash import run_car_wash_simulation, CarWashConfig


class TestCarWashSimulation:

    def test_runs_to_completion(self):
        """Car wash simulation runs without errors."""
        config = CarWashConfig(duration_s=300.0, seed=42)
        result = run_car_wash_simulation(config)

        assert result.summary.total_events_processed > 0
        assert result.sink.count > 0

    def test_all_stations_process_cars(self):
        """All pipeline stages process cars."""
        config = CarWashConfig(duration_s=300.0, seed=42)
        result = run_car_wash_simulation(config)

        for name, station in result.stations.items():
            assert station.cars_processed > 0, f"{name} processed no cars"

    def test_pipeline_ordering(self):
        """Cars flow through pipeline in order."""
        config = CarWashConfig(duration_s=300.0, seed=42)
        result = run_car_wash_simulation(config)

        # Pre-rinse should process at least as many as later stages
        assert result.stations["PreRinse"].cars_processed >= result.sink.count

    def test_reasonable_throughput(self):
        """Throughput is reasonable for the configuration."""
        config = CarWashConfig(duration_s=600.0, seed=123)
        result = run_car_wash_simulation(config)

        # With ~2 cars/min arrival, expect some completed in 10 minutes
        assert result.sink.count > 0

    def test_latency_is_positive(self):
        """All completed cars have positive cycle time."""
        config = CarWashConfig(duration_s=300.0, seed=42)
        result = run_car_wash_simulation(config)

        assert result.sink.count > 0
        assert result.sink.mean_latency() > 0
