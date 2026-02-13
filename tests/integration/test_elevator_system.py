"""Integration test for elevator system simulation."""

from __future__ import annotations

from examples.elevator_system import run_elevator_simulation, ElevatorConfig


class TestElevatorSimulation:

    def test_runs_to_completion(self):
        config = ElevatorConfig(duration_s=600.0, seed=42)
        result = run_elevator_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_passengers_transported(self):
        config = ElevatorConfig(duration_s=600.0, seed=42)
        result = run_elevator_simulation(config)
        assert result.sink.count > 0

    def test_all_elevators_used(self):
        config = ElevatorConfig(duration_s=1200.0, seed=88)
        result = run_elevator_simulation(config)
        for elev in result.elevators:
            assert elev.trips > 0, f"{elev.name} had no trips"
