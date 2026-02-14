"""Integration test for urgent care simulation."""

from __future__ import annotations

from examples.urgent_care import run_urgent_care_simulation, UrgentCareConfig


class TestUrgentCareSimulation:

    def test_runs_to_completion(self):
        config = UrgentCareConfig(duration_s=3600.0, seed=42)
        result = run_urgent_care_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_patients_treated(self):
        config = UrgentCareConfig(duration_s=3600.0, seed=42)
        result = run_urgent_care_simulation(config)
        assert result.sink.count > 0

    def test_routing_by_severity(self):
        config = UrgentCareConfig(duration_s=7200.0, seed=88)
        result = run_urgent_care_simulation(config)
        assert result.router.total_routed > 0
        assert result.trauma_bays._treated > 0
        assert result.exam_rooms.processed > 0
