"""Integration test for pharmacy simulation."""

from __future__ import annotations

from examples.pharmacy import run_pharmacy_simulation, PharmacyConfig


class TestPharmacySimulation:

    def test_runs_to_completion(self):
        config = PharmacyConfig(duration_s=3600.0, seed=42)
        result = run_pharmacy_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_prescriptions_filled(self):
        config = PharmacyConfig(duration_s=3600.0, seed=42)
        result = run_pharmacy_simulation(config)
        assert result.sink.count > 0

    def test_verification_produces_rework(self):
        config = PharmacyConfig(duration_s=7200.0, seed=88)
        result = run_pharmacy_simulation(config)
        assert result.verification.failed > 0
        # DataEntry processed more than arrivals (rework)
        assert result.data_entry.processed >= result.dropoff.processed
