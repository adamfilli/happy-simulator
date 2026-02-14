"""Integration test for blood bank simulation."""

from __future__ import annotations

from examples.blood_bank import run_blood_bank_simulation, BloodBankConfig


class TestBloodBankSimulation:

    def test_runs_to_completion(self):
        config = BloodBankConfig(duration_s=3600.0, seed=42)
        result = run_blood_bank_simulation(config)
        assert result.summary.total_events_processed > 0

    def test_donations_processed(self):
        config = BloodBankConfig(duration_s=3600.0, seed=42)
        result = run_blood_bank_simulation(config)
        assert result.donation_station._processed > 0

    def test_split_merge_completes(self):
        config = BloodBankConfig(duration_s=7200.0, seed=88)
        result = run_blood_bank_simulation(config)
        assert result.split_merge.stats.merges_completed > 0
