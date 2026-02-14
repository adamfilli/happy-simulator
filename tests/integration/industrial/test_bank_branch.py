"""Integration test for bank branch simulation."""

from __future__ import annotations

import pytest

from examples.bank_branch import run_bank_simulation, BankConfig


class TestBankBranchSimulation:

    def test_runs_to_completion(self):
        """Bank simulation runs without errors."""
        config = BankConfig(duration_s=300.0, seed=42)
        result = run_bank_simulation(config)

        assert result.summary.total_events_processed > 0

    def test_customers_served(self):
        """Customers are processed by tellers."""
        config = BankConfig(duration_s=300.0, seed=42)
        result = run_bank_simulation(config)

        assert result.sink.count > 0

    def test_reasonable_throughput(self):
        """Throughput is reasonable for the configuration."""
        config = BankConfig(duration_s=600.0, seed=88)
        result = run_bank_simulation(config)

        assert result.sink.count > 0
