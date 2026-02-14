"""Integration test for call center simulation."""

from __future__ import annotations

import pytest

from examples.call_center import run_call_center_simulation, CallCenterConfig


class TestCallCenterSimulation:

    def test_runs_to_completion(self):
        """Call center simulation runs without errors."""
        config = CallCenterConfig(duration_s=300.0, seed=42)
        result = run_call_center_simulation(config)

        assert result.summary.total_events_processed > 0

    def test_calls_flow_through_ivr(self):
        """Calls pass through IVR before routing."""
        config = CallCenterConfig(duration_s=300.0, seed=42)
        result = run_call_center_simulation(config)

        assert result.ivr.calls_processed > 0

    def test_all_pools_handle_calls(self):
        """All agent pools receive and handle calls."""
        config = CallCenterConfig(duration_s=600.0, arrival_rate=0.1, seed=42)
        result = run_call_center_simulation(config)

        for name, pool in result.pools.items():
            assert pool.calls_handled > 0 or pool.reneging_stats.reneged > 0, \
                f"{name} pool had no activity"

    def test_reneging_occurs(self):
        """Some customers abandon when wait is too long."""
        config = CallCenterConfig(
            duration_s=600.0, arrival_rate=0.15,
            mean_patience_s=60.0,  # Very impatient
            morning_agents=2,
            seed=42,
        )
        result = run_call_center_simulation(config)

        assert result.abandoned.count > 0

    def test_reasonable_throughput(self):
        """Throughput is reasonable for the configuration."""
        config = CallCenterConfig(duration_s=600.0, seed=77)
        result = run_call_center_simulation(config)

        assert result.sink.count > 0
