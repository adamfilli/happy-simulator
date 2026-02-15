"""Tests for MCP server tool implementations.

Tests the simulation functions in tools.py which don't require the mcp SDK.
"""

import json

from happysimulator.ai.result import SimulationResult, SweepResult
from happysimulator.mcp.tools import (
    DISTRIBUTIONS_INFO,
    format_distributions,
    format_response,
    run_pipeline_simulation,
    run_queue_simulation,
)


class TestRunQueueSimulation:
    def test_basic_queue(self):
        result = run_queue_simulation(
            arrival_rate=10,
            service_rate=12,
            duration=50,
            seed=42,
        )
        assert isinstance(result, SimulationResult)
        assert result.summary.total_events_processed > 0
        assert result.latency is not None
        assert result.latency.count() > 0

    def test_mmc_queue(self):
        result = run_queue_simulation(
            arrival_rate=50,
            service_rate=12,
            servers=5,
            duration=50,
            seed=42,
        )
        assert isinstance(result, SimulationResult)
        assert result.summary.total_events_processed > 0

    def test_with_seed_runs_successfully(self):
        """Seeded run should produce valid results."""
        result = run_queue_simulation(
            arrival_rate=10,
            service_rate=12,
            duration=20,
            seed=123,
        )
        assert result.summary.total_events_processed > 0
        assert result.latency.count() > 0


class TestRunPipelineSimulation:
    def test_two_stage_pipeline(self):
        result = run_pipeline_simulation(
            stages=[
                {"name": "WebServer", "concurrency": 4, "service_time": 0.05},
                {"name": "Database", "concurrency": 2, "service_time": 0.02},
            ],
            source_rate=10,
            duration=50,
            seed=42,
        )
        assert isinstance(result, SimulationResult)
        assert "WebServer" in result.queue_depth
        assert "Database" in result.queue_depth
        assert result.latency.count() > 0


class TestFormatResponse:
    def test_format_response_valid_json(self):
        result = run_queue_simulation(
            arrival_rate=10,
            service_rate=12,
            duration=20,
            seed=42,
        )
        text = format_response(result)
        parsed = json.loads(text)
        assert "prompt_context" in parsed
        assert "data" in parsed
        assert "summary" in parsed["data"]


class TestSweepViaInternals:
    def test_sweep_produces_sweep_result(self):
        values = [6, 8, 10]
        results = [
            run_queue_simulation(
                arrival_rate=rate,
                service_rate=12,
                duration=30,
                seed=42,
            )
            for rate in values
        ]
        sweep = SweepResult(
            parameter_name="arrival_rate",
            parameter_values=values,
            results=results,
        )
        text = sweep.to_prompt_context()
        assert "arrival_rate" in text
        assert len(sweep.results) == 3


class TestFormatDistributions:
    def test_format_distributions(self):
        text = format_distributions()
        assert "ConstantLatency" in text
        assert "ExponentialLatency" in text
        assert "UniformDistribution" in text

    def test_distributions_info_is_list(self):
        assert isinstance(DISTRIBUTIONS_INFO, list)
        assert len(DISTRIBUTIONS_INFO) >= 4
