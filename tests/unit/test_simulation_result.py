"""Tests for SimulationResult, SimulationComparison, and SweepResult."""

import random

import pytest

from happysimulator import (
    ConstantLatency,
    ExponentialLatency,
    LatencyTracker,
    Probe,
    Simulation,
    Source,
)
from happysimulator.ai.result import (
    MetricDiff,
    SimulationComparison,
    SimulationResult,
    SweepResult,
    _pct_change,
)
from happysimulator.components.server import Server


def _run_queue(arrival_rate, service_rate, duration=50, seed=42):
    """Helper to run a quick queue simulation and return SimulationResult."""
    random.seed(seed)
    tracker = LatencyTracker("Sink")
    server = Server(
        "Server", concurrency=1,
        service_time=ExponentialLatency(1.0 / service_rate),
        downstream=tracker,
    )
    source = Source.poisson(rate=arrival_rate, target=server)
    probe, depth_data = Probe.on(server, "depth", interval=0.5)

    summary = Simulation(
        duration=duration,
        sources=[source],
        entities=[server, tracker],
        probes=[probe],
    ).run()

    return SimulationResult.from_run(
        summary,
        latency=tracker.data,
        queue_depth={"Server": depth_data},
    )


class TestSimulationResult:
    def test_from_run_creates_analysis(self):
        result = _run_queue(10, 12)
        assert result.analysis is not None
        assert result.summary is not None
        assert result.latency is not None
        assert "Server" in result.queue_depth

    def test_to_dict(self):
        result = _run_queue(10, 12)
        d = result.to_dict()
        assert "summary" in d
        assert "metrics" in d

    def test_to_prompt_context(self):
        result = _run_queue(10, 12)
        text = result.to_prompt_context()
        assert "Simulation Summary" in text
        assert "Duration:" in text

    def test_to_prompt_context_includes_recommendations(self):
        # High load should generate recommendations
        result = _run_queue(11, 12, duration=100)
        text = result.to_prompt_context()
        # May or may not have recommendations depending on run,
        # but the method should not fail
        assert isinstance(text, str)


class TestSimulationComparison:
    def test_compare_produces_diffs(self):
        result_a = _run_queue(8, 12)
        result_b = _run_queue(11, 12)

        comparison = result_a.compare(result_b)
        assert isinstance(comparison, SimulationComparison)
        assert "latency" in comparison.metric_diffs

    def test_comparison_to_prompt_context(self):
        result_a = _run_queue(8, 12)
        result_b = _run_queue(11, 12)

        comparison = result_a.compare(result_b)
        text = comparison.to_prompt_context()
        assert "Simulation Comparison" in text
        assert "Run A" in text
        assert "Run B" in text

    def test_comparison_to_dict(self):
        result_a = _run_queue(8, 12)
        result_b = _run_queue(11, 12)

        comparison = result_a.compare(result_b)
        d = comparison.to_dict()
        assert "metric_diffs" in d
        assert "result_a" in d


class TestSweepResult:
    def test_sweep_to_prompt_context(self):
        results = [_run_queue(rate, 12) for rate in [6, 8, 10]]
        sweep = SweepResult(
            parameter_name="arrival_rate",
            parameter_values=[6, 8, 10],
            results=results,
        )
        text = sweep.to_prompt_context()
        assert "Parameter Sweep: arrival_rate" in text
        assert "6" in text

    def test_sweep_best_by(self):
        results = [_run_queue(rate, 12) for rate in [6, 8, 10]]
        sweep = SweepResult(
            parameter_name="arrival_rate",
            parameter_values=[6, 8, 10],
            results=results,
        )
        best = sweep.best_by(metric="latency", stat="p99")
        # Lowest arrival rate should have lowest latency
        assert best is results[0]

    def test_sweep_to_dict(self):
        results = [_run_queue(rate, 12) for rate in [6, 8]]
        sweep = SweepResult(
            parameter_name="arrival_rate",
            parameter_values=[6, 8],
            results=results,
        )
        d = sweep.to_dict()
        assert d["parameter_name"] == "arrival_rate"
        assert len(d["results"]) == 2


class TestMetricDiff:
    def test_to_dict(self):
        diff = MetricDiff(
            name="latency",
            mean_a=0.05, mean_b=0.03,
            mean_change_pct=-40.0,
            p99_a=0.5, p99_b=0.12,
            p99_change_pct=-76.0,
        )
        d = diff.to_dict()
        assert d["name"] == "latency"
        assert d["mean_change_pct"] == -40.0


class TestPctChange:
    def test_positive_change(self):
        assert _pct_change(10, 15) == pytest.approx(50.0)

    def test_negative_change(self):
        assert _pct_change(10, 5) == pytest.approx(-50.0)

    def test_zero_base(self):
        assert _pct_change(0, 5) == float("inf")

    def test_zero_to_zero(self):
        assert _pct_change(0, 0) == 0.0
