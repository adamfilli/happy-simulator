"""Tests for the recommendations engine."""

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
from happysimulator.ai.insights import (
    Recommendation,
    generate_recommendations,
)
from happysimulator.ai.result import SimulationResult
from happysimulator.components.server import Server


def _run_queue(arrival_rate, service_rate, servers=1, duration=100, seed=42):
    """Helper to run a queue simulation and return SimulationResult."""
    random.seed(seed)
    tracker = LatencyTracker("Sink")
    server = Server(
        "Server", concurrency=servers,
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


class TestRecommendation:
    def test_to_dict(self):
        rec = Recommendation(
            category="capacity",
            description="System is overloaded",
            confidence="high",
            suggested_change="Add more servers",
        )
        d = rec.to_dict()
        assert d["category"] == "capacity"
        assert d["confidence"] == "high"


class TestGenerateRecommendations:
    def test_returns_list(self):
        result = _run_queue(8, 12)
        recs = generate_recommendations(result)
        assert isinstance(recs, list)
        assert all(isinstance(r, Recommendation) for r in recs)

    def test_underutilized_system_gets_recommendation(self):
        """Very low arrival rate relative to service rate should be flagged."""
        result = _run_queue(1, 50, duration=50)
        recs = generate_recommendations(result)
        categories = [r.category for r in recs]
        # Should suggest reducing capacity
        assert "capacity" in categories

    def test_saturated_system_gets_recommendation(self):
        """Arrival rate exceeding service rate should flag saturation."""
        result = _run_queue(15, 12, duration=100)
        recs = generate_recommendations(result)
        # Should detect saturation or tail latency
        assert len(recs) > 0

    def test_recommendations_included_in_result(self):
        """from_run() should auto-generate recommendations."""
        result = _run_queue(15, 12, duration=100)
        assert isinstance(result.recommendations, list)

    def test_recommendations_in_prompt_context(self):
        """to_prompt_context() should include recommendations if present."""
        result = _run_queue(15, 12, duration=100)
        if result.recommendations:
            text = result.to_prompt_context()
            assert "Recommendations" in text
