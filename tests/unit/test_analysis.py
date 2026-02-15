"""Tests for the analysis package: phases, trace_analysis, and report."""

import pytest

from happysimulator.analysis.phases import Phase, detect_phases
from happysimulator.analysis.report import (
    Anomaly,
    CausalChain,
    MetricSummary,
    SimulationAnalysis,
    analyze,
)
from happysimulator.analysis.trace_analysis import (
    EventLifecycle,
    list_event_lifecycles,
    trace_event_lifecycle,
)
from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.recorder import InMemoryTraceRecorder
from happysimulator.instrumentation.summary import SimulationSummary

# === Helpers ===


def _make_data(pairs: list[tuple[float, float]]) -> Data:
    d = Data()
    for t, v in pairs:
        d.add_stat(v, Instant.from_seconds(t))
    return d


def _make_summary(**kwargs) -> SimulationSummary:
    defaults = {
        "duration_s": 100.0,
        "total_events_processed": 1000,
        "events_per_second": 10.0,
        "wall_clock_seconds": 1.0,
    }
    defaults.update(kwargs)
    return SimulationSummary(**defaults)


# === Phase Detection ===


class TestDetectPhases:
    def test_stable_data_single_phase(self):
        """Constant data should produce a single stable phase."""
        d = _make_data([(float(i), 10.0) for i in range(100)])
        phases = detect_phases(d, window_s=10.0)
        assert len(phases) >= 1
        assert all(p.label == "stable" for p in phases)

    def test_two_distinct_levels(self):
        """Data with a clear level shift should produce multiple phases."""
        pairs = [(float(i), 10.0) for i in range(50)]
        pairs += [(float(i), 100.0) for i in range(50, 100)]
        d = _make_data(pairs)
        phases = detect_phases(d, window_s=5.0, threshold=2.0)
        assert len(phases) >= 2

    def test_empty_data(self):
        phases = detect_phases(Data())
        assert phases == []

    def test_single_sample(self):
        d = _make_data([(0.0, 42.0)])
        phases = detect_phases(d)
        assert len(phases) == 0 or len(phases) == 1

    def test_phase_properties(self):
        d = _make_data([(float(i), 10.0) for i in range(50)])
        phases = detect_phases(d, window_s=10.0)
        for p in phases:
            assert p.duration_s > 0
            assert p.start_s < p.end_s
            assert isinstance(p.label, str)

    def test_phase_to_dict(self):
        p = Phase(start_s=0.0, end_s=10.0, mean=5.0, std=1.0, label="stable")
        d = p.to_dict()
        assert d["start_s"] == 0.0
        assert d["end_s"] == 10.0
        assert d["duration_s"] == 10.0
        assert d["label"] == "stable"

    def test_phase_duration(self):
        p = Phase(start_s=5.0, end_s=15.0, mean=1.0, std=0.1, label="stable")
        assert p.duration_s == 10.0

    def test_spike_detection(self):
        """A sudden spike should trigger a phase transition."""
        pairs = [(float(i), 10.0) for i in range(30)]
        pairs += [(float(i), 200.0) for i in range(30, 40)]
        pairs += [(float(i), 10.0) for i in range(40, 70)]
        d = _make_data(pairs)
        phases = detect_phases(d, window_s=5.0, threshold=2.0)
        # Should detect the spike as a separate phase
        assert len(phases) >= 2
        labels = [p.label for p in phases]
        assert any(l in ("degraded", "overloaded") for l in labels)


# === Trace Analysis ===


class TestTraceAnalysis:
    def test_event_lifecycle_basic(self):
        recorder = InMemoryTraceRecorder()
        recorder.record(
            time=Instant.from_seconds(1.0),
            kind="simulation.schedule",
            event_id="evt-1",
            event_type="Request",
        )
        recorder.record(
            time=Instant.from_seconds(1.5),
            kind="simulation.dequeue",
            event_id="evt-1",
            event_type="Request",
        )

        lc = trace_event_lifecycle(recorder, "evt-1")
        assert lc is not None
        assert lc.event_id == "evt-1"
        assert lc.event_type == "Request"
        assert lc.scheduled_at == Instant.from_seconds(1.0)
        assert lc.dequeued_at == Instant.from_seconds(1.5)

    def test_event_lifecycle_wait_time(self):
        recorder = InMemoryTraceRecorder()
        recorder.record(
            time=Instant.from_seconds(1.0),
            kind="simulation.schedule",
            event_id="evt-1",
            event_type="Request",
        )
        recorder.record(
            time=Instant.from_seconds(3.0),
            kind="simulation.dequeue",
            event_id="evt-1",
        )

        lc = trace_event_lifecycle(recorder, "evt-1")
        assert lc.wait_time is not None
        assert lc.wait_time.to_seconds() == pytest.approx(2.0)

    def test_event_lifecycle_not_found(self):
        recorder = InMemoryTraceRecorder()
        lc = trace_event_lifecycle(recorder, "nonexistent")
        assert lc is None

    def test_event_lifecycle_to_dict(self):
        lc = EventLifecycle(
            event_id="evt-1",
            event_type="Request",
            scheduled_at=Instant.from_seconds(1.0),
            dequeued_at=Instant.from_seconds(2.0),
        )
        d = lc.to_dict()
        assert d["event_id"] == "evt-1"
        assert d["scheduled_at_s"] == 1.0
        assert d["dequeued_at_s"] == 2.0
        assert d["wait_time_s"] == 1.0

    def test_event_lifecycle_str(self):
        lc = EventLifecycle(event_id="evt-1", event_type="Request")
        text = str(lc)
        assert "evt-1" in text
        assert "Request" in text

    def test_list_event_lifecycles(self):
        recorder = InMemoryTraceRecorder()
        for i in range(5):
            recorder.record(
                time=Instant.from_seconds(float(i)),
                kind="simulation.schedule",
                event_id=f"evt-{i}",
                event_type="Request",
            )

        lifecycles = list_event_lifecycles(recorder)
        assert len(lifecycles) == 5

    def test_list_event_lifecycles_filtered(self):
        recorder = InMemoryTraceRecorder()
        recorder.record(
            time=Instant.Epoch,
            kind="simulation.schedule",
            event_id="req-1",
            event_type="Request",
        )
        recorder.record(
            time=Instant.Epoch,
            kind="simulation.schedule",
            event_id="resp-1",
            event_type="Response",
        )

        requests = list_event_lifecycles(recorder, event_type="Request")
        assert len(requests) == 1
        assert requests[0].event_id == "req-1"


# === Report / analyze() ===


class TestMetricSummary:
    def test_to_dict(self):
        ms = MetricSummary(
            name="latency",
            count=100,
            mean=0.05,
            std=0.02,
            min=0.01,
            max=0.2,
            p50=0.04,
            p95=0.09,
            p99=0.15,
        )
        d = ms.to_dict()
        assert d["name"] == "latency"
        assert d["count"] == 100


class TestAnomaly:
    def test_to_dict(self):
        a = Anomaly(
            time_s=55.0,
            metric="latency",
            description="spike",
            severity="warning",
            context={"value": 0.5},
        )
        d = a.to_dict()
        assert d["time_s"] == 55.0
        assert d["severity"] == "warning"


class TestCausalChain:
    def test_to_dict(self):
        c = CausalChain(
            trigger_description="load spike",
            effects=["queue buildup", "latency increase"],
            duration_s=15.0,
        )
        d = c.to_dict()
        assert d["trigger"] == "load spike"
        assert len(d["effects"]) == 2


class TestAnalyze:
    def test_basic_analysis(self):
        summary = _make_summary()
        latency = _make_data([(float(i), 0.05 + i * 0.001) for i in range(100)])

        result = analyze(summary, latency=latency)

        assert isinstance(result, SimulationAnalysis)
        assert result.summary is summary
        assert "latency" in result.metrics
        assert result.metrics["latency"].count == 100

    def test_named_metrics(self):
        summary = _make_summary()
        custom = _make_data([(float(i), float(i)) for i in range(50)])

        result = analyze(summary, custom_metric=custom)

        assert "custom_metric" in result.metrics
        assert result.metrics["custom_metric"].count == 50

    def test_empty_data(self):
        summary = _make_summary()
        result = analyze(summary)
        assert result.metrics == {}
        assert result.phases == {}

    def test_phase_detection_in_analysis(self):
        """Analysis should detect phases in provided data."""
        pairs = [(float(i), 10.0) for i in range(50)]
        pairs += [(float(i), 200.0) for i in range(50, 100)]
        d = _make_data(pairs)
        summary = _make_summary()

        result = analyze(summary, latency=d, phase_window_s=10.0)

        assert "latency" in result.phases
        assert len(result.phases["latency"]) >= 2

    def test_per_phase_metric_breakdown(self):
        """MetricSummary should include per-phase breakdown."""
        pairs = [(float(i), 10.0) for i in range(50)]
        pairs += [(float(i), 200.0) for i in range(50, 100)]
        d = _make_data(pairs)
        summary = _make_summary()

        result = analyze(summary, latency=d, phase_window_s=10.0)

        ms = result.metrics["latency"]
        assert len(ms.by_phase) > 0

    def test_anomaly_detection(self):
        """Large deviations should be detected as anomalies."""
        # Normal values with a spike
        pairs = [(float(i), 10.0) for i in range(100)]
        # Insert anomalous window
        for i in range(50, 55):
            pairs[i] = (float(i), 500.0)
        d = _make_data(pairs)
        summary = _make_summary()

        result = analyze(summary, latency=d, anomaly_threshold=2.0)

        # Should detect the spike as an anomaly
        assert len(result.anomalies) > 0

    def test_to_dict(self):
        summary = _make_summary()
        latency = _make_data([(float(i), float(i) * 0.01) for i in range(50)])
        result = analyze(summary, latency=latency)

        d = result.to_dict()
        assert "summary" in d
        assert "metrics" in d
        assert "anomalies" in d
        assert "causal_chains" in d


class TestSimulationAnalysis:
    def test_to_prompt_context(self):
        summary = _make_summary()
        pairs = [(float(i), 10.0) for i in range(50)]
        pairs += [(float(i), 200.0) for i in range(50, 100)]
        latency = _make_data(pairs)

        result = analyze(summary, latency=latency)
        ctx = result.to_prompt_context()

        assert "Simulation Summary" in ctx
        assert "100.00s" in ctx

    def test_to_prompt_context_respects_max_tokens(self):
        summary = _make_summary()
        # Create a lot of data to test truncation
        latency = _make_data([(float(i), float(i)) for i in range(1000)])

        result = analyze(summary, latency=latency)
        ctx = result.to_prompt_context(max_tokens=100)

        assert len(ctx) <= 400 + 20  # 100 * 4 chars + truncation marker

    def test_to_prompt_context_includes_anomalies(self):
        summary = _make_summary()
        analysis = SimulationAnalysis(
            summary=summary,
            anomalies=[
                Anomaly(
                    time_s=55.0, metric="latency", description="spike detected", severity="warning"
                ),
            ],
        )
        ctx = analysis.to_prompt_context()
        assert "Anomalies" in ctx
        assert "spike detected" in ctx

    def test_to_prompt_context_includes_causal_chains(self):
        summary = _make_summary()
        analysis = SimulationAnalysis(
            summary=summary,
            causal_chains=[
                CausalChain(
                    trigger_description="load spike at t=55s",
                    effects=["queue grew", "latency increased"],
                    duration_s=15.0,
                ),
            ],
        )
        ctx = analysis.to_prompt_context()
        assert "Causal Chains" in ctx
        assert "load spike" in ctx
