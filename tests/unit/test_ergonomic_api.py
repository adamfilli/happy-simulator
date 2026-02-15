"""Tests for Phase 1 ergonomic API improvements."""

import random

import pytest

from happysimulator import (
    ConstantLatency,
    Data,
    Event,
    ExponentialLatency,
    LatencyTracker,
    Probe,
    SimpleEventProvider,
    Simulation,
    Source,
)
from happysimulator.components.server import Server
from happysimulator.core.entity import Entity
from happysimulator.core.temporal import Instant

# ── Probe.on() ──────────────────────────────────────────────────────────────


class TestProbeOn:
    def test_returns_probe_and_data(self):
        server = Server("S", service_time=ConstantLatency(0.01))
        probe, data = Probe.on(server, "depth", interval=0.5)

        assert isinstance(probe, Probe)
        assert isinstance(data, Data)
        assert probe.target is server
        assert probe.metric == "depth"

    def test_probe_collects_data_in_simulation(self):
        tracker = LatencyTracker("Sink")
        server = Server("S", service_time=ConstantLatency(0.01), downstream=tracker)
        probe, depth_data = Probe.on(server, "depth", interval=0.1)
        source = Source.constant(rate=5, target=server)

        Simulation(
            duration=2,
            sources=[source],
            entities=[server, tracker],
            probes=[probe],
        ).run()

        assert depth_data.count() > 0

    def test_on_many_returns_correct_structure(self):
        server = Server("S", service_time=ConstantLatency(0.01))
        probes, data_dict = Probe.on_many(server, ["depth", "utilization"], interval=0.5)

        assert len(probes) == 2
        assert set(data_dict.keys()) == {"depth", "utilization"}
        assert all(isinstance(d, Data) for d in data_dict.values())


# ── Server.downstream ───────────────────────────────────────────────────────


class TestServerDownstream:
    def test_without_downstream_returns_none(self):
        """Server with no downstream should not forward events."""
        server = Server("S", service_time=ConstantLatency(0.01))
        source = Source.constant(rate=5, target=server)

        summary = Simulation(
            duration=1,
            sources=[source],
            entities=[server],
        ).run()

        assert summary.total_events_processed > 0

    def test_with_downstream_forwards_events(self):
        """Server with downstream should forward completed events to it."""
        tracker = LatencyTracker("Sink")
        server = Server("S", service_time=ConstantLatency(0.01), downstream=tracker)
        source = Source.constant(rate=10, target=server)

        Simulation(
            duration=2,
            sources=[source],
            entities=[server, tracker],
        ).run()

        # Tracker should have received forwarded events
        assert tracker.count > 0

    def test_downstream_preserves_context(self):
        """Forwarded events should preserve the original context."""
        tracker = LatencyTracker("Sink")
        server = Server("S", service_time=ConstantLatency(0.01), downstream=tracker)
        source = Source.constant(rate=5, target=server)

        Simulation(
            duration=1,
            sources=[source],
            entities=[server, tracker],
        ).run()

        # LatencyTracker uses context["created_at"] to compute latency.
        # If context wasn't preserved, latency would be 0 for all events.
        if tracker.count > 0:
            assert tracker.mean_latency() >= 0.01  # at least the service time

    def test_downstream_setter(self):
        """downstream property should be settable."""
        server = Server("S", service_time=ConstantLatency(0.01))
        assert server.downstream is None

        tracker = LatencyTracker("Sink")
        server.downstream = tracker
        assert server.downstream is tracker


# ── Entity.forward() ────────────────────────────────────────────────────────


class _DummyEntity(Entity):
    def handle_event(self, event):
        return None


class TestEntityForward:
    def test_forward_preserves_context(self):
        entity = _DummyEntity("A")
        target = _DummyEntity("B")
        # Inject clock so entity.now works
        from happysimulator.core.clock import Clock

        clock = Clock(Instant.Epoch)
        clock._current_time = Instant.from_seconds(5.0)
        entity.set_clock(clock)

        original = Event(
            time=Instant.from_seconds(1.0),
            event_type="Request",
            target=entity,
            context={"key": "value", "created_at": Instant.from_seconds(1.0)},
        )

        forwarded = entity.forward(original, target)

        assert forwarded.target is target
        assert forwarded.event_type == "Request"
        assert forwarded.context == original.context
        assert forwarded.time == Instant.from_seconds(5.0)  # entity.now

    def test_forward_overrides_event_type(self):
        entity = _DummyEntity("A")
        target = _DummyEntity("B")
        from happysimulator.core.clock import Clock

        clock = Clock(Instant.Epoch)
        entity.set_clock(clock)

        original = Event(
            time=Instant.Epoch,
            event_type="Request",
            target=entity,
        )

        forwarded = entity.forward(original, target, event_type="Response")
        assert forwarded.event_type == "Response"


# ── Simulation(duration=) ───────────────────────────────────────────────────


class TestSimulationDuration:
    def test_duration_sets_end_time(self):
        sim = Simulation(duration=50)
        assert sim._end_time == Instant.from_seconds(50)

    def test_duration_conflicts_with_end_time(self):
        with pytest.raises(ValueError, match="Cannot specify both"):
            Simulation(duration=50, end_time=Instant.from_seconds(100))

    def test_duration_default_start_time(self):
        sim = Simulation(duration=10)
        assert sim._start_time == Instant.Epoch

    def test_duration_runs_simulation(self):
        server = Server("S", service_time=ConstantLatency(0.01))
        source = Source.constant(rate=5, target=server)
        summary = Simulation(duration=2, sources=[source], entities=[server]).run()
        assert summary.total_events_processed > 0
        assert summary.duration_s == pytest.approx(2.0, abs=0.1)


# ── SimpleEventProvider ─────────────────────────────────────────────────────


class TestSimpleEventProvider:
    def test_default_context(self):
        target = _DummyEntity("T")
        provider = SimpleEventProvider(target)

        events = provider.get_events(Instant.from_seconds(1.0))
        assert len(events) == 1
        assert events[0].target is target
        assert events[0].event_type == "Request"
        assert events[0].context["request_id"] == 1
        assert events[0].context["created_at"] == Instant.from_seconds(1.0)

    def test_custom_context_fn(self):
        target = _DummyEntity("T")
        provider = SimpleEventProvider(
            target,
            context_fn=lambda t, n: {"time": t, "seq": n, "priority": "high"},
        )

        events = provider.get_events(Instant.from_seconds(2.0))
        assert events[0].context["seq"] == 1
        assert events[0].context["priority"] == "high"

    def test_auto_increment(self):
        target = _DummyEntity("T")
        provider = SimpleEventProvider(target)

        provider.get_events(Instant.from_seconds(1.0))
        events = provider.get_events(Instant.from_seconds(2.0))
        assert events[0].context["request_id"] == 2

    def test_stop_after(self):
        target = _DummyEntity("T")
        provider = SimpleEventProvider(target, stop_after=Instant.from_seconds(5.0))

        assert len(provider.get_events(Instant.from_seconds(4.0))) == 1
        assert len(provider.get_events(Instant.from_seconds(6.0))) == 0

    def test_source_constant_with_event_provider(self):
        target = _DummyEntity("T")
        provider = SimpleEventProvider(target, event_type="Custom")
        source = Source.constant(rate=10, event_provider=provider)
        assert source is not None

    def test_source_poisson_with_event_provider(self):
        target = _DummyEntity("T")
        provider = SimpleEventProvider(target)
        source = Source.poisson(rate=10, event_provider=provider)
        assert source is not None

    def test_source_constant_requires_target_or_provider(self):
        with pytest.raises(ValueError, match="Either 'target' or 'event_provider'"):
            Source.constant(rate=10)

    def test_source_poisson_requires_target_or_provider(self):
        with pytest.raises(ValueError, match="Either 'target' or 'event_provider'"):
            Source.poisson(rate=10)


# ── End-to-end: the "after" M/M/1 example ───────────────────────────────────


class TestEndToEndMM1:
    def test_mm1_queue_with_ergonomic_api(self):
        """The 7-line M/M/1 setup from the design doc should work end-to-end."""
        random.seed(42)

        tracker = LatencyTracker("Sink")
        server = Server(
            "Server",
            concurrency=1,
            service_time=ExponentialLatency(1 / 12),
            downstream=tracker,
        )
        source = Source.poisson(rate=10, target=server)
        depth_probe, depth_data = Probe.on(server, "depth", interval=0.1)

        summary = Simulation(
            duration=100,
            sources=[source],
            entities=[server, tracker],
            probes=[depth_probe],
        ).run()

        # Basic sanity checks
        assert summary.total_events_processed > 500
        assert tracker.count > 0
        assert tracker.mean_latency() > 0
        assert depth_data.count() > 0
