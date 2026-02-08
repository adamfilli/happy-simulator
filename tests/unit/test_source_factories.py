"""Unit tests for Source factory methods.

Tests Source.constant(), .poisson(), .with_profile(), _SimpleEventProvider,
stop_after behavior, and backward compatibility with the manual constructor.
"""

from __future__ import annotations

import pytest

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
    SpikeProfile,
)
from happysimulator.load.source import _SimpleEventProvider
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider


# ---------------------------------------------------------------------------
# Helper: simple collector entity
# ---------------------------------------------------------------------------

class Collector(Entity):
    """Collects received events for test assertions."""

    def __init__(self):
        super().__init__("Collector")
        self.received: list[Event] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.received.append(event)
        return []


# ---------------------------------------------------------------------------
# _SimpleEventProvider
# ---------------------------------------------------------------------------

class TestSimpleEventProvider:

    def test_generates_event_with_correct_fields(self):
        target = Collector()
        provider = _SimpleEventProvider(target, "Request")
        t = Instant.from_seconds(1.0)

        events = provider.get_events(t)

        assert len(events) == 1
        evt = events[0]
        assert evt.event_type == "Request"
        assert evt.target is target
        assert evt.time == t
        assert evt.context["created_at"] == t
        assert evt.context["request_id"] == 1

    def test_auto_increments_request_id(self):
        target = Collector()
        provider = _SimpleEventProvider(target, "Ping")
        t = Instant.from_seconds(0.0)

        events1 = provider.get_events(t)
        events2 = provider.get_events(t)
        events3 = provider.get_events(t)

        assert events1[0].context["request_id"] == 1
        assert events2[0].context["request_id"] == 2
        assert events3[0].context["request_id"] == 3

    def test_custom_event_type(self):
        target = Collector()
        provider = _SimpleEventProvider(target, "HealthCheck")

        events = provider.get_events(Instant.Epoch)
        assert events[0].event_type == "HealthCheck"

    def test_stop_after_returns_empty(self):
        target = Collector()
        stop = Instant.from_seconds(5.0)
        provider = _SimpleEventProvider(target, "Request", stop_after=stop)

        # Before stop_after: should generate
        assert len(provider.get_events(Instant.from_seconds(4.0))) == 1

        # At stop_after: should generate (stop_after is exclusive via >)
        assert len(provider.get_events(Instant.from_seconds(5.0))) == 1

        # After stop_after: should stop
        assert len(provider.get_events(Instant.from_seconds(5.1))) == 0

    def test_no_stop_after_generates_indefinitely(self):
        target = Collector()
        provider = _SimpleEventProvider(target, "Request", stop_after=None)

        for i in range(100):
            events = provider.get_events(Instant.from_seconds(float(i)))
            assert len(events) == 1


# ---------------------------------------------------------------------------
# Source.constant()
# ---------------------------------------------------------------------------

class TestSourceConstant:

    def test_creates_source_instance(self):
        target = Collector()
        source = Source.constant(rate=10, target=target)

        assert isinstance(source, Source)

    def test_uses_constant_arrival_provider(self):
        target = Collector()
        source = Source.constant(rate=10, target=target)

        assert isinstance(source._time_provider, ConstantArrivalTimeProvider)

    def test_default_name(self):
        target = Collector()
        source = Source.constant(rate=10, target=target)

        assert source.name == "Source"

    def test_custom_name(self):
        target = Collector()
        source = Source.constant(rate=10, target=target, name="Traffic")

        assert source.name == "Traffic"

    def test_generates_events_in_simulation(self):
        target = Collector()
        source = Source.constant(rate=10, target=target)

        sim = Simulation(
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        # At rate=10 over 1 second, expect 10 events
        assert len(target.received) == 10

    def test_custom_event_type(self):
        target = Collector()
        source = Source.constant(rate=5, target=target, event_type="HealthCheck")

        sim = Simulation(
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        assert all(e.event_type == "HealthCheck" for e in target.received)

    def test_stop_after_float(self):
        target = Collector()
        source = Source.constant(rate=10, target=target, stop_after=0.5)

        sim = Simulation(
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        # Should generate events only for the first 0.5s
        assert len(target.received) == 5

    def test_stop_after_instant(self):
        target = Collector()
        stop = Instant.from_seconds(0.5)
        source = Source.constant(rate=10, target=target, stop_after=stop)

        sim = Simulation(
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        assert len(target.received) == 5

    def test_events_have_created_at_and_request_id(self):
        target = Collector()
        source = Source.constant(rate=2, target=target)

        sim = Simulation(
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        for i, evt in enumerate(target.received, start=1):
            assert "created_at" in evt.context
            assert evt.context["request_id"] == i


# ---------------------------------------------------------------------------
# Source.poisson()
# ---------------------------------------------------------------------------

class TestSourcePoisson:

    def test_creates_source_instance(self):
        target = Collector()
        source = Source.poisson(rate=10, target=target)

        assert isinstance(source, Source)

    def test_uses_poisson_arrival_provider(self):
        target = Collector()
        source = Source.poisson(rate=10, target=target)

        assert isinstance(source._time_provider, PoissonArrivalTimeProvider)

    def test_generates_events_in_simulation(self):
        target = Collector()
        source = Source.poisson(rate=100, target=target)

        sim = Simulation(
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        # Poisson with rate=100 over 1s should generate ~100 events (stochastic)
        assert 50 < len(target.received) < 200

    def test_stop_after(self):
        target = Collector()
        source = Source.poisson(rate=100, target=target, stop_after=0.5)

        sim = Simulation(
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        # All received events should have been created at or before t=0.5
        for evt in target.received:
            assert evt.context["created_at"] <= Instant.from_seconds(0.5)


# ---------------------------------------------------------------------------
# Source.with_profile()
# ---------------------------------------------------------------------------

class TestSourceWithProfile:

    def test_with_spike_profile_poisson(self):
        target = Collector()
        profile = SpikeProfile(
            baseline_rate=10.0,
            spike_rate=50.0,
            warmup_s=1.0,
            spike_duration_s=1.0,
        )
        source = Source.with_profile(
            profile=profile, target=target, poisson=True
        )

        assert isinstance(source._time_provider, PoissonArrivalTimeProvider)

    def test_with_spike_profile_constant(self):
        target = Collector()
        profile = SpikeProfile(
            baseline_rate=10.0,
            spike_rate=50.0,
            warmup_s=1.0,
            spike_duration_s=1.0,
        )
        source = Source.with_profile(
            profile=profile, target=target, poisson=False
        )

        assert isinstance(source._time_provider, ConstantArrivalTimeProvider)

    def test_generates_events_in_simulation(self):
        target = Collector()
        profile = SpikeProfile(
            baseline_rate=10.0,
            spike_rate=50.0,
            warmup_s=1.0,
            spike_duration_s=1.0,
        )
        source = Source.with_profile(
            profile=profile, target=target, poisson=False, stop_after=3.0
        )

        sim = Simulation(
            end_time=Instant.from_seconds(4.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        # warmup 1s @ 10/s = 10, spike 1s @ 50/s = 50, recovery 1s @ 10/s = 10
        assert len(target.received) == 70

    def test_custom_name_and_event_type(self):
        target = Collector()
        profile = SpikeProfile()
        source = Source.with_profile(
            profile=profile,
            target=target,
            name="LoadGen",
            event_type="Query",
        )

        assert source.name == "LoadGen"
        assert isinstance(source._event_provider, _SimpleEventProvider)
        assert source._event_provider._event_type == "Query"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_manual_constructor_still_works(self):
        """The original 4-object construction pattern must still work."""
        from happysimulator import (
            ConstantArrivalTimeProvider,
            ConstantRateProfile,
            EventProvider,
        )

        target = Collector()

        class MyProvider(EventProvider):
            def get_events(self, time: Instant) -> list[Event]:
                return [
                    Event(
                        time=time,
                        event_type="Custom",
                        target=target,
                        context={"custom_field": "value"},
                    )
                ]

        provider = MyProvider()
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=5), start_time=Instant.Epoch
        )
        source = Source(
            name="ManualSource",
            event_provider=provider,
            arrival_time_provider=arrival,
        )

        sim = Simulation(
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[target],
        )
        sim.run()

        assert len(target.received) == 5
        assert all(e.context["custom_field"] == "value" for e in target.received)


# ---------------------------------------------------------------------------
# _resolve_stop_after
# ---------------------------------------------------------------------------

class TestResolveStopAfter:

    def test_none_returns_none(self):
        assert Source._resolve_stop_after(None) is None

    def test_float_returns_instant(self):
        result = Source._resolve_stop_after(10.0)
        assert isinstance(result, Instant)
        assert result == Instant.from_seconds(10.0)

    def test_instant_passthrough(self):
        instant = Instant.from_seconds(42.0)
        result = Source._resolve_stop_after(instant)
        assert result is instant
