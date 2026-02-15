"""Tests for LatencyTracker and ThroughputTracker."""

import pytest

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)
from happysimulator.instrumentation.collectors import LatencyTracker, ThroughputTracker
from happysimulator.load.event_provider import EventProvider


class _SimpleEventProvider(EventProvider):
    """Generates events targeting a specific entity."""

    def __init__(self, target: Entity):
        self._target = target

    def get_events(self, time: Instant) -> list[Event]:
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={"created_at": time},
            )
        ]


class TestLatencyTracker:
    def test_records_latency(self):
        """Latency should be event.time - created_at."""
        tracker = LatencyTracker("test")
        tracker.set_clock(type("Clock", (), {"now": Instant.from_seconds(1.0)})())

        created = Instant.from_seconds(0.5)
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Completed",
            target=tracker,
            context={"created_at": created},
        )
        tracker.handle_event(event)

        assert tracker.count == 1
        assert tracker.data.count() == 1
        assert tracker.data.raw_values()[0] == pytest.approx(0.5, abs=0.001)

    def test_p50_p99(self):
        tracker = LatencyTracker("test")
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())

        for i in range(100):
            t = Instant.from_seconds(float(i))
            created = Instant.from_seconds(float(i) - float(i) * 0.01)
            event = Event(
                time=t,
                event_type="Done",
                target=tracker,
                context={"created_at": created},
            )
            tracker.handle_event(event)

        assert tracker.count == 100
        assert tracker.p50() >= 0
        assert tracker.p99() >= tracker.p50()

    def test_mean_latency(self):
        tracker = LatencyTracker("test")
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())

        # Two events: latency 0.1s and 0.3s
        for latency_s, t_s in [(0.1, 1.0), (0.3, 2.0)]:
            event = Event(
                time=Instant.from_seconds(t_s),
                event_type="Done",
                target=tracker,
                context={"created_at": Instant.from_seconds(t_s - latency_s)},
            )
            tracker.handle_event(event)

        assert tracker.mean_latency() == pytest.approx(0.2, abs=0.001)

    def test_summary_bucketing(self):
        tracker = LatencyTracker("test")
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())

        # 3 events in first second, 2 in second
        for t_s in [0.1, 0.5, 0.9, 1.1, 1.5]:
            event = Event(
                time=Instant.from_seconds(t_s),
                event_type="Done",
                target=tracker,
                context={"created_at": Instant.from_seconds(t_s - 0.05)},
            )
            tracker.handle_event(event)

        b = tracker.summary(window_s=1.0)
        assert len(b) == 2
        assert b.counts() == [3, 2]

    def test_returns_empty_events(self):
        tracker = LatencyTracker()
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Done",
            target=tracker,
            context={"created_at": Instant.Epoch},
        )
        result = tracker.handle_event(event)
        assert result == []

    def test_in_simulation(self):
        """LatencyTracker works as a downstream entity in a simulation."""
        tracker = LatencyTracker("Sink")
        provider = _SimpleEventProvider(tracker)
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=10.0), start_time=Instant.Epoch
        )
        source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[tracker],
        )
        sim.run()

        assert tracker.count >= 9  # ~10 events in 1s


class TestThroughputTracker:
    def test_counts_events(self):
        tracker = ThroughputTracker("test")
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(float(i) * 0.1),
                event_type="Request",
                target=tracker,
            )
            tracker.handle_event(event)

        assert tracker.count == 5
        assert tracker.data.count() == 5

    def test_throughput_bucketing(self):
        tracker = ThroughputTracker("test")
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())

        # 3 events in [0,1), 2 events in [1,2)
        for t_s in [0.1, 0.3, 0.9, 1.1, 1.8]:
            event = Event(
                time=Instant.from_seconds(t_s),
                event_type="Request",
                target=tracker,
            )
            tracker.handle_event(event)

        b = tracker.throughput(window_s=1.0)
        assert len(b) == 2
        assert b.sums() == [3.0, 2.0]

    def test_returns_empty_events(self):
        tracker = ThroughputTracker()
        tracker.set_clock(type("Clock", (), {"now": Instant.Epoch})())
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Request",
            target=tracker,
        )
        result = tracker.handle_event(event)
        assert result == []
