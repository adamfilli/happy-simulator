"""Tests for SimulationSummary and EntitySummary."""

import pytest

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    FIFOQueue,
    Instant,
    QueuedResource,
    Simulation,
    Source,
)
from happysimulator.instrumentation.summary import (
    EntitySummary,
    QueueStats,
    SimulationSummary,
)
from happysimulator.load.event_provider import EventProvider
from typing import Generator


class _CountingSink(Entity):
    """Simple sink that counts events."""

    def __init__(self, name: str = "Sink"):
        super().__init__(name)
        self.count: int = 0

    def handle_event(self, event: Event) -> list[Event]:
        self.count += 1
        return []


class _SimpleServer(QueuedResource):
    """Minimal queued resource for testing."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.stats_processed: int = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield 0.01
        self.stats_processed += 1
        return [Event(
            time=self.now,
            event_type="Done",
            target=self.downstream,
            context=event.context,
        )]


class _SimpleProvider(EventProvider):
    def __init__(self, target: Entity):
        self._target = target

    def get_events(self, time: Instant) -> list[Event]:
        return [Event(time=time, event_type="Request", target=self._target)]


class TestSimulationSummary:
    def test_str_format(self):
        s = SimulationSummary(
            duration_s=10.0,
            total_events_processed=100,
            events_per_second=10.0,
            wall_clock_seconds=0.5,
        )
        text = str(s)
        assert "10.00s" in text
        assert "100" in text

    def test_to_dict(self):
        s = SimulationSummary(
            duration_s=10.0,
            total_events_processed=100,
            events_per_second=10.0,
            wall_clock_seconds=0.5,
        )
        d = s.to_dict()
        assert d["duration_s"] == 10.0
        assert d["total_events_processed"] == 100

    def test_with_entity_summaries(self):
        s = SimulationSummary(
            duration_s=10.0,
            total_events_processed=100,
            events_per_second=10.0,
            wall_clock_seconds=0.5,
            entities={
                "Server": EntitySummary(
                    name="Server",
                    entity_type="MM1Server",
                    events_handled=50,
                    queue_stats=QueueStats(peak_depth=0, total_accepted=50, total_dropped=0),
                ),
            },
        )
        text = str(s)
        assert "Server" in text
        assert "accepted=50" in text

    def test_entity_summary_to_dict(self):
        es = EntitySummary(
            name="Server",
            entity_type="MyServer",
            events_handled=42,
            queue_stats=QueueStats(peak_depth=10, total_accepted=42, total_dropped=3),
        )
        d = es.to_dict()
        assert d["name"] == "Server"
        assert d["queue"]["total_accepted"] == 42
        assert d["queue"]["total_dropped"] == 3

    def test_entity_summary_without_queue(self):
        es = EntitySummary(name="Sink", entity_type="Sink", events_handled=10)
        d = es.to_dict()
        assert "queue" not in d


class TestSimulationRunReturnsSummary:
    def test_run_returns_summary(self):
        sink = _CountingSink("Sink")
        provider = _SimpleProvider(sink)
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=10.0), start_time=Instant.Epoch
        )
        source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[sink],
        )
        summary = sim.run()

        assert isinstance(summary, SimulationSummary)
        assert summary.total_events_processed > 0
        assert summary.duration_s == pytest.approx(1.0, abs=0.2)
        assert summary.wall_clock_seconds > 0
        assert summary.events_per_second > 0

    def test_summary_accessible_after_run(self):
        sink = _CountingSink("Sink")
        provider = _SimpleProvider(sink)
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=10.0), start_time=Instant.Epoch
        )
        source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[sink],
        )
        assert sim.summary is None
        summary = sim.run()
        assert sim.summary is summary

    def test_summary_includes_queued_resource(self):
        sink = _CountingSink("Sink")
        server = _SimpleServer("Server", downstream=sink)
        provider = _SimpleProvider(server)
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=10.0), start_time=Instant.Epoch
        )
        source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[server, sink],
        )
        summary = sim.run()

        assert "Server" in summary.entities
        server_summary = summary.entities["Server"]
        assert server_summary.entity_type == "_SimpleServer"
        assert server_summary.queue_stats is not None
        assert server_summary.queue_stats.total_accepted > 0

    def test_empty_simulation(self):
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
        )
        summary = sim.run()
        assert summary.total_events_processed == 0
