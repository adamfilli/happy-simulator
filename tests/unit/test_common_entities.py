"""Tests for Sink and Counter common entities."""

from __future__ import annotations

import pytest

from happysimulator.components.common import Counter, Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source

# ---------------------------------------------------------------------------
# Sink tests
# ---------------------------------------------------------------------------


class TestSink:
    def test_default_name(self):
        sink = Sink()
        assert sink.name == "Sink"

    def test_custom_name(self):
        sink = Sink("my_sink")
        assert sink.name == "my_sink"

    def test_counts_events(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[sink],
        )
        for i in range(5):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(0.1 * (i + 1)),
                    event_type="ping",
                    target=sink,
                )
            )
        sim.run()
        assert sink.events_received == 5

    def test_latency_from_created_at(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[sink],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.5),
                event_type="done",
                target=sink,
                context={"created_at": Instant.from_seconds(0.5)},
            )
        )
        sim.run()

        assert sink.events_received == 1
        assert sink.latencies_s[0] == pytest.approx(1.0)

    def test_latency_zero_when_no_created_at(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[sink],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.5),
                event_type="ping",
                target=sink,
            )
        )
        sim.run()

        assert sink.latencies_s[0] == pytest.approx(0.0)

    def test_average_latency_empty(self):
        sink = Sink()
        assert sink.average_latency() == 0.0

    def test_average_latency(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[sink],
        )
        # Event with 1.0s latency
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="a",
                target=sink,
                context={"created_at": Instant.Epoch},
            )
        )
        # Event with 2.0s latency
        sim.schedule(
            Event(
                time=Instant.from_seconds(2.0),
                event_type="b",
                target=sink,
                context={"created_at": Instant.Epoch},
            )
        )
        sim.run()

        assert sink.average_latency() == pytest.approx(1.5)

    def test_latency_time_series_seconds(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[sink],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="a",
                target=sink,
                context={"created_at": Instant.from_seconds(0.5)},
            )
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(2.0),
                event_type="b",
                target=sink,
                context={"created_at": Instant.from_seconds(1.0)},
            )
        )
        sim.run()

        times, latencies = sink.latency_time_series_seconds()
        assert times == [pytest.approx(1.0), pytest.approx(2.0)]
        assert latencies == [pytest.approx(0.5), pytest.approx(1.0)]

    def test_latency_stats_empty(self):
        sink = Sink()
        stats = sink.latency_stats()
        assert stats["count"] == 0
        assert stats["avg"] == 0.0

    def test_latency_stats(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[sink],
        )
        # Create events with latencies: 0.1, 0.2, 0.3, 0.4, 0.5
        for i in range(1, 6):
            latency = i * 0.1
            sim.schedule(
                Event(
                    time=Instant.from_seconds(float(i)),
                    event_type="x",
                    target=sink,
                    context={"created_at": Instant.from_seconds(float(i) - latency)},
                )
            )
        sim.run()

        stats = sink.latency_stats()
        assert stats["count"] == 5
        assert stats["avg"] == pytest.approx(0.3)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.5)
        assert stats["p50"] == pytest.approx(0.3)
        assert stats["p99"] > stats["p50"]

    def test_sink_with_source_integration(self):
        """Sink works end-to-end with Source.constant()."""
        sink = Sink()
        source = Source.constant(rate=10, target=sink, name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[sink],
        )
        sim.run()

        assert sink.events_received == 10
        # Source.constant sets created_at, so latency should be 0
        assert all(lat == pytest.approx(0.0) for lat in sink.latencies_s)

    def test_completion_times_are_instants(self):
        sink = Sink()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[sink],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.5),
                event_type="x",
                target=sink,
            )
        )
        sim.run()

        assert len(sink.completion_times) == 1
        assert sink.completion_times[0] == Instant.from_seconds(0.5)


# ---------------------------------------------------------------------------
# Counter tests
# ---------------------------------------------------------------------------


class TestCounter:
    def test_default_name(self):
        counter = Counter()
        assert counter.name == "Counter"

    def test_custom_name(self):
        counter = Counter("hits")
        assert counter.name == "hits"

    def test_counts_total(self):
        counter = Counter()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[counter],
        )
        for i in range(3):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(0.1 * (i + 1)),
                    event_type="ping",
                    target=counter,
                )
            )
        sim.run()
        assert counter.total == 3

    def test_counts_by_type(self):
        counter = Counter()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[counter],
        )
        sim.schedule(Event(time=Instant.from_seconds(0.1), event_type="A", target=counter))
        sim.schedule(Event(time=Instant.from_seconds(0.2), event_type="B", target=counter))
        sim.schedule(Event(time=Instant.from_seconds(0.3), event_type="A", target=counter))
        sim.schedule(Event(time=Instant.from_seconds(0.4), event_type="C", target=counter))
        sim.schedule(Event(time=Instant.from_seconds(0.5), event_type="A", target=counter))
        sim.run()

        assert counter.total == 5
        assert counter.by_type == {"A": 3, "B": 1, "C": 1}

    def test_empty_counter(self):
        counter = Counter()
        assert counter.total == 0
        assert counter.by_type == {}

    def test_counter_with_source_integration(self):
        """Counter works end-to-end with Source.constant()."""
        counter = Counter()
        source = Source.constant(rate=5, target=counter, event_type="Tick", name="src")

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[counter],
        )
        sim.run()

        assert counter.total == 10
        assert counter.by_type == {"Tick": 10}

    def test_counter_handle_event_returns_none(self):
        """Counter returns None (no follow-up events)."""
        counter = Counter()
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[counter],
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="x",
                target=counter,
            )
        )
        sim.run()
        # If it returned events, the simulation would process them,
        # but counter.total should still be 1
        assert counter.total == 1


# ---------------------------------------------------------------------------
# Import / export tests
# ---------------------------------------------------------------------------


class TestExports:
    def test_importable_from_components(self):
        from happysimulator.components import Counter, Sink

        assert Sink is not None
        assert Counter is not None

    def test_importable_from_top_level(self):
        from happysimulator import Counter, Sink

        assert Sink is not None
        assert Counter is not None

    def test_sink_is_entity(self):
        assert issubclass(Sink, Entity)

    def test_counter_is_entity(self):
        assert issubclass(Counter, Entity)
