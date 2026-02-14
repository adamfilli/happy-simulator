"""Tests for InspectionStation component."""

from __future__ import annotations

import random

import pytest

from happysimulator.components.industrial.inspection import InspectionStation
from happysimulator.components.common import Sink, Counter
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestInspectionBasics:

    def test_creates_with_defaults(self):
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")
        station = InspectionStation("inspect", pass_sink, fail_sink)
        assert station.name == "inspect"
        assert station.pass_rate == 0.95
        assert station.inspection_time == 0.1
        assert station.inspected == 0

    def test_all_pass_with_rate_1(self):
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")
        station = InspectionStation(
            "inspect", pass_sink, fail_sink, pass_rate=1.0, inspection_time=0.01
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[station, pass_sink, fail_sink],
        )
        for i in range(10):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Item",
                    target=station,
                )
            )
        sim.run()

        assert station.inspected == 10
        assert station.passed == 10
        assert station.failed == 0
        assert pass_sink.events_received == 10
        assert fail_sink.events_received == 0

    def test_all_fail_with_rate_0(self):
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")
        station = InspectionStation(
            "inspect", pass_sink, fail_sink, pass_rate=0.0, inspection_time=0.01
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[station, pass_sink, fail_sink],
        )
        for i in range(10):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Item",
                    target=station,
                )
            )
        sim.run()

        assert station.inspected == 10
        assert station.passed == 0
        assert station.failed == 10
        assert pass_sink.events_received == 0
        assert fail_sink.events_received == 10

    def test_probabilistic_routing(self):
        random.seed(42)
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")
        station = InspectionStation(
            "inspect", pass_sink, fail_sink,
            pass_rate=0.8, inspection_time=0.001,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[station, pass_sink, fail_sink],
        )
        n = 100
        for i in range(n):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.01),
                    event_type="Item",
                    target=station,
                )
            )
        sim.run()

        assert station.inspected == n
        assert station.passed + station.failed == n
        # With pass_rate=0.8 over 100 items, expect ~80 passes
        assert 60 < station.passed < 95

    def test_stats_snapshot(self):
        pass_sink = Sink("pass")
        fail_sink = Sink("fail")
        station = InspectionStation(
            "inspect", pass_sink, fail_sink, pass_rate=1.0, inspection_time=0.01
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[station, pass_sink, fail_sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=station))
        sim.run()

        stats = station.stats
        assert stats.inspected == 1
        assert stats.passed == 1
        assert stats.failed == 0
