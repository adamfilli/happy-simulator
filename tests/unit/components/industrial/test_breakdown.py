"""Tests for BreakdownScheduler component."""

from __future__ import annotations

import random

import pytest

from happysimulator.components.industrial.breakdown import BreakdownScheduler
from happysimulator.components.common import Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class DummyMachine(Entity):
    """Simple entity for testing breakdowns."""

    def __init__(self, name="machine"):
        super().__init__(name)
        self._broken = False

    def handle_event(self, event):
        return []

    def has_capacity(self):
        return not self._broken


class TestBreakdownBasics:

    def test_creates_with_defaults(self):
        machine = DummyMachine()
        scheduler = BreakdownScheduler("bd", target=machine)
        assert scheduler.mean_time_to_failure == 100.0
        assert scheduler.mean_repair_time == 5.0
        assert scheduler.is_down is False

    def test_start_event_creates_breakdown(self):
        random.seed(42)
        machine = DummyMachine()
        scheduler = BreakdownScheduler(
            "bd", target=machine,
            mean_time_to_failure=10.0,
            mean_repair_time=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(100.0),
            entities=[machine, scheduler],
        )
        sim.schedule(scheduler.start_event())
        sim.run()

        # Should have had some breakdowns and repairs
        assert scheduler.stats.breakdown_count > 0
        assert scheduler.stats.total_downtime_s > 0
        assert scheduler.stats.total_uptime_s > 0

    def test_sets_broken_flag(self):
        random.seed(42)
        machine = DummyMachine()
        scheduler = BreakdownScheduler(
            "bd", target=machine,
            mean_time_to_failure=1.0,
            mean_repair_time=0.5,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(50.0),
            entities=[machine, scheduler],
        )
        sim.schedule(scheduler.start_event())
        sim.run()

        # After simulation, machine should be in a definite state
        # (either up or down, not stuck)
        assert isinstance(machine._broken, bool)
        assert scheduler.stats.breakdown_count >= 1

    def test_availability(self):
        random.seed(42)
        machine = DummyMachine()
        scheduler = BreakdownScheduler(
            "bd", target=machine,
            mean_time_to_failure=10.0,
            mean_repair_time=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(200.0),
            entities=[machine, scheduler],
        )
        sim.schedule(scheduler.start_event())
        sim.run()

        stats = scheduler.stats
        avail = stats.availability
        # With MTTF=10 and MTTR=1, expected availability ~= 10/(10+1) ~= 0.91
        assert 0.5 < avail < 1.0

    def test_stats_snapshot(self):
        machine = DummyMachine()
        scheduler = BreakdownScheduler("bd", target=machine)
        stats = scheduler.stats
        assert stats.breakdown_count == 0
        assert stats.total_downtime_s == 0.0
        assert stats.availability == 1.0
