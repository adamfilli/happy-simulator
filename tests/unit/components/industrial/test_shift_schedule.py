"""Tests for ShiftSchedule and ShiftedServer components."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.shift_schedule import (
    Shift,
    ShiftSchedule,
    ShiftedServer,
)
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestShiftSchedule:

    def test_creates_schedule(self):
        schedule = ShiftSchedule(
            shifts=[
                Shift(0, 8, 2),
                Shift(8, 16, 4),
                Shift(16, 24, 2),
            ]
        )
        assert len(schedule.shifts) == 3

    def test_capacity_at_within_shift(self):
        schedule = ShiftSchedule(
            shifts=[Shift(0, 10, 2), Shift(10, 20, 4)]
        )
        assert schedule.capacity_at(5.0) == 2
        assert schedule.capacity_at(15.0) == 4

    def test_capacity_at_boundary(self):
        schedule = ShiftSchedule(
            shifts=[Shift(0, 10, 2), Shift(10, 20, 4)]
        )
        assert schedule.capacity_at(0.0) == 2
        assert schedule.capacity_at(10.0) == 4

    def test_default_capacity_in_gaps(self):
        schedule = ShiftSchedule(
            shifts=[Shift(10, 20, 4)],
            default_capacity=1,
        )
        assert schedule.capacity_at(5.0) == 1
        assert schedule.capacity_at(25.0) == 1
        assert schedule.capacity_at(15.0) == 4

    def test_transition_times(self):
        schedule = ShiftSchedule(
            shifts=[Shift(0, 8, 2), Shift(8, 16, 4)]
        )
        times = schedule.transition_times()
        assert times == [0, 8, 16]


class TestShiftedServer:

    def test_creates_with_schedule(self):
        schedule = ShiftSchedule(shifts=[Shift(0, 100, 2)])
        server = ShiftedServer("server", schedule=schedule)
        assert server.current_capacity == 2
        assert server.processed == 0

    def test_processes_items(self):
        sink = Sink()
        schedule = ShiftSchedule(shifts=[Shift(0, 100, 2)])
        server = ShiftedServer(
            "server", schedule=schedule,
            service_time=0.01, downstream=sink,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[server, sink],
        )
        for i in range(5):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Item",
                    target=server,
                )
            )
        sim.run()

        assert server.processed == 5
        assert sink.events_received == 5

    def test_capacity_changes_at_shift_boundary(self):
        # Verify that the shift schedule correctly reports capacity at different times
        schedule = ShiftSchedule(
            shifts=[Shift(0, 5, 1), Shift(5, 100, 3)],
        )
        assert schedule.capacity_at(2.0) == 1
        assert schedule.capacity_at(5.0) == 3
        assert schedule.capacity_at(50.0) == 3

        # Track capacity changes via an event hook
        capacity_log: list[tuple[float, int]] = []

        server = ShiftedServer("server", schedule=schedule, service_time=0.01)
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            entities=[server],
        )
        sim.schedule(
            Event(time=Instant.Epoch, event_type="Item", target=server)
        )

        # Hook to record capacity changes
        sim.control.on_event(
            lambda event: capacity_log.append(
                (event.time.to_seconds(), server.current_capacity)
            )
            if event.event_type == "_ShiftChange"
            else None
        )
        sim.run()

        # The shift change at t=5 should have set capacity to 3
        shift_at_5 = [c for t, c in capacity_log if t == 5.0]
        assert shift_at_5 == [3]

    def test_no_downstream(self):
        schedule = ShiftSchedule(shifts=[Shift(0, 100, 2)])
        server = ShiftedServer("server", schedule=schedule, service_time=0.01)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[server],
        )
        sim.schedule(
            Event(time=Instant.Epoch, event_type="Item", target=server)
        )
        sim.run()

        assert server.processed == 1
