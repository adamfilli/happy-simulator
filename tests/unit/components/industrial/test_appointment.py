"""Tests for AppointmentScheduler component."""

from __future__ import annotations

import random

from happysimulator.components.common import Sink
from happysimulator.components.industrial.appointment import AppointmentScheduler
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestAppointmentBasics:
    def test_creates_with_parameters(self):
        sink = Sink()
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=[1.0, 2.0, 3.0],
        )
        assert scheduler.no_show_rate == 0.0

    def test_generates_all_arrivals(self):
        sink = Sink()
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=[1.0, 2.0, 3.0],
            no_show_rate=0.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[scheduler, sink],
        )
        for e in scheduler.start_events():
            sim.schedule(e)
        sim.run()

        assert sink.events_received == 3
        stats = scheduler.stats
        assert stats.arrivals == 3
        assert stats.no_shows == 0

    def test_no_shows(self):
        sink = Sink()
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=[1.0, 2.0, 3.0],
            no_show_rate=1.0,  # All no-show
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[scheduler, sink],
        )
        for e in scheduler.start_events():
            sim.schedule(e)
        sim.run()

        assert sink.events_received == 0
        stats = scheduler.stats
        assert stats.arrivals == 0
        assert stats.no_shows == 3

    def test_probabilistic_no_shows(self):
        random.seed(42)
        sink = Sink()
        appointments = [float(i) for i in range(1, 51)]
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=appointments,
            no_show_rate=0.2,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(55.0),
            entities=[scheduler, sink],
        )
        for e in scheduler.start_events():
            sim.schedule(e)
        sim.run()

        stats = scheduler.stats
        assert stats.total_scheduled == 50
        assert stats.arrivals + stats.no_shows == 50
        # Expect ~40 arrivals, ~10 no-shows
        assert 25 < stats.arrivals < 50

    def test_arrival_times_match_appointments(self):
        sink = Sink()
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=[1.0, 3.0, 5.0],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            entities=[scheduler, sink],
        )
        for e in scheduler.start_events():
            sim.schedule(e)
        sim.run()

        assert sink.completion_times[0] == Instant.from_seconds(1.0)
        assert sink.completion_times[1] == Instant.from_seconds(3.0)
        assert sink.completion_times[2] == Instant.from_seconds(5.0)

    def test_custom_event_type(self):
        sink = Sink()
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=[1.0],
            event_type="Reservation",
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[scheduler, sink],
        )
        for e in scheduler.start_events():
            sim.schedule(e)
        sim.run()

        assert sink.events_received == 1

    def test_stats_snapshot(self):
        sink = Sink()
        scheduler = AppointmentScheduler(
            "appt",
            target=sink,
            appointments=[1.0, 2.0],
        )
        stats = scheduler.stats
        assert stats.total_scheduled == 2
        assert stats.arrivals == 0
        assert stats.no_shows == 0
