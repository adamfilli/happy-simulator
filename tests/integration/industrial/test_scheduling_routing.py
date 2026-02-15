"""Integration tests for scheduling and routing components.

Tests AppointmentScheduler, ConditionalRouter, and GateController
in multi-component pipelines.
"""

from __future__ import annotations

import random

from happysimulator.components.common import Sink
from happysimulator.components.industrial.appointment import AppointmentScheduler
from happysimulator.components.industrial.conditional_router import ConditionalRouter
from happysimulator.components.industrial.gate_controller import GateController
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class TestAppointmentSchedulingWithNoShows:
    """AppointmentScheduler → Sink with probabilistic no-shows."""

    def test_arrivals_and_no_shows_match_schedule(self):
        random.seed(42)
        sink = Sink("clinic")
        appointments = [float(i) for i in range(1, 31)]
        scheduler = AppointmentScheduler(
            "appointments",
            target=sink,
            appointments=appointments,
            no_show_rate=0.2,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=35.0,
            entities=[scheduler, sink],
        )
        for e in scheduler.start_events():
            sim.schedule(e)
        sim.run()

        stats = scheduler.stats
        assert stats.total_scheduled == 30
        assert stats.arrivals + stats.no_shows == 30
        assert stats.arrivals == sink.events_received
        # With 20% no-show rate over 30 appointments, expect some no-shows
        assert stats.no_shows > 0
        assert stats.arrivals > 0


class TestConditionalRoutingByContextField:
    """Source → ConditionalRouter.by_context_field() → multiple Sinks."""

    def test_routes_events_correctly(self):
        sink_express = Sink("express")
        sink_standard = Sink("standard")
        sink_economy = Sink("economy")
        default_sink = Sink("default")

        router = ConditionalRouter.by_context_field(
            "router",
            "tier",
            {
                "express": sink_express,
                "standard": sink_standard,
                "economy": sink_economy,
            },
            default=default_sink,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=2.0,
            entities=[router, sink_express, sink_standard, sink_economy, default_sink],
        )

        tiers = [
            "express",
            "standard",
            "economy",
            "standard",
            "express",
            "economy",
            "economy",
            "unknown",
        ]
        for i, tier in enumerate(tiers):
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="Order",
                    target=router,
                    context={"tier": tier},
                )
            )
        sim.run()

        assert sink_express.events_received == 2
        assert sink_standard.events_received == 2
        assert sink_economy.events_received == 3
        assert default_sink.events_received == 1
        assert router.total_routed == 8


class TestGateControllerWithSchedule:
    """Source → GateController → Sink with open/close schedule."""

    def test_items_queued_when_closed_flushed_when_opened(self):
        sink = Sink("output")
        gate = GateController(
            "gate",
            downstream=sink,
            schedule=[(2.0, 4.0), (6.0, 8.0)],
            initially_open=False,
        )

        source = Source.constant(rate=5.0, target=gate, stop_after=9.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            sources=[source],
            entities=[gate, sink],
        )
        for ev in gate.start_events():
            sim.schedule(ev)
        sim.run()

        # Items should have been queued while closed, then flushed during open windows
        assert sink.events_received > 0
        assert gate.stats.queued_while_closed > 0
        assert gate.stats.passed_through > 0


class TestCombinedRoutingAndGate:
    """ConditionalRouter → GateController → Sink.

    End-to-end flow through routing and gate components with mixed traffic.
    """

    def test_end_to_end_flow(self):
        sink = Sink("completed")
        gate = GateController(
            "gate",
            downstream=sink,
            schedule=[(0.0, 20.0)],  # Open the whole time
            initially_open=False,
        )

        urgent_sink = Sink("urgent")
        router = ConditionalRouter.by_context_field(
            "router",
            "priority",
            {
                "urgent": urgent_sink,
                "normal": gate,
            },
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=15.0,
            entities=[router, gate, sink, urgent_sink],
        )
        for ev in gate.start_events():
            sim.schedule(ev)

        # Send 10 events — half urgent, half normal
        for i in range(10):
            priority = "urgent" if i % 2 == 0 else "normal"
            sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.5 + 0.5),
                    event_type="Visit",
                    target=router,
                    context={"priority": priority},
                )
            )
        sim.run()

        # Urgent goes direct to urgent_sink, normal goes through gate to sink
        assert urgent_sink.events_received + sink.events_received == 10
        assert urgent_sink.events_received == 5
        assert sink.events_received == 5
