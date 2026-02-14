"""Tests for GateController component."""

from __future__ import annotations

import pytest

from happysimulator.components.industrial.gate_controller import GateController
from happysimulator.components.common import Sink
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestGateControllerBasics:

    def test_creates_initially_open(self):
        sink = Sink()
        gate = GateController("gate", downstream=sink, initially_open=True)
        assert gate.is_open is True

    def test_creates_initially_closed(self):
        sink = Sink()
        gate = GateController("gate", downstream=sink, initially_open=False)
        assert gate.is_open is False

    def test_passes_through_when_open(self):
        sink = Sink()
        gate = GateController("gate", downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[gate, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=gate))
        sim.run()

        assert sink.events_received == 1
        assert gate.stats.passed_through == 1

    def test_queues_when_closed(self):
        sink = Sink()
        gate = GateController("gate", downstream=sink, initially_open=False)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[gate, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=gate))
        sim.run()

        assert sink.events_received == 0
        assert gate.queue_depth == 1
        assert gate.stats.queued_while_closed == 1


class TestGateControllerSchedule:

    def test_schedule_open_close(self):
        sink = Sink()
        gate = GateController(
            "gate", downstream=sink,
            schedule=[(1.0, 3.0)],
            initially_open=False,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[gate, sink],
        )
        for ev in gate.start_events():
            sim.schedule(ev)

        # Event arrives while closed (t=0.5)
        sim.schedule(Event(time=Instant.from_seconds(0.5), event_type="Item", target=gate))
        # Event arrives while open (t=2.0)
        sim.schedule(Event(time=Instant.from_seconds(2.0), event_type="Item", target=gate))
        sim.run()

        # Both should have passed through (first flushed on open, second during open)
        assert sink.events_received == 2

    def test_rejects_when_queue_full(self):
        sink = Sink()
        gate = GateController(
            "gate", downstream=sink, initially_open=False, queue_capacity=1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[gate, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=gate))
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=gate))
        sim.run()

        assert gate.queue_depth == 1
        assert gate.stats.rejected == 1

    def test_programmatic_open(self):
        sink = Sink()
        gate = GateController("gate", downstream=sink, initially_open=False)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[gate, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Item", target=gate))
        # Programmatically open after queuing
        sim.schedule(
            Event.once(
                time=Instant.from_seconds(1.0),
                event_type="Open",
                fn=lambda e: gate.open(),
            )
        )
        sim.run()

        assert sink.events_received == 1

    def test_stats_snapshot(self):
        sink = Sink()
        gate = GateController("gate", downstream=sink)
        stats = gate.stats
        assert stats.passed_through == 0
        assert stats.queued_while_closed == 0
        assert stats.rejected == 0
        assert stats.is_open is True
