from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

from happysimulator.entities.entity import Entity
from happysimulator.entities.queue import Queue
from happysimulator.entities.queue_driver import QueueDriver
from happysimulator.entities.queue_policy import FIFOQueue
from happysimulator.events.event import Event
from happysimulator.simulation import Simulation
from happysimulator.tracing.recorder import InMemoryTraceRecorder
from happysimulator.utils.instant import Instant

import logging

def test_queue_driver_single_event_trace_flow(caplog):
    """Single-event trace: Queue notifies driver, driver polls, server processes, driver polls again."""

    @dataclass
    class YieldingServer(Entity):
        """A minimal server that yields once (simulated service time)."""

        name: str = "Server"
        service_time_s: float = 1.0

        _in_flight: int = field(default=0, init=False)
        stats_processed: int = field(default=0, init=False)

        def has_capacity(self) -> bool:
            return self._in_flight == 0

        def handle_event(self, event: Event) -> Generator[Instant, None, list[Event]]:
            self._in_flight += 1
            yield self.service_time_s, None
            self._in_flight -= 1
            self.stats_processed += 1
            return []

    trace = InMemoryTraceRecorder()
    server = YieldingServer(service_time_s=1.0)
    driver = QueueDriver(name="Driver", queue=None, server=server)
    queue = Queue(name="RequestQueue", egress=driver, policy=FIFOQueue())
    driver.queue = queue

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(5.0),
        sources=[],
        entities=[queue, driver, server],
        trace_recorder=trace,
    )

    sim.schedule(Event(time=Instant.Epoch, event_type="Request", target=queue))

    with caplog.at_level(logging.WARNING, logger="happysimulator.simulation"):
        sim.run()

    assert queue.stats_accepted == 1
    assert server.stats_processed == 1
    assert not any("Time travel detected" in rec.message for rec in caplog.records)

    scheduled = [
        s for s in trace.spans
        if s["kind"] == "simulation.schedule" and s.get("event_type") in {"QUEUE_NOTIFY", "QUEUE_POLL"}
    ]
    scheduled_types = {s.get("event_type") for s in scheduled}
    assert "QUEUE_NOTIFY" in scheduled_types
    assert "QUEUE_POLL" in scheduled_types

    poll_times = [
        s["data"]["scheduled_time"]
        for s in scheduled
        if s.get("event_type") == "QUEUE_POLL"
    ]
    assert Instant.Epoch in poll_times
    assert Instant.from_seconds(1.0) in poll_times


def test_queue_driver_overload_serializes_requests(caplog):
    """Two requests arrive at t=0; server concurrency=1 forces serialization.

    We validate serialization by verifying that the driver schedules follow-up
    polls at exactly one service-time apart (t=1s, then t=2s), meaning the second
    request only starts after the first finishes.
    """

    @dataclass
    class YieldingServer(Entity):
        """A minimal server that yields once (simulated service time)."""

        name: str = "Server"
        service_time_s: float = 1.0

        _in_flight: int = field(default=0, init=False)
        stats_processed: int = field(default=0, init=False)

        def has_capacity(self) -> bool:
            return self._in_flight == 0

        def handle_event(self, event: Event) -> Generator[Instant, None, list[Event]]:
            self._in_flight += 1
            yield self.service_time_s, None
            self._in_flight -= 1
            self.stats_processed += 1
            return []

    trace = InMemoryTraceRecorder()
    server = YieldingServer(service_time_s=1.0)
    driver = QueueDriver(name="Driver", queue=None, server=server)
    queue = Queue(name="RequestQueue", egress=driver, policy=FIFOQueue())
    driver.queue = queue

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(5.0),
        sources=[],
        entities=[queue, driver, server],
        trace_recorder=trace,
    )

    sim.schedule(Event(time=Instant.Epoch, event_type="Request", target=queue))
    sim.schedule(Event(time=Instant.Epoch, event_type="Request", target=queue))

    with caplog.at_level(logging.WARNING, logger="happysimulator.simulation"):
        sim.run()

    assert queue.stats_accepted == 2
    assert server.stats_processed == 2
    assert not any("Time travel detected" in rec.message for rec in caplog.records)

    scheduled_polls = [
        s for s in trace.spans
        if s["kind"] == "simulation.schedule" and s.get("event_type") == "QUEUE_POLL"
    ]

    poll_times = [s["data"]["scheduled_time"] for s in scheduled_polls]
    assert Instant.Epoch in poll_times
    assert Instant.from_seconds(1.0) in poll_times
    assert Instant.from_seconds(2.0) in poll_times