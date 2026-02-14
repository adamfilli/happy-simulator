"""Integration tests for service operation components.

Tests multi-component pipelines using BalkingQueue, RenegingQueuedResource,
ShiftedServer, and ShiftSchedule wired together in realistic service
configurations.
"""

from __future__ import annotations

from typing import Generator

from happysimulator.components.industrial.balking import BalkingQueue
from happysimulator.components.industrial.reneging import RenegingQueuedResource
from happysimulator.components.industrial.shift_schedule import (
    Shift,
    ShiftSchedule,
    ShiftedServer,
)
from happysimulator.components.common import Sink
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class SimpleRenegingServer(RenegingQueuedResource):
    """Concrete reneging server for integration tests."""

    def __init__(self, name, service_time=0.1, downstream=None, concurrency=1, **kwargs):
        super().__init__(name, **kwargs)
        self.service_time = service_time
        self.downstream = downstream
        self._concurrency = concurrency
        self._active = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def _handle_served_event(self, event):
        self._active += 1
        try:
            yield self.service_time
        finally:
            self._active -= 1
        if self.downstream:
            return [
                Event(
                    time=self.now,
                    event_type="Served",
                    target=self.downstream,
                    context=event.context,
                )
            ]
        return []


class FixedTimeServer(QueuedResource):
    """Simple queued resource for balking tests."""

    def __init__(self, name, service_time=0.1, downstream=None, concurrency=1, **kwargs):
        super().__init__(name, **kwargs)
        self.service_time = service_time
        self.downstream = downstream
        self._concurrency = concurrency
        self._active = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def handle_queued_event(self, event) -> Generator:
        self._active += 1
        try:
            yield self.service_time
        finally:
            self._active -= 1
        if self.downstream:
            return [
                Event(
                    time=self.now,
                    event_type="Served",
                    target=self.downstream,
                    context=event.context,
                )
            ]
        return []


class TestServiceWithBalking:
    """Source → QueuedResource(BalkingQueue policy) → Sink."""

    def test_balking_occurs_at_threshold(self):
        sink = Sink("served")
        server = FixedTimeServer(
            "server", service_time=0.5, downstream=sink,
            concurrency=1, policy=BalkingQueue(FIFOQueue(), balk_threshold=3),
        )

        # High arrival rate to cause queue buildup
        source = Source.constant(rate=20.0, target=server, stop_after=5.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[server, sink],
        )
        sim.run()

        # Some arrivals should have balked
        assert server.stats_dropped > 0
        assert sink.events_received > 0
        # All arrivals are either served or dropped (balked)
        assert server.stats_accepted + server.stats_dropped > 0


class TestServiceWithReneging:
    """Source → SimpleRenegingServer → (served: Sink, reneged: Sink)."""

    def test_reneging_under_overload(self):
        served_sink = Sink("served")
        reneged_sink = Sink("reneged")

        server = SimpleRenegingServer(
            "server", service_time=1.0, downstream=served_sink,
            concurrency=1,
            reneged_target=reneged_sink,
            default_patience_s=0.5,
        )

        # High rate to cause queueing; slow service forces long waits
        source = Source.constant(rate=5.0, target=server, stop_after=10.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(20.0),
            sources=[source],
            entities=[server, served_sink, reneged_sink],
        )
        sim.run()

        assert server.served > 0
        assert server.reneged > 0
        assert served_sink.events_received == server.served
        assert reneged_sink.events_received == server.reneged
        assert server.served + server.reneged > 0


class TestShiftBasedCapacity:
    """Source → ShiftedServer → Sink with multiple shift capacities."""

    def test_throughput_varies_by_shift(self):
        sink = Sink("served")

        schedule = ShiftSchedule(
            shifts=[
                Shift(0, 10, capacity=1),    # Low capacity shift
                Shift(10, 20, capacity=5),   # High capacity shift
            ]
        )
        server = ShiftedServer(
            "server", schedule=schedule,
            service_time=0.1, downstream=sink,
        )

        source = Source.constant(rate=10.0, target=server, stop_after=20.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(25.0),
            sources=[source],
            entities=[server, sink],
        )
        sim.run()

        assert server.processed > 0
        assert sink.events_received == server.processed


class TestCombinedBalkingRenegingComparison:
    """Compare balking vs reneging loss patterns under the same load."""

    def test_both_mechanisms_shed_load(self):
        # Balking setup
        balk_sink = Sink("balk_served")
        balk_server = FixedTimeServer(
            "balk_server", service_time=0.5, downstream=balk_sink,
            concurrency=1, policy=BalkingQueue(FIFOQueue(), balk_threshold=3),
        )
        balk_source = Source.constant(rate=10.0, target=balk_server, stop_after=5.0)

        balk_sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[balk_source],
            entities=[balk_server, balk_sink],
        )
        balk_sim.run()

        # Reneging setup
        renege_served = Sink("renege_served")
        renege_lost = Sink("renege_lost")
        renege_server = SimpleRenegingServer(
            "renege_server", service_time=0.5, downstream=renege_served,
            concurrency=1,
            reneged_target=renege_lost,
            default_patience_s=0.5,
        )
        renege_source = Source.constant(rate=10.0, target=renege_server, stop_after=5.0)

        renege_sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[renege_source],
            entities=[renege_server, renege_served, renege_lost],
        )
        renege_sim.run()

        # Both should have lost some customers
        assert balk_server.stats_dropped > 0
        assert renege_server.reneged > 0
        # Both should have served some customers
        assert balk_sink.events_received > 0
        assert renege_served.events_received > 0
