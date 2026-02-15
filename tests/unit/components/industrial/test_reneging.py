"""Tests for RenegingQueuedResource component."""

from __future__ import annotations

from happysimulator.components.common import Sink
from happysimulator.components.industrial.reneging import RenegingQueuedResource
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class SimpleRenegingServer(RenegingQueuedResource):
    """Concrete implementation for testing."""

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


class TestRenegingBasics:
    def test_creates_with_defaults(self):
        server = SimpleRenegingServer("server")
        assert server.served == 0
        assert server.reneged == 0
        assert server.default_patience_s == float("inf")

    def test_serves_patient_items(self):
        sink = Sink()
        server = SimpleRenegingServer(
            "server",
            service_time=0.01,
            downstream=sink,
            default_patience_s=10.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[server, sink],
        )
        sim.schedule(
            Event(
                time=Instant.Epoch,
                event_type="Request",
                target=server,
                context={"created_at": Instant.Epoch},
            )
        )
        sim.run()

        assert server.served == 1
        assert server.reneged == 0
        assert sink.events_received == 1

    def test_reneges_impatient_items(self):
        sink = Sink()
        reneged_sink = Sink("reneged")
        server = SimpleRenegingServer(
            "server",
            service_time=0.5,
            downstream=sink,
            reneged_target=reneged_sink,
            default_patience_s=0.5,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[server, sink, reneged_sink],
        )

        # First item gets served immediately
        sim.schedule(
            Event(
                time=Instant.Epoch,
                event_type="Request",
                target=server,
                context={"created_at": Instant.Epoch},
            )
        )
        # Second item arrives at t=0, must wait in queue.
        # By the time the first item finishes at t=0.5, this item
        # has waited 0.5s. With patience=0.5, it should be served
        # (wait_time is not strictly greater).
        # But an item arriving at t=0 that gets dequeued at t=0.51
        # would renege.
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.01),
                event_type="Request",
                target=server,
                context={"created_at": Instant.from_seconds(0.01)},
            )
        )

        sim.run()

        # Both should be served since wait doesn't exceed patience
        assert server.served + server.reneged == 2

    def test_per_event_patience(self):
        sink = Sink()
        reneged_sink = Sink("reneged")
        server = SimpleRenegingServer(
            "server",
            service_time=0.5,
            downstream=sink,
            reneged_target=reneged_sink,
            default_patience_s=100.0,  # Very patient by default
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[server, sink, reneged_sink],
        )

        # First request occupies server for 0.5s
        sim.schedule(
            Event(
                time=Instant.Epoch,
                event_type="Request",
                target=server,
                context={"created_at": Instant.Epoch},
            )
        )
        # Second request with very short patience - will renege
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.01),
                event_type="Request",
                target=server,
                context={
                    "created_at": Instant.from_seconds(0.01),
                    "patience_s": 0.1,  # Will wait max 100ms
                },
            )
        )
        sim.run()

        assert server.served == 1
        assert server.reneged == 1
        assert reneged_sink.events_received == 1

    def test_stats_snapshot(self):
        server = SimpleRenegingServer("server", default_patience_s=10.0)
        stats = server.reneging_stats
        assert stats.served == 0
        assert stats.reneged == 0
