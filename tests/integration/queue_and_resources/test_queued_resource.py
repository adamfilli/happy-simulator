from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List

import pytest

from happysimulator.instrumentation.probe import Probe
from happysimulator.core.entity import Entity
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.event import Event
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class RecordingSink(Entity):
    def __init__(self, name: str = "Sink"):
        super().__init__(name)
        self.events_received = 0
        self.received_request_ids: list[int] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.events_received += 1
        self.received_request_ids.append(int(event.context.get("request_id", -1)))
        return []


class FixedTimeResource(QueuedResource):
    """A small queued resource used for tests."""

    def __init__(
        self,
        *,
        name: str = "Resource",
        service_time_s: float = 0.01,
        concurrency: int = 1,
        policy: FIFOQueue,
        downstream: Entity | None = None,
    ):
        super().__init__(name=name, policy=policy)
        self.service_time_s = float(service_time_s)
        self.concurrency = int(concurrency)
        self.downstream = downstream

        self._in_flight = 0
        self.stats_processed = 0

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._in_flight += 1
        yield self.service_time_s
        self._in_flight -= 1

        self.stats_processed += 1
        if self.downstream is None:
            return []

        return [
            self.forward(event, self.downstream, event_type="Completed")
        ]


def test_queued_resource_processes_work_end_to_end() -> None:
    duration_s = 1.0
    drain_s = 1.0

    sink = RecordingSink()
    resource = FixedTimeResource(
        service_time_s=0.02,
        concurrency=1,
        policy=FIFOQueue(),
        downstream=sink,
    )

    depth_probe, depth_data = Probe.on(resource, "depth", interval=0.05)

    source = Source.constant(rate=5.0, target=resource, stop_after=duration_s)

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=[resource, sink],
        probes=[depth_probe],
    )
    sim.run()

    assert sink.events_received > 0
    assert sink.events_received == resource.stats_processed
    assert resource.stats_dropped == 0
    assert resource.stats_accepted == sink.events_received

    assert depth_data.values
    assert depth_data.values[-1][1] == 0  # drained


def test_queued_resource_fifo_preserves_order_when_single_threaded() -> None:
    duration_s = 0.25
    drain_s = 1.0

    sink = RecordingSink()
    resource = FixedTimeResource(
        service_time_s=0.02,
        concurrency=1,
        policy=FIFOQueue(),
        downstream=sink,
    )

    source = Source.constant(rate=20.0, target=resource, stop_after=duration_s)

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=[resource, sink],
        probes=[],
    )
    sim.run()

    assert sink.events_received == resource.stats_processed
    assert sink.received_request_ids == list(range(1, sink.events_received + 1))
