from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List

import pytest

from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.probe import Probe
from happysimulator.core.entity import Entity
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.event import Event
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:  # noqa: ARG002 - profile is constant
        return float(self.rate_per_s)


class RequestProvider(EventProvider):
    """Generates request events targeting a queued resource."""

    def __init__(self, target: Entity, *, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after
        self.generated_requests = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated_requests += 1
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated_requests,
                },
            )
        ]


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
            Event(
                time=self.now,
                event_type="Completed",
                target=self.downstream,
                context=event.context,
            )
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

    depth_data = Data()
    depth_probe = Probe(
        target=resource,
        metric="depth",
        data=depth_data,
        interval=0.05,
        start_time=Instant.Epoch,
    )

    provider = RequestProvider(resource, stop_after=Instant.from_seconds(duration_s))
    arrival = ConstantArrivalTimeProvider(ConstantRateProfile(rate_per_s=5.0), start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + drain_s),
        sources=[source],
        entities=[resource, sink],
        probes=[depth_probe],
    )
    sim.run()

    assert provider.generated_requests > 0
    assert sink.events_received == provider.generated_requests
    assert resource.stats_processed == sink.events_received
    assert resource.stats_dropped == 0
    assert resource.stats_accepted == provider.generated_requests

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

    provider = RequestProvider(resource, stop_after=Instant.from_seconds(duration_s))
    arrival = ConstantArrivalTimeProvider(ConstantRateProfile(rate_per_s=20.0), start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + drain_s),
        sources=[source],
        entities=[resource, sink],
        probes=[],
    )
    sim.run()

    assert sink.events_received == provider.generated_requests
    assert sink.received_request_ids == list(range(1, provider.generated_requests + 1))