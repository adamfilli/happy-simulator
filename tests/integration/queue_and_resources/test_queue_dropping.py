from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.components.queue import Queue
from happysimulator.components.queue_driver import QueueDriver
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


def test_queue_drops_when_capacity_is_one_and_three_arrive() -> None:
    """Queue capacity=1 drops the third request when server concurrency=1.

    Scenario:
    - Three requests arrive at t=0.
    - Driver immediately polls and delivers one request to server at t=0.
    - One additional request can sit in the queue (capacity=1).
    - The third request is rejected (dropped).
    """

    @dataclass
    class YieldingServer(Entity):
        name: str = "Worker"
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

    server = YieldingServer(service_time_s=1.0)
    driver = QueueDriver(name="Driver", queue=None, target=server)
    queue = Queue(name="RequestQueue", egress=driver, policy=FIFOQueue(capacity=1))
    driver.queue = queue

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[],
        entities=[queue, driver, server],
        probes=[],
    )

    sim.schedule(Event(time=Instant.Epoch + 0.001, event_type="Request", target=queue))
    sim.schedule(Event(time=Instant.Epoch + 0.002, event_type="Request", target=queue))
    sim.schedule(Event(time=Instant.Epoch + 0.003, event_type="Request", target=queue))

    sim.run()

    # With capacity=1 and concurrency=1, exactly one request should be dropped.
    assert queue.stats_accepted == 2
    assert queue.stats_dropped == 1
    assert server.stats_processed == 2
