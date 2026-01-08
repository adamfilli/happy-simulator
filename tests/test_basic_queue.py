"""Test: Queue -> QueueDriver -> Server design with increasing load.

Scenario:
- A Source generates requests with a ramp-up load profile (increasing over time)
- Requests flow through: Queue -> QueueDriver -> ConcurrencyLimitedServer -> Sink
- The server has a concurrency limit, so requests queue up as load increases
- A Probe tracks queue depth over time
- The Sink tracks time-in-system (latency) for each completed request

This test validates the decoupled Queue/QueueDriver/Server design where:
- Queue buffers work and notifies the driver
- QueueDriver polls the queue when server has capacity
- Server is queue-agnostic and just implements has_capacity()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List

from happysimulator.entities.entity import Entity
from happysimulator.entities.queue import Queue
from happysimulator.entities.queue_driver import QueueDriver
from happysimulator.entities.queue_policy import FIFOQueue
from happysimulator.events.event import Event
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.simulation import Simulation
from happysimulator.utils.instant import Instant

@dataclass
class ConcurrencyLimitedServer(Entity):
    """
    A server that processes requests with a fixed latency and concurrency limit.
    
    Implements has_capacity() so QueueDriver knows when to poll for more work.
    """
    name: str = "Server"
    concurrency: int = 1
    service_time_s: float = 1.0
    downstream: Entity = None  # Sink for completed requests
    
    _in_flight: int = field(default=0, init=False)
    
    # Statistics
    stats_processed: int = field(default=0, init=False)

    def has_capacity(self) -> bool:
        """Return True if server can accept more work."""
        return self._in_flight < self.concurrency

    def handle_event(self, event: Event) -> Generator[Instant, None, list[Event]]:
        """Process a request: simulate service time, then forward to downstream."""
        self._in_flight += 1
        
        # Simulate service time
        yield Instant.from_seconds(self.service_time_s)
        
        self._in_flight -= 1
        self.stats_processed += 1
        
        # Forward to downstream (sink)
        if self.downstream is not None:
            # Preserve context for latency tracking
            completed = Event(
                time=event.time + Instant.from_seconds(self.service_time_s),
                event_type="Completed",
                target=self.downstream,
                context=event.context,
            )
            return [completed]
        return []

class LatencyTrackingSink(Entity):
    """
    Final destination that records completion times and calculates latency.
    
    Latency = time event reaches sink - time event was created (from context).
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.events_received: int = 0
        self.latencies_s: list[float] = []
        self.completion_times_s: list[float] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.events_received += 1
        
        # Calculate time in system, 
        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()
        
        self.latencies_s.append(latency_s)
        self.completion_times_s.append(event.time.to_seconds())
        
        return []

    def average_latency(self) -> float:
        if not self.latencies_s:
            return 0.0
        return sum(self.latencies_s) / len(self.latencies_s)

    def percentile(self, p: float) -> float:
        """Return the pth percentile (0.0 to 1.0) of latencies."""
        if not self.latencies_s:
            return 0.0
        ordered = sorted(self.latencies_s)
        idx = int(p * (len(ordered) - 1))
        return ordered[idx]

class RequestProvider(EventProvider):
    """Generates request events targeting the queue."""

    def __init__(self, queue: Entity):
        super().__init__()
        self._queue = queue

    def get_events(self, time: Instant) -> List[Event]:
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._queue,
            )
        ]

def test_queue_driver_basic_flow():
    """Basic test: verify events flow through Queue -> QueueDriver -> Server -> Sink."""
    # Setup entities
    sink = LatencyTrackingSink(name="Sink")
    server = ConcurrencyLimitedServer(
        name="Server",
        concurrency=1,
        service_time_s=1,
        downstream=sink,
    )
    driver = QueueDriver(
        name="Driver",
        queue=None,  # Set after queue is created
        server=server,
    )
    queue = Queue(
        name="RequestQueue",
        egress=driver,
        policy=FIFOQueue(),
    )
    driver.queue = queue  # Complete the circular reference
    
    # Create a simple constant-rate source
    @dataclass(frozen=True)
    class ConstantProfile(Profile):
        rate: float
        def get_rate(self, time: Instant) -> float:
            return self.rate
    
    provider = RequestProvider(queue)
    arrival = ConstantArrivalTimeProvider(ConstantProfile(rate=5.0), start_time=Instant.Epoch)
    source = Source(
        name="RequestSource",
        event_provider=provider,
        arrival_time_provider=arrival,
    )
        
    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(60.0),
        sources=[source],
        entities=[queue, driver, server, sink]
    )
    
    sim.run()
    
    # Verify flow worked
    assert sink.events_received > 0, "Sink should have received completed events"
    assert queue.stats_accepted > 0, "Queue should have accepted events"
    print(f"Queue accepted: {queue.stats_accepted}, Sink received: {sink.events_received}")
    print(f"Average latency: {sink.average_latency():.3f}s")