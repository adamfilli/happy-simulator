"""Visual debugger example — bursty M/M/1 queue.

Launch with:
    python examples/visual_debugger.py

Opens a browser at http://127.0.0.1:8765 showing the entity graph.
Step through the simulation to watch queue depth rise during bursts
and drain during quiet periods.
"""

from dataclasses import dataclass

from happysimulator.core.simulation import Simulation
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source
from happysimulator.load.profile import Profile
from happysimulator.components.common import Sink
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.probe import Probe
from happysimulator.visual import serve, Chart


@dataclass(frozen=True)
class BurstProfile(Profile):
    """Alternates between high and low request rates on a repeating cycle."""

    high_rate: float = 20.0
    high_duration_s: float = 2.0
    low_rate: float = 5.0
    low_duration_s: float = 3.0

    def get_rate(self, time: Instant) -> float:
        cycle = self.high_duration_s + self.low_duration_s
        t_in_cycle = time.to_seconds() % cycle
        return self.high_rate if t_in_cycle < self.high_duration_s else self.low_rate


class Server(QueuedResource):
    """Single-server queue with concurrency limit so the queue actually builds up."""

    def __init__(self, name: str, downstream, service_time: float = 0.08, concurrency: int = 1):
        super().__init__(name)
        self._downstream = downstream
        self._service_time = service_time
        self._concurrency = concurrency
        self._in_flight = 0

    def has_capacity(self) -> bool:
        return self._in_flight < self._concurrency

    def handle_queued_event(self, event):
        self._in_flight += 1
        try:
            yield self._service_time
        finally:
            self._in_flight -= 1
        return [Event(
            time=self.now,
            event_type="Response",
            target=self._downstream,
            context=event.context,
        )]


sink = Sink("Sink")
server = Server("Server", downstream=sink, service_time=0.08)

# Bursty load: 20 req/s for 2s, then 5 req/s for 3s (repeating)
# Server capacity is 12.5 req/s, so queue builds during bursts and drains between
source = Source.with_profile(
    profile=BurstProfile(),
    target=server,
    event_type="Request",
    poisson=False,
)

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1)

sim = Simulation(
    sources=[source],
    entities=[server, sink],
    probes=[depth_probe],
    end_time=Instant.from_seconds(60.0),
)

serve(sim, charts=[
    Chart(depth_data, title="Queue Depth", y_label="items"),
    Chart(depth_data, title="P99 Queue Depth",
          transform="p99", window_s=1.0, y_label="items", color="#f59e0b"),
    Chart.from_probe(depth_probe, title="Mean Queue Depth",
          transform="mean", window_s=0.5, color="#10b981"),
])
