"""Visual debugger example — M/M/1 queue.

Launch with:
    python examples/visual_debugger.py

Opens a browser at http://127.0.0.1:8765 showing the entity graph.
Step through the simulation to watch queue depth, latency, and
throughput change in real time.
"""

from happysimulator.core.simulation import Simulation
from happysimulator.core.event import Event
from happysimulator.load.source import Source
from happysimulator.components.common import Sink
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.probe import Probe
from happysimulator.core.control.breakpoints import MetricBreakpoint
from happysimulator.visual import serve


class Server(QueuedResource):
    """Simple single-server queue with fixed processing time."""

    def __init__(self, name: str, downstream, service_time: float = 0.08):
        super().__init__(name)
        self._downstream = downstream
        self._service_time = service_time

    def handle_queued_event(self, event):
        yield self._service_time
        return [Event(
            time=self.now,
            event_type="Response",
            target=self._downstream,
            context=event.context,
        )]


sink = Sink("Sink")
server = Server("Server", downstream=sink, service_time=0.08)
source = Source.constant(rate=10, target=server, event_type="Request")

depth_data = Data()
depth_probe = Probe(target=server, metric="depth", data=depth_data, interval=0.1)

sim = Simulation(
    sources=[source],
    entities=[server, sink],
    probes=[depth_probe],
)

sim.control.add_breakpoint(MetricBreakpoint(
    entity_name="Sink",
    attribute="events_received",
    operator="ge",
    threshold=100,
))

serve(sim)
