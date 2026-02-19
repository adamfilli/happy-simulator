"""Visual debugger widget demo — custom entity animations.

Launch with:
    python examples/visual/widget_demo.py

Opens a browser at http://127.0.0.1:8765 showing the entity graph.
The Server node renders with a slots widget (filled circles = active slots),
and the Buffer node renders with a queue widget (filled squares = queued items).
"""

from happysimulator.components.common import Sink
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.components.server.server import Server
from happysimulator.core.simulation import Simulation
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.instrumentation.probe import Probe
from happysimulator.load.source import Source
from happysimulator.visual import Chart, serve


class Buffer(QueuedResource):
    """Pass-through buffer that visually shows queue depth."""

    def __init__(self, name, downstream, capacity=20):
        super().__init__(name, policy=FIFOQueue(capacity=capacity))
        self._downstream = downstream

    def handle_queued_event(self, event):
        yield 0.001  # minimal processing
        return [self.forward(event, self._downstream)]

    def downstream_entities(self):
        return [self._downstream]

    def visual_widget(self):
        return {"type": "queue", "depth": "depth"}


# Build pipeline: Source -> Buffer -> Server -> Sink
sink = Sink()
server = Server(
    "Processor",
    concurrency=4,
    service_time=ConstantLatency(0.05),
    downstream=sink,
)
buffer = Buffer("Buffer", downstream=server)
source = Source.poisson(rate=50, target=buffer, name="Traffic")

probe, data = Probe.on(server, "depth", interval=0.1)

sim = Simulation(
    sources=[source],
    entities=[buffer, server, sink],
    probes=[probe],
    duration=60.0,
)

serve(sim, charts=[Chart(data, title="Processor Queue Depth")])
