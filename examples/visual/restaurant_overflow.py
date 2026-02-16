"""Restaurant overflow — visual debugger demo.

Simulates a 30-minute lunch rush where customers prefer Restaurant A, but
when its queue backs up the host diverts overflow to Restaurant B.

Launch with:
    python examples/visual/restaurant_overflow.py

Opens a browser at http://127.0.0.1:8765.  Click Play and watch the queue
depths on the Dashboard — time axes are displayed in minutes.
"""

from happysimulator.components.common import Sink
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.distributions import ExponentialLatency
from happysimulator.instrumentation.probe import Probe
from happysimulator.load.source import Source
from happysimulator.visual import Chart, serve


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class Restaurant(QueuedResource):
    """A restaurant with a fixed number of seats and servers."""

    def __init__(
        self,
        name: str,
        downstream: Entity,
        seats: int = 5,
        servers: int = 1,
        mean_service_time: float = 180.0,
        seed: int = 0,
    ):
        super().__init__(name)
        self._downstream = downstream
        self._seats = seats
        self._servers = servers
        self._service_time = ExponentialLatency(mean=mean_service_time, seed=seed)
        self._in_flight = 0

    def has_capacity(self) -> bool:
        return self._in_flight < self._servers

    def handle_queued_event(self, event):
        self._in_flight += 1
        try:
            yield self._service_time.sample()
        finally:
            self._in_flight -= 1
        return [self.forward(event, self._downstream, event_type="Satisfied")]


class Host(Entity):
    """Routes customers to Restaurant A, overflowing to B when A is busy."""

    def __init__(
        self,
        name: str,
        restaurant_a: Restaurant,
        restaurant_b: Restaurant,
        overflow_threshold: int = 3,
    ):
        super().__init__(name)
        self._restaurant_a = restaurant_a
        self._restaurant_b = restaurant_b
        self._overflow_threshold = overflow_threshold
        self.routed_to_b: int = 0
        # Expose targets for topology discovery
        self.targets = [restaurant_a, restaurant_b]

    def handle_event(self, event):
        if self._restaurant_a.depth >= self._overflow_threshold:
            self.routed_to_b += 1
            return [self.forward(event, self._restaurant_b, event_type="Seated")]
        return [self.forward(event, self._restaurant_a, event_type="Seated")]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

sink = Sink("Satisfied")

restaurant_a = Restaurant(
    "Restaurant A",
    downstream=sink,
    seats=5,
    servers=1,
    mean_service_time=180.0,  # ~3 min
    seed=1,
)
restaurant_b = Restaurant(
    "Restaurant B",
    downstream=sink,
    seats=10,
    servers=2,
    mean_service_time=240.0,  # ~4 min
    seed=2,
)

host = Host("Host", restaurant_a, restaurant_b, overflow_threshold=3)

# ~2 customers per minute (rate is per second internally)
source = Source.poisson(
    rate=2 / 60,
    target=host,
    event_type="Customer",
    stop_after=1800.0,  # 30 min of arrivals
    seed=42,
)

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

depth_a_probe, depth_a_data = Probe.on(restaurant_a, "depth", interval=6.0)
depth_b_probe, depth_b_data = Probe.on(restaurant_b, "depth", interval=6.0)
overflow_probe, overflow_data = Probe.on(host, "routed_to_b", interval=6.0)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

sim = Simulation(
    sources=[source],
    entities=[host, restaurant_a, restaurant_b, sink],
    probes=[depth_a_probe, depth_b_probe, overflow_probe],
    duration=2100.0,  # 35 min total (30 min arrivals + 5 min drain)
)

serve(
    sim,
    charts=[
        Chart(depth_a_data, title="Restaurant A Queue", y_label="customers", color="#3b82f6"),
        Chart(depth_b_data, title="Restaurant B Queue", y_label="customers", color="#10b981"),
        Chart(overflow_data, title="Customers Diverted to B", y_label="count", color="#f59e0b"),
    ],
    time_unit="min",
)
