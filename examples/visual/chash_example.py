"""Consistent hashing load balancer — visual debugger demo.

Simulates 3 backend nodes behind a consistent-hash load balancer.
Traffic follows a dynamic profile: a warm-up ramp, a steady phase, a spike,
and a cool-down — so you can watch how consistent hashing distributes load
as traffic changes.

Launch with:
    python examples/visual/chash_example.py

Opens a browser at http://127.0.0.1:8765.
"""

import random

from happysimulator.components.common import Sink
from happysimulator.components.load_balancer.load_balancer import LoadBalancer
from happysimulator.components.load_balancer.strategies import ConsistentHash
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions import ExponentialLatency
from happysimulator.instrumentation.probe import Probe
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.visual import Chart, serve


# ---------------------------------------------------------------------------
# Dynamic traffic profile
# ---------------------------------------------------------------------------


class DynamicTrafficProfile(Profile):
    """Multi-phase traffic that ramps up, spikes, and cools down.

    Timeline (seconds):
      0-15   ramp from 5 → 30 req/s
      15-45  steady at 30 req/s
      45-60  spike to 80 req/s
      60-75  cool down from 80 → 20 req/s
      75-90  steady at 20 req/s
    """

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        if t < 15:
            return 5 + (30 - 5) * (t / 15)
        if t < 45:
            return 30
        if t < 60:
            return 80
        if t < 75:
            frac = (t - 60) / 15
            return 80 + (20 - 80) * frac
        return 20


# ---------------------------------------------------------------------------
# Event provider that assigns random client IDs
# ---------------------------------------------------------------------------

NUM_CLIENTS = 200


class ClientRequestProvider(EventProvider):
    """Generates requests with random client_id metadata for hashing."""

    def __init__(self, target, stop_after: Instant | None = None, seed: int = 42):
        self._target = target
        self._stop_after = stop_after
        self._rng = random.Random(seed)

    def get_events(self, time: Instant) -> Event | list[Event]:
        if self._stop_after and time >= self._stop_after:
            return []
        client_id = self._rng.randint(0, NUM_CLIENTS - 1)
        return [Event(
            time=time,
            event_type="Request",
            target=self._target,
            context={"metadata": {"client_id": str(client_id)}},
        )]


# ---------------------------------------------------------------------------
# Backend node
# ---------------------------------------------------------------------------


class Node(QueuedResource):
    """A backend server with limited concurrency and exponential service time."""

    def __init__(self, name: str, downstream, concurrency: int = 3, mean_service: float = 0.1):
        super().__init__(name)
        self._downstream = downstream
        self._concurrency = concurrency
        self._service = ExponentialLatency(mean_latency=mean_service)
        self._in_flight = 0

    def has_capacity(self) -> bool:
        return self._in_flight < self._concurrency

    def handle_queued_event(self, event):
        self._in_flight += 1
        try:
            yield self._service.get_latency(self.now).to_seconds()
        finally:
            self._in_flight -= 1
        return [self.forward(event, self._downstream, event_type="Response")]


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------

sink = Sink("Sink")

nodes = [Node(f"Node {i}", downstream=sink) for i in range(3)]

lb = LoadBalancer(
    name="Router",
    backends=nodes,
    strategy=ConsistentHash(virtual_nodes=150),
)

source = Source.with_profile(
    profile=DynamicTrafficProfile(),
    name="Traffic",
    poisson=True,
    event_provider=ClientRequestProvider(
        target=lb,
        stop_after=Instant.from_seconds(90),
    ),
)

# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

lb_probe, lb_data = Probe.on(lb, "_requests_received", interval=0.5)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

sim = Simulation(
    sources=[source],
    entities=[lb, *nodes, sink],
    probes=[lb_probe],
    duration=100.0,
)

serve(
    sim,
    charts=[
        Chart(lb_data, title="Total Requests", y_label="count", color="#6366f1"),
    ],
)
