"""Inductor burst suppression — visual debugger example.

Demonstrates the Inductor's EWMA-based smoothing across three traffic
patterns: step-up, linear ramp, and periodic bursts.  Unlike a rate
limiter the Inductor has **no throughput cap** — it only resists rapid
rate *changes*.

Launch with:
    python examples/performance/inductor_burst_suppression.py

Opens a browser at http://127.0.0.1:8765 with four charts showing
input/output throughput, EWMA rate estimate, and queue depth.
"""

from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.components.rate_limiter.inductor import Inductor
from happysimulator.instrumentation.collectors import ThroughputTracker
from happysimulator.instrumentation.data import Data
from happysimulator.instrumentation.probe import Probe
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.visual import serve, Chart


# -- Inline passthrough counter -----------------------------------------------

class InputCounter(Entity):
    """Counts arrivals (1.0 per event) then forwards to downstream."""

    def __init__(self, name: str, downstream: Entity) -> None:
        super().__init__(name)
        self.data = Data()
        self._downstream = downstream

    def handle_event(self, event: Event) -> list[Event]:
        self.data.add_stat(1.0, event.time)
        return [Event(
            time=event.time,
            event_type=event.event_type,
            target=self._downstream,
            context=event.context,
        )]


# -- Multi-phase profile ------------------------------------------------------

@dataclass(frozen=True)
class InductorShowcaseProfile(Profile):
    """Three demo phases separated by cooldown periods at base rate.

    Phase 1 (0–40s):   Step-up from 10 to 50 req/s at t=15s
    Cooldown (40–50s):  10 req/s
    Phase 2 (50–100s):  Linear ramp 10→50 req/s
    Cooldown (100–110s): 10 req/s
    Phase 3 (110–180s): Periodic bursts (10/80 req/s, period=10s, burst=2s)
    """

    base_rate: float = 10.0

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()

        # Phase 1: step-up
        if t < 15.0:
            return self.base_rate
        if t < 40.0:
            return 50.0

        # Cooldown
        if t < 50.0:
            return self.base_rate

        # Phase 2: linear ramp 10→50 over 50s
        if t < 100.0:
            fraction = (t - 50.0) / 50.0
            return self.base_rate + fraction * (50.0 - self.base_rate)

        # Cooldown
        if t < 110.0:
            return self.base_rate

        # Phase 3: periodic bursts (period=10s, burst first 2s at 80, rest at 10)
        t_in_cycle = (t - 110.0) % 10.0
        return 80.0 if t_in_cycle < 2.0 else self.base_rate


# -- Wiring --------------------------------------------------------------------

output_tracker = ThroughputTracker("OutputTracker")
inductor = Inductor("Inductor", downstream=output_tracker, time_constant=2.0)
input_counter = InputCounter("InputCounter", downstream=inductor)

source = Source.with_profile(
    profile=InductorShowcaseProfile(),
    target=input_counter,
    event_type="Request",
    poisson=False,
    name="LoadGenerator",
)

# Probes on inductor internals
rate_data = Data()
rate_probe = Probe(target=inductor, metric="estimated_rate", data=rate_data, interval=0.1)

depth_data = Data()
depth_probe = Probe(target=inductor, metric="queue_depth", data=depth_data, interval=0.1)

sim = Simulation(
    sources=[source],
    entities=[input_counter, inductor, output_tracker],
    probes=[rate_probe, depth_probe],
    end_time=Instant.from_seconds(180.0),
)

serve(sim, charts=[
    Chart(input_counter.data, title="Input Throughput",
          transform="rate", window_s=1.0, y_label="req/s", color="#6366f1"),
    Chart(output_tracker.data, title="Output Throughput",
          transform="rate", window_s=1.0, y_label="req/s", color="#10b981"),
    Chart(rate_data, title="EWMA Rate Estimate",
          transform="raw", y_label="req/s", color="#f59e0b"),
    Chart(depth_data, title="Queue Depth",
          transform="raw", y_label="items", color="#ef4444"),
])
