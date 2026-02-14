"""Auto-scaler example: traffic spike -> scaling lag -> stabilization.

Demonstrates scaling lag when load suddenly increases, showing latency
spikes during the delay before new instances come online.

## Architecture

```
    Source (variable rate: 10 -> 50 -> 10 req/s)
        |
        v
    LoadBalancer (RoundRobin)
        |                     AutoScaler
        v                       (evaluates every 5s)
    [Servers]                   |
        |                       v
      Sink              server_factory creates new servers
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from happysimulator import (
    ConstantLatency,
    Entity,
    Event,
    EventProvider,
    Instant,
    LatencyTracker,
    PoissonArrivalTimeProvider,
    Profile,
    Simulation,
    Source,
)
from happysimulator.components.deployment import AutoScaler, TargetUtilization
from happysimulator.components.load_balancer import LoadBalancer, RoundRobin
from happysimulator.components.server import Server


@dataclass(frozen=True)
class SpikeProfile(Profile):
    """Traffic profile with a spike in the middle."""

    base_rate: float = 10.0
    spike_rate: float = 50.0
    spike_start: float = 20.0
    spike_end: float = 40.0

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        if self.spike_start <= t < self.spike_end:
            return self.spike_rate
        return self.base_rate


class RequestProvider(EventProvider):
    """Generates request events targeting the load balancer."""

    def __init__(self, target: Entity, stop_after: float = 60.0):
        self._target = target
        self._stop_after = Instant.from_seconds(stop_after)
        self._count = 0

    def get_events(self, time: Instant) -> list[Event]:
        if time > self._stop_after:
            return []
        self._count += 1
        return [Event(
            time=time, event_type="Request", target=self._target,
            context={"created_at": time},
        )]


def make_server(name: str) -> Server:
    """Factory for creating new server instances."""
    return Server(name=name, concurrency=5, service_time=ConstantLatency(0.05))


def run_auto_scaler_demo(
    duration_s: float = 80.0,
    seed: int = 42,
) -> None:
    """Run auto-scaler demo with traffic spike."""
    random.seed(seed)

    # Initial setup: 2 servers behind load balancer
    servers = [make_server(f"server_{i}") for i in range(2)]
    sink = LatencyTracker(name="Sink")
    lb = LoadBalancer(name="LB", backends=servers, strategy=RoundRobin())

    scaler = AutoScaler(
        name="AutoScaler",
        load_balancer=lb,
        server_factory=make_server,
        policy=TargetUtilization(target=0.6),
        min_instances=2,
        max_instances=10,
        evaluation_interval=5.0,
        scale_out_cooldown=10.0,
        scale_in_cooldown=20.0,
    )

    # Traffic source
    profile = SpikeProfile()
    provider = RequestProvider(lb, stop_after=duration_s - 10.0)
    arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(
        name="Traffic", event_provider=provider,
        arrival_time_provider=arrival,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s),
        sources=[source],
        entities=[lb, scaler, sink] + servers,
    )
    sim.schedule(scaler.start())
    summary = sim.run()

    # Results
    print("=" * 60)
    print("AUTO-SCALER SIMULATION RESULTS")
    print("=" * 60)
    print(f"\nTraffic profile:")
    print(f"  Base rate: {profile.base_rate} req/s")
    print(f"  Spike: {profile.spike_rate} req/s ({profile.spike_start}-{profile.spike_end}s)")

    print(f"\nScaler stats:")
    print(f"  Evaluations: {scaler.stats.evaluations}")
    print(f"  Scale-out: {scaler.stats.scale_out_count} ({scaler.stats.instances_added} added)")
    print(f"  Scale-in: {scaler.stats.scale_in_count} ({scaler.stats.instances_removed} removed)")
    print(f"  Cooldown blocks: {scaler.stats.cooldown_blocks}")
    print(f"  Final backend count: {len(lb.all_backends)}")

    print(f"\nScaling history:")
    for event in scaler.scaling_history:
        print(f"  t={event.time.to_seconds():.1f}s: {event.action} "
              f"{event.from_count}->{event.to_count} ({event.reason})")

    print(f"\n{summary}")
    print("=" * 60)


if __name__ == "__main__":
    run_auto_scaler_demo()
