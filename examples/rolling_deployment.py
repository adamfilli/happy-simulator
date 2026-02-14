"""Rolling deployment example: replacing backends with health checking.

Demonstrates latency impact during a rolling deployment where backends
are replaced one-by-one with health checking at each step.

## Architecture

```
    Source (constant 20 req/s)
        |
        v
    LoadBalancer (RoundRobin)
        |                     RollingDeployer
        v                       (batch_size=1)
    [Servers v1]                  |
        |                         v
      Sink                  Replaces with v2 servers
```
"""

from __future__ import annotations

import random

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantLatency,
    ConstantRateProfile,
    Entity,
    Event,
    EventProvider,
    Instant,
    LatencyTracker,
    Simulation,
    Source,
)
from happysimulator.components.deployment import RollingDeployer
from happysimulator.components.load_balancer import LoadBalancer, RoundRobin
from happysimulator.components.server import Server


class RequestProvider(EventProvider):
    """Generates request events."""

    def __init__(self, target: Entity, stop_after: float = 30.0):
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
    """Factory for v2 servers (slightly faster)."""
    return Server(name=name, concurrency=3, service_time=ConstantLatency(0.04))


def run_rolling_deployment(
    duration_s: float = 40.0,
    seed: int = 42,
) -> None:
    """Run rolling deployment demo."""
    random.seed(seed)

    # Initial v1 servers
    v1_servers = [
        Server(f"server_v1_{i}", concurrency=3, service_time=ConstantLatency(0.05))
        for i in range(3)
    ]
    sink = LatencyTracker(name="Sink")
    lb = LoadBalancer(name="LB", backends=v1_servers, strategy=RoundRobin())

    deployer = RollingDeployer(
        name="Deployer",
        load_balancer=lb,
        server_factory=make_server,
        batch_size=1,
        health_check_interval=2.0,
        healthy_threshold=2,
        max_failures=3,
    )

    # Traffic
    provider = RequestProvider(lb, stop_after=duration_s - 5.0)
    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=20.0), start_time=Instant.Epoch,
    )
    source = Source(
        name="Traffic", event_provider=provider,
        arrival_time_provider=arrival,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s),
        sources=[source],
        entities=[lb, deployer, sink] + v1_servers,
    )

    # Start deployment at t=5s
    from happysimulator.core.temporal import Duration
    deploy_event = Event(
        time=Instant.from_seconds(5.0),
        event_type="_rolling_deploy_start",
        target=deployer,
        context={},
    )
    sim.schedule(deploy_event)
    summary = sim.run()

    # Results
    print("=" * 60)
    print("ROLLING DEPLOYMENT RESULTS")
    print("=" * 60)
    print(f"\nDeployment state: {deployer.state.status}")
    print(f"  Total instances: {deployer.state.total_instances}")
    print(f"  Replaced: {deployer.state.replaced}")
    print(f"  Failed: {deployer.state.failed}")

    print(f"\nDeployer stats:")
    print(f"  Deployments started: {deployer.stats.deployments_started}")
    print(f"  Deployments completed: {deployer.stats.deployments_completed}")
    print(f"  Deployments rolled back: {deployer.stats.deployments_rolled_back}")
    print(f"  Instances replaced: {deployer.stats.instances_replaced}")
    print(f"  Health checks: {deployer.stats.health_checks_performed} performed, "
          f"{deployer.stats.health_checks_passed} passed, "
          f"{deployer.stats.health_checks_failed} failed")

    print(f"\nCurrent backends: {[b.name for b in lb.all_backends]}")

    print(f"\n{summary}")
    print("=" * 60)


if __name__ == "__main__":
    run_rolling_deployment()
