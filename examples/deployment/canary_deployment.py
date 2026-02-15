"""Canary deployment example: progressive traffic shift with monitoring.

Demonstrates a canary deployment that progressively shifts traffic
from baseline servers to a canary instance, rolling back if the
canary degrades.

## Architecture

```
    Source (constant 30 req/s)
        |
        v
    LoadBalancer (WeightedRoundRobin)
        |                     CanaryDeployer
        v                       (stages: 5% -> 25% -> 100%)
    [Baseline Servers]            |
    [Canary Server]               v
        |                   Evaluates canary health
      Sink                  at each stage
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
from happysimulator.components.deployment import (
    CanaryDeployer,
    CanaryStage,
)
from happysimulator.components.load_balancer import LoadBalancer, WeightedRoundRobin
from happysimulator.components.server import Server


class RequestProvider(EventProvider):
    """Generates request events."""

    def __init__(self, target: Entity, stop_after: float = 50.0):
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


class AlwaysHealthyEvaluator:
    """Simple evaluator that always reports canary as healthy."""

    def is_healthy(self, canary: Entity, baseline_backends: list[Entity]) -> bool:
        return True


def make_server(name: str) -> Server:
    """Factory for canary servers."""
    return Server(name=name, concurrency=3, service_time=ConstantLatency(0.04))


def run_canary_deployment(
    duration_s: float = 60.0,
    seed: int = 42,
) -> None:
    """Run canary deployment demo."""
    random.seed(seed)

    # Baseline servers
    baseline_servers = [
        Server(f"baseline_{i}", concurrency=3, service_time=ConstantLatency(0.05))
        for i in range(3)
    ]
    sink = LatencyTracker(name="Sink")
    lb = LoadBalancer(
        name="LB", backends=baseline_servers,
        strategy=WeightedRoundRobin(),
    )

    deployer = CanaryDeployer(
        name="CanaryDeployer",
        load_balancer=lb,
        server_factory=make_server,
        stages=[
            CanaryStage(traffic_percentage=0.05, evaluation_period=5.0),
            CanaryStage(traffic_percentage=0.25, evaluation_period=5.0),
            CanaryStage(traffic_percentage=1.0, evaluation_period=5.0),
        ],
        metric_evaluator=AlwaysHealthyEvaluator(),
        evaluation_interval=2.0,
    )

    # Traffic
    provider = RequestProvider(lb, stop_after=duration_s - 5.0)
    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=30.0), start_time=Instant.Epoch,
    )
    source = Source(
        name="Traffic", event_provider=provider,
        arrival_time_provider=arrival,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s,
        sources=[source],
        entities=[lb, deployer, sink] + baseline_servers,
    )

    # Start deployment at t=5s
    deploy_event = Event(
        time=Instant.from_seconds(5.0),
        event_type="_canary_deploy_start",
        target=deployer,
        context={},
    )
    sim.schedule(deploy_event)
    summary = sim.run()

    # Results
    print("=" * 60)
    print("CANARY DEPLOYMENT RESULTS")
    print("=" * 60)
    print(f"\nDeployment state: {deployer.state.status}")
    print(f"  Current stage: {deployer.state.current_stage}/{deployer.state.total_stages}")
    print(f"  Canary traffic: {deployer.state.canary_traffic_pct*100:.0f}%")

    print(f"\nDeployer stats:")
    print(f"  Deployments started: {deployer.stats.deployments_started}")
    print(f"  Deployments completed: {deployer.stats.deployments_completed}")
    print(f"  Deployments rolled back: {deployer.stats.deployments_rolled_back}")
    print(f"  Stages completed: {deployer.stats.stages_completed}")
    print(f"  Evaluations: {deployer.stats.evaluations_performed} "
          f"({deployer.stats.evaluations_passed} passed, "
          f"{deployer.stats.evaluations_failed} failed)")

    print(f"\nCurrent backends: {[b.name for b in lb.all_backends]}")

    print(f"\n{summary}")
    print("=" * 60)


if __name__ == "__main__":
    run_canary_deployment()
