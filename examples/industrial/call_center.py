"""Call center simulation with IVR, skill-based routing, and shift changes.

## Architecture Diagram

```
                          CALL CENTER
    +-----------------------------------------------------------+
    |                                                           |
    |  Source -> IVR (30s) -> Router -> Sales Queue   -> Sink   |
    | (Poisson)              (skill)    Support Queue           |
    |                                   Billing Queue           |
    |                                                           |
    |  Shift schedule: Morning(8), Afternoon(12), Evening(4)   |
    |  Customer abandonment (reneging with exp patience)        |
    +-----------------------------------------------------------+
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator import (
    Entity,
    Event,
    FIFOQueue,
    Instant,
    LatencyTracker,
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.industrial import (
    RenegingQueuedResource,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class CallCenterConfig:
    duration_s: float = 28800.0  # 8 hours
    arrival_rate: float = 0.05  # calls per second (~180/hr)
    ivr_time: float = 30.0
    # Skill routing probabilities
    sales_pct: float = 0.30
    support_pct: float = 0.50
    billing_pct: float = 0.20
    # Service times (seconds)
    sales_service: float = 300.0
    support_service: float = 480.0
    billing_service: float = 180.0
    # Patience
    mean_patience_s: float = 300.0  # 5 min average patience
    # Shifts (agents per queue)
    morning_agents: int = 8
    afternoon_agents: int = 12
    evening_agents: int = 4
    seed: int = 42


class IVRMenu(QueuedResource):
    """Interactive Voice Response - fixed delay then route to queue."""

    def __init__(self, name: str, router: Entity, ivr_time: float):
        super().__init__(name, policy=FIFOQueue())
        self.router = router
        self.ivr_time = ivr_time
        self.calls_processed = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.ivr_time
        self.calls_processed += 1
        # Assign skill category
        r = random.random()
        if r < 0.30:
            skill = "sales"
        elif r < 0.80:
            skill = "support"
        else:
            skill = "billing"
        ctx = dict(event.context)
        ctx["skill"] = skill
        ctx["patience_s"] = random.expovariate(1.0 / 300.0)
        return [Event(time=self.now, event_type="Call", target=self.router, context=ctx)]


class SkillRouter(Entity):
    """Routes calls to the appropriate skill queue."""

    def __init__(self, name: str, queues: dict[str, Entity]):
        super().__init__(name)
        self.queues = queues

    def handle_event(self, event: Event) -> list[Event]:
        skill = event.context.get("skill", "support")
        target = self.queues.get(skill, next(iter(self.queues.values())))
        return [self.forward(event, target)]


class AgentPool(RenegingQueuedResource):
    """Pool of call center agents with reneging support."""

    def __init__(
        self,
        name: str,
        service_time: float,
        num_agents: int,
        downstream: Entity,
        reneged_target: Entity,
    ):
        super().__init__(
            name,
            reneged_target=reneged_target,
            default_patience_s=300.0,
            policy=FIFOQueue(),
        )
        self.service_time_s = service_time
        self._concurrency = num_agents
        self._active = 0
        self.downstream = downstream
        self.calls_handled = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def set_agents(self, count: int) -> None:
        self._concurrency = count

    def _handle_served_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._active += 1
        try:
            yield random.expovariate(1.0 / self.service_time_s)
        finally:
            self._active -= 1
        self.calls_handled += 1
        return [self.forward(event, self.downstream, event_type="Completed")]


@dataclass
class CallCenterResult:
    sink: LatencyTracker
    abandoned: LatencyTracker
    ivr: IVRMenu
    pools: dict[str, AgentPool]
    config: CallCenterConfig
    summary: SimulationSummary


def run_call_center_simulation(config: CallCenterConfig | None = None) -> CallCenterResult:
    if config is None:
        config = CallCenterConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Completed")
    abandoned = LatencyTracker("Abandoned")

    # Agent pools
    pools = {
        "sales": AgentPool("Sales", config.sales_service, config.morning_agents, sink, abandoned),
        "support": AgentPool(
            "Support", config.support_service, config.morning_agents, sink, abandoned
        ),
        "billing": AgentPool(
            "Billing", config.billing_service, config.morning_agents, sink, abandoned
        ),
    }

    router = SkillRouter("Router", pools)
    ivr = IVRMenu("IVR", router, config.ivr_time)

    source = Source.poisson(
        rate=config.arrival_rate,
        target=ivr,
        event_type="IncomingCall",
        name="Calls",
        stop_after=config.duration_s,
    )

    entities = [ivr, router, *pools.values(), sink, abandoned]

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 1800,
        sources=[source],
        entities=entities,
    )
    summary = sim.run()

    return CallCenterResult(
        sink=sink,
        abandoned=abandoned,
        ivr=ivr,
        pools=pools,
        config=config,
        summary=summary,
    )


def print_summary(result: CallCenterResult) -> None:
    print("\n" + "=" * 60)
    print("CALL CENTER SIMULATION RESULTS")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Duration: {result.config.duration_s / 3600:.0f} hours")
    print(f"  Arrival rate: {result.config.arrival_rate * 3600:.0f} calls/hr")

    print(f"\nIVR: {result.ivr.calls_processed} calls processed")

    print("\nAgent Pools:")
    for name, pool in result.pools.items():
        stats = pool.reneging_stats
        print(
            f"  {name.title()}: handled={pool.calls_handled}, "
            f"served={stats.served}, reneged={stats.reneged}"
        )

    print("\nOutcomes:")
    print(f"  Completed: {result.sink.count}")
    print(f"  Abandoned: {result.abandoned.count}")
    total = result.sink.count + result.abandoned.count
    if total > 0:
        print(f"  Answer rate: {result.sink.count / total * 100:.1f}%")

    if result.sink.count > 0:
        print("\nService Levels:")
        print(f"  Avg wait+handle: {result.sink.mean_latency() / 60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Call center simulation")
    parser.add_argument("--duration", type=float, default=28800.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = CallCenterConfig(duration_s=args.duration, seed=args.seed)
    result = run_call_center_simulation(config)
    print_summary(result)
