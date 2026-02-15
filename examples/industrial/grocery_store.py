"""Grocery store checkout simulation with express lanes and self-checkout.

## Architecture Diagram

```
                       GROCERY STORE CHECKOUT
    +-----------------------------------------------------------+
    |                                                           |
    |  Source -> Chooser -> Regular Lane 1-4 (FIFO)    -> Sink  |
    | (Poisson)          -> Express Lane   (max 15 items)       |
    |                    -> Self-Checkout 1-6                   |
    |                                                           |
    |  Customer balking when all queues > threshold             |
    |  Self-checkout jams (breakdowns)                          |
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
    BreakdownScheduler,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class GroceryConfig:
    duration_s: float = 7200.0  # 2 hours
    arrival_rate: float = 0.1  # customers/sec (~360/hr)
    num_regular: int = 4
    num_self_checkout: int = 6
    express_item_limit: int = 15
    regular_service_time: float = 120.0  # 2 min avg
    express_service_time: float = 45.0
    self_checkout_time: float = 90.0
    balk_threshold: int = 6
    # Self-checkout jam parameters
    jam_mttf: float = 600.0  # 10 min between jams
    jam_mttr: float = 60.0  # 1 min to fix
    seed: int = 42


class CheckoutLane(QueuedResource):
    """Single checkout lane."""

    def __init__(self, name: str, service_time: float, downstream: Entity, policy=None):
        super().__init__(name, policy=policy or FIFOQueue())
        self.mean_service_time = service_time
        self.downstream = downstream
        self.customers_served = 0
        self._broken = False
        self._active = 0

    def has_capacity(self) -> bool:
        return not self._broken and self._active < 1

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._active += 1
        try:
            items = event.context.get("items", 10)
            # Service time scales with items
            base = self.mean_service_time * (items / 15.0)
            yield random.expovariate(1.0 / max(base, 10.0))
            self.customers_served += 1
        finally:
            self._active -= 1
        return [self.forward(event, self.downstream, event_type="Done")]


class LaneChooser(Entity):
    """Routes customers to shortest queue, respecting express rules."""

    def __init__(
        self,
        name: str,
        regular: list[CheckoutLane],
        express: CheckoutLane,
        self_checkouts: list[CheckoutLane],
        express_limit: int,
        balked_target: Entity,
        balk_threshold: int = 6,
    ):
        super().__init__(name)
        self.regular = regular
        self.express = express
        self.self_checkouts = self_checkouts
        self.express_limit = express_limit
        self.balked_target = balked_target
        self.balk_threshold = balk_threshold
        self.routed = 0
        self.balked = 0

    def handle_event(self, event: Event) -> list[Event]:
        items = random.randint(1, 40)
        ctx = dict(event.context)
        ctx["items"] = items

        # Find shortest queue across all options
        candidates: list[tuple[int, CheckoutLane]] = [(lane.depth, lane) for lane in self.regular]

        if items <= self.express_limit:
            candidates.append((self.express.depth, self.express))

        candidates.extend(
            (sc.depth, sc) for sc in self.self_checkouts if not getattr(sc, "_broken", False)
        )

        if not candidates:
            self.balked += 1
            return []

        # Check balking: if shortest queue is still long, customer leaves
        candidates.sort(key=lambda x: x[0])
        shortest_depth = candidates[0][0]

        if shortest_depth >= self.balk_threshold and random.random() < 0.7:  # 70% chance of balking
            self.balked += 1
            return [
                Event(time=self.now, event_type="Balked", target=self.balked_target, context=ctx)
            ]

        # Choose shortest
        target = candidates[0][1]
        self.routed += 1
        return [Event(time=self.now, event_type="Checkout", target=target, context=ctx)]


@dataclass
class GroceryResult:
    sink: LatencyTracker
    balked_sink: LatencyTracker
    chooser: LaneChooser
    regular_lanes: list[CheckoutLane]
    express: CheckoutLane
    self_checkouts: list[CheckoutLane]
    config: GroceryConfig
    summary: SimulationSummary


def run_grocery_simulation(config: GroceryConfig | None = None) -> GroceryResult:
    if config is None:
        config = GroceryConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Completed")
    balked_sink = LatencyTracker("Balked")

    # Create lanes
    regular = [
        CheckoutLane(f"Regular{i + 1}", config.regular_service_time, sink)
        for i in range(config.num_regular)
    ]
    express = CheckoutLane("Express", config.express_service_time, sink)
    self_checkouts = [
        CheckoutLane(f"SelfCheckout{i + 1}", config.self_checkout_time, sink)
        for i in range(config.num_self_checkout)
    ]

    # Breakdowns for self-checkouts
    breakdowns = []
    for sc in self_checkouts:
        bd = BreakdownScheduler(
            f"{sc.name}_BD",
            target=sc,
            mean_time_to_failure=config.jam_mttf,
            mean_repair_time=config.jam_mttr,
        )
        breakdowns.append(bd)

    chooser = LaneChooser(
        "Chooser",
        regular,
        express,
        self_checkouts,
        config.express_item_limit,
        balked_sink,
        balk_threshold=config.balk_threshold,
    )

    source = Source.poisson(
        rate=config.arrival_rate,
        target=chooser,
        event_type="Customer",
        name="Arrivals",
        stop_after=config.duration_s,
    )

    entities = [chooser, *regular, express, *self_checkouts, *breakdowns, sink, balked_sink]

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 600,
        sources=[source],
        entities=entities,
    )
    for bd in breakdowns:
        sim.schedule(bd.start_event())
    summary = sim.run()

    return GroceryResult(
        sink=sink,
        balked_sink=balked_sink,
        chooser=chooser,
        regular_lanes=regular,
        express=express,
        self_checkouts=self_checkouts,
        config=config,
        summary=summary,
    )


def print_summary(result: GroceryResult) -> None:
    print("\n" + "=" * 60)
    print("GROCERY STORE SIMULATION RESULTS")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Duration: {result.config.duration_s / 60:.0f} minutes")
    print(
        f"  Lanes: {result.config.num_regular} regular, 1 express, "
        f"{result.config.num_self_checkout} self-checkout"
    )

    print("\nCustomer Routing:")
    print(f"  Routed: {result.chooser.routed}")
    print(f"  Balked: {result.chooser.balked}")

    print("\nLane Performance:")
    for lane in result.regular_lanes:
        print(f"  {lane.name}: {lane.customers_served} served")
    print(f"  {result.express.name}: {result.express.customers_served} served")
    for sc in result.self_checkouts:
        print(f"  {sc.name}: {sc.customers_served} served")

    print("\nOverall:")
    print(f"  Completed: {result.sink.count}")
    if result.sink.count > 0:
        print(f"  Avg checkout time: {result.sink.mean_latency() / 60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grocery store simulation")
    parser.add_argument("--duration", type=float, default=7200.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = GroceryConfig(duration_s=args.duration, seed=args.seed)
    result = run_grocery_simulation(config)
    print_summary(result)
