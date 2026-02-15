"""Coffee shop discrete-event simulation with walk-in and mobile orders.

Walk-in and mobile customers arrive via Poisson processes. Mobile orders
get priority at the counter. A ConditionalRouter splits by drink type
(drip, espresso, blended). Drip uses a BatchProcessor (brew pot of 12),
while espresso and blended are individual QueuedResource stations.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                      COFFEE SHOP SIMULATION                            |
+-----------------------------------------------------------------------+

  +----------+   +----------+                     +-----------+
  | Walk-in  |-->|          |   +--------+ drip   | BatchProc |--+
  | Source   |   | Order    |-->| Drink  |------->| (pot: 12) |  |
  +----------+   | Counter  |   | Router |        +-----------+  |
  +----------+   |(Priority)|   |        | espresso +----------+ |  +------+
  | Mobile   |-->|          |   |        |--------->| Espresso |->->| Sink |
  | Source   |   +----------+   |        |          | Station  | |  +------+
  +----------+                  |        | blended  +----------+ |
                                |        |--------->| Blended  |-+
                                +--------+          | Station  |
                                                    +----------+
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
    EventProvider,
    FIFOQueue,
    Instant,
    LatencyTracker,
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.industrial import BatchProcessor, ConditionalRouter
from happysimulator.components.queue_policy import PriorityQueue

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class CoffeeShopConfig:
    """Configuration for the coffee shop simulation."""

    duration_s: float = 3600.0
    walkin_rate_per_min: float = 2.0
    mobile_rate_per_min: float = 1.0
    drip_pct: float = 0.40
    espresso_pct: float = 0.40
    blended_pct: float = 0.20
    order_time: float = 15.0
    batch_size: int = 12
    batch_brew_time: float = 180.0
    espresso_time: float = 45.0
    blended_time: float = 90.0
    seed: int = 42


DRINK_TYPES = ["drip", "espresso", "blended"]


# =============================================================================
# Event Providers
# =============================================================================


class CustomerProvider(EventProvider):
    """Generates customer arrival events."""

    def __init__(
        self,
        target: Entity,
        order_type: str,
        drink_pcts: list[float],
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._order_type = order_type
        self._drink_pcts = drink_pcts
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        r = random.random()
        cumulative = 0.0
        drink_type = DRINK_TYPES[-1]
        for dt, pct in zip(DRINK_TYPES, self._drink_pcts, strict=False):
            cumulative += pct
            if r < cumulative:
                drink_type = dt
                break

        return [
            Event(
                time=time,
                event_type="Order",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "drink_type": drink_type,
                    "order_type": self._order_type,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class OrderCounter(QueuedResource):
    """Counter where orders are taken. Mobile orders get priority."""

    def __init__(self, name: str, order_time: float, downstream: Entity):
        super().__init__(
            name,
            policy=PriorityQueue(
                key=lambda e: 0.0 if e.context.get("order_type") == "mobile" else 1.0
            ),
        )
        self.order_time = order_time
        self.downstream = downstream
        self._processed = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.order_time
        self._processed += 1
        return [self.forward(event, self.downstream, event_type="Order")]


class CoffeeStation(QueuedResource):
    """Generic coffee preparation station."""

    def __init__(self, name: str, service_time: float, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.service_time = service_time
        self.downstream = downstream
        self._processed = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.service_time
        self._processed += 1
        return [self.forward(event, self.downstream, event_type="Served")]


# =============================================================================
# Result
# =============================================================================


@dataclass
class CoffeeShopResult:
    """Results from the coffee shop simulation."""

    sink: LatencyTracker
    counter: OrderCounter
    router: ConditionalRouter
    drip_batch: BatchProcessor
    espresso_station: CoffeeStation
    blended_station: CoffeeStation
    walkin_provider: CustomerProvider
    mobile_provider: CustomerProvider
    config: CoffeeShopConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_coffee_shop_simulation(config: CoffeeShopConfig | None = None) -> CoffeeShopResult:
    """Run the coffee shop simulation."""
    if config is None:
        config = CoffeeShopConfig()

    random.seed(config.seed)

    # Build pipeline from end to start
    sink = LatencyTracker("Sink")

    espresso_station = CoffeeStation("EspressoStation", config.espresso_time, sink)
    blended_station = CoffeeStation("BlendedStation", config.blended_time, sink)
    drip_batch = BatchProcessor(
        "DripBrewer",
        downstream=sink,
        batch_size=config.batch_size,
        process_time=config.batch_brew_time,
        timeout_s=300.0,
    )

    router = ConditionalRouter.by_context_field(
        "DrinkRouter",
        "drink_type",
        {
            "drip": drip_batch,
            "espresso": espresso_station,
            "blended": blended_station,
        },
    )

    counter = OrderCounter("OrderCounter", config.order_time, router)

    stop_after = Instant.from_seconds(config.duration_s)
    drink_pcts = [config.drip_pct, config.espresso_pct, config.blended_pct]

    walkin_provider = CustomerProvider(counter, "walkin", drink_pcts, stop_after)
    mobile_provider = CustomerProvider(counter, "mobile", drink_pcts, stop_after)

    from happysimulator.load.profile import ConstantRateProfile
    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider

    walkin_source = Source(
        name="WalkIns",
        event_provider=walkin_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.walkin_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )
    mobile_source = Source(
        name="MobileOrders",
        event_provider=mobile_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.mobile_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 600)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[walkin_source, mobile_source],
        entities=[counter, router, drip_batch, espresso_station, blended_station, sink],
    )

    summary = sim.run()

    return CoffeeShopResult(
        sink=sink,
        counter=counter,
        router=router,
        drip_batch=drip_batch,
        espresso_station=espresso_station,
        blended_station=blended_station,
        walkin_provider=walkin_provider,
        mobile_provider=mobile_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: CoffeeShopResult) -> None:
    """Print a formatted summary of the coffee shop simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("COFFEE SHOP SIMULATION RESULTS")
    print("=" * 65)

    print("\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 60:.0f} minutes")
    print(f"  Walk-in rate:        {config.walkin_rate_per_min:.1f}/min")
    print(f"  Mobile rate:         {config.mobile_rate_per_min:.1f}/min")

    walkins = result.walkin_provider.generated
    mobiles = result.mobile_provider.generated
    total = walkins + mobiles

    print("\nCustomer Flow:")
    print(f"  Walk-in customers:   {walkins}")
    print(f"  Mobile customers:    {mobiles}")
    print(f"  Total:               {total}")
    print(f"  Orders processed:    {result.counter._processed}")

    print("\nDrink Routing:")
    for name, count in result.router.routed_counts.items():
        print(f"  {name:20s} {count}")
    print(f"  Dropped:             {result.router.dropped}")

    print("\nStation Stats:")
    print(f"  Drip batches:        {result.drip_batch.batches_processed}")
    print(f"  Drip items:          {result.drip_batch.items_processed}")
    print(f"  Espresso served:     {result.espresso_station._processed}")
    print(f"  Blended served:      {result.blended_station._processed}")

    completed = result.sink.count
    if completed > 0:
        print("\nEnd-to-End Latency:")
        print(f"  Completed:           {completed}")
        print(f"  Mean:    {result.sink.mean_latency():.1f}s")
        print(f"  p50:     {result.sink.p50():.1f}s")
        print(f"  p99:     {result.sink.p99():.1f}s")

    print(f"\n{result.summary}")
    print("=" * 65)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Coffee shop simulation")
    parser.add_argument("--duration", type=float, default=3600.0, help="Duration in seconds")
    parser.add_argument("--walkin-rate", type=float, default=2.0, help="Walk-in rate per minute")
    parser.add_argument(
        "--mobile-rate", type=float, default=1.0, help="Mobile order rate per minute"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = CoffeeShopConfig(
        duration_s=args.duration,
        walkin_rate_per_min=args.walkin_rate,
        mobile_rate_per_min=args.mobile_rate,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running coffee shop simulation...")
    result = run_coffee_shop_simulation(cfg)
    print_summary(result)
