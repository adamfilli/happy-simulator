"""Drive-through restaurant discrete-event simulation.

Single-lane pipeline: OrderBoard → Kitchen → PaymentWindow → PickupWindow.
ConditionalRouter splits simple vs complex orders to fast/slow kitchen
paths. BalkingQueue at entrance rejects customers when queue is long.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                    DRIVE-THROUGH SIMULATION                            |
+-----------------------------------------------------------------------+

  +---------+   +-----------+   +----------+  simple   +-----------+
  | Source  |-->| Balking   |-->| Order    |---------->| Fast      |--+
  |(Poisson)|   | Queue     |   | Board    |           | Kitchen   |  |
  +---------+   | (thresh=6)|   +----------+  complex  | (60s)     |  |
                +-----------+       |       --------->+-----------+  |
                                    |                  | Slow      |--+
                                    |                  | Kitchen   |  |
                                    |                  | (180s)    |  |
                                    +----+-------------+-----------+  |
                                         |                            |
                                    +---------+   +----------+   +------+
                                    | Payment |-->| Pickup   |-->| Sink |
                                    | Window  |   | Window   |   |      |
                                    | (30s)   |   | (15s)    |   +------+
                                    +---------+   +----------+
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    LatencyTracker,
    Probe,
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter
from happysimulator.components.industrial import BalkingQueue, ConditionalRouter


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class DriveThruConfig:
    """Configuration for the drive-through simulation."""

    duration_s: float = 3600.0
    arrival_rate_per_min: float = 1.5
    balk_threshold: int = 6
    simple_pct: float = 0.60
    order_time: float = 30.0
    fast_kitchen_time: float = 60.0
    slow_kitchen_time: float = 180.0
    payment_time: float = 30.0
    pickup_time: float = 15.0
    seed: int = 42


# =============================================================================
# Event Provider
# =============================================================================


class CarProvider(EventProvider):
    """Generates drive-through car arrival events."""

    def __init__(self, target: Entity, simple_pct: float, stop_after: Instant | None = None):
        self._target = target
        self._simple_pct = simple_pct
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        order_type = "simple" if random.random() < self._simple_pct else "complex"

        return [
            Event(
                time=time,
                event_type="Car",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "order_type": order_type,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class Station(QueuedResource):
    """Generic drive-through station with configurable service time."""

    def __init__(self, name: str, service_time: float, downstream: Entity, policy=None):
        super().__init__(name, policy=policy or FIFOQueue())
        self.service_time = service_time
        self.downstream = downstream
        self._processed = 0

    @property
    def processed(self) -> int:
        return self._processed

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.service_time
        self._processed += 1
        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.downstream,
                context=event.context,
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class DriveThruResult:
    """Results from the drive-through simulation."""

    sink: LatencyTracker
    order_board: Station
    fast_kitchen: Station
    slow_kitchen: Station
    payment: Station
    pickup: Station
    router: ConditionalRouter
    balking_queue: BalkingQueue
    car_provider: CarProvider
    config: DriveThruConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_drive_through_simulation(config: DriveThruConfig | None = None) -> DriveThruResult:
    """Run the drive-through simulation."""
    if config is None:
        config = DriveThruConfig()

    random.seed(config.seed)

    # Build pipeline from end to start
    sink = LatencyTracker("Sink")
    pickup = Station("PickupWindow", config.pickup_time, sink)
    payment = Station("PaymentWindow", config.payment_time, pickup)
    fast_kitchen = Station("FastKitchen", config.fast_kitchen_time, payment)
    slow_kitchen = Station("SlowKitchen", config.slow_kitchen_time, payment)

    router = ConditionalRouter.by_context_field(
        "KitchenRouter",
        "order_type",
        {"simple": fast_kitchen, "complex": slow_kitchen},
    )

    balking_queue = BalkingQueue(inner=FIFOQueue(), balk_threshold=config.balk_threshold)
    order_board = Station("OrderBoard", config.order_time, router, policy=balking_queue)

    stop_after = Instant.from_seconds(config.duration_s)
    car_provider = CarProvider(order_board, config.simple_pct, stop_after)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    source = Source(
        name="Cars",
        event_provider=car_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.arrival_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 600)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[order_board, router, fast_kitchen, slow_kitchen, payment, pickup, sink],
    )

    summary = sim.run()

    return DriveThruResult(
        sink=sink,
        order_board=order_board,
        fast_kitchen=fast_kitchen,
        slow_kitchen=slow_kitchen,
        payment=payment,
        pickup=pickup,
        router=router,
        balking_queue=balking_queue,
        car_provider=car_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: DriveThruResult) -> None:
    """Print a formatted summary of the drive-through simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("DRIVE-THROUGH SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 60:.0f} minutes")
    print(f"  Arrival rate:        {config.arrival_rate_per_min:.1f} cars/min")
    print(f"  Balk threshold:      {config.balk_threshold}")

    total = result.car_provider.generated
    balked = result.balking_queue.balked

    print(f"\nCustomer Flow:")
    print(f"  Arrived:             {total}")
    print(f"  Balked:              {balked} ({100 * balked / max(total, 1):.1f}%)")
    print(f"  Orders taken:        {result.order_board.processed}")

    print(f"\nKitchen Routing:")
    for name, count in result.router.routed_counts.items():
        print(f"  {name:20s} {count}")

    print(f"\nStation Throughput:")
    print(f"  Order board:         {result.order_board.processed}")
    print(f"  Fast kitchen:        {result.fast_kitchen.processed}")
    print(f"  Slow kitchen:        {result.slow_kitchen.processed}")
    print(f"  Payment:             {result.payment.processed}")
    print(f"  Pickup:              {result.pickup.processed}")

    completed = result.sink.count
    if completed > 0:
        print(f"\nEnd-to-End Latency:")
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
    parser = argparse.ArgumentParser(description="Drive-through simulation")
    parser.add_argument("--duration", type=float, default=3600.0, help="Duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=1.5, help="Cars per minute")
    parser.add_argument("--balk-threshold", type=int, default=6, help="Balk threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = DriveThruConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        balk_threshold=args.balk_threshold,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running drive-through simulation...")
    result = run_drive_through_simulation(cfg)
    print_summary(result)
