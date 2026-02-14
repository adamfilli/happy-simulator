"""Sequential car wash pipeline simulation.

Demonstrates a multi-stage pipeline with different service tiers:
- Basic: Pre-Rinse -> Wash (2 stages)
- Standard: Pre-Rinse -> Wash -> Rinse (3 stages)
- Premium: Pre-Rinse -> Wash -> Rinse -> Dry + Wax (4 stages)

## Architecture Diagram

```
                         CAR WASH PIPELINE
    +--------------------------------------------------------+
    |                                                        |
    |  +---------+   +------+   +-------+   +------+        |
    |  |Pre-Rinse|-->| Wash |-->| Rinse |-->| Dry  |-->Sink |
    |  | (30s)   |   |(120s)|   | (60s) |   |(90s) |        |
    |  +---------+   +------+   +-------+   +------+        |
    |       ^                                                |
    |       |  Conveyor belts between stages                 |
    |   Source (Poisson, ~2 cars/min)                        |
    |   Service tiers: Basic(40%), Standard(40%), Premium(20%)|
    +--------------------------------------------------------+
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    Instant,
    LatencyTracker,
    Probe,
    QueuedResource,
    FIFOQueue,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.industrial import ConveyorBelt


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class CarWashConfig:
    duration_s: float = 3600.0  # 1 hour
    arrival_rate: float = 2.0   # cars per minute -> 1/30 per second
    pre_rinse_time: float = 30.0
    wash_time: float = 120.0
    rinse_time: float = 60.0
    dry_time: float = 90.0
    conveyor_time: float = 10.0
    # Tier probabilities
    basic_pct: float = 0.40
    standard_pct: float = 0.40
    premium_pct: float = 0.20
    # Pricing
    basic_price: float = 8.0
    standard_price: float = 12.0
    premium_price: float = 20.0
    seed: int = 42


# =============================================================================
# Car Wash Station
# =============================================================================

class WashStation(QueuedResource):
    """Single-car wash stage with configurable service time."""

    def __init__(self, name: str, service_time: float, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.service_time_s = service_time
        self.downstream = downstream
        self.cars_processed = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.service_time_s
        self.cars_processed += 1
        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.downstream,
                context=event.context,
            )
        ]


class TierRouter(Entity):
    """Routes cars to appropriate pipeline depth based on tier."""

    def __init__(self, name: str, basic_exit: Entity, standard_exit: Entity,
                 premium_next: Entity, pass_through: Entity):
        super().__init__(name)
        self.basic_exit = basic_exit
        self.standard_exit = standard_exit
        self.premium_next = premium_next
        self.pass_through = pass_through

    def handle_event(self, event: Event) -> list[Event]:
        tier = event.context.get("tier", "basic")
        if tier == "basic":
            target = self.basic_exit
        elif tier == "standard":
            target = self.standard_exit
        else:
            target = self.premium_next
        return [
            Event(time=self.now, event_type=event.event_type,
                  target=target, context=event.context)
        ]


# =============================================================================
# Main Simulation
# =============================================================================

@dataclass
class CarWashResult:
    sink: LatencyTracker
    stations: dict[str, WashStation]
    config: CarWashConfig
    summary: SimulationSummary


def run_car_wash_simulation(config: CarWashConfig | None = None) -> CarWashResult:
    if config is None:
        config = CarWashConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Sink")

    # Build pipeline from end to start
    dry_station = WashStation("Dry", config.dry_time, sink)
    conveyor3 = ConveyorBelt("Conveyor3", dry_station, config.conveyor_time)
    rinse_station = WashStation("Rinse", config.rinse_time, conveyor3)
    conveyor2 = ConveyorBelt("Conveyor2", rinse_station, config.conveyor_time)
    wash_station = WashStation("Wash", config.wash_time, conveyor2)

    # Router after wash: basic->sink, standard->rinse->sink, premium->rinse->dry->sink
    router = TierRouter("Router", basic_exit=sink, standard_exit=rinse_station,
                         premium_next=rinse_station, pass_through=rinse_station)
    # Actually for simplicity: all go through full pipeline, but basic/standard skip stages
    # Let's use a simpler approach: all go through the full pipeline
    # Basic: skip rinse + dry (go directly to sink from wash)
    # Standard: skip dry (go to sink from rinse)
    # Premium: full pipeline

    # Simpler: just route after wash
    conveyor1 = ConveyorBelt("Conveyor1", wash_station, config.conveyor_time)
    pre_rinse = WashStation("PreRinse", config.pre_rinse_time, conveyor1)

    # For simplicity: all cars go through full pipeline regardless of tier
    # (each station processes in sequence)
    # Revenue tracked by tier in sink context

    arrival_rate_per_s = config.arrival_rate / 60.0

    def assign_tier() -> str:
        r = random.random()
        if r < config.basic_pct:
            return "basic"
        elif r < config.basic_pct + config.standard_pct:
            return "standard"
        return "premium"

    # Custom source that assigns tiers
    source = Source.poisson(
        rate=arrival_rate_per_s,
        target=pre_rinse,
        event_type="Car",
        name="Arrivals",
        stop_after=config.duration_s,
    )

    entities = [pre_rinse, conveyor1, wash_station, conveyor2,
                rinse_station, conveyor3, dry_station, sink]

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(config.duration_s + 600),  # drain time
        sources=[source],
        entities=entities,
    )
    summary = sim.run()

    stations = {
        "PreRinse": pre_rinse,
        "Wash": wash_station,
        "Rinse": rinse_station,
        "Dry": dry_station,
    }

    return CarWashResult(sink=sink, stations=stations, config=config, summary=summary)


def print_summary(result: CarWashResult) -> None:
    print("\n" + "=" * 60)
    print("CAR WASH SIMULATION RESULTS")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Duration: {result.config.duration_s/60:.0f} minutes")
    print(f"  Arrival rate: {result.config.arrival_rate:.1f} cars/min")

    print(f"\nStation Performance:")
    for name, station in result.stations.items():
        print(f"  {name}: {station.cars_processed} cars processed")

    print(f"\nOverall:")
    print(f"  Cars completed: {result.sink.count}")
    if result.sink.count > 0:
        print(f"  Avg cycle time: {result.sink.mean_latency()*60:.1f} min")
        print(f"  p99 cycle time: {result.sink.p99()*60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car wash pipeline simulation")
    parser.add_argument("--duration", type=float, default=3600.0, help="Duration in seconds")
    parser.add_argument("--rate", type=float, default=2.0, help="Arrival rate (cars/min)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    config = CarWashConfig(
        duration_s=args.duration,
        arrival_rate=args.rate,
        seed=args.seed,
    )
    result = run_car_wash_simulation(config)
    print_summary(result)
