"""Elevator system discrete-event simulation.

Sources per floor (different rates) → FloorQueues → ElevatorDispatcher
→ GateController (doors) → ElevatorCar (Resource, 8 capacity, travel
time proportional to floors) → destination Sink. 3 elevators, 10 floors.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                     ELEVATOR SYSTEM SIMULATION                         |
+-----------------------------------------------------------------------+

  Floor 0: Source --> +                      +---> Elevator 0 (cap=8)
  Floor 1: Source --> |   +-----------+      |     travel_time = |floors| * 3s
  Floor 2: Source --> +-->| Dispatcher|------+---> Elevator 1 (cap=8)
  ...                |   | (assigns  |      |
  Floor 9: Source --> +   |  elevator)|      +---> Elevator 2 (cap=8)
                          +-----------+                    |
                                                           v
                                                       +------+
                                                       | Sink |
                                                       +------+
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
    Resource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ElevatorConfig:
    """Configuration for the elevator system simulation."""

    duration_s: float = 7200.0        # 2 hours
    num_floors: int = 10
    num_elevators: int = 3
    elevator_capacity: int = 8
    floor_travel_time: float = 3.0    # seconds per floor
    door_time: float = 5.0            # door open/close
    lobby_rate_per_min: float = 2.0   # ground floor
    upper_rate_per_min: float = 0.3   # per upper floor
    seed: int = 42


# =============================================================================
# Event Provider
# =============================================================================


class FloorProvider(EventProvider):
    """Generates passenger events from a specific floor."""

    def __init__(self, target: Entity, floor: int, num_floors: int,
                 stop_after: Instant | None = None):
        self._target = target
        self._floor = floor
        self._num_floors = num_floors
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        # Random destination (different from origin)
        dest = self._floor
        while dest == self._floor:
            dest = random.randint(0, self._num_floors - 1)

        return [
            Event(
                time=time,
                event_type="Passenger",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "origin_floor": self._floor,
                    "dest_floor": dest,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class ElevatorCar(Entity):
    """Single elevator car that transports passengers."""

    def __init__(self, name: str, capacity: int, floor_travel_time: float,
                 door_time: float, downstream: Entity):
        super().__init__(name)
        self._capacity = capacity
        self.floor_travel_time = floor_travel_time
        self.door_time = door_time
        self.downstream = downstream
        self._current_floor = 0
        self._trips = 0
        self._passengers_carried = 0

    @property
    def trips(self) -> int:
        return self._trips

    @property
    def passengers_carried(self) -> int:
        return self._passengers_carried

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        origin = event.context.get("origin_floor", 0)
        dest = event.context.get("dest_floor", 0)

        # Travel to origin floor
        floors_to_origin = abs(origin - self._current_floor)
        if floors_to_origin > 0:
            yield floors_to_origin * self.floor_travel_time

        # Door open + boarding
        yield self.door_time

        self._current_floor = origin

        # Travel to destination
        floors_to_dest = abs(dest - origin)
        if floors_to_dest > 0:
            yield floors_to_dest * self.floor_travel_time

        # Door open + alighting
        yield self.door_time

        self._current_floor = dest
        self._trips += 1
        self._passengers_carried += 1

        return [
            Event(
                time=self.now,
                event_type="Arrived",
                target=self.downstream,
                context=event.context,
            )
        ]


class ElevatorDispatcher(Entity):
    """Dispatches passengers to elevators round-robin."""

    def __init__(self, name: str, elevators: list[ElevatorCar]):
        super().__init__(name)
        self.elevators = elevators
        self._next_elevator = 0
        self._dispatched = 0

    def handle_event(self, event: Event) -> list[Event]:
        elevator = self.elevators[self._next_elevator]
        self._next_elevator = (self._next_elevator + 1) % len(self.elevators)
        self._dispatched += 1

        return [
            Event(
                time=self.now,
                event_type="Passenger",
                target=elevator,
                context=event.context,
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class ElevatorResult:
    """Results from the elevator system simulation."""

    sink: LatencyTracker
    dispatcher: ElevatorDispatcher
    elevators: list[ElevatorCar]
    floor_providers: list[FloorProvider]
    config: ElevatorConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_elevator_simulation(config: ElevatorConfig | None = None) -> ElevatorResult:
    """Run the elevator system simulation."""
    if config is None:
        config = ElevatorConfig()

    random.seed(config.seed)

    sink = LatencyTracker("Sink")

    elevators = [
        ElevatorCar(
            f"Elevator_{i}", config.elevator_capacity,
            config.floor_travel_time, config.door_time, sink,
        )
        for i in range(config.num_elevators)
    ]

    dispatcher = ElevatorDispatcher("Dispatcher", elevators)

    stop_after = Instant.from_seconds(config.duration_s)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    sources: list[Source] = []
    floor_providers: list[FloorProvider] = []

    for floor in range(config.num_floors):
        rate = config.lobby_rate_per_min if floor == 0 else config.upper_rate_per_min
        provider = FloorProvider(dispatcher, floor, config.num_floors, stop_after)
        floor_providers.append(provider)

        sources.append(
            Source(
                name=f"Floor_{floor}",
                event_provider=provider,
                arrival_time_provider=PoissonArrivalTimeProvider(
                    ConstantRateProfile(rate=rate / 60.0),
                    start_time=Instant.Epoch,
                ),
            )
        )

    end_time = Instant.from_seconds(config.duration_s + 600)

    all_entities: list[Entity] = [dispatcher, sink]
    all_entities.extend(elevators)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=sources,
        entities=all_entities,
    )

    summary = sim.run()

    return ElevatorResult(
        sink=sink,
        dispatcher=dispatcher,
        elevators=elevators,
        floor_providers=floor_providers,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: ElevatorResult) -> None:
    """Print a formatted summary of the elevator simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("ELEVATOR SYSTEM SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 60:.0f} minutes")
    print(f"  Floors:              {config.num_floors}")
    print(f"  Elevators:           {config.num_elevators}")
    print(f"  Capacity:            {config.elevator_capacity} per car")
    print(f"  Lobby rate:          {config.lobby_rate_per_min:.1f}/min")
    print(f"  Upper floor rate:    {config.upper_rate_per_min:.1f}/min each")

    total = sum(p.generated for p in result.floor_providers)
    print(f"\nPassenger Flow:")
    print(f"  Total passengers:    {total}")
    print(f"  Dispatched:          {result.dispatcher._dispatched}")
    print(f"  From lobby:          {result.floor_providers[0].generated}")
    print(f"  From upper floors:   {sum(p.generated for p in result.floor_providers[1:])}")

    print(f"\nElevator Stats:")
    for elev in result.elevators:
        print(f"  {elev.name}: trips={elev.trips}, passengers={elev.passengers_carried}")

    completed = result.sink.count
    if completed > 0:
        print(f"\nWait + Travel Latency:")
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
    parser = argparse.ArgumentParser(description="Elevator system simulation")
    parser.add_argument("--duration", type=float, default=7200.0, help="Duration in seconds")
    parser.add_argument("--floors", type=int, default=10, help="Number of floors")
    parser.add_argument("--elevators", type=int, default=3, help="Number of elevators")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = ElevatorConfig(
        duration_s=args.duration,
        num_floors=args.floors,
        num_elevators=args.elevators,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running elevator system simulation...")
    result = run_elevator_simulation(cfg)
    print_summary(result)
