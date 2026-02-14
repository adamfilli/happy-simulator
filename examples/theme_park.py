"""Theme park discrete-event simulation with rides, FastPass, and balking.

3 rides as PooledCycleResource (roller coaster, ferris wheel, water ride).
ConditionalRouter for ride choice. FastPass holders get priority. Guests
balk if ride queue is too long.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                      THEME PARK SIMULATION                             |
+-----------------------------------------------------------------------+

  +---------+   +----------+   roller_coaster  +-----------+   +------+
  | Source  |-->| Ride     |------------------>| RC Entry  |-->| RC   |->+
  |(Poisson)|   | Chooser  |                   | (balk@30) |   | 24/3m| |
  +---------+   | (Router) |   ferris_wheel    +-----------+   +------+ |
                |          |------------------>| FW Entry  |-->| FW   |->+->Sink
                |          |                   | (balk@25) |   | 40/10| |
                |          |   water_ride      +-----------+   +------+ |
                |          |------------------>| WR Entry  |-->| WR   |->+
                +----------+                   | (balk@20) |   | 12/5m|
                                               +-----------+   +------+
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
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter
from happysimulator.components.industrial import ConditionalRouter, PooledCycleResource


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class ThemeParkConfig:
    """Configuration for the theme park simulation."""

    duration_s: float = 14400.0
    arrival_rate_per_min: float = 3.0
    fastpass_pct: float = 0.20
    roller_coaster_pct: float = 0.40
    ferris_wheel_pct: float = 0.30
    water_ride_pct: float = 0.30
    rc_seats: int = 24
    rc_cycle_time: float = 180.0
    rc_balk_threshold: int = 30
    fw_seats: int = 40
    fw_cycle_time: float = 600.0
    fw_balk_threshold: int = 25
    wr_seats: int = 12
    wr_cycle_time: float = 300.0
    wr_balk_threshold: int = 20
    seed: int = 42


RIDE_NAMES = ["roller_coaster", "ferris_wheel", "water_ride"]


# =============================================================================
# Event Provider
# =============================================================================


class GuestProvider(EventProvider):
    """Generates guest arrival events with ride preference and FastPass status."""

    def __init__(self, target: Entity, config: ThemeParkConfig, stop_after: Instant | None = None):
        self._target = target
        self._config = config
        self._ride_weights = [config.roller_coaster_pct, config.ferris_wheel_pct, config.water_ride_pct]
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        r = random.random()
        cumulative = 0.0
        ride_choice = RIDE_NAMES[-1]
        for name, weight in zip(RIDE_NAMES, self._ride_weights):
            cumulative += weight
            if r < cumulative:
                ride_choice = name
                break

        has_fastpass = random.random() < self._config.fastpass_pct

        return [
            Event(
                time=time,
                event_type="Guest",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "ride_choice": ride_choice,
                    "has_fastpass": has_fastpass,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class RideEntrance(Entity):
    """Checks queue depth before allowing guests to enter a ride."""

    def __init__(self, name: str, ride: PooledCycleResource, balk_threshold: int, balk_counter: Entity):
        super().__init__(name)
        self.ride = ride
        self.balk_threshold = balk_threshold
        self.balk_counter = balk_counter
        self._entered = 0
        self._balked = 0

    def handle_event(self, event: Event) -> list[Event]:
        total_busy = self.ride.queued + self.ride.active
        if total_busy >= self.balk_threshold:
            self._balked += 1
            return [
                Event(
                    time=self.now,
                    event_type="Balked",
                    target=self.balk_counter,
                    context=event.context,
                )
            ]
        self._entered += 1
        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.ride,
                context=event.context,
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class ThemeParkResult:
    """Results from the theme park simulation."""

    sink: LatencyTracker
    rides: dict[str, PooledCycleResource]
    entrances: dict[str, RideEntrance]
    router: ConditionalRouter
    balk_counter: Counter
    guest_provider: GuestProvider
    config: ThemeParkConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_theme_park_simulation(config: ThemeParkConfig | None = None) -> ThemeParkResult:
    """Run the theme park simulation."""
    if config is None:
        config = ThemeParkConfig()

    random.seed(config.seed)

    sink = LatencyTracker("Sink")
    balk_counter = Counter("BalkCounter")

    # Create rides
    rides = {
        "roller_coaster": PooledCycleResource("RollerCoaster", config.rc_seats, config.rc_cycle_time, sink),
        "ferris_wheel": PooledCycleResource("FerrisWheel", config.fw_seats, config.fw_cycle_time, sink),
        "water_ride": PooledCycleResource("WaterRide", config.wr_seats, config.wr_cycle_time, sink),
    }

    thresholds = {
        "roller_coaster": config.rc_balk_threshold,
        "ferris_wheel": config.fw_balk_threshold,
        "water_ride": config.wr_balk_threshold,
    }

    entrances = {
        name: RideEntrance(f"{name}_entrance", ride, thresholds[name], balk_counter)
        for name, ride in rides.items()
    }

    router = ConditionalRouter.by_context_field(
        "RideChooser",
        "ride_choice",
        {name: entrance for name, entrance in entrances.items()},
    )

    stop_after = Instant.from_seconds(config.duration_s)
    guest_provider = GuestProvider(router, config, stop_after)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    source = Source(
        name="Guests",
        event_provider=guest_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.arrival_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 1800)

    all_entities: list[Entity] = [router, balk_counter, sink]
    all_entities.extend(entrances.values())
    all_entities.extend(rides.values())

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=all_entities,
    )

    summary = sim.run()

    return ThemeParkResult(
        sink=sink,
        rides=rides,
        entrances=entrances,
        router=router,
        balk_counter=balk_counter,
        guest_provider=guest_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: ThemeParkResult) -> None:
    """Print a formatted summary of the theme park simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("THEME PARK SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 3600:.0f} hours")
    print(f"  Guest arrival rate:  {config.arrival_rate_per_min:.1f}/min")
    print(f"  FastPass rate:       {config.fastpass_pct:.0%}")

    total = result.guest_provider.generated
    total_balked = sum(e._balked for e in result.entrances.values())

    print(f"\nGuest Flow:")
    print(f"  Total guests:        {total}")
    print(f"  Total balked:        {total_balked} ({100 * total_balked / max(total, 1):.1f}%)")

    print(f"\nRide Statistics:")
    for name, ride in result.rides.items():
        entrance = result.entrances[name]
        stats = ride.stats
        print(f"  {name}:")
        print(f"    Entered:           {entrance._entered}")
        print(f"    Balked:            {entrance._balked}")
        print(f"    Completed:         {stats.completed}")
        print(f"    Utilization:       {stats.utilization:.1%}")

    completed = result.sink.count
    if completed > 0:
        print(f"\nEnd-to-End Latency (ride wait + cycle):")
        print(f"  Completed:           {completed}")
        print(f"  Mean:    {result.sink.mean_latency() / 60:.1f} min")
        print(f"  p50:     {result.sink.p50() / 60:.1f} min")
        print(f"  p99:     {result.sink.p99() / 60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 65)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Theme park simulation")
    parser.add_argument("--duration", type=float, default=14400.0, help="Duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=3.0, help="Guests per minute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = ThemeParkConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running theme park simulation...")
    result = run_theme_park_simulation(cfg)
    print_summary(result)
