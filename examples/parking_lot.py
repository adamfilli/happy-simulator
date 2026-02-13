"""Parking lot simulation with time-varying arrivals and revenue tracking.

Models a parking lot with regular and premium spots as separate Resources.
Arrivals follow a time-varying profile (morning ramp, midday peak, evening
decline). Cars stay for a random duration, then depart. When the lot is full,
arriving customers balk. Revenue is tracked per spot type.

## Architecture Diagram

```
                         PARKING LOT SIMULATION
    +---------------------------------------------------------------+
    |                                                               |
    |  Source -----> ParkingAttendant                                |
    | (time-varying   |      |                                      |
    |  profile)       |      +-- premium? --> acquire PremiumSpots  |
    |                 |                           |                  |
    |                 +-- regular --> acquire RegularSpots           |
    |                                     |                         |
    |                                     v                         |
    |                               ParkedCar                       |
    |                              (hold spot for random duration)   |
    |                                     |                         |
    |                                     v                         |
    |                               release spot --> DepartureSink  |
    |                                                               |
    |  Balking: try_acquire fails -> BalkCounter                    |
    |                                                               |
    |  Resources: 100 regular spots, 20 premium spots               |
    |  Revenue: $3/hr regular, $6/hr premium                        |
    +---------------------------------------------------------------+
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Generator

from happysimulator import (
    Entity,
    Event,
    Instant,
    LatencyTracker,
    Profile,
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
class ParkingConfig:
    duration_s: float = 7200.0      # 2 hours
    num_regular: int = 100
    num_premium: int = 20
    premium_probability: float = 0.15  # 15% want premium
    # Stay durations (seconds)
    mean_stay_s: float = 3600.0     # 1 hour average
    min_stay_s: float = 600.0       # 10 min minimum
    max_stay_s: float = 10800.0     # 3 hour maximum
    # Pricing per hour
    regular_rate_per_hr: float = 3.0
    premium_rate_per_hr: float = 6.0
    # Arrival profile rates (per second)
    morning_rate: float = 0.08      # ~288/hr early
    peak_rate: float = 0.15         # ~540/hr midday
    evening_rate: float = 0.04      # ~144/hr late
    # Profile timing (as fraction of duration)
    peak_start_frac: float = 0.25
    peak_end_frac: float = 0.70
    seed: int = 42


# =============================================================================
# Time-Varying Arrival Profile
# =============================================================================

class ParkingArrivalProfile(Profile):
    """Three-phase arrival profile: morning ramp, midday peak, evening decline."""

    def __init__(self, config: ParkingConfig):
        self.morning_rate = config.morning_rate
        self.peak_rate = config.peak_rate
        self.evening_rate = config.evening_rate
        self.peak_start = config.duration_s * config.peak_start_frac
        self.peak_end = config.duration_s * config.peak_end_frac
        self.duration = config.duration_s

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        if t < self.peak_start:
            # Ramp from morning to peak
            frac = t / self.peak_start if self.peak_start > 0 else 1.0
            return self.morning_rate + frac * (self.peak_rate - self.morning_rate)
        elif t < self.peak_end:
            return self.peak_rate
        elif t < self.duration:
            # Decline from peak to evening
            remaining = self.duration - self.peak_end
            frac = (t - self.peak_end) / remaining if remaining > 0 else 1.0
            return self.peak_rate - frac * (self.peak_rate - self.evening_rate)
        return self.evening_rate


# =============================================================================
# Revenue Tracker
# =============================================================================

class RevenueTracker:
    """Tracks parking revenue by spot type."""

    def __init__(self, regular_rate: float, premium_rate: float):
        self.regular_rate_per_s = regular_rate / 3600.0
        self.premium_rate_per_s = premium_rate / 3600.0
        self.regular_revenue = 0.0
        self.premium_revenue = 0.0
        self.regular_cars = 0
        self.premium_cars = 0

    def record_departure(self, spot_type: str, stay_duration_s: float) -> float:
        if spot_type == "premium":
            revenue = stay_duration_s * self.premium_rate_per_s
            self.premium_revenue += revenue
            self.premium_cars += 1
        else:
            revenue = stay_duration_s * self.regular_rate_per_s
            self.regular_revenue += revenue
            self.regular_cars += 1
        return revenue

    @property
    def total_revenue(self) -> float:
        return self.regular_revenue + self.premium_revenue


# =============================================================================
# Parking Attendant (routes cars, handles balking)
# =============================================================================

class ParkingAttendant(Entity):
    """Routes arriving cars to appropriate spot type, with balking on full lot."""

    def __init__(self, name: str, regular_spots: Resource, premium_spots: Resource,
                 parked_car_handler: Entity, balk_counter: Counter,
                 premium_probability: float):
        super().__init__(name)
        self.regular_spots = regular_spots
        self.premium_spots = premium_spots
        self.parked_car_handler = parked_car_handler
        self.balk_counter = balk_counter
        self.premium_probability = premium_probability
        self.cars_admitted = 0
        self.cars_balked = 0

    def handle_event(self, event: Event) -> list[Event]:
        ctx = dict(event.context)

        # Decide spot preference
        wants_premium = random.random() < self.premium_probability

        if wants_premium:
            # Try premium first, fall back to regular
            grant = self.premium_spots.try_acquire(1)
            if grant is not None:
                ctx["spot_type"] = "premium"
                ctx["spot_grant"] = grant
                self.cars_admitted += 1
                return [
                    Event(time=self.now, event_type="Parked",
                          target=self.parked_car_handler, context=ctx)
                ]
            # Premium full, try regular
            grant = self.regular_spots.try_acquire(1)
            if grant is not None:
                ctx["spot_type"] = "regular"
                ctx["spot_grant"] = grant
                self.cars_admitted += 1
                return [
                    Event(time=self.now, event_type="Parked",
                          target=self.parked_car_handler, context=ctx)
                ]
        else:
            # Regular only
            grant = self.regular_spots.try_acquire(1)
            if grant is not None:
                ctx["spot_type"] = "regular"
                ctx["spot_grant"] = grant
                self.cars_admitted += 1
                return [
                    Event(time=self.now, event_type="Parked",
                          target=self.parked_car_handler, context=ctx)
                ]

        # Lot full - customer balks
        self.cars_balked += 1
        return [
            Event(time=self.now, event_type="Balked",
                  target=self.balk_counter, context=ctx)
        ]


# =============================================================================
# Parked Car Handler (holds spot for random duration, then departs)
# =============================================================================

class ParkedCarHandler(Entity):
    """Holds a parking spot for a random duration, then releases it."""

    def __init__(self, name: str, downstream: Entity, revenue_tracker: RevenueTracker,
                 mean_stay_s: float, min_stay_s: float, max_stay_s: float):
        super().__init__(name)
        self.downstream = downstream
        self.revenue = revenue_tracker
        self.mean_stay_s = mean_stay_s
        self.min_stay_s = min_stay_s
        self.max_stay_s = max_stay_s
        self.departures = 0

    def handle_event(self, event: Event) -> Generator:
        # Random stay duration (clamped exponential)
        stay = random.expovariate(1.0 / self.mean_stay_s)
        stay = max(self.min_stay_s, min(stay, self.max_stay_s))

        yield stay

        # Release spot
        grant = event.context.get("spot_grant")
        if grant is not None:
            grant.release()

        # Record revenue
        spot_type = event.context.get("spot_type", "regular")
        self.revenue.record_departure(spot_type, stay)
        self.departures += 1

        return [
            Event(time=self.now, event_type="Departed",
                  target=self.downstream, context=event.context)
        ]


# =============================================================================
# Main Simulation
# =============================================================================

@dataclass
class ParkingResult:
    sink: LatencyTracker
    balk_counter: Counter
    attendant: ParkingAttendant
    parked_handler: ParkedCarHandler
    regular_spots: Resource
    premium_spots: Resource
    revenue: RevenueTracker
    config: ParkingConfig
    summary: SimulationSummary


def run_parking_simulation(config: ParkingConfig | None = None) -> ParkingResult:
    if config is None:
        config = ParkingConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Departures")
    balk_counter = Counter("Balked")

    # Resources
    regular_spots = Resource("RegularSpots", capacity=config.num_regular)
    premium_spots = Resource("PremiumSpots", capacity=config.num_premium)

    # Revenue tracker
    revenue = RevenueTracker(config.regular_rate_per_hr, config.premium_rate_per_hr)

    # Build pipeline
    parked_handler = ParkedCarHandler(
        "ParkedCars", downstream=sink, revenue_tracker=revenue,
        mean_stay_s=config.mean_stay_s,
        min_stay_s=config.min_stay_s,
        max_stay_s=config.max_stay_s,
    )

    attendant = ParkingAttendant(
        "Attendant", regular_spots=regular_spots, premium_spots=premium_spots,
        parked_car_handler=parked_handler, balk_counter=balk_counter,
        premium_probability=config.premium_probability,
    )

    # Time-varying arrival source
    profile = ParkingArrivalProfile(config)
    source = Source.with_profile(
        profile=profile, target=attendant,
        event_type="CarArrival", poisson=True,
        name="Arrivals", stop_after=config.duration_s,
    )

    entities = [
        attendant, parked_handler, regular_spots, premium_spots,
        sink, balk_counter,
    ]

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(config.duration_s + config.max_stay_s + 600),
        sources=[source],
        entities=entities,
    )
    summary = sim.run()

    return ParkingResult(
        sink=sink, balk_counter=balk_counter, attendant=attendant,
        parked_handler=parked_handler, regular_spots=regular_spots,
        premium_spots=premium_spots, revenue=revenue,
        config=config, summary=summary,
    )


def print_summary(result: ParkingResult) -> None:
    print("\n" + "=" * 65)
    print("PARKING LOT SIMULATION RESULTS")
    print("=" * 65)

    c = result.config
    print(f"\nConfiguration:")
    print(f"  Duration: {c.duration_s/60:.0f} minutes")
    print(f"  Spots: {c.num_regular} regular, {c.num_premium} premium")
    print(f"  Pricing: ${c.regular_rate_per_hr:.2f}/hr regular, "
          f"${c.premium_rate_per_hr:.2f}/hr premium")

    print(f"\nTraffic:")
    total_arrivals = result.attendant.cars_admitted + result.attendant.cars_balked
    print(f"  Total arrivals: {total_arrivals}")
    print(f"  Admitted: {result.attendant.cars_admitted}")
    print(f"  Balked (lot full): {result.attendant.cars_balked} "
          f"({result.attendant.cars_balked / max(total_arrivals, 1) * 100:.1f}%)")

    print(f"\nSpot Utilization:")
    rs = result.regular_spots.stats
    ps = result.premium_spots.stats
    print(f"  Regular: {rs.peak_utilization*100:.0f}% peak, "
          f"{rs.contentions} contentions")
    print(f"  Premium: {ps.peak_utilization*100:.0f}% peak, "
          f"{ps.contentions} contentions")

    print(f"\nRevenue:")
    rev = result.revenue
    print(f"  Regular: ${rev.regular_revenue:.2f} ({rev.regular_cars} cars)")
    print(f"  Premium: ${rev.premium_revenue:.2f} ({rev.premium_cars} cars)")
    print(f"  Total: ${rev.total_revenue:.2f}")
    if result.parked_handler.departures > 0:
        avg_revenue = rev.total_revenue / result.parked_handler.departures
        print(f"  Avg per car: ${avg_revenue:.2f}")

    print(f"\nDepartures:")
    print(f"  Total: {result.sink.count}")
    if result.sink.count > 0:
        print(f"  Avg stay: {result.sink.mean_latency()/60:.1f} min")
        print(f"  p99 stay: {result.sink.p99()/60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parking lot simulation")
    parser.add_argument("--duration", type=float, default=7200.0,
                        help="Duration in seconds (default: 7200)")
    parser.add_argument("--regular", type=int, default=100,
                        help="Number of regular spots (default: 100)")
    parser.add_argument("--premium", type=int, default=20,
                        help="Number of premium spots (default: 20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    config = ParkingConfig(
        duration_s=args.duration,
        num_regular=args.regular,
        num_premium=args.premium,
        seed=args.seed,
    )
    result = run_parking_simulation(config)
    print_summary(result)
