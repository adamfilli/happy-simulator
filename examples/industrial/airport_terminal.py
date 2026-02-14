"""Airport terminal discrete-event simulation.

Passengers are routed by ticket class to check-in counters, then through
baggage handling (ConveyorBelt), security (PriorityQueue for TSA PreCheck),
gate lounge, and boarding (PooledCycleResource). Staffing varies by shift.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                    AIRPORT TERMINAL SIMULATION                         |
+-----------------------------------------------------------------------+

  +---------+   +----------+  economy   +----------+
  | Source  |-->| Ticket   |----------->| Econ     |--+
  |(Poisson)|   | Router   |  business  | Check-in |  |
  +---------+   |          |----------->+----------+  |   +---------+
                |          |  first     | Biz      |--+-->| Baggage |
                |          |----------->| Check-in |  |   | Belt    |
                +----------+            +----------+  |   | (300s)  |
                                        | First    |--+   +----+----+
                                        | Check-in |            |
                                        +----------+       +----v----+
                                                           | Security|
              +------+   +---------+   +---------+        |(PreCheck|
              | Sink |<--| Board   |<--| Gate    |<-------| prio)   |
              |      |   | (seats) |   | Lounge  |        +---------+
              +------+   +---------+   +---------+
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
from happysimulator.components.industrial import (
    ConditionalRouter,
    ConveyorBelt,
    PooledCycleResource,
    Shift,
    ShiftSchedule,
    ShiftedServer,
)
from happysimulator.components.queue_policy import PriorityQueue


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class AirportConfig:
    """Configuration for the airport terminal simulation."""

    duration_s: float = 14400.0
    arrival_rate_per_min: float = 2.0
    economy_pct: float = 0.70
    business_pct: float = 0.20
    first_pct: float = 0.10
    precheck_pct: float = 0.30
    checkin_time: float = 120.0
    baggage_belt_time: float = 300.0
    security_time: float = 180.0
    lounge_time: float = 60.0
    boarding_seats: int = 180
    boarding_cycle_time: float = 1800.0
    seed: int = 42


TICKET_CLASSES = ["economy", "business", "first"]


# =============================================================================
# Event Provider
# =============================================================================


class PassengerProvider(EventProvider):
    """Generates passenger arrival events."""

    def __init__(self, target: Entity, config: AirportConfig, stop_after: Instant | None = None):
        self._target = target
        self._config = config
        self._class_weights = [config.economy_pct, config.business_pct, config.first_pct]
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        r = random.random()
        cumulative = 0.0
        ticket_class = TICKET_CLASSES[-1]
        for tc, weight in zip(TICKET_CLASSES, self._class_weights):
            cumulative += weight
            if r < cumulative:
                ticket_class = tc
                break

        has_precheck = random.random() < self._config.precheck_pct

        return [
            Event(
                time=time,
                event_type="Passenger",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "ticket_class": ticket_class,
                    "has_precheck": has_precheck,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class Station(QueuedResource):
    """Generic airport station."""

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
class AirportResult:
    """Results from the airport terminal simulation."""

    sink: LatencyTracker
    router: ConditionalRouter
    checkin_counters: dict[str, Station]
    baggage_belt: ConveyorBelt
    security: Station
    gate_lounge: Station
    boarding: PooledCycleResource
    pax_provider: PassengerProvider
    config: AirportConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_airport_simulation(config: AirportConfig | None = None) -> AirportResult:
    """Run the airport terminal simulation."""
    if config is None:
        config = AirportConfig()

    random.seed(config.seed)

    sink = LatencyTracker("Sink")

    boarding = PooledCycleResource(
        "Boarding", config.boarding_seats, config.boarding_cycle_time, sink,
    )

    gate_lounge = Station("GateLounge", config.lounge_time, boarding)

    # Security with PreCheck priority
    security = Station(
        "Security", config.security_time, gate_lounge,
        policy=PriorityQueue(
            key=lambda e: 0.0 if e.context.get("has_precheck") else 1.0
        ),
    )

    baggage_belt = ConveyorBelt("BaggageBelt", security, config.baggage_belt_time)

    # Check-in counters per class
    checkin_counters = {
        tc: Station(f"CheckIn_{tc}", config.checkin_time, baggage_belt)
        for tc in TICKET_CLASSES
    }

    router = ConditionalRouter.by_context_field(
        "TicketRouter",
        "ticket_class",
        {tc: counter for tc, counter in checkin_counters.items()},
    )

    stop_after = Instant.from_seconds(config.duration_s)
    pax_provider = PassengerProvider(router, config, stop_after)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    source = Source(
        name="Passengers",
        event_provider=pax_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.arrival_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 3600)

    all_entities: list[Entity] = [router, baggage_belt, security, gate_lounge, boarding, sink]
    all_entities.extend(checkin_counters.values())

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=all_entities,
    )

    summary = sim.run()

    return AirportResult(
        sink=sink,
        router=router,
        checkin_counters=checkin_counters,
        baggage_belt=baggage_belt,
        security=security,
        gate_lounge=gate_lounge,
        boarding=boarding,
        pax_provider=pax_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: AirportResult) -> None:
    """Print a formatted summary of the airport simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("AIRPORT TERMINAL SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 3600:.0f} hours")
    print(f"  Arrival rate:        {config.arrival_rate_per_min:.1f}/min")
    print(f"  PreCheck rate:       {config.precheck_pct:.0%}")

    total = result.pax_provider.generated

    print(f"\nPassenger Flow:")
    print(f"  Total passengers:    {total}")
    for name, count in result.router.routed_counts.items():
        print(f"  {name:20s} {count}")

    print(f"\nStation Throughput:")
    for tc, counter in result.checkin_counters.items():
        print(f"  CheckIn {tc:10s}   {counter.processed}")
    print(f"  Baggage belt:        {result.baggage_belt.items_transported}")
    print(f"  Security:            {result.security.processed}")
    print(f"  Gate lounge:         {result.gate_lounge.processed}")
    print(f"  Boarded:             {result.boarding.completed}")

    completed = result.sink.count
    if completed > 0:
        print(f"\nEnd-to-End Latency:")
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
    parser = argparse.ArgumentParser(description="Airport terminal simulation")
    parser.add_argument("--duration", type=float, default=14400.0, help="Duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=2.0, help="Passengers per minute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = AirportConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running airport terminal simulation...")
    result = run_airport_simulation(cfg)
    print_summary(result)
