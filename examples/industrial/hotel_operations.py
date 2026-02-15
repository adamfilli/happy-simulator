"""Hotel operations discrete-event simulation.

AppointmentScheduler (reservations) + Source (walk-ins) → FrontDesk →
Rooms (Resource, 80 rooms) → CheckOut → Housekeeping (ShiftSchedule).
GateController blocks check-in during 11am-3pm room turnover window.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                    HOTEL OPERATIONS SIMULATION                         |
+-----------------------------------------------------------------------+

  +-----------+
  | Appt      |-->+
  | Scheduler |   |   +--------+   +-----------+   +-------+
  +-----------+   +-->| Gate   |-->| FrontDesk |-->| Rooms |--+
  +-----------+   |   | (no    |   | (check-in)|   | (80)  |  |
  | Walk-in   |-->+   |check-in|   +-----------+   | Res.  |  |
  | Source    |       |11a-3p) |                    +-------+  |
  +-----------+       +--------+                               |
                                                               |
              +------+   +--------------+   +----------+       |
              | Sink |<--| Housekeeping |<--| CheckOut |<------+
              |      |   | (day shift)  |   |          |
              +------+   +--------------+   +----------+
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
from happysimulator.components.industrial import (
    AppointmentScheduler,
    GateController,
    Shift,
    ShiftSchedule,
    ShiftedServer,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class HotelConfig:
    """Configuration for the hotel operations simulation."""

    duration_s: float = 86400.0       # 24 hours
    walkin_rate_per_hour: float = 2.0
    num_rooms: int = 80
    stay_duration_s: float = 28800.0  # 8 hour average stay
    checkin_time: float = 300.0       # 5 min
    checkout_time: float = 180.0      # 3 min
    housekeeping_time: float = 1800.0 # 30 min per room
    # Gate: no check-in from 11am-3pm (turnover window)
    gate_close_s: float = 39600.0     # 11:00 AM
    gate_open_s: float = 54000.0      # 3:00 PM
    # Appointments every 30 min from 2pm-10pm
    appt_start_s: float = 50400.0     # 2:00 PM
    appt_end_s: float = 79200.0       # 10:00 PM
    appt_interval_s: float = 1800.0   # every 30 min
    no_show_rate: float = 0.10
    # Housekeeping shift: 8am-6pm
    hk_shift_start: float = 28800.0   # 8:00 AM
    hk_shift_end: float = 64800.0     # 6:00 PM
    hk_staff: int = 4
    seed: int = 42


# =============================================================================
# Event Provider
# =============================================================================


class WalkinProvider(EventProvider):
    """Generates walk-in guest events."""

    def __init__(self, target: Entity, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        return [
            Event(
                time=time,
                event_type="Guest",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "guest_type": "walkin",
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class FrontDesk(QueuedResource):
    """Check-in desk."""

    def __init__(self, name: str, checkin_time: float, rooms: Resource,
                 stay_duration: float, checkout_entity: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.checkin_time = checkin_time
        self.rooms = rooms
        self.stay_duration = stay_duration
        self.checkout_entity = checkout_entity
        self._checked_in = 0
        self._turned_away = 0

    def handle_queued_event(self, event: Event) -> Generator:
        yield self.checkin_time

        grant = self.rooms.try_acquire(1)
        if grant is None:
            self._turned_away += 1
            return []

        self._checked_in += 1

        # Guest stays for a random duration
        stay = random.expovariate(1.0 / self.stay_duration)
        yield stay

        grant.release()
        return [
            self.forward(event, self.checkout_entity, event_type="CheckOut")
        ]


class CheckOutDesk(QueuedResource):
    """Check-out processing."""

    def __init__(self, name: str, checkout_time: float, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.checkout_time = checkout_time
        self.downstream = downstream
        self._processed = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.checkout_time
        self._processed += 1
        return [
            self.forward(event, self.downstream, event_type="Housekeeping")
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class HotelResult:
    """Results from the hotel operations simulation."""

    sink: LatencyTracker
    gate: GateController
    front_desk: FrontDesk
    rooms: Resource
    checkout: CheckOutDesk
    housekeeping: ShiftedServer
    appointments: AppointmentScheduler
    walkin_provider: WalkinProvider
    config: HotelConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_hotel_simulation(config: HotelConfig | None = None) -> HotelResult:
    """Run the hotel operations simulation."""
    if config is None:
        config = HotelConfig()

    random.seed(config.seed)

    sink = LatencyTracker("Sink")

    # Housekeeping with shift schedule
    hk_schedule = ShiftSchedule(
        shifts=[Shift(config.hk_shift_start, config.hk_shift_end, config.hk_staff)],
        default_capacity=0,
    )
    housekeeping = ShiftedServer(
        "Housekeeping",
        schedule=hk_schedule,
        service_time=config.housekeeping_time,
        downstream=sink,
    )

    checkout = CheckOutDesk("CheckOut", config.checkout_time, housekeeping)

    rooms = Resource("Rooms", capacity=config.num_rooms)
    front_desk = FrontDesk(
        "FrontDesk", config.checkin_time, rooms, config.stay_duration_s, checkout,
    )

    # Gate blocks check-in during turnover
    gate = GateController(
        "CheckInGate",
        downstream=front_desk,
        schedule=[(config.gate_open_s, config.gate_close_s)],
        initially_open=True,
    )

    # Appointment scheduler
    appt_times = []
    t = config.appt_start_s
    while t <= config.appt_end_s:
        appt_times.append(t)
        t += config.appt_interval_s

    appointments = AppointmentScheduler(
        "Reservations",
        target=gate,
        appointments=appt_times,
        no_show_rate=config.no_show_rate,
        event_type="Guest",
    )

    # Walk-in source
    stop_after = Instant.from_seconds(config.duration_s)
    walkin_provider = WalkinProvider(gate, stop_after)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    walkin_source = Source(
        name="WalkIns",
        event_provider=walkin_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.walkin_rate_per_hour / 3600.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 43200)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[walkin_source],
        entities=[gate, front_desk, rooms, checkout, housekeeping, appointments, sink],
    )

    # Schedule appointment events and gate schedule
    for ev in appointments.start_events():
        sim.schedule(ev)
    for ev in gate.start_events():
        sim.schedule(ev)

    summary = sim.run()

    return HotelResult(
        sink=sink,
        gate=gate,
        front_desk=front_desk,
        rooms=rooms,
        checkout=checkout,
        housekeeping=housekeeping,
        appointments=appointments,
        walkin_provider=walkin_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: HotelResult) -> None:
    """Print a formatted summary of the hotel simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("HOTEL OPERATIONS SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 3600:.0f} hours")
    print(f"  Rooms:               {config.num_rooms}")
    print(f"  Walk-in rate:        {config.walkin_rate_per_hour:.1f}/hour")
    print(f"  Appointments:        {len(result.appointments.appointments)}")
    print(f"  No-show rate:        {config.no_show_rate:.0%}")

    walkins = result.walkin_provider.generated
    appt_stats = result.appointments.stats

    print(f"\nGuest Flow:")
    print(f"  Walk-ins:            {walkins}")
    print(f"  Appointments:        {appt_stats.arrivals} (no-shows: {appt_stats.no_shows})")
    print(f"  Gate passed:         {result.gate.stats.passed_through}")
    print(f"  Gate queued:         {result.gate.stats.queued_while_closed}")
    print(f"  Checked in:          {result.front_desk._checked_in}")
    print(f"  Turned away:         {result.front_desk._turned_away}")
    print(f"  Checked out:         {result.checkout._processed}")

    room_stats = result.rooms.stats
    print(f"\nRoom Stats:")
    print(f"  Utilization:         {room_stats.utilization:.1%}")
    print(f"  Housekeeping done:   {result.housekeeping.processed}")

    completed = result.sink.count
    if completed > 0:
        print(f"\nEnd-to-End Latency:")
        print(f"  Completed:           {completed}")
        print(f"  Mean:    {result.sink.mean_latency() / 3600:.1f} hours")

    print(f"\n{result.summary}")
    print("=" * 65)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hotel operations simulation")
    parser.add_argument("--duration", type=float, default=86400.0, help="Duration in seconds")
    parser.add_argument("--rooms", type=int, default=80, help="Number of rooms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = HotelConfig(
        duration_s=args.duration,
        num_rooms=args.rooms,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running hotel operations simulation...")
    result = run_hotel_simulation(cfg)
    print_summary(result)
