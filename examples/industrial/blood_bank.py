"""Blood bank discrete-event simulation with parallel testing and perishable storage.

Donors arrive by appointment → DonationStation → SplitMerge fans out to
3 parallel tests (typing, infection, antibody) → ConditionalRouter (all
pass?) → PerishableInventory (42-day shelf life). Separate demand source
consumes from inventory.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                      BLOOD BANK SIMULATION                             |
+-----------------------------------------------------------------------+

  +-----------+   +----------+   +----------+   +---------+
  | Appt      |-->| Donation |-->| SplitMrg |-->| Result  |--> pass -->+
  | Scheduler |   | Station  |   | (3 tests)|   | Router  |            |
  +-----------+   | (600s)   |   +----+-----+   +---------+            |
                  +----------+        |              |                  |
                                +-----+-----+       | fail             |
                                | Type | Inf |       v                  |
                                | Test | Test| +---------+             |
                                | (60s)|(120s)| | Reject  |            |
                                +------+-----+ | Counter |            |
                                | Antibody   |  +---------+            |
                                | Test (90s) |                         |
                                +------------+                         |
                                                                       v
  +-----------+                                    +-------------------+
  | Demand    |--- consume -->                     | Perishable        |
  | Source    |                                    | Inventory (42-day)|
  +-----------+                                    +-------------------+
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
    AppointmentScheduler,
    ConditionalRouter,
    PerishableInventory,
    SplitMerge,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BloodBankConfig:
    """Configuration for the blood bank simulation."""

    duration_s: float = 28800.0       # 8 hours
    donation_time: float = 600.0      # 10 min
    type_test_time: float = 60.0
    infection_test_time: float = 120.0
    antibody_test_time: float = 90.0
    test_pass_rate: float = 0.95
    shelf_life_s: float = 3628800.0   # 42 days
    spoilage_check_s: float = 3600.0
    initial_stock: int = 50
    reorder_point: int = 10
    order_quantity: int = 0           # no external reorder
    demand_rate_per_hour: float = 3.0
    # Appointments every 15 min from 8am-4pm
    appt_start_s: float = 0.0
    appt_interval_s: float = 900.0
    no_show_rate: float = 0.05
    seed: int = 42


# =============================================================================
# Entities
# =============================================================================


class DonationStation(QueuedResource):
    """Blood donation station."""

    def __init__(self, name: str, donation_time: float, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.donation_time = donation_time
        self.downstream = downstream
        self._processed = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.donation_time
        self._processed += 1
        return [
            self.forward(event, self.downstream, event_type="BloodUnit")
        ]


class TestLab(Entity):
    """Individual test lab that resolves a reply_future after testing."""

    def __init__(self, name: str, test_time: float, pass_rate: float):
        super().__init__(name)
        self.test_time = test_time
        self.pass_rate = pass_rate
        self._tested = 0
        self._passed = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.test_time

        self._tested += 1
        passed = random.random() < self.pass_rate
        if passed:
            self._passed += 1

        reply_future = event.context.get("reply_future")
        if reply_future is not None:
            reply_future.resolve({"passed": passed, "test": self.name})

        return []


class DemandProvider(EventProvider):
    """Generates blood demand events."""

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
                event_type="Demand",
                target=self._target,
                context={"quantity": 1},
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class BloodBankResult:
    """Results from the blood bank simulation."""

    donation_station: DonationStation
    split_merge: SplitMerge
    test_labs: list[TestLab]
    router: ConditionalRouter
    inventory: PerishableInventory
    reject_counter: Counter
    appointments: AppointmentScheduler
    demand_provider: DemandProvider
    config: BloodBankConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_blood_bank_simulation(config: BloodBankConfig | None = None) -> BloodBankResult:
    """Run the blood bank simulation."""
    if config is None:
        config = BloodBankConfig()

    random.seed(config.seed)

    reject_counter = Counter("Rejected")

    inventory = PerishableInventory(
        "BloodStorage",
        initial_stock=config.initial_stock,
        shelf_life_s=config.shelf_life_s,
        spoilage_check_interval_s=config.spoilage_check_s,
        reorder_point=config.reorder_point,
        order_quantity=config.order_quantity,
    )

    # Router: check if all sub_results passed
    def all_tests_passed(event: Event) -> bool:
        sub_results = event.context.get("sub_results", [])
        return all(r.get("passed", False) for r in sub_results)

    router = ConditionalRouter(
        "ResultRouter",
        routes=[(all_tests_passed, inventory)],
        default=reject_counter,
    )

    # Test labs
    type_test = TestLab("TypeTest", config.type_test_time, config.test_pass_rate)
    infection_test = TestLab("InfectionTest", config.infection_test_time, config.test_pass_rate)
    antibody_test = TestLab("AntibodyTest", config.antibody_test_time, config.test_pass_rate)
    test_labs = [type_test, infection_test, antibody_test]

    split_merge = SplitMerge(
        "TestSuite",
        targets=test_labs,
        downstream=router,
    )

    donation_station = DonationStation("DonationStation", config.donation_time, split_merge)

    # Appointments
    appt_times = []
    t = config.appt_start_s
    while t < config.duration_s:
        appt_times.append(t)
        t += config.appt_interval_s

    appointments = AppointmentScheduler(
        "DonorAppointments",
        target=donation_station,
        appointments=appt_times,
        no_show_rate=config.no_show_rate,
    )

    # Demand source
    stop_after = Instant.from_seconds(config.duration_s)
    demand_provider = DemandProvider(inventory, stop_after)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    demand_source = Source(
        name="Demand",
        event_provider=demand_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.demand_rate_per_hour / 3600.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 3600)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[demand_source],
        entities=[
            donation_station, split_merge,
            type_test, infection_test, antibody_test,
            router, inventory, reject_counter, appointments,
        ],
    )

    for ev in appointments.start_events():
        sim.schedule(ev)
    sim.schedule(inventory.start_event())

    summary = sim.run()

    return BloodBankResult(
        donation_station=donation_station,
        split_merge=split_merge,
        test_labs=test_labs,
        router=router,
        inventory=inventory,
        reject_counter=reject_counter,
        appointments=appointments,
        demand_provider=demand_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: BloodBankResult) -> None:
    """Print a formatted summary of the blood bank simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("BLOOD BANK SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 3600:.0f} hours")
    print(f"  Appointments:        {len(result.appointments.appointments)}")
    print(f"  Demand rate:         {config.demand_rate_per_hour:.1f}/hour")

    appt_stats = result.appointments.stats
    print(f"\nDonor Flow:")
    print(f"  Scheduled:           {appt_stats.total_scheduled}")
    print(f"  Arrived:             {appt_stats.arrivals}")
    print(f"  No-shows:            {appt_stats.no_shows}")
    print(f"  Donated:             {result.donation_station._processed}")

    sm_stats = result.split_merge.stats
    print(f"\nTesting:")
    print(f"  Splits initiated:    {sm_stats.splits_initiated}")
    print(f"  Merges completed:    {sm_stats.merges_completed}")
    for lab in result.test_labs:
        print(f"  {lab.name:20s} tested={lab._tested}, passed={lab._passed}")

    print(f"\nRouting:")
    print(f"  Accepted (all pass): {result.router.total_routed}")
    print(f"  Rejected (any fail): {result.reject_counter.total}")

    inv = result.inventory.stats
    print(f"\nInventory:")
    print(f"  Current stock:       {inv.current_stock}")
    print(f"  Total consumed:      {inv.total_consumed}")
    print(f"  Total spoiled:       {inv.total_spoiled}")
    print(f"  Stockouts:           {inv.stockouts}")
    print(f"  Waste rate:          {inv.waste_rate:.1%}")

    print(f"\nDemand:")
    print(f"  Requests:            {result.demand_provider.generated}")

    print(f"\n{result.summary}")
    print("=" * 65)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blood bank simulation")
    parser.add_argument("--duration", type=float, default=28800.0, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = BloodBankConfig(
        duration_s=args.duration,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running blood bank simulation...")
    result = run_blood_bank_simulation(cfg)
    print_summary(result)
