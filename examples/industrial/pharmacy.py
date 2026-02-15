"""Pharmacy discrete-event simulation with verification rework and perishable inventory.

5-stage pipeline: DropOff → DataEntry → PharmacistVerify (92% pass, 8% fail
rework back to DataEntry) → Filling → Pickup. Controlled substances are
tracked in a PerishableInventory with 30-day shelf life.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                       PHARMACY SIMULATION                              |
+-----------------------------------------------------------------------+

  +---------+   +---------+   +-----------+   +-----------+   +---------+
  | Source  |-->| DropOff |-->| DataEntry |-->| Pharmacist|-->| Filling |--+
  |(Poisson)|   | (15s)   |   | (60s)     |   | Verify    |   | (120s)  |  |
  +---------+   +---------+   +-----------+   | (92% pass)|   +---------+  |
                                              +-----------+                |
                                              | fail (8%) |                |
                                              +-----+-----+               |
                                                    |                      |
                                                    +---> rework           |
                                                     (back to DataEntry)   |
                                                                           |
                                 +---------+   +-------------------+       |
                                 | Pickup  |<--| Perishable        |<------+
                                 | (30s)   |   | Inventory (30-day)|
                                 +----+----+   +-------------------+
                                      |
                                   +------+
                                   | Sink |
                                   +------+
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
from happysimulator.components.industrial import InspectionStation, PerishableInventory

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class PharmacyConfig:
    """Configuration for the pharmacy simulation."""

    duration_s: float = 28800.0
    arrival_rate_per_min: float = 0.5
    dropoff_time: float = 15.0
    data_entry_time: float = 60.0
    verification_time: float = 45.0
    verification_pass_rate: float = 0.92
    filling_time: float = 120.0
    pickup_time: float = 30.0
    initial_stock: int = 200
    shelf_life_s: float = 2592000.0
    spoilage_check_s: float = 3600.0
    reorder_point: int = 50
    order_quantity: int = 100
    lead_time: float = 7200.0
    seed: int = 42


# =============================================================================
# Event Provider
# =============================================================================


class PrescriptionProvider(EventProvider):
    """Generates prescription arrival events."""

    def __init__(self, target: Entity, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        rx_type = "controlled" if random.random() < 0.30 else "generic"

        return [
            Event(
                time=time,
                event_type="Prescription",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "rx_type": rx_type,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class Station(QueuedResource):
    """Generic pharmacy station with configurable service time."""

    def __init__(self, name: str, service_time: float, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.service_time = service_time
        self.downstream = downstream
        self._processed = 0

    @property
    def processed(self) -> int:
        return self._processed

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.service_time
        self._processed += 1
        return [self.forward(event, self.downstream)]


# =============================================================================
# Result
# =============================================================================


@dataclass
class PharmacyResult:
    """Results from the pharmacy simulation."""

    sink: LatencyTracker
    dropoff: Station
    data_entry: Station
    verification: InspectionStation
    filling: Station
    inventory: PerishableInventory
    pickup: Station
    rx_provider: PrescriptionProvider
    config: PharmacyConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_pharmacy_simulation(config: PharmacyConfig | None = None) -> PharmacyResult:
    """Run the pharmacy simulation."""
    if config is None:
        config = PharmacyConfig()

    random.seed(config.seed)

    # Build pipeline from end to start
    sink = LatencyTracker("Sink")
    pickup = Station("Pickup", config.pickup_time, sink)

    inventory = PerishableInventory(
        "Inventory",
        initial_stock=config.initial_stock,
        shelf_life_s=config.shelf_life_s,
        spoilage_check_interval_s=config.spoilage_check_s,
        reorder_point=config.reorder_point,
        order_quantity=config.order_quantity,
        lead_time=config.lead_time,
        downstream=pickup,
    )

    filling = Station("Filling", config.filling_time, inventory)

    # DataEntry needs to be defined before verification (rework loop target)
    # but also needs verification as downstream. Use a placeholder then fix.
    data_entry = Station("DataEntry", config.data_entry_time, None)  # type: ignore

    verification = InspectionStation(
        "PharmacistVerify",
        pass_target=filling,
        fail_target=data_entry,
        inspection_time=config.verification_time,
        pass_rate=config.verification_pass_rate,
    )
    data_entry.downstream = verification

    dropoff = Station("DropOff", config.dropoff_time, data_entry)

    stop_after = Instant.from_seconds(config.duration_s)
    rx_provider = PrescriptionProvider(dropoff, stop_after)

    from happysimulator.load.profile import ConstantRateProfile
    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider

    source = Source(
        name="Prescriptions",
        event_provider=rx_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.arrival_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 1800)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[dropoff, data_entry, verification, filling, inventory, pickup, sink],
    )

    sim.schedule(inventory.start_event())

    summary = sim.run()

    return PharmacyResult(
        sink=sink,
        dropoff=dropoff,
        data_entry=data_entry,
        verification=verification,
        filling=filling,
        inventory=inventory,
        pickup=pickup,
        rx_provider=rx_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: PharmacyResult) -> None:
    """Print a formatted summary of the pharmacy simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("PHARMACY SIMULATION RESULTS")
    print("=" * 65)

    print("\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 3600:.0f} hours")
    print(f"  Arrival rate:        {config.arrival_rate_per_min:.1f}/min")
    print(f"  Verification rate:   {config.verification_pass_rate:.0%} pass")

    total = result.rx_provider.generated

    print("\nPrescription Flow:")
    print(f"  Arrived:             {total}")
    print(f"  Dropped off:         {result.dropoff.processed}")
    print(f"  Data entries:        {result.data_entry.processed} (incl. rework)")
    v_stats = result.verification.stats
    print(
        f"  Verified:            {v_stats.inspected} (pass: {v_stats.passed}, fail: {v_stats.failed})"
    )
    print(f"  Filled:              {result.filling.processed}")
    print(f"  Picked up:           {result.pickup.processed}")

    inv = result.inventory.stats
    print("\nInventory:")
    print(f"  Current stock:       {inv.current_stock}")
    print(f"  Consumed:            {inv.total_consumed}")
    print(f"  Spoiled:             {inv.total_spoiled}")
    print(f"  Stockouts:           {inv.stockouts}")
    print(f"  Reorders:            {inv.reorders}")
    print(f"  Waste rate:          {inv.waste_rate:.1%}")

    completed = result.sink.count
    if completed > 0:
        print("\nEnd-to-End Latency:")
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
    parser = argparse.ArgumentParser(description="Pharmacy simulation")
    parser.add_argument("--duration", type=float, default=28800.0, help="Duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=0.5, help="Prescriptions per minute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PharmacyConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running pharmacy simulation...")
    result = run_pharmacy_simulation(cfg)
    print_summary(result)
