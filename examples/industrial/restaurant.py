"""Full-service restaurant simulation with reservations, walk-ins, and kitchen pipeline.

Tables are modeled as two Resource pools (two-tops and four-tops). Reservations
arrive via AppointmentScheduler; walk-ins via Poisson Source and renege after 20
minutes if no table is available. A host stand (RenegingQueuedResource) seats
guests then routes them through a kitchen pipeline: Prep -> Cook -> Plate, each
a QueuedResource stage. After plating, the dining phase begins and eventually
the table is released.

## Architecture Diagram

```
                           RESTAURANT SIMULATION
    +------------------------------------------------------------------+
    |                                                                  |
    |  AppointmentScheduler --+                                        |
    |   (reservations)        |                                        |
    |                         v                                        |
    |  Source (walk-ins) --> HostStand --> TableSeating                 |
    |   (Poisson)           (renege 20m)   (acquire table Resource)    |
    |                                        |                         |
    |                                        v                         |
    |                          Prep --> Cook --> Plate --> Dining       |
    |                         (QR)     (QR)     (QR)    (hold table)   |
    |                                                      |           |
    |                                                      v           |
    |                                                  release table   |
    |                                                      |           |
    |                                                      v           |
    |                                                    Sink          |
    |                                                                  |
    |  Resources: 15 two-tops, 5 four-tops                             |
    +------------------------------------------------------------------+
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
    FIFOQueue,
    Instant,
    LatencyTracker,
    QueuedResource,
    Resource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter
from happysimulator.components.industrial import (
    AppointmentScheduler,
    RenegingQueuedResource,
)

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RestaurantConfig:
    duration_s: float = 7200.0  # 2 hours of dinner service
    walkin_rate: float = 0.02  # walk-ins per second (~72/hr)
    num_two_tops: int = 15
    num_four_tops: int = 5
    reservation_interval_min: float = 5.0  # one reservation every 5 min
    no_show_rate: float = 0.10
    walkin_patience_s: float = 1200.0  # 20 min patience for walk-ins
    # Kitchen times (seconds)
    prep_time: float = 180.0  # 3 min
    cook_time: float = 600.0  # 10 min
    plate_time: float = 60.0  # 1 min
    dining_time: float = 1800.0  # 30 min average dining
    # Staffing
    num_prep_cooks: int = 3
    num_line_cooks: int = 4
    num_platers: int = 2
    seed: int = 42


# =============================================================================
# Host Stand (RenegingQueuedResource)
# =============================================================================


class HostStand(RenegingQueuedResource):
    """Walk-in queue where guests renege after patience expires.

    Once a guest is seated, they are forwarded to table seating.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        reneged_target: Entity,
        default_patience_s: float,
        concurrency: int,
    ):
        super().__init__(
            name,
            reneged_target=reneged_target,
            default_patience_s=default_patience_s,
            policy=FIFOQueue(),
        )
        self.downstream = downstream
        self._concurrency = concurrency
        self._active = 0
        self.guests_seated = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def _handle_served_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._active += 1
        # Minimal seating delay
        yield 30.0
        self._active -= 1
        self.guests_seated += 1
        return [self.forward(event, self.downstream, event_type="Seat")]


# =============================================================================
# Table Seating (acquires table Resource, routes to kitchen)
# =============================================================================


class TableSeating(Entity):
    """Acquires a table resource and sends order to kitchen.

    Parties of 1-2 use two-tops; parties of 3-4 use four-tops.
    """

    def __init__(
        self,
        name: str,
        two_tops: Resource,
        four_tops: Resource,
        kitchen_entry: Entity,
        dining_stage: Entity,
    ):
        super().__init__(name)
        self.two_tops = two_tops
        self.four_tops = four_tops
        self.kitchen_entry = kitchen_entry
        self.dining_stage = dining_stage
        self.tables_assigned = 0

    def handle_event(self, event: Event) -> Generator:
        party_size = event.context.get("party_size", 2)
        ctx = dict(event.context)

        # Choose table type
        if party_size <= 2:
            grant = yield self.two_tops.acquire(1)
            ctx["table_type"] = "two_top"
        else:
            grant = yield self.four_tops.acquire(1)
            ctx["table_type"] = "four_top"

        ctx["table_grant"] = grant
        self.tables_assigned += 1

        return [Event(time=self.now, event_type="Order", target=self.kitchen_entry, context=ctx)]


# =============================================================================
# Kitchen Pipeline Stages
# =============================================================================


class KitchenStation(QueuedResource):
    """Kitchen processing stage with limited concurrency."""

    def __init__(self, name: str, service_time: float, downstream: Entity, concurrency: int):
        super().__init__(name, policy=FIFOQueue())
        self.service_time_s = service_time
        self.downstream = downstream
        self._concurrency = concurrency
        self._active = 0
        self.orders_processed = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._active += 1
        try:
            yield random.expovariate(1.0 / self.service_time_s)
        finally:
            self._active -= 1
        self.orders_processed += 1
        return [self.forward(event, self.downstream)]


# =============================================================================
# Dining Stage (holds table, then releases)
# =============================================================================


class DiningStage(Entity):
    """Simulates guests dining, then releases the table."""

    def __init__(
        self,
        name: str,
        two_tops: Resource,
        four_tops: Resource,
        downstream: Entity,
        mean_dining_time: float,
    ):
        super().__init__(name)
        self.two_tops = two_tops
        self.four_tops = four_tops
        self.downstream = downstream
        self.mean_dining_time = mean_dining_time
        self.guests_finished = 0

    def handle_event(self, event: Event) -> Generator:
        # Dining duration
        dining_time = random.expovariate(1.0 / self.mean_dining_time)
        yield dining_time

        # Release the table
        grant = event.context.get("table_grant")
        if grant is not None:
            grant.release()

        self.guests_finished += 1
        return [self.forward(event, self.downstream, event_type="Finished")]


# =============================================================================
# Main Simulation
# =============================================================================


@dataclass
class RestaurantResult:
    sink: LatencyTracker
    reneged: Counter
    host_stand: HostStand
    table_seating: TableSeating
    kitchen_stations: dict[str, KitchenStation]
    dining: DiningStage
    two_tops: Resource
    four_tops: Resource
    scheduler: AppointmentScheduler
    config: RestaurantConfig
    summary: SimulationSummary


def run_restaurant_simulation(config: RestaurantConfig | None = None) -> RestaurantResult:
    if config is None:
        config = RestaurantConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Sink")
    reneged_counter = Counter("Reneged")

    # Table resources
    two_tops = Resource("TwoTops", capacity=config.num_two_tops)
    four_tops = Resource("FourTops", capacity=config.num_four_tops)

    # Build pipeline from end to start
    dining = DiningStage("Dining", two_tops, four_tops, sink, config.dining_time)

    plate_station = KitchenStation("Plate", config.plate_time, dining, config.num_platers)
    cook_station = KitchenStation("Cook", config.cook_time, plate_station, config.num_line_cooks)
    prep_station = KitchenStation("Prep", config.prep_time, cook_station, config.num_prep_cooks)

    table_seating = TableSeating("TableSeating", two_tops, four_tops, prep_station, dining)

    host_stand = HostStand(
        "HostStand",
        downstream=table_seating,
        reneged_target=reneged_counter,
        default_patience_s=config.walkin_patience_s,
        concurrency=3,  # 3 hosts can seat simultaneously
    )

    # Walk-in source
    walkin_source = Source.poisson(
        rate=config.walkin_rate,
        target=host_stand,
        event_type="WalkIn",
        name="WalkIns",
        stop_after=config.duration_s,
    )

    # Reservation scheduler: one reservation every interval
    num_reservations = int(config.duration_s / (config.reservation_interval_min * 60))
    appointment_times = [i * config.reservation_interval_min * 60 for i in range(num_reservations)]
    scheduler = AppointmentScheduler(
        "Reservations",
        target=host_stand,
        appointments=appointment_times,
        no_show_rate=config.no_show_rate,
        event_type="Reservation",
    )

    kitchen_stations = {
        "Prep": prep_station,
        "Cook": cook_station,
        "Plate": plate_station,
    }

    entities = [
        host_stand,
        table_seating,
        prep_station,
        cook_station,
        plate_station,
        dining,
        two_tops,
        four_tops,
        sink,
        reneged_counter,
        scheduler,
    ]

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 3600,  # 1hr drain
        sources=[walkin_source],
        entities=entities,
    )

    # Schedule reservation events
    for e in scheduler.start_events():
        sim.schedule(e)

    summary = sim.run()

    return RestaurantResult(
        sink=sink,
        reneged=reneged_counter,
        host_stand=host_stand,
        table_seating=table_seating,
        kitchen_stations=kitchen_stations,
        dining=dining,
        two_tops=two_tops,
        four_tops=four_tops,
        scheduler=scheduler,
        config=config,
        summary=summary,
    )


def print_summary(result: RestaurantResult) -> None:
    print("\n" + "=" * 65)
    print("RESTAURANT SIMULATION RESULTS")
    print("=" * 65)

    c = result.config
    print("\nConfiguration:")
    print(f"  Duration: {c.duration_s / 60:.0f} minutes")
    print(f"  Walk-in rate: {c.walkin_rate * 3600:.0f}/hr")
    print(f"  Tables: {c.num_two_tops} two-tops, {c.num_four_tops} four-tops")
    print(
        f"  Reservations: every {c.reservation_interval_min:.0f} min "
        f"({c.no_show_rate * 100:.0f}% no-show)"
    )

    sched = result.scheduler.stats
    print("\nArrivals:")
    print(f"  Reservations scheduled: {sched.total_scheduled}")
    print(f"  Reservations arrived: {sched.arrivals}")
    print(f"  No-shows: {sched.no_shows}")
    print(f"  Walk-ins seated: {result.host_stand.guests_seated}")

    rs = result.host_stand.reneging_stats
    print(f"  Walk-ins reneged: {rs.reneged}")

    print("\nKitchen Performance:")
    for name, station in result.kitchen_stations.items():
        print(f"  {name}: {station.orders_processed} orders")

    print("\nTable Utilization:")
    ts = result.two_tops.stats
    fs = result.four_tops.stats
    print(f"  Two-tops: {ts.peak_utilization * 100:.0f}% peak, {ts.contentions} contentions")
    print(f"  Four-tops: {fs.peak_utilization * 100:.0f}% peak, {fs.contentions} contentions")

    print("\nDining:")
    print(f"  Guests finished: {result.dining.guests_finished}")

    print("\nOverall:")
    print(f"  Completed meals: {result.sink.count}")
    if result.sink.count > 0:
        print(f"  Avg end-to-end: {result.sink.mean_latency() / 60:.1f} min")
        print(f"  p99 end-to-end: {result.sink.p99() / 60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant simulation")
    parser.add_argument(
        "--duration", type=float, default=7200.0, help="Duration in seconds (default: 7200)"
    )
    parser.add_argument(
        "--walkin-rate",
        type=float,
        default=0.02,
        help="Walk-in arrival rate per second (default: 0.02)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    config = RestaurantConfig(
        duration_s=args.duration,
        walkin_rate=args.walkin_rate,
        seed=args.seed,
    )
    result = run_restaurant_simulation(config)
    print_summary(result)
