"""Warehouse order fulfillment simulation with batch picking and zone routing.

Orders flow through a three-stage pipeline: Pick -> Pack -> Ship. The picking
stage uses BatchProcessor to accumulate 10 orders before dispatching a batch
pick run. Shared Resources model pickers, pack stations, and shipping docks.
QueuedResource stages model each processing step. Zone-based picking assigns
different pick times depending on warehouse zone.

## Architecture Diagram

```
                    WAREHOUSE ORDER FULFILLMENT
    +---------------------------------------------------------------+
    |                                                               |
    |  Source -----> ZoneRouter -----> BatchPicker (batch=10)        |
    | (Poisson)      (assigns zone)    (accumulate then pick)       |
    |                                      |                        |
    |                                      v                        |
    |                               PickStation (QR)                |
    |                              (acquire picker Resource)        |
    |                                      |                        |
    |                                      v                        |
    |                               PackStation (QR)                |
    |                              (acquire pack_station Resource)  |
    |                                      |                        |
    |                                      v                        |
    |                               ShipStation (QR)                |
    |                              (acquire dock Resource)          |
    |                                      |                        |
    |                                      v                        |
    |                                    Sink                       |
    |                                                               |
    |  Resources: 8 pickers, 4 pack stations, 2 shipping docks     |
    |  Zones: A (fast), B (medium), C (slow/bulk)                   |
    +---------------------------------------------------------------+
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
from happysimulator.components.industrial import BatchProcessor

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class WarehouseConfig:
    duration_s: float = 7200.0  # 2 hour shift
    order_rate: float = 0.1  # orders per second (~360/hr)
    # Batch picking
    batch_size: int = 10
    batch_timeout_s: float = 120.0  # flush partial batch after 2 min
    batch_process_time: float = 30.0  # batch formation overhead
    # Resources
    num_pickers: int = 8
    num_pack_stations: int = 4
    num_docks: int = 2
    # Zone pick times (seconds)
    zone_a_time: float = 60.0  # fast-moving items (30%)
    zone_b_time: float = 120.0  # medium items (50%)
    zone_c_time: float = 240.0  # bulk/slow items (20%)
    # Pack and ship times
    pack_time: float = 90.0  # 1.5 min per order
    ship_time: float = 60.0  # 1 min per order
    seed: int = 42


# Zone probabilities
ZONE_WEIGHTS = {"A": 0.30, "B": 0.50, "C": 0.20}


# =============================================================================
# Zone Router
# =============================================================================


class ZoneRouter(Entity):
    """Assigns warehouse zone to each order and forwards to batch picker."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self.downstream = downstream
        self.orders_routed = 0
        self.zone_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0}

    def handle_event(self, event: Event) -> list[Event]:
        # Assign zone based on weighted probability
        r = random.random()
        cumulative = 0.0
        zone = "C"
        for z, weight in ZONE_WEIGHTS.items():
            cumulative += weight
            if r < cumulative:
                zone = z
                break

        ctx = dict(event.context)
        ctx["zone"] = zone
        ctx["items"] = random.randint(1, 8)  # 1-8 items per order

        self.orders_routed += 1
        self.zone_counts[zone] = self.zone_counts.get(zone, 0) + 1

        return [Event(time=self.now, event_type="Order", target=self.downstream, context=ctx)]


# =============================================================================
# Pick Station (with Resource for pickers)
# =============================================================================


class PickStation(QueuedResource):
    """Picking stage that acquires a picker and picks by zone time."""

    def __init__(
        self,
        name: str,
        pickers: Resource,
        downstream: Entity,
        zone_times: dict[str, float],
        concurrency: int,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.pickers = pickers
        self.downstream = downstream
        self.zone_times = zone_times
        self._concurrency = concurrency
        self._active = 0
        self.orders_picked = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def handle_queued_event(self, event: Event) -> Generator:
        self._active += 1

        # Acquire a picker
        picker_grant = yield self.pickers.acquire(1)

        # Pick time depends on zone
        zone = event.context.get("zone", "B")
        base_time = self.zone_times.get(zone, 120.0)
        items = event.context.get("items", 3)
        # Scale by number of items (roughly)
        pick_time = base_time * (0.5 + 0.5 * items / 5.0)
        yield random.expovariate(1.0 / pick_time)

        picker_grant.release()
        self._active -= 1
        self.orders_picked += 1

        return [self.forward(event, self.downstream, event_type="Picked")]


# =============================================================================
# Pack Station (with Resource for pack stations)
# =============================================================================


class PackStation(QueuedResource):
    """Packing stage that acquires a pack station resource."""

    def __init__(
        self, name: str, stations: Resource, downstream: Entity, pack_time: float, concurrency: int
    ):
        super().__init__(name, policy=FIFOQueue())
        self.stations = stations
        self.downstream = downstream
        self.pack_time_s = pack_time
        self._concurrency = concurrency
        self._active = 0
        self.orders_packed = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def handle_queued_event(self, event: Event) -> Generator:
        self._active += 1

        station_grant = yield self.stations.acquire(1)

        items = event.context.get("items", 3)
        # Packing time scales with item count
        pack_time = self.pack_time_s * (0.6 + 0.4 * items / 5.0)
        yield random.expovariate(1.0 / pack_time)

        station_grant.release()
        self._active -= 1
        self.orders_packed += 1

        return [self.forward(event, self.downstream, event_type="Packed")]


# =============================================================================
# Ship Station (with Resource for docks)
# =============================================================================


class ShipStation(QueuedResource):
    """Shipping stage that acquires a dock resource."""

    def __init__(
        self, name: str, docks: Resource, downstream: Entity, ship_time: float, concurrency: int
    ):
        super().__init__(name, policy=FIFOQueue())
        self.docks = docks
        self.downstream = downstream
        self.ship_time_s = ship_time
        self._concurrency = concurrency
        self._active = 0
        self.orders_shipped = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def handle_queued_event(self, event: Event) -> Generator:
        self._active += 1

        dock_grant = yield self.docks.acquire(1)
        yield random.expovariate(1.0 / self.ship_time_s)

        dock_grant.release()
        self._active -= 1
        self.orders_shipped += 1

        return [self.forward(event, self.downstream, event_type="Shipped")]


# =============================================================================
# Main Simulation
# =============================================================================


@dataclass
class WarehouseResult:
    sink: LatencyTracker
    zone_router: ZoneRouter
    batch_picker: BatchProcessor
    pick_station: PickStation
    pack_station: PackStation
    ship_station: ShipStation
    pickers: Resource
    pack_stations_res: Resource
    docks: Resource
    config: WarehouseConfig
    summary: SimulationSummary


def run_warehouse_simulation(config: WarehouseConfig | None = None) -> WarehouseResult:
    if config is None:
        config = WarehouseConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Shipped")

    # Resources
    pickers = Resource("Pickers", capacity=config.num_pickers)
    pack_stations_res = Resource("PackStations", capacity=config.num_pack_stations)
    docks = Resource("Docks", capacity=config.num_docks)

    # Build pipeline from end to start
    ship_station = ShipStation(
        "ShipStation",
        docks=docks,
        downstream=sink,
        ship_time=config.ship_time,
        concurrency=config.num_docks,
    )

    pack_station = PackStation(
        "PackStation",
        stations=pack_stations_res,
        downstream=ship_station,
        pack_time=config.pack_time,
        concurrency=config.num_pack_stations,
    )

    zone_times = {
        "A": config.zone_a_time,
        "B": config.zone_b_time,
        "C": config.zone_c_time,
    }

    pick_station = PickStation(
        "PickStation",
        pickers=pickers,
        downstream=pack_station,
        zone_times=zone_times,
        concurrency=config.num_pickers,
    )

    # Batch picker accumulates orders before dispatching to pick station
    batch_picker = BatchProcessor(
        "BatchPicker",
        downstream=pick_station,
        batch_size=config.batch_size,
        process_time=config.batch_process_time,
        timeout_s=config.batch_timeout_s,
    )

    zone_router = ZoneRouter("ZoneRouter", downstream=batch_picker)

    source = Source.poisson(
        rate=config.order_rate,
        target=zone_router,
        event_type="Order",
        name="Orders",
        stop_after=config.duration_s,
    )

    entities = [
        zone_router,
        batch_picker,
        pick_station,
        pack_station,
        ship_station,
        pickers,
        pack_stations_res,
        docks,
        sink,
    ]

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 1800,  # 30min drain
        sources=[source],
        entities=entities,
    )
    summary = sim.run()

    return WarehouseResult(
        sink=sink,
        zone_router=zone_router,
        batch_picker=batch_picker,
        pick_station=pick_station,
        pack_station=pack_station,
        ship_station=ship_station,
        pickers=pickers,
        pack_stations_res=pack_stations_res,
        docks=docks,
        config=config,
        summary=summary,
    )


def print_summary(result: WarehouseResult) -> None:
    print("\n" + "=" * 65)
    print("WAREHOUSE ORDER FULFILLMENT SIMULATION RESULTS")
    print("=" * 65)

    c = result.config
    print("\nConfiguration:")
    print(f"  Duration: {c.duration_s / 60:.0f} minutes")
    print(f"  Order rate: {c.order_rate * 3600:.0f} orders/hr")
    print(f"  Batch size: {c.batch_size}")
    print(
        f"  Resources: {c.num_pickers} pickers, {c.num_pack_stations} pack stations, "
        f"{c.num_docks} docks"
    )

    print("\nZone Distribution:")
    for zone, count in sorted(result.zone_router.zone_counts.items()):
        pct = count / max(result.zone_router.orders_routed, 1) * 100
        print(f"  Zone {zone}: {count} orders ({pct:.0f}%)")

    bp = result.batch_picker.stats
    print("\nBatch Picking:")
    print(f"  Batches formed: {bp.batches_processed}")
    print(f"  Items batch-picked: {bp.items_processed}")
    print(f"  Timeouts (partial batches): {bp.timeouts}")

    print("\nPipeline Throughput:")
    print(f"  Picked: {result.pick_station.orders_picked}")
    print(f"  Packed: {result.pack_station.orders_packed}")
    print(f"  Shipped: {result.ship_station.orders_shipped}")

    print("\nResource Utilization:")
    ps = result.pickers.stats
    pss = result.pack_stations_res.stats
    ds = result.docks.stats
    print(f"  Pickers: {ps.peak_utilization * 100:.0f}% peak, {ps.contentions} contentions")
    print(f"  Pack stations: {pss.peak_utilization * 100:.0f}% peak, {pss.contentions} contentions")
    print(f"  Docks: {ds.peak_utilization * 100:.0f}% peak, {ds.contentions} contentions")

    print("\nOverall:")
    print(f"  Orders completed: {result.sink.count}")
    if result.sink.count > 0:
        print(f"  Avg fulfillment time: {result.sink.mean_latency() / 60:.1f} min")
        print(f"  p99 fulfillment time: {result.sink.p99() / 60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Warehouse fulfillment simulation")
    parser.add_argument(
        "--duration", type=float, default=7200.0, help="Duration in seconds (default: 7200)"
    )
    parser.add_argument(
        "--order-rate", type=float, default=0.1, help="Order rate per second (default: 0.1)"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch pick size (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    config = WarehouseConfig(
        duration_s=args.duration,
        order_rate=args.order_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    result = run_warehouse_simulation(config)
    print_summary(result)
