"""Laundromat discrete-event simulation with washers, dryers, and folding tables.

Customers arrive, use washers (35min), dryers (45min), and folding tables
(10min). Customers renege if washer wait exceeds 15 minutes.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                      LAUNDROMAT SIMULATION                             |
+-----------------------------------------------------------------------+

  +---------+   +---------------+   +---------------+   +---------+
  | Source  |-->| Washers       |-->| Dryers        |-->| Folding |-->+------+
  |(Poisson)|   | (8 units,     |   | (6 units,     |   | Tables  |   | Sink |
  +---------+   |  35min cycle) |   |  45min cycle) |   | (4,Res) |   +------+
                | renege: 15min |   +---------------+   | ~10min  |
                +---------------+                       +---------+
                      |
                      | reneged
                      v
                +----------+
                | Reneged  |
                | Counter  |
                +----------+
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
    Resource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter
from happysimulator.components.industrial import PooledCycleResource, RenegingQueuedResource


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class LaundromatConfig:
    """Configuration for the laundromat simulation."""

    duration_s: float = 14400.0
    arrival_rate_per_min: float = 0.8
    num_washers: int = 8
    wash_cycle_time: float = 2100.0
    num_dryers: int = 6
    dry_cycle_time: float = 2700.0
    num_folding_tables: int = 4
    fold_time: float = 600.0
    patience_s: float = 900.0
    seed: int = 42


# =============================================================================
# Event Provider
# =============================================================================


class CustomerProvider(EventProvider):
    """Generates laundromat customer arrival events."""

    def __init__(self, target: Entity, patience_s: float, stop_after: Instant | None = None):
        self._target = target
        self._patience_s = patience_s
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        return [
            Event(
                time=time,
                event_type="Customer",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "patience_s": self._patience_s,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class WasherStation(RenegingQueuedResource):
    """Bank of washers with reneging support."""

    def __init__(
        self,
        name: str,
        num_washers: int,
        wash_time: float,
        downstream: Entity,
        reneged_target: Entity | None = None,
        default_patience_s: float = float("inf"),
    ):
        super().__init__(
            name,
            reneged_target=reneged_target,
            default_patience_s=default_patience_s,
            policy=FIFOQueue(),
        )
        self._num_washers = num_washers
        self.wash_time = wash_time
        self.downstream = downstream
        self._active = 0
        self._processed = 0

    @property
    def processed(self) -> int:
        return self._processed

    def has_capacity(self) -> bool:
        return self._active < self._num_washers

    def _handle_served_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._active += 1
        try:
            yield self.wash_time
        finally:
            self._active -= 1

        self._processed += 1
        return [
            Event(
                time=self.now,
                event_type="Washed",
                target=self.downstream,
                context=event.context,
            )
        ]


class FoldingArea(Entity):
    """Folding tables using Resource for contended capacity."""

    def __init__(self, name: str, tables: Resource, fold_time: float, downstream: Entity):
        super().__init__(name)
        self.tables = tables
        self.fold_time = fold_time
        self.downstream = downstream
        self._processed = 0

    def handle_event(self, event: Event) -> Generator:
        grant = yield self.tables.acquire(1)
        yield self.fold_time
        grant.release()
        self._processed += 1
        return [
            Event(
                time=self.now,
                event_type="Done",
                target=self.downstream,
                context=event.context,
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class LaundromatResult:
    """Results from the laundromat simulation."""

    sink: LatencyTracker
    washers: WasherStation
    dryers: PooledCycleResource
    folding_tables: Resource
    folding_area: FoldingArea
    reneged_counter: Counter
    customer_provider: CustomerProvider
    config: LaundromatConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_laundromat_simulation(config: LaundromatConfig | None = None) -> LaundromatResult:
    """Run the laundromat simulation."""
    if config is None:
        config = LaundromatConfig()

    random.seed(config.seed)

    # Build pipeline from end to start
    sink = LatencyTracker("Sink")

    folding_tables = Resource("FoldingTables", capacity=config.num_folding_tables)
    folding_area = FoldingArea("FoldingArea", folding_tables, config.fold_time, sink)

    dryers = PooledCycleResource(
        "Dryers",
        pool_size=config.num_dryers,
        cycle_time=config.dry_cycle_time,
        downstream=folding_area,
    )

    reneged_counter = Counter("RenegedCounter")

    washers = WasherStation(
        "Washers",
        num_washers=config.num_washers,
        wash_time=config.wash_cycle_time,
        downstream=dryers,
        reneged_target=reneged_counter,
        default_patience_s=config.patience_s,
    )

    stop_after = Instant.from_seconds(config.duration_s)
    customer_provider = CustomerProvider(washers, config.patience_s, stop_after)

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    source = Source(
        name="Arrivals",
        event_provider=customer_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=config.arrival_rate_per_min / 60.0),
            start_time=Instant.Epoch,
        ),
    )

    end_time = Instant.from_seconds(config.duration_s + 7200)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[washers, dryers, folding_area, folding_tables, reneged_counter, sink],
    )

    summary = sim.run()

    return LaundromatResult(
        sink=sink,
        washers=washers,
        dryers=dryers,
        folding_tables=folding_tables,
        folding_area=folding_area,
        reneged_counter=reneged_counter,
        customer_provider=customer_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: LaundromatResult) -> None:
    """Print a formatted summary of the laundromat simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("LAUNDROMAT SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 60:.0f} minutes")
    print(f"  Arrival rate:        {config.arrival_rate_per_min:.1f}/min")
    print(f"  Washers:             {config.num_washers} ({config.wash_cycle_time / 60:.0f} min cycle)")
    print(f"  Dryers:              {config.num_dryers} ({config.dry_cycle_time / 60:.0f} min cycle)")
    print(f"  Folding tables:      {config.num_folding_tables}")
    print(f"  Patience:            {config.patience_s / 60:.0f} min")

    total = result.customer_provider.generated
    reneged = result.washers.reneged

    print(f"\nCustomer Flow:")
    print(f"  Arrived:             {total}")
    print(f"  Reneged:             {reneged} ({100 * reneged / max(total, 1):.1f}%)")
    print(f"  Washed:              {result.washers.processed}")
    print(f"  Dried:               {result.dryers.completed}")
    print(f"  Folded:              {result.folding_area._processed}")

    dryer_stats = result.dryers.stats
    print(f"\nDryer Utilization:     {dryer_stats.utilization:.1%}")

    tbl_stats = result.folding_tables.stats
    print(f"Folding Utilization:   {tbl_stats.utilization:.1%}")

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
    parser = argparse.ArgumentParser(description="Laundromat simulation")
    parser.add_argument("--duration", type=float, default=14400.0, help="Duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=0.8, help="Customers per minute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = LaundromatConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running laundromat simulation...")
    result = run_laundromat_simulation(cfg)
    print_summary(result)
