"""Assembly line manufacturing simulation.

Four-stage pipeline with quality inspection, rework loop, and batch packaging.

## Architecture Diagram

```
                    MANUFACTURING LINE
    +-------------------------------------------------------+
    |                                                       |
    |  Source -> Cut -> [belt] -> Assemble -> [belt] ->     |
    |                                                       |
    |           Inspect --pass--> [belt] -> Package -> Sink |
    |              |                          (batch=12)    |
    |              +--fail--> [belt] -> Assemble (rework)   |
    |                                                       |
    |  BreakdownScheduler attached to Cut station           |
    +-------------------------------------------------------+
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Generator

from happysimulator import (
    Entity, Event, Instant, LatencyTracker, QueuedResource,
    FIFOQueue, Simulation, SimulationSummary, Source,
)
from happysimulator.components.industrial import (
    ConveyorBelt, InspectionStation, BatchProcessor, BreakdownScheduler,
)


@dataclass(frozen=True)
class ManufacturingConfig:
    duration_s: float = 3600.0
    arrival_rate: float = 0.5  # parts per second
    cut_time: float = 1.5
    assemble_time: float = 3.0
    inspect_time: float = 1.0
    package_time: float = 5.0
    conveyor_time: float = 2.0
    defect_rate: float = 0.05
    batch_size: int = 12
    mttf: float = 300.0  # mean time to failure for cut station
    mttr: float = 30.0   # mean time to repair
    seed: int = 42


class WorkStation(QueuedResource):
    """Simple manufacturing station with configurable service time."""

    def __init__(self, name: str, service_time: float, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.service_time_s = service_time
        self.downstream = downstream
        self.parts_processed = 0
        self._broken = False

    def has_capacity(self) -> bool:
        return not self._broken

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.service_time_s
        self.parts_processed += 1
        return [
            self.forward(event, self.downstream)
        ]


@dataclass
class ManufacturingResult:
    sink: LatencyTracker
    stations: dict[str, WorkStation]
    inspector: InspectionStation
    packager: BatchProcessor
    breakdown: BreakdownScheduler
    config: ManufacturingConfig
    summary: SimulationSummary


def run_manufacturing_simulation(config: ManufacturingConfig | None = None) -> ManufacturingResult:
    if config is None:
        config = ManufacturingConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Sink")

    # Build pipeline from end to start
    packager = BatchProcessor(
        "Package", downstream=sink,
        batch_size=config.batch_size,
        process_time=config.package_time,
        timeout_s=30.0,
    )
    belt_to_pack = ConveyorBelt("Belt_ToPack", packager, config.conveyor_time)

    # Inspection with rework loop
    # fail_target will be set to assemble station (circular reference)
    rework_sink = LatencyTracker("ReworkSink")  # placeholder

    inspector = InspectionStation(
        "Inspect",
        pass_target=belt_to_pack,
        fail_target=rework_sink,  # will be overridden
        inspection_time=config.inspect_time,
        pass_rate=1.0 - config.defect_rate,
    )

    belt_to_inspect = ConveyorBelt("Belt_ToInspect", inspector, config.conveyor_time)
    assemble = WorkStation("Assemble", config.assemble_time, belt_to_inspect)

    # Fix rework loop: failed items go back to assemble via conveyor
    rework_belt = ConveyorBelt("Belt_Rework", assemble, config.conveyor_time)
    inspector.fail_target = rework_belt

    belt_to_assemble = ConveyorBelt("Belt_ToAssemble", assemble, config.conveyor_time)
    cut_station = WorkStation("Cut", config.cut_time, belt_to_assemble)

    # Breakdown scheduler for cut station
    breakdown = BreakdownScheduler(
        "CutBreakdowns", target=cut_station,
        mean_time_to_failure=config.mttf,
        mean_repair_time=config.mttr,
    )

    source = Source.poisson(
        rate=config.arrival_rate, target=cut_station,
        event_type="Part", name="PartSource",
        stop_after=config.duration_s,
    )

    entities = [cut_station, belt_to_assemble, assemble, belt_to_inspect,
                inspector, belt_to_pack, packager, rework_belt, sink,
                rework_sink, breakdown]

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 300,
        sources=[source],
        entities=entities,
    )
    sim.schedule(breakdown.start_event())
    summary = sim.run()

    stations = {"Cut": cut_station, "Assemble": assemble}

    return ManufacturingResult(
        sink=sink, stations=stations, inspector=inspector,
        packager=packager, breakdown=breakdown,
        config=config, summary=summary,
    )


def print_summary(result: ManufacturingResult) -> None:
    print("\n" + "=" * 60)
    print("MANUFACTURING LINE SIMULATION RESULTS")
    print("=" * 60)

    c = result.config
    print(f"\nConfiguration:")
    print(f"  Duration: {c.duration_s/60:.0f} minutes")
    print(f"  Arrival rate: {c.arrival_rate:.2f} parts/s")
    print(f"  Defect rate: {c.defect_rate*100:.1f}%")
    print(f"  Batch size: {c.batch_size}")

    print(f"\nStation Performance:")
    for name, station in result.stations.items():
        print(f"  {name}: {station.parts_processed} parts")

    insp = result.inspector.stats
    print(f"\nInspection:")
    print(f"  Inspected: {insp.inspected}")
    print(f"  Passed: {insp.passed} ({insp.passed/max(insp.inspected,1)*100:.1f}%)")
    print(f"  Failed (rework): {insp.failed}")

    pkg = result.packager.stats
    print(f"\nPackaging:")
    print(f"  Batches: {pkg.batches_processed}")
    print(f"  Items packaged: {pkg.items_processed}")

    bd = result.breakdown.stats
    print(f"\nBreakdowns:")
    print(f"  Count: {bd.breakdown_count}")
    print(f"  Availability: {bd.availability*100:.1f}%")

    print(f"\nOverall:")
    print(f"  Finished goods: {result.sink.count}")
    if result.sink.count > 0:
        print(f"  Avg cycle time: {result.sink.mean_latency():.1f}s")

    print(f"\n{result.summary}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manufacturing line simulation")
    parser.add_argument("--duration", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    config = ManufacturingConfig(duration_s=args.duration, seed=args.seed)
    result = run_manufacturing_simulation(config)
    print_summary(result)
