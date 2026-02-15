"""Emergency room simulation with triage priority and shared resources.

## Architecture Diagram

```
                        EMERGENCY ROOM
    +----------------------------------------------------------+
    |                                                          |
    |  Source -> Triage -> PriorityQueue -> Treatment -> Sink  |
    | (Poisson)  (nurse)    (1-5 priority)   (doctors)         |
    |                                                          |
    |  Shared Resources: 3 doctors, 6 nurses, 10 beds, 1 CT   |
    +----------------------------------------------------------+
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
    PriorityQueue,
    QueuedResource,
    Resource,
    Simulation,
    SimulationSummary,
    Source,
)

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class ERConfig:
    duration_s: float = 28800.0  # 8 hours
    arrival_rate: float = 0.02  # patients per second (~72/hr)
    num_doctors: int = 3
    num_nurses: int = 6
    num_beds: int = 10
    triage_time: float = 120.0  # 2 minutes
    seed: int = 42


# Treatment times by priority (1=critical, 5=minor)
TREATMENT_TIMES = {
    1: 3600.0,  # 60 min (critical)
    2: 1800.0,  # 30 min
    3: 900.0,  # 15 min
    4: 600.0,  # 10 min
    5: 300.0,  # 5 min
}

PRIORITY_WEIGHTS = {1: 0.05, 2: 0.15, 3: 0.30, 4: 0.30, 5: 0.20}


class TriageNurse(QueuedResource):
    """Assigns priority and forwards to treatment queue."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.triaged = 0

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield 120.0  # triage assessment time

        # Assign priority
        r = random.random()
        cumulative = 0.0
        priority = 5
        for p, weight in sorted(PRIORITY_WEIGHTS.items()):
            cumulative += weight
            if r < cumulative:
                priority = p
                break

        self.triaged += 1
        ctx = dict(event.context)
        ctx["priority"] = priority
        ctx["treatment_time"] = random.expovariate(1.0 / TREATMENT_TIMES[priority])

        return [Event(time=self.now, event_type="Treatment", target=self.downstream, context=ctx)]


class TreatmentRoom(QueuedResource):
    """Treats patients using shared doctor and bed resources."""

    def __init__(
        self, name: str, doctors: Resource, beds: Resource, downstream: Entity, concurrency: int = 3
    ):
        super().__init__(
            name,
            policy=PriorityQueue(key=lambda e: e.context.get("priority", 5)),
        )
        self.doctors = doctors
        self.beds = beds
        self.downstream = downstream
        self._concurrency = concurrency
        self._active = 0
        self.treated = 0

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def handle_queued_event(self, event: Event) -> Generator:
        self._active += 1

        # Acquire resources
        doc_grant = yield self.doctors.acquire(1)
        bed_grant = yield self.beds.acquire(1)

        treatment_time = event.context.get("treatment_time", 600.0)
        yield treatment_time

        doc_grant.release()
        bed_grant.release()
        self._active -= 1
        self.treated += 1

        return [self.forward(event, self.downstream, event_type="Discharged")]


@dataclass
class ERResult:
    sink: LatencyTracker
    triage: TriageNurse
    treatment: TreatmentRoom
    doctors: Resource
    beds: Resource
    config: ERConfig
    summary: SimulationSummary


def run_er_simulation(config: ERConfig | None = None) -> ERResult:
    if config is None:
        config = ERConfig()
    random.seed(config.seed)

    sink = LatencyTracker("Discharged")
    doctors = Resource("Doctors", capacity=config.num_doctors)
    beds = Resource("Beds", capacity=config.num_beds)

    treatment = TreatmentRoom(
        "Treatment",
        doctors=doctors,
        beds=beds,
        downstream=sink,
        concurrency=config.num_doctors,
    )
    triage = TriageNurse("Triage", downstream=treatment)

    source = Source.poisson(
        rate=config.arrival_rate,
        target=triage,
        event_type="Patient",
        name="Arrivals",
        stop_after=config.duration_s,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + 7200,
        sources=[source],
        entities=[triage, treatment, doctors, beds, sink],
    )
    summary = sim.run()

    return ERResult(
        sink=sink,
        triage=triage,
        treatment=treatment,
        doctors=doctors,
        beds=beds,
        config=config,
        summary=summary,
    )


def print_summary(result: ERResult) -> None:
    print("\n" + "=" * 60)
    print("EMERGENCY ROOM SIMULATION RESULTS")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Duration: {result.config.duration_s / 3600:.0f} hours")
    print(f"  Doctors: {result.config.num_doctors}, Beds: {result.config.num_beds}")

    print("\nPatient Flow:")
    print(f"  Triaged: {result.triage.triaged}")
    print(f"  Treated: {result.treatment.treated}")
    print(f"  Discharged: {result.sink.count}")

    print("\nResource Utilization:")
    ds = result.doctors.stats
    bs = result.beds.stats
    print(f"  Doctors: {ds.peak_utilization * 100:.0f}% peak, {ds.contentions} contentions")
    print(f"  Beds: {bs.peak_utilization * 100:.0f}% peak, {bs.contentions} contentions")

    if result.sink.count > 0:
        print("\nLatency:")
        print(f"  Avg: {result.sink.mean_latency() / 60:.1f} min")
        print(f"  p99: {result.sink.p99() / 60:.1f} min")

    print(f"\n{result.summary}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency room simulation")
    parser.add_argument("--duration", type=float, default=28800.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ERConfig(duration_s=args.duration, seed=args.seed)
    result = run_er_simulation(config)
    print_summary(result)
