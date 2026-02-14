"""Urgent care clinic discrete-event simulation with triage and preemption.

Patients arrive → Reception → Triage (assigns priority) → ConditionalRouter
(critical/non-critical) → critical: PreemptibleResource (2 trauma bays,
preempts minor cases) / non-critical: ExamRooms (4 rooms) → Treatment → Sink.
Non-critical patients renege after 90min.

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                    URGENT CARE SIMULATION                              |
+-----------------------------------------------------------------------+

  +---------+   +-----------+   +---------+   +----------+ critical
  | Source  |-->| Reception |-->| Triage  |-->| Priority |--------+
  |(Poisson)|   | (60s)     |   | (120s)  |   | Router   |        |
  +---------+   +-----------+   +---------+   +----------+        v
                                                   |        +-----------+
                                                   |        | Trauma    |
                                            non-   |        | Bays (2)  |
                                           critical|        | preemptive|
                                                   |        +-----+-----+
                                                   v              |
                                              +-----------+       |
                                              | Exam Rooms|       |
                                              | (4 rooms) |       |
                                              | renege:90m|       |
                                              +-----+-----+       |
                                                    |              |
                                                    v              v
                                              +-----------+   +------+
                                              | Treatment |-->| Sink |
                                              | (300s)    |   |      |
                                              +-----------+   +------+
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
    PreemptibleResource,
    RenegingQueuedResource,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class UrgentCareConfig:
    """Configuration for the urgent care simulation."""

    duration_s: float = 28800.0       # 8 hours
    arrival_rate_per_min: float = 0.5
    critical_pct: float = 0.20
    reception_time: float = 60.0
    triage_time: float = 120.0
    num_trauma_bays: int = 2
    trauma_treatment_time: float = 1200.0  # 20 min
    num_exam_rooms: int = 4
    exam_time: float = 600.0          # 10 min
    treatment_time: float = 300.0     # 5 min
    renege_patience_s: float = 5400.0 # 90 min
    seed: int = 42


# =============================================================================
# Event Provider
# =============================================================================


class PatientProvider(EventProvider):
    """Generates patient arrival events."""

    def __init__(self, target: Entity, critical_pct: float,
                 patience_s: float, stop_after: Instant | None = None):
        self._target = target
        self._critical_pct = critical_pct
        self._patience_s = patience_s
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        is_critical = random.random() < self._critical_pct

        return [
            Event(
                time=time,
                event_type="Patient",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "severity": "critical" if is_critical else "non-critical",
                    "patience_s": self._patience_s,
                },
            )
        ]


# =============================================================================
# Entities
# =============================================================================


class Station(QueuedResource):
    """Generic station with configurable service time."""

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
        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.downstream,
                context=event.context,
            )
        ]


class TraumaBays(Entity):
    """Trauma bays using PreemptibleResource. Critical gets high priority."""

    def __init__(self, name: str, resource: PreemptibleResource,
                 treatment_time: float, downstream: Entity):
        super().__init__(name)
        self.resource = resource
        self.treatment_time = treatment_time
        self.downstream = downstream
        self._treated = 0
        self._preempted = 0

    def handle_event(self, event: Event) -> Generator:
        severity = event.context.get("severity", "non-critical")
        priority = 0.0 if severity == "critical" else 10.0

        preempted_flag = [False]

        def on_preempt():
            preempted_flag[0] = True

        grant = yield self.resource.acquire(
            amount=1, priority=priority, preempt=True, on_preempt=on_preempt,
        )

        if grant.preempted:
            self._preempted += 1
            return []

        yield self.treatment_time

        grant.release()
        self._treated += 1

        return [
            Event(
                time=self.now,
                event_type="Treated",
                target=self.downstream,
                context=event.context,
            )
        ]


class ExamRooms(RenegingQueuedResource):
    """Exam rooms where non-critical patients may renege."""

    def __init__(self, name: str, num_rooms: int, exam_time: float,
                 downstream: Entity, reneged_target: Entity | None = None,
                 default_patience_s: float = float("inf")):
        super().__init__(
            name,
            reneged_target=reneged_target,
            default_patience_s=default_patience_s,
            policy=FIFOQueue(),
        )
        self._num_rooms = num_rooms
        self.exam_time = exam_time
        self.downstream = downstream
        self._active = 0
        self._processed = 0

    @property
    def processed(self) -> int:
        return self._processed

    def has_capacity(self) -> bool:
        return self._active < self._num_rooms

    def _handle_served_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._active += 1
        try:
            yield self.exam_time
        finally:
            self._active -= 1

        self._processed += 1
        return [
            Event(
                time=self.now,
                event_type="Examined",
                target=self.downstream,
                context=event.context,
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class UrgentCareResult:
    """Results from the urgent care simulation."""

    sink: LatencyTracker
    reception: Station
    triage: Station
    router: ConditionalRouter
    trauma_resource: PreemptibleResource
    trauma_bays: TraumaBays
    exam_rooms: ExamRooms
    treatment: Station
    reneged_counter: Counter
    patient_provider: PatientProvider
    config: UrgentCareConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_urgent_care_simulation(config: UrgentCareConfig | None = None) -> UrgentCareResult:
    """Run the urgent care simulation."""
    if config is None:
        config = UrgentCareConfig()

    random.seed(config.seed)

    sink = LatencyTracker("Sink")
    treatment = Station("Treatment", config.treatment_time, sink)
    reneged_counter = Counter("RenegedCounter")

    # Trauma bays with preemption
    trauma_resource = PreemptibleResource("TraumaResource", config.num_trauma_bays)
    trauma_bays = TraumaBays("TraumaBays", trauma_resource, config.trauma_treatment_time, treatment)

    # Exam rooms with reneging
    exam_rooms = ExamRooms(
        "ExamRooms",
        num_rooms=config.num_exam_rooms,
        exam_time=config.exam_time,
        downstream=treatment,
        reneged_target=reneged_counter,
        default_patience_s=config.renege_patience_s,
    )

    # Router: critical → trauma, non-critical → exam rooms
    router = ConditionalRouter.by_context_field(
        "TriageRouter",
        "severity",
        {"critical": trauma_bays, "non-critical": exam_rooms},
    )

    triage = Station("Triage", config.triage_time, router)
    reception = Station("Reception", config.reception_time, triage)

    stop_after = Instant.from_seconds(config.duration_s)
    patient_provider = PatientProvider(
        reception, config.critical_pct, config.renege_patience_s, stop_after,
    )

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    source = Source(
        name="Patients",
        event_provider=patient_provider,
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
        entities=[
            reception, triage, router,
            trauma_resource, trauma_bays,
            exam_rooms, treatment,
            reneged_counter, sink,
        ],
    )

    summary = sim.run()

    return UrgentCareResult(
        sink=sink,
        reception=reception,
        triage=triage,
        router=router,
        trauma_resource=trauma_resource,
        trauma_bays=trauma_bays,
        exam_rooms=exam_rooms,
        treatment=treatment,
        reneged_counter=reneged_counter,
        patient_provider=patient_provider,
        config=config,
        summary=summary,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: UrgentCareResult) -> None:
    """Print a formatted summary of the urgent care simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("URGENT CARE SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 3600:.0f} hours")
    print(f"  Arrival rate:        {config.arrival_rate_per_min:.1f}/min")
    print(f"  Critical rate:       {config.critical_pct:.0%}")
    print(f"  Trauma bays:         {config.num_trauma_bays}")
    print(f"  Exam rooms:          {config.num_exam_rooms}")
    print(f"  Renege patience:     {config.renege_patience_s / 60:.0f} min")

    total = result.patient_provider.generated

    print(f"\nPatient Flow:")
    print(f"  Arrived:             {total}")
    print(f"  Reception:           {result.reception.processed}")
    print(f"  Triage:              {result.triage.processed}")
    for name, count in result.router.routed_counts.items():
        print(f"  Routed to {name:12s} {count}")

    print(f"\nTrauma Bays:")
    ts = result.trauma_resource.stats
    print(f"  Treated:             {result.trauma_bays._treated}")
    print(f"  Preempted:           {result.trauma_bays._preempted}")
    print(f"  Total preemptions:   {ts.preemptions}")

    print(f"\nExam Rooms:")
    print(f"  Examined:            {result.exam_rooms.processed}")
    print(f"  Reneged:             {result.exam_rooms.reneged}")

    print(f"\nTreatment:")
    print(f"  Treated:             {result.treatment.processed}")

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
    parser = argparse.ArgumentParser(description="Urgent care simulation")
    parser.add_argument("--duration", type=float, default=28800.0, help="Duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=0.5, help="Patients per minute")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = UrgentCareConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running urgent care simulation...")
    result = run_urgent_care_simulation(cfg)
    print_summary(result)
