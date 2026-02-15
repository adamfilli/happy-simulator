"""CPU scheduling: FairShare vs PriorityPreemptive under mixed workloads.

This example demonstrates how CPU scheduling policy affects task
completion and context switch overhead. The key insight:
PriorityPreemptive provides better latency for high-priority tasks
at the cost of more context switches and potential starvation of
low-priority work.

## Architecture Diagram

```
    Source (constant rate)
        |
        v
    TaskSubmitter ──> CPUScheduler (FairShare)    ──> Sink
    TaskSubmitter ──> CPUScheduler (Priority)     ──> Sink
```

## Key Metrics

- Tasks completed
- Context switch count and overhead fraction
- Total CPU time vs overhead time
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
    SimulationSummary,
    Sink,
    Source,
)
from happysimulator.components.infrastructure import (
    CPUScheduler,
    CPUSchedulerStats,
    FairShare,
    PriorityPreemptive,
)


# =============================================================================
# Custom Entity
# =============================================================================


class TaskSubmitter(Entity):
    """Submits CPU tasks with varying priorities and durations."""

    def __init__(
        self,
        name: str,
        *,
        cpu: CPUScheduler,
        downstream: Entity | None = None,
    ) -> None:
        super().__init__(name)
        self._cpu = cpu
        self._downstream = downstream
        self._task_count: int = 0

    @property
    def task_count(self) -> int:
        return self._task_count

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._task_count += 1
        priority = random.choice([0, 0, 0, 1, 1, 5, 10])
        cpu_time = random.uniform(0.001, 0.02)

        yield from self._cpu.execute(
            f"task-{self._task_count}",
            cpu_time_s=cpu_time,
            priority=priority,
        )

        if self._downstream:
            return [self.forward(event, self._downstream, event_type="Done")]
        return []


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SchedulerResult:
    policy_name: str
    stats: CPUSchedulerStats
    summary: SimulationSummary


@dataclass
class SimulationResult:
    fair_share: SchedulerResult
    priority: SchedulerResult
    duration_s: float


def _run_policy(
    policy_name: str,
    cpu: CPUScheduler,
    *,
    duration_s: float,
    rate: float,
    seed: int | None,
) -> SchedulerResult:
    if seed is not None:
        random.seed(seed)

    sink = Sink()
    submitter = TaskSubmitter(f"Submitter_{policy_name}", cpu=cpu, downstream=sink)

    source = Source.constant(
        rate=rate,
        target=submitter,
        event_type="Submit",
        stop_after=Instant.from_seconds(duration_s),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[cpu, submitter, sink],
    )
    summary = sim.run()

    return SchedulerResult(
        policy_name=policy_name,
        stats=cpu.stats,
        summary=summary,
    )


def run_simulation(
    *,
    duration_s: float = 5.0,
    rate: float = 200.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Compare FairShare and PriorityPreemptive scheduling."""
    fair = _run_policy(
        "FairShare",
        CPUScheduler("CPU_Fair", policy=FairShare(quantum_s=0.005)),
        duration_s=duration_s, rate=rate, seed=seed,
    )
    priority = _run_policy(
        "PriorityPreemptive",
        CPUScheduler("CPU_Priority", policy=PriorityPreemptive(quantum_s=0.005)),
        duration_s=duration_s, rate=rate, seed=seed,
    )

    return SimulationResult(
        fair_share=fair,
        priority=priority,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    print("\n" + "=" * 72)
    print("CPU SCHEDULING: FairShare vs PriorityPreemptive")
    print("=" * 72)

    f, p = result.fair_share.stats, result.priority.stats
    header = f"{'Metric':<35} {'FairShare':>15} {'Priority':>15}"
    print(f"\n{header}")
    print("-" * len(header))
    print(f"{'Tasks completed':<35} {f.tasks_completed:>15,} {p.tasks_completed:>15,}")
    print(f"{'Context switches':<35} {f.context_switches:>15,} {p.context_switches:>15,}")
    print(f"{'Total CPU time (s)':<35} {f.total_cpu_time_s:>15.4f} {p.total_cpu_time_s:>15.4f}")
    print(f"{'Context switch overhead (s)':<35} {f.total_context_switch_overhead_s:>15.6f} {p.total_context_switch_overhead_s:>15.6f}")
    print(f"{'Overhead fraction':<35} {f.overhead_fraction:>15.4%} {p.overhead_fraction:>15.4%}")
    print(f"{'Total wait time (s)':<35} {f.total_wait_time_s:>15.4f} {p.total_wait_time_s:>15.4f}")
    print(f"{'Peak queue depth':<35} {f.peak_queue_depth:>15} {p.peak_queue_depth:>15}")

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  FairShare gives equal time slices (round-robin), ensuring fairness")
    print("  but potentially delaying urgent tasks. PriorityPreemptive serves")
    print("  high-priority tasks first, reducing their latency but causing more")
    print("  context switches and potentially starving low-priority work.")
    print("\n" + "=" * 72)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CPU scheduling comparison")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    print("Running CPU scheduling comparison...")
    result = run_simulation(duration_s=args.duration, seed=seed)
    print_summary(result)
