"""Work-stealing pool example: comparing tail latency under skewed workload.

Demonstrates how work-stealing reduces tail latency compared to a
simple shared-queue approach when tasks have variable processing times.

## Architecture

```
    Source (100 req/s, skewed processing times)
        |
        v
    WorkStealingPool (4 workers)
        |
        v
      Sink (latency tracking)
```

Tasks have bimodal processing times: 90% are fast (10ms), 10% are
slow (100ms). Work-stealing helps because idle workers steal from
neighbors with long-running tasks queued up.
"""

from __future__ import annotations

import random

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    EventProvider,
    Instant,
    LatencyTracker,
    Simulation,
    Source,
)
from happysimulator.components.scheduling import WorkStealingPool


class SkewedTaskProvider(EventProvider):
    """Generates tasks with bimodal processing times."""

    def __init__(self, target: Entity, fast_pct: float = 0.9,
                 fast_time: float = 0.01, slow_time: float = 0.1):
        self._target = target
        self._fast_pct = fast_pct
        self._fast_time = fast_time
        self._slow_time = slow_time
        self._count = 0

    def get_events(self, time: Instant) -> list[Event]:
        self._count += 1
        is_fast = random.random() < self._fast_pct
        processing_time = self._fast_time if is_fast else self._slow_time

        return [Event(
            time=time, event_type="Task", target=self._target,
            context={
                "created_at": time,
                "metadata": {
                    "processing_time": processing_time,
                    "task_id": self._count,
                    "is_fast": is_fast,
                },
            },
        )]


def run_work_stealing_demo(
    duration_s: float = 10.0,
    num_workers: int = 4,
    rate: float = 100.0,
    seed: int = 42,
) -> None:
    """Run work-stealing pool under skewed workload."""
    random.seed(seed)

    sink = LatencyTracker(name="Sink")
    pool = WorkStealingPool(
        name="Pool", num_workers=num_workers, downstream=sink,
        default_processing_time=0.01,
    )

    provider = SkewedTaskProvider(pool)
    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=rate), start_time=Instant.Epoch,
    )
    source = Source(
        name="Traffic", event_provider=provider,
        arrival_time_provider=arrival,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 2.0,
        sources=[source],
        entities=[pool] + pool.workers + [sink],
    )
    summary = sim.run()

    # Results
    print("=" * 60)
    print("WORK-STEALING POOL RESULTS")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Workers: {num_workers}")
    print(f"  Rate: {rate} tasks/s")
    print(f"  Task mix: 90% fast (10ms), 10% slow (100ms)")

    print(f"\nPool stats:")
    print(f"  Tasks submitted: {pool.stats.tasks_submitted}")
    print(f"  Tasks completed: {pool.stats.tasks_completed}")
    print(f"  Total steals: {pool.stats.total_steals}")
    print(f"  Steal attempts: {pool.stats.total_steal_attempts}")

    print(f"\nPer-worker breakdown:")
    for i, ws in enumerate(pool.worker_stats):
        print(f"  Worker {i}: completed={ws.tasks_completed}, "
              f"stolen={ws.tasks_stolen}, "
              f"processing={ws.total_processing_time:.2f}s")

    if sink.count > 0:
        print(f"\nLatency:")
        print(f"  Count: {sink.count}")
        print(f"  Mean: {sink.mean_latency()*1000:.1f}ms")
        print(f"  p50:  {sink.p50()*1000:.1f}ms")
        print(f"  p99:  {sink.p99()*1000:.1f}ms")

    print(f"\n{summary}")
    print("=" * 60)


if __name__ == "__main__":
    run_work_stealing_demo()
