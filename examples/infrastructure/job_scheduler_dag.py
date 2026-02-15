"""DAG-based job scheduler example: Extract -> Transform -> Load.

Demonstrates dependency-driven execution ordering where downstream
jobs only fire after their upstream dependencies complete.

## Architecture

```
    JobScheduler (tick every 1s)
        |
        v
    [Extract] --depends--> [Transform] --depends--> [Load]
      (0.5s)                  (0.3s)                (0.2s)
```

Each job runs at a 5-second interval. Transform waits for Extract
to complete, and Load waits for Transform.
"""

from __future__ import annotations

import random
from typing import Generator

from happysimulator import Entity, Event, Instant, Simulation
from happysimulator.components.scheduling import JobDefinition, JobScheduler


class ETLWorker(Entity):
    """Worker that processes ETL jobs with a configurable delay."""

    def __init__(self, name: str, delay: float = 0.1):
        super().__init__(name)
        self._delay = delay
        self.jobs_processed = 0
        self.execution_times: list[float] = []

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        start = self.now.to_seconds()
        yield self._delay
        self.jobs_processed += 1
        self.execution_times.append(start)


def run_dag_scheduler(duration_s: float = 30.0, seed: int = 42) -> None:
    """Run a DAG scheduler with Extract -> Transform -> Load pipeline."""
    random.seed(seed)

    # Create ETL workers
    extract_worker = ETLWorker("Extract", delay=0.5)
    transform_worker = ETLWorker("Transform", delay=0.3)
    load_worker = ETLWorker("Load", delay=0.2)

    # Create scheduler
    scheduler = JobScheduler(name="ETL_Scheduler", tick_interval=1.0)

    # Define DAG: Extract -> Transform -> Load
    scheduler.add_job(JobDefinition(
        name="extract", target=extract_worker, event_type="Extract",
        interval=5.0, priority=10,
    ))
    scheduler.add_job(JobDefinition(
        name="transform", target=transform_worker, event_type="Transform",
        interval=5.0, priority=5, depends_on=["extract"],
    ))
    scheduler.add_job(JobDefinition(
        name="load", target=load_worker, event_type="Load",
        interval=5.0, priority=1, depends_on=["transform"],
    ))

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s,
        entities=[scheduler, extract_worker, transform_worker, load_worker],
    )
    sim.schedule(scheduler.start())
    summary = sim.run()

    # Print results
    print("=" * 60)
    print("DAG JOB SCHEDULER RESULTS")
    print("=" * 60)
    print(f"\nDuration: {duration_s}s")
    print(f"Scheduler ticks: {scheduler.stats.ticks}")
    print(f"Jobs triggered: {scheduler.stats.jobs_triggered}")
    print(f"Jobs completed: {scheduler.stats.jobs_completed}")
    print(f"Jobs skipped (dependency): {scheduler.stats.jobs_skipped_dependency}")
    print(f"Jobs skipped (running): {scheduler.stats.jobs_skipped_running}")

    print(f"\nPer-job execution counts:")
    print(f"  Extract:   {extract_worker.jobs_processed} runs")
    print(f"  Transform: {transform_worker.jobs_processed} runs")
    print(f"  Load:      {load_worker.jobs_processed} runs")

    if extract_worker.execution_times and transform_worker.execution_times:
        print(f"\nExecution timeline (first 5 of each):")
        print(f"  Extract:   {[f'{t:.1f}s' for t in extract_worker.execution_times[:5]]}")
        print(f"  Transform: {[f'{t:.1f}s' for t in transform_worker.execution_times[:5]]}")
        print(f"  Load:      {[f'{t:.1f}s' for t in load_worker.execution_times[:5]]}")

    print(f"\n{summary}")
    print("=" * 60)


if __name__ == "__main__":
    run_dag_scheduler()
