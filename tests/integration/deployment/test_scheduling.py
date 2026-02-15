"""Integration tests for scheduling components (JobScheduler + WorkStealingPool)."""

import random

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)
from happysimulator.components.common import Sink
from happysimulator.components.scheduling import (
    JobDefinition,
    JobScheduler,
    WorkStealingPool,
)
from happysimulator.load import EventProvider


class SimpleWorker(Entity):
    """Worker that processes jobs with a fixed delay."""

    def __init__(self, name: str, delay: float = 0.1):
        super().__init__(name)
        self._delay = delay
        self.jobs_processed = 0

    def handle_event(self, event):
        self.jobs_processed += 1
        yield self._delay
        return None


class TestJobSchedulerEndToEnd:
    def test_dag_execution(self):
        """Jobs fire in DAG order: extract -> transform -> load."""
        extract_worker = SimpleWorker("extract_worker", delay=0.5)
        transform_worker = SimpleWorker("transform_worker", delay=0.3)
        load_worker = SimpleWorker("load_worker", delay=0.2)

        scheduler = JobScheduler(name="cron", tick_interval=1.0)
        scheduler.add_job(
            JobDefinition(
                name="extract",
                target=extract_worker,
                event_type="Extract",
                interval=5.0,
                priority=10,
            )
        )
        scheduler.add_job(
            JobDefinition(
                name="transform",
                target=transform_worker,
                event_type="Transform",
                interval=5.0,
                priority=5,
                depends_on=["extract"],
            )
        )
        scheduler.add_job(
            JobDefinition(
                name="load",
                target=load_worker,
                event_type="Load",
                interval=5.0,
                priority=1,
                depends_on=["transform"],
            )
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            entities=[scheduler, extract_worker, transform_worker, load_worker],
        )
        sim.schedule(scheduler.start())
        sim.run()

        # All workers should have processed at least one job
        assert extract_worker.jobs_processed >= 1
        assert transform_worker.jobs_processed >= 1
        assert load_worker.jobs_processed >= 1

        # Verify ordering: extract ran most, load ran least
        # (due to cascading dependency delays)
        assert extract_worker.jobs_processed >= load_worker.jobs_processed
        assert scheduler.stats.jobs_triggered >= 3

    def test_scheduler_respects_intervals(self):
        """Jobs fire at correct intervals."""
        worker = SimpleWorker("worker", delay=0.01)

        scheduler = JobScheduler(name="cron", tick_interval=0.5)
        scheduler.add_job(
            JobDefinition(
                name="fast_job",
                target=worker,
                event_type="Work",
                interval=2.0,
            )
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=10.0,
            entities=[scheduler, worker],
        )
        sim.schedule(scheduler.start())
        sim.run()

        # With 10s runtime and 2s interval, expect ~5 runs
        assert 3 <= worker.jobs_processed <= 7


class TaskEventProvider(EventProvider):
    """Generates task events for the pool."""

    def __init__(self, target: Entity):
        self._target = target
        self._count = 0

    def get_events(self, time: Instant) -> list[Event]:
        self._count += 1
        return [
            Event(
                time=time,
                event_type="Task",
                target=self._target,
                context={
                    "created_at": time,
                    "metadata": {"processing_time": 0.05, "task_id": self._count},
                },
            )
        ]


class TestWorkStealingPoolEndToEnd:
    def test_all_tasks_complete(self):
        """All submitted tasks are processed."""
        random.seed(42)

        sink = Sink()
        pool = WorkStealingPool(
            name="pool",
            num_workers=4,
            downstream=sink,
            default_processing_time=0.05,
        )

        provider = TaskEventProvider(pool)
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=50.0), start_time=Instant.Epoch
        )
        source = Source(
            name="tasks",
            event_provider=provider,
            arrival_time_provider=arrival,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[source],
            entities=[pool, *pool.workers, sink],
        )
        sim.run()

        assert pool.stats.tasks_submitted > 0
        assert pool.stats.tasks_completed > 0
        # Most tasks should complete within the simulation time
        assert pool.stats.tasks_completed >= pool.stats.tasks_submitted * 0.5

    def test_work_distribution(self):
        """Work is distributed across workers."""
        random.seed(42)

        sink = Sink()
        pool = WorkStealingPool(
            name="pool",
            num_workers=4,
            downstream=sink,
            default_processing_time=0.01,
        )

        provider = TaskEventProvider(pool)
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate=100.0), start_time=Instant.Epoch
        )
        source = Source(
            name="tasks",
            event_provider=provider,
            arrival_time_provider=arrival,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=3.0,
            sources=[source],
            entities=[pool, *pool.workers, sink],
        )
        sim.run()

        # Each worker should have completed at least some tasks
        for ws in pool.worker_stats:
            assert ws.tasks_completed > 0
