"""Cron-like job scheduler with priority ordering and DAG dependencies.

Provides periodic job scheduling with dependency tracking, allowing
simulation of batch processing pipelines, ETL workflows, and cron-based
task management.

Example:
    from happysimulator.components.scheduling import JobScheduler, JobDefinition

    scheduler = JobScheduler(name="cron", tick_interval=1.0)
    scheduler.add_job(JobDefinition(
        name="extract", target=extractor, event_type="Run",
        interval=10.0, priority=10,
    ))
    scheduler.add_job(JobDefinition(
        name="transform", target=transformer, event_type="Run",
        interval=10.0, priority=5, depends_on=["extract"],
    ))

    sim = Simulation(entities=[scheduler, extractor, transformer, ...])
    sim.schedule(scheduler.start())
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass
class JobDefinition:
    """Definition of a scheduled job.

    Attributes:
        name: Unique job identifier.
        target: Entity to receive the job event.
        event_type: Event type string for the dispatched event.
        interval: Seconds between job executions.
        priority: Higher values execute first when multiple jobs are due.
        depends_on: Names of jobs that must complete before this one runs.
        context: Extra metadata to include in job events.
        enabled: Whether the job is active.
    """

    name: str
    target: Entity
    event_type: str
    interval: float
    priority: int = 0
    depends_on: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class JobState:
    """Runtime state for a scheduled job."""

    last_run_time: Instant | None = None
    last_completion_time: Instant | None = None
    is_running: bool = False
    run_count: int = 0
    failure_count: int = 0


@dataclass(frozen=True)
class JobSchedulerStats:
    """Statistics tracked by JobScheduler."""

    ticks: int = 0
    jobs_triggered: int = 0
    jobs_completed: int = 0
    jobs_skipped_dependency: int = 0
    jobs_skipped_running: int = 0


class JobScheduler(Entity):
    """Periodic job scheduler with priority and DAG dependencies.

    Uses a self-perpetuating tick loop to check for due jobs at regular
    intervals. Jobs are sorted by priority (highest first) and checked
    for DAG dependency satisfaction before firing.

    Attributes:
        name: Scheduler identifier.
        tick_interval: Seconds between scheduler ticks.
        stats: Frozen statistics snapshot (via property).
    """

    def __init__(self, name: str, tick_interval: float = 1.0):
        """Initialize the job scheduler.

        Args:
            name: Scheduler identifier.
            tick_interval: Seconds between evaluation ticks.

        Raises:
            ValueError: If tick_interval is not positive.
        """
        super().__init__(name)

        if tick_interval <= 0:
            raise ValueError(f"tick_interval must be > 0, got {tick_interval}")

        self._tick_interval = tick_interval
        self._jobs: dict[str, JobDefinition] = {}
        self._job_states: dict[str, JobState] = {}
        self._is_running = False
        self._ticks = 0
        self._jobs_triggered = 0
        self._jobs_completed = 0
        self._jobs_skipped_dependency = 0
        self._jobs_skipped_running = 0

        logger.debug(
            "[%s] JobScheduler initialized: tick_interval=%.1fs",
            name,
            tick_interval,
        )

    @property
    def tick_interval(self) -> float:
        """Seconds between scheduler ticks."""
        return self._tick_interval

    @property
    def job_names(self) -> list[str]:
        """Names of all registered jobs."""
        return list(self._jobs.keys())

    @property
    def running_jobs(self) -> list[str]:
        """Names of currently running jobs."""
        return [name for name, state in self._job_states.items() if state.is_running]

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is active."""
        return self._is_running

    @property
    def stats(self) -> JobSchedulerStats:
        """Frozen snapshot of current statistics."""
        return JobSchedulerStats(
            ticks=self._ticks,
            jobs_triggered=self._jobs_triggered,
            jobs_completed=self._jobs_completed,
            jobs_skipped_dependency=self._jobs_skipped_dependency,
            jobs_skipped_running=self._jobs_skipped_running,
        )

    def add_job(self, job: JobDefinition) -> None:
        """Register a job with the scheduler.

        Args:
            job: Job definition to add.

        Raises:
            ValueError: If a job with the same name already exists.
        """
        if job.name in self._jobs:
            raise ValueError(f"Job '{job.name}' already exists")
        self._jobs[job.name] = job
        self._job_states[job.name] = JobState()
        logger.debug(
            "[%s] Added job: %s (interval=%.1fs, priority=%d)",
            self.name,
            job.name,
            job.interval,
            job.priority,
        )

    def remove_job(self, name: str) -> None:
        """Remove a job from the scheduler.

        Args:
            name: Name of the job to remove.
        """
        self._jobs.pop(name, None)
        self._job_states.pop(name, None)

    def enable_job(self, name: str) -> None:
        """Enable a disabled job."""
        if name in self._jobs:
            self._jobs[name].enabled = True

    def disable_job(self, name: str) -> None:
        """Disable a job without removing it."""
        if name in self._jobs:
            self._jobs[name].enabled = False

    def get_job_state(self, name: str) -> JobState | None:
        """Get the runtime state of a job."""
        return self._job_states.get(name)

    def start(self) -> Event:
        """Start the scheduler tick loop.

        Returns:
            The first tick event to schedule.
        """
        self._is_running = True
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_scheduler_tick",
            target=self,
            context={},
        )

    def stop(self) -> None:
        """Stop the scheduler."""
        self._is_running = False
        logger.info("[%s] Scheduler stopped", self.name)

    def handle_event(self, event: Event) -> list[Event] | None:
        """Handle scheduler events.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule.
        """
        if event.event_type == "_scheduler_tick":
            return self._run_tick()
        if event.event_type == "_job_complete":
            return self._handle_job_complete(event)
        return None

    def _run_tick(self) -> list[Event]:
        """Execute a scheduler tick: check all due jobs."""
        if not self._is_running:
            return []

        self._ticks += 1
        result_events: list[Event] = []

        # Collect due jobs, sort by priority (highest first)
        due_jobs = self._get_due_jobs()
        due_jobs.sort(key=lambda j: j.priority, reverse=True)

        for job in due_jobs:
            state = self._job_states[job.name]

            # Skip if already running
            if state.is_running:
                self._jobs_skipped_running += 1
                continue

            # Check DAG dependencies
            if not self._deps_satisfied(job):
                self._jobs_skipped_dependency += 1
                continue

            # Fire the job
            job_event = Event(
                time=self.now,
                event_type=job.event_type,
                target=job.target,
                context={
                    **job.context,
                    "metadata": {
                        **job.context.get("metadata", {}),
                        "_job_name": job.name,
                        "_scheduler": self.name,
                    },
                },
            )

            # Add completion hook
            job_name = job.name

            def on_complete(finish_time: Instant, _name=job_name) -> Event:
                return Event(
                    time=finish_time,
                    event_type="_job_complete",
                    target=self,
                    context={"metadata": {"job_name": _name}},
                )

            job_event.add_completion_hook(on_complete)

            state.is_running = True
            state.last_run_time = self.now
            state.run_count += 1
            self._jobs_triggered += 1

            result_events.append(job_event)
            logger.debug("[%s] Triggered job: %s", self.name, job.name)

        # Schedule next tick
        next_tick = Event(
            time=self.now + Duration.from_seconds(self._tick_interval),
            event_type="_scheduler_tick",
            target=self,
            daemon=True,
            context={},
        )
        result_events.append(next_tick)

        return result_events

    def _get_due_jobs(self) -> list[JobDefinition]:
        """Return jobs that are due to run."""
        due = []
        for name, job in self._jobs.items():
            if not job.enabled:
                continue
            state = self._job_states[name]
            if state.last_run_time is None:
                # Never run: due immediately
                due.append(job)
            else:
                elapsed = (self.now - state.last_run_time).to_seconds()
                if elapsed >= job.interval:
                    due.append(job)
        return due

    def _deps_satisfied(self, job: JobDefinition) -> bool:
        """Check if all DAG dependencies are satisfied.

        A dependency is satisfied when:
        - The dependency job exists and is not currently running
        - The dependency has completed at least once
        - The dependency's last_completion_time is after this job's last_run_time
          (or this job has never run)
        """
        for dep_name in job.depends_on:
            if dep_name not in self._job_states:
                return False
            dep_state = self._job_states[dep_name]
            # Dependency must not be running
            if dep_state.is_running:
                return False
            # Dependency must have completed at least once
            if dep_state.last_completion_time is None:
                return False
            # If this job has run before, dep must have completed after
            state = self._job_states[job.name]
            if (
                state.last_run_time is not None
                and dep_state.last_completion_time <= state.last_run_time
            ):
                return False
        return True

    def _handle_job_complete(self, event: Event) -> None:
        """Handle job completion."""
        job_name = event.context.get("metadata", {}).get("job_name")
        if job_name and job_name in self._job_states:
            state = self._job_states[job_name]
            state.is_running = False
            state.last_completion_time = self.now
            self._jobs_completed += 1
            logger.debug("[%s] Job completed: %s", self.name, job_name)
