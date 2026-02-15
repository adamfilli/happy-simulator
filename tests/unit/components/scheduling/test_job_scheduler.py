"""Unit tests for JobScheduler."""

import pytest

from happysimulator.components.scheduling.job_scheduler import (
    JobDefinition,
    JobScheduler,
)
from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class DummyTarget(Entity):
    """Simple entity that records received events."""

    def __init__(self, name: str = "target"):
        super().__init__(name)
        self.received: list[Event] = []

    def handle_event(self, event):
        self.received.append(event)
        return


def make_scheduler(tick_interval: float = 1.0, time: float = 0.0) -> tuple[JobScheduler, Clock]:
    clock = Clock(Instant.from_seconds(time))
    scheduler = JobScheduler(name="scheduler", tick_interval=tick_interval)
    scheduler.set_clock(clock)
    return scheduler, clock


class TestJobSchedulerCreation:
    def test_basic_creation(self):
        scheduler, _ = make_scheduler()
        assert scheduler.name == "scheduler"
        assert scheduler.tick_interval == 1.0
        assert scheduler.job_names == []
        assert not scheduler.is_running

    def test_invalid_tick_interval(self):
        with pytest.raises(ValueError):
            JobScheduler(name="bad", tick_interval=0)
        with pytest.raises(ValueError):
            JobScheduler(name="bad", tick_interval=-1)


class TestJobManagement:
    def test_add_job(self):
        scheduler, _ = make_scheduler()
        target = DummyTarget()
        job = JobDefinition(name="j1", target=target, event_type="Run", interval=5.0)
        scheduler.add_job(job)
        assert "j1" in scheduler.job_names

    def test_add_duplicate_raises(self):
        scheduler, _ = make_scheduler()
        target = DummyTarget()
        job = JobDefinition(name="j1", target=target, event_type="Run", interval=5.0)
        scheduler.add_job(job)
        with pytest.raises(ValueError):
            scheduler.add_job(job)

    def test_remove_job(self):
        scheduler, _ = make_scheduler()
        target = DummyTarget()
        scheduler.add_job(JobDefinition(name="j1", target=target, event_type="Run", interval=5.0))
        scheduler.remove_job("j1")
        assert "j1" not in scheduler.job_names

    def test_enable_disable(self):
        scheduler, _ = make_scheduler()
        target = DummyTarget()
        job = JobDefinition(name="j1", target=target, event_type="Run", interval=5.0)
        scheduler.add_job(job)
        scheduler.disable_job("j1")
        assert not job.enabled
        scheduler.enable_job("j1")
        assert job.enabled

    def test_get_job_state(self):
        scheduler, _ = make_scheduler()
        target = DummyTarget()
        scheduler.add_job(JobDefinition(name="j1", target=target, event_type="Run", interval=5.0))
        state = scheduler.get_job_state("j1")
        assert state is not None
        assert state.run_count == 0
        assert not state.is_running


class TestStartStop:
    def test_start_returns_event(self):
        scheduler, _ = make_scheduler()
        event = scheduler.start()
        assert event.event_type == "_scheduler_tick"
        assert event.target is scheduler
        assert scheduler.is_running

    def test_stop(self):
        scheduler, _ = make_scheduler()
        scheduler.start()
        scheduler.stop()
        assert not scheduler.is_running


class TestTickExecution:
    def test_tick_fires_due_job(self):
        scheduler, clock = make_scheduler()
        target = DummyTarget()
        target.set_clock(clock)
        scheduler.add_job(
            JobDefinition(
                name="j1",
                target=target,
                event_type="Run",
                interval=5.0,
            )
        )
        scheduler._is_running = True

        # First tick: job never run, should fire
        tick_event = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick_event)

        # Should produce job event + next tick
        job_events = [e for e in result if e.event_type == "Run"]
        tick_events = [e for e in result if e.event_type == "_scheduler_tick"]
        assert len(job_events) == 1
        assert len(tick_events) == 1
        assert scheduler.stats.jobs_triggered == 1

    def test_tick_not_running_returns_empty(self):
        scheduler, _ = make_scheduler()
        tick_event = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick_event)
        assert result == []

    def test_disabled_job_skipped(self):
        scheduler, clock = make_scheduler()
        target = DummyTarget()
        target.set_clock(clock)
        job = JobDefinition(name="j1", target=target, event_type="Run", interval=5.0, enabled=False)
        scheduler.add_job(job)
        scheduler._is_running = True

        tick_event = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick_event)

        job_events = [e for e in result if e.event_type == "Run"]
        assert len(job_events) == 0

    def test_job_not_due_skipped(self):
        scheduler, clock = make_scheduler()
        target = DummyTarget()
        target.set_clock(clock)
        scheduler.add_job(
            JobDefinition(
                name="j1",
                target=target,
                event_type="Run",
                interval=5.0,
            )
        )
        scheduler._is_running = True

        # First tick triggers the job
        tick1 = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        scheduler.handle_event(tick1)
        assert scheduler.stats.jobs_triggered == 1

        # Complete the job so it's no longer running
        clock._current_time = Instant.from_seconds(0.5)
        complete = Event(
            time=Instant.from_seconds(0.5),
            event_type="_job_complete",
            target=scheduler,
            context={"metadata": {"job_name": "j1"}},
        )
        scheduler.handle_event(complete)

        # Second tick at t=1s: job interval is 5s, not yet due
        clock._current_time = Instant.from_seconds(1.0)
        tick2 = Event(
            time=Instant.from_seconds(1.0),
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick2)
        job_events = [e for e in result if e.event_type == "Run"]
        # Job not due (only 1s elapsed of 5s interval)
        assert len(job_events) == 0
        assert scheduler.stats.jobs_triggered == 1  # unchanged

    def test_running_job_skipped_when_due(self):
        scheduler, clock = make_scheduler()
        target = DummyTarget()
        target.set_clock(clock)
        scheduler.add_job(
            JobDefinition(
                name="j1",
                target=target,
                event_type="Run",
                interval=5.0,
            )
        )
        scheduler._is_running = True

        # Trigger the job at t=0
        tick1 = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        scheduler.handle_event(tick1)

        # At t=6s the job is due again but still running (no completion)
        clock._current_time = Instant.from_seconds(6.0)
        tick2 = Event(
            time=Instant.from_seconds(6.0),
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick2)
        job_events = [e for e in result if e.event_type == "Run"]
        assert len(job_events) == 0
        assert scheduler.stats.jobs_skipped_running == 1


class TestPriorityOrdering:
    def test_higher_priority_fires_first(self):
        scheduler, clock = make_scheduler()
        t1 = DummyTarget("t1")
        t2 = DummyTarget("t2")
        t1.set_clock(clock)
        t2.set_clock(clock)

        scheduler.add_job(
            JobDefinition(
                name="low",
                target=t1,
                event_type="RunLow",
                interval=5.0,
                priority=1,
            )
        )
        scheduler.add_job(
            JobDefinition(
                name="high",
                target=t2,
                event_type="RunHigh",
                interval=5.0,
                priority=10,
            )
        )
        scheduler._is_running = True

        tick_event = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick_event)
        job_events = [e for e in result if e.event_type in ("RunLow", "RunHigh")]

        # Both should fire, high priority first
        assert len(job_events) == 2
        assert job_events[0].event_type == "RunHigh"
        assert job_events[1].event_type == "RunLow"


class TestDAGDependencies:
    def test_dependency_blocks_until_completed(self):
        scheduler, clock = make_scheduler()
        t1 = DummyTarget("t1")
        t2 = DummyTarget("t2")
        t1.set_clock(clock)
        t2.set_clock(clock)

        scheduler.add_job(
            JobDefinition(
                name="extract",
                target=t1,
                event_type="Extract",
                interval=10.0,
            )
        )
        scheduler.add_job(
            JobDefinition(
                name="transform",
                target=t2,
                event_type="Transform",
                interval=10.0,
                depends_on=["extract"],
            )
        )
        scheduler._is_running = True

        # Tick: extract fires, transform blocked (extract hasn't completed)
        tick1 = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick1)
        job_types = [e.event_type for e in result if e.event_type in ("Extract", "Transform")]
        assert "Extract" in job_types
        assert "Transform" not in job_types
        assert scheduler.stats.jobs_skipped_dependency == 1

    def test_dependency_unblocks_after_completion(self):
        scheduler, clock = make_scheduler()
        t1 = DummyTarget("t1")
        t2 = DummyTarget("t2")
        t1.set_clock(clock)
        t2.set_clock(clock)

        scheduler.add_job(
            JobDefinition(
                name="extract",
                target=t1,
                event_type="Extract",
                interval=10.0,
            )
        )
        scheduler.add_job(
            JobDefinition(
                name="transform",
                target=t2,
                event_type="Transform",
                interval=10.0,
                depends_on=["extract"],
            )
        )
        scheduler._is_running = True

        # First tick: trigger extract (transform blocked by deps)
        tick1 = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        scheduler.handle_event(tick1)

        # Complete extract at t=1s
        clock._current_time = Instant.from_seconds(1.0)
        complete_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="_job_complete",
            target=scheduler,
            context={"metadata": {"job_name": "extract"}},
        )
        scheduler.handle_event(complete_event)

        # Tick at t=2s: extract not due yet (interval=10s), transform is due
        # and extract completed after transform's last_run_time (None) → deps satisfied
        clock._current_time = Instant.from_seconds(2.0)
        tick2 = Event(
            time=Instant.from_seconds(2.0),
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick2)
        job_types = [e.event_type for e in result if e.event_type in ("Extract", "Transform")]
        assert "Transform" in job_types
        assert "Extract" not in job_types  # not due yet


class TestCompletionHooks:
    def test_completion_updates_state(self):
        scheduler, clock = make_scheduler()
        target = DummyTarget()
        target.set_clock(clock)
        scheduler.add_job(
            JobDefinition(
                name="j1",
                target=target,
                event_type="Run",
                interval=5.0,
            )
        )
        scheduler._is_running = True

        # Trigger job
        tick = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        scheduler.handle_event(tick)
        state = scheduler.get_job_state("j1")
        assert state.is_running
        assert state.run_count == 1

        # Complete job
        clock._current_time = Instant.from_seconds(1.0)
        complete = Event(
            time=Instant.from_seconds(1.0),
            event_type="_job_complete",
            target=scheduler,
            context={"metadata": {"job_name": "j1"}},
        )
        scheduler.handle_event(complete)
        assert not state.is_running
        assert state.last_completion_time == Instant.from_seconds(1.0)
        assert scheduler.stats.jobs_completed == 1

    def test_job_event_has_completion_hook(self):
        scheduler, clock = make_scheduler()
        target = DummyTarget()
        target.set_clock(clock)
        scheduler.add_job(
            JobDefinition(
                name="j1",
                target=target,
                event_type="Run",
                interval=5.0,
            )
        )
        scheduler._is_running = True

        tick = Event(
            time=Instant.Epoch,
            event_type="_scheduler_tick",
            target=scheduler,
            context={},
        )
        result = scheduler.handle_event(tick)
        job_events = [e for e in result if e.event_type == "Run"]
        assert len(job_events) == 1
        assert len(job_events[0].on_complete) > 0
