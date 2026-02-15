"""Unit tests for CPUScheduler."""

import pytest

from happysimulator.components.infrastructure.cpu_scheduler import (
    CPUScheduler,
    CPUSchedulerStats,
    CPUTask,
    FairShare,
    PriorityPreemptive,
)
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestSchedulingPolicies:
    def test_fair_share_selects_first(self):
        policy = FairShare()
        tasks = [CPUTask("a"), CPUTask("b")]
        assert policy.select_next(tasks).task_id == "a"

    def test_fair_share_empty(self):
        policy = FairShare()
        assert policy.select_next([]) is None

    def test_fair_share_quantum(self):
        policy = FairShare(quantum_s=0.02)
        task = CPUTask("a")
        assert policy.time_quantum_s(task) == 0.02

    def test_fair_share_invalid_quantum(self):
        with pytest.raises(ValueError, match="quantum_s must be > 0"):
            FairShare(quantum_s=0.0)

    def test_priority_selects_highest(self):
        policy = PriorityPreemptive()
        tasks = [CPUTask("low", priority=1), CPUTask("high", priority=10)]
        assert policy.select_next(tasks).task_id == "high"

    def test_priority_empty(self):
        policy = PriorityPreemptive()
        assert policy.select_next([]) is None

    def test_priority_invalid_quantum(self):
        with pytest.raises(ValueError, match="quantum_s must be > 0"):
            PriorityPreemptive(quantum_s=-1.0)


class TestCPUScheduler:
    def _make_scheduler(self, **kwargs) -> tuple[CPUScheduler, Simulation]:
        cpu = CPUScheduler("test_cpu", **kwargs)
        sim = Simulation(
            start_time=Instant.from_seconds(0),
            end_time=Instant.from_seconds(100),
            entities=[cpu],
        )
        return cpu, sim

    def test_creation_defaults(self):
        cpu = CPUScheduler("cpu")
        assert cpu.name == "cpu"
        assert cpu.ready_queue_depth == 0

    def test_stats_initial(self):
        cpu, _sim = self._make_scheduler()
        stats = cpu.stats
        assert isinstance(stats, CPUSchedulerStats)
        assert stats.tasks_completed == 0
        assert stats.context_switches == 0
        assert stats.total_cpu_time_s == 0.0

    def test_single_task_completes(self):
        cpu, _sim = self._make_scheduler()
        gen = cpu.execute("task-1", cpu_time_s=0.02)

        # Exhaust the generator
        values = []
        try:
            while True:
                values.append(next(gen))
        except StopIteration:
            pass

        assert cpu.stats.tasks_completed == 1
        assert cpu.stats.total_cpu_time_s == pytest.approx(0.02)

    def test_task_yields_positive_delays(self):
        cpu, _sim = self._make_scheduler()
        gen = cpu.execute("task-1", cpu_time_s=0.05)

        values = []
        try:
            while True:
                values.append(next(gen))
        except StopIteration:
            pass

        assert all(v > 0 for v in values)

    def test_overhead_fraction_zero_single_task(self):
        cpu, _sim = self._make_scheduler()
        gen = cpu.execute("task-1", cpu_time_s=0.02)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Single task, no context switches needed
        assert cpu.stats.context_switches == 0
        assert cpu.stats.overhead_fraction == 0.0

    def test_repr(self):
        cpu, _sim = self._make_scheduler()
        assert "test_cpu" in repr(cpu)

    def test_handle_event_is_noop(self):
        cpu, _sim = self._make_scheduler()
        from happysimulator.core.event import Event

        event = Event(
            time=Instant.from_seconds(1),
            event_type="Test",
            target=cpu,
        )
        result = cpu.handle_event(event)
        assert result is None

    def test_peak_queue_depth(self):
        cpu, _sim = self._make_scheduler()
        # Start a task
        gen = cpu.execute("task-1", cpu_time_s=0.02)
        next(gen)  # start executing
        assert cpu.stats.peak_queue_depth >= 1
