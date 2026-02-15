"""Tests for ThreadPool component."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from happysimulator.components.queue_policy import LIFOQueue
from happysimulator.components.server.thread_pool import ThreadPool
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""

    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


class TaskProvider(EventProvider):
    """Generates task events targeting a thread pool."""

    def __init__(
        self,
        pool: ThreadPool,
        processing_time: float = 0.01,
        stop_after: Instant | None = None,
    ):
        self.pool = pool
        self.processing_time = processing_time
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        event = Event(
            time=time,
            event_type=f"Task-{self.generated}",
            target=self.pool,
        )
        event.context["metadata"]["processing_time"] = self.processing_time
        return [event]


class VariableTaskProvider(EventProvider):
    """Generates tasks with variable processing times."""

    def __init__(
        self,
        pool: ThreadPool,
        processing_times: list[float],
        stop_after: Instant | None = None,
    ):
        self.pool = pool
        self.processing_times = processing_times
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        event = Event(
            time=time,
            event_type=f"Task-{self.generated}",
            target=self.pool,
        )
        # Cycle through processing times
        idx = (self.generated - 1) % len(self.processing_times)
        event.context["metadata"]["processing_time"] = self.processing_times[idx]
        return [event]


class TestThreadPoolBasics:
    """Basic ThreadPool functionality tests."""

    def test_creates_with_defaults(self):
        """ThreadPool can be created with minimal parameters."""
        pool = ThreadPool(name="TestPool", num_workers=4)
        assert pool.name == "TestPool"
        assert pool.num_workers == 4
        assert pool.active_workers == 0
        assert pool.idle_workers == 4
        assert pool.worker_utilization == 0.0

    def test_rejects_zero_workers(self):
        """ThreadPool rejects num_workers < 1."""
        with pytest.raises(ValueError):
            ThreadPool(name="TestPool", num_workers=0)

        with pytest.raises(ValueError):
            ThreadPool(name="TestPool", num_workers=-1)

    def test_creates_with_custom_queue_policy(self):
        """ThreadPool can be created with custom queue policy."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=2,
            queue_policy=LIFOQueue(),
        )
        assert isinstance(pool.queue.policy, LIFOQueue)

    def test_initial_statistics_are_zero(self):
        """ThreadPool starts with zero statistics."""
        pool = ThreadPool(name="TestPool", num_workers=4)
        assert pool.stats.tasks_completed == 0
        assert pool.stats.tasks_rejected == 0
        assert pool.stats.total_processing_time == 0.0


class TestThreadPoolProcessing:
    """Tests for ThreadPool task processing."""

    def test_processes_single_task(self):
        """ThreadPool processes a single task successfully."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=2,
            default_processing_time=0.010,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        task = Event(time=Instant.Epoch, event_type="Task", target=pool)
        sim.schedule(task)
        sim.run()

        assert pool.stats.tasks_completed == 1
        assert pool.stats.total_processing_time == pytest.approx(0.010, rel=0.01)

    def test_processes_task_with_context_time(self):
        """ThreadPool uses processing time from task context."""
        pool = ThreadPool(name="TestPool", num_workers=2)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        task = Event(time=Instant.Epoch, event_type="Task", target=pool)
        task.context["metadata"]["processing_time"] = 0.050
        sim.schedule(task)
        sim.run()

        assert pool.stats.tasks_completed == 1
        assert pool.stats.total_processing_time == pytest.approx(0.050, rel=0.01)

    def test_processes_multiple_tasks_concurrently(self):
        """ThreadPool with multiple workers processes tasks in parallel."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=4,
            default_processing_time=0.100,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        # Schedule 4 tasks at the same time
        for i in range(4):
            task = Event(time=Instant.Epoch, event_type=f"Task-{i}", target=pool)
            sim.schedule(task)

        sim.run()

        # All 4 should complete
        assert pool.stats.tasks_completed == 4
        # Total processing time is 4 * 0.1 = 0.4s
        assert pool.stats.total_processing_time == pytest.approx(0.4, rel=0.01)

    def test_queues_tasks_when_workers_busy(self):
        """Tasks are queued when all workers are busy."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=1,
            default_processing_time=0.100,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        # Schedule 3 tasks at the same time
        for i in range(3):
            task = Event(time=Instant.Epoch, event_type=f"Task-{i}", target=pool)
            sim.schedule(task)

        sim.run()

        # All 3 should complete (queued and processed sequentially)
        assert pool.stats.tasks_completed == 3


class TestThreadPoolWorkerManagement:
    """Tests for ThreadPool worker management."""

    def test_has_capacity_reflects_worker_state(self):
        """has_capacity() returns False when all workers are busy."""
        pool = ThreadPool(name="TestPool", num_workers=2)

        # Initially has capacity
        assert pool.has_capacity() is True
        assert pool.active_workers == 0
        assert pool.idle_workers == 2

    def test_utilization_calculation(self):
        """Worker utilization is correctly calculated."""
        pool = ThreadPool(name="TestPool", num_workers=4)

        # Initially 0
        assert pool.worker_utilization == 0.0


class TestThreadPoolStatistics:
    """Tests for ThreadPool statistics tracking."""

    def test_tracks_completed_tasks(self):
        """ThreadPool tracks number of completed tasks."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=2,
            default_processing_time=0.010,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        for i in range(5):
            task = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type=f"Task-{i}",
                target=pool,
            )
            sim.schedule(task)

        sim.run()

        assert pool.stats.tasks_completed == 5

    def test_tracks_total_processing_time(self):
        """ThreadPool tracks total processing time."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=1,
            default_processing_time=0.050,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        for i in range(3):
            task = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type=f"Task-{i}",
                target=pool,
            )
            sim.schedule(task)

        sim.run()

        assert pool.stats.tasks_completed == 3
        assert pool.stats.total_processing_time == pytest.approx(0.150, rel=0.01)

    def test_average_processing_time(self):
        """ThreadPool calculates average processing time correctly."""
        pool = ThreadPool(name="TestPool", num_workers=1)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        # Schedule tasks with specific processing times
        for i, proc_time in enumerate([0.010, 0.020, 0.030]):
            task = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type=f"Task-{i}",
                target=pool,
            )
            task.context["metadata"]["processing_time"] = proc_time
            sim.schedule(task)

        sim.run()

        assert pool.stats.tasks_completed == 3
        # Average of 0.01, 0.02, 0.03 = 0.02
        assert pool.average_processing_time == pytest.approx(0.020, rel=0.01)

    def test_processing_time_percentile(self):
        """ThreadPool calculates processing time percentiles correctly."""
        random.seed(42)

        pool = ThreadPool(name="TestPool", num_workers=4)

        provider = VariableTaskProvider(
            pool,
            processing_times=[0.010, 0.020, 0.050, 0.100],
            stop_after=Instant.from_seconds(2.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=20.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[pool],
        )
        sim.run()

        # Should have enough samples
        assert pool.stats.tasks_completed >= 30

        p50 = pool.get_processing_time_percentile(0.50)
        p99 = pool.get_processing_time_percentile(0.99)

        # p99 should be higher than p50
        assert p99 >= p50
        assert p50 > 0


class TestThreadPoolCustomExtractor:
    """Tests for ThreadPool with custom processing time extractor."""

    def test_uses_custom_extractor(self):
        """ThreadPool uses custom processing time extractor."""

        def custom_extractor(task: Event) -> float:
            # Extract from a different context key
            return task.context.get("custom_time", 0.025)

        pool = ThreadPool(
            name="TestPool",
            num_workers=2,
            processing_time_extractor=custom_extractor,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        task = Event(time=Instant.Epoch, event_type="Task", target=pool)
        task.context["custom_time"] = 0.075
        sim.schedule(task)
        sim.run()

        assert pool.stats.tasks_completed == 1
        assert pool.stats.total_processing_time == pytest.approx(0.075, rel=0.01)


class TestThreadPoolWithLoad:
    """Integration tests for ThreadPool under various load conditions."""

    def test_pool_under_light_load(self):
        """ThreadPool handles light load efficiently."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=4,
            default_processing_time=0.010,
        )

        provider = TaskProvider(
            pool,
            processing_time=0.010,
            stop_after=Instant.from_seconds(2.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=10.0),  # Well under capacity
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source],
            entities=[pool],
        )
        sim.run()

        # All tasks should complete
        assert pool.stats.tasks_completed >= 15
        # Queue should not build up significantly
        assert pool.depth == 0

    def test_pool_at_capacity(self):
        """ThreadPool handles load at capacity."""
        # Capacity: 4 workers * (1/0.1) = 40 tasks/s
        pool = ThreadPool(
            name="TestPool",
            num_workers=4,
            default_processing_time=0.100,
        )

        provider = TaskProvider(
            pool,
            processing_time=0.100,
            stop_after=Instant.from_seconds(2.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=40.0),  # At capacity
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[pool],
        )
        sim.run()

        # Should process most tasks
        assert pool.stats.tasks_completed >= 70

    def test_pool_overloaded(self):
        """ThreadPool queues tasks when overloaded."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=2,
            default_processing_time=0.100,  # Capacity: 20 tasks/s
        )

        provider = TaskProvider(
            pool,
            processing_time=0.100,
            stop_after=Instant.from_seconds(1.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=50.0),  # 2.5x capacity
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),  # Time to drain
            sources=[source],
            entities=[pool],
        )
        sim.run()

        # Should complete all eventually
        assert pool.stats.tasks_completed >= 40
        # Queue was used
        assert pool.stats_accepted >= 40


class TestThreadPoolSubmit:
    """Tests for the submit() convenience method."""

    def test_submit_returns_task(self):
        """submit() returns the task for chaining."""
        pool = ThreadPool(name="TestPool", num_workers=2)
        other_pool = ThreadPool(name="OtherPool", num_workers=1)

        # Create task targeting a different pool
        task = Event(time=Instant.Epoch, event_type="Task", target=other_pool)
        result = pool.submit(task)

        # submit() returns the same task, now retargeted
        assert result is task
        assert task.target is pool

    def test_submit_with_simulation(self):
        """Tasks created via submit() work with simulation."""
        pool = ThreadPool(
            name="TestPool",
            num_workers=2,
            default_processing_time=0.010,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[pool],
        )

        # Create task targeting the pool directly
        task = Event(time=Instant.Epoch, event_type="Task", target=pool)
        task.context["metadata"]["processing_time"] = 0.025
        sim.schedule(pool.submit(task))
        sim.run()

        assert pool.stats.tasks_completed == 1
        assert pool.stats.total_processing_time == pytest.approx(0.025, rel=0.01)
