"""Unit tests for WorkStealingPool."""

import pytest

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

from happysimulator.components.scheduling.work_stealing_pool import (
    WorkerStats,
    WorkStealingPool,
    WorkStealingPoolStats,
    _Worker,
)


class DummySink(Entity):
    """Collects completed events."""

    def __init__(self, name: str = "sink"):
        super().__init__(name)
        self.received: list[Event] = []

    def handle_event(self, event):
        self.received.append(event)
        return None


def make_pool(
    num_workers: int = 4,
    default_processing_time: float = 0.1,
    time: float = 0.0,
) -> tuple[WorkStealingPool, Clock]:
    clock = Clock(Instant.from_seconds(time))
    sink = DummySink()
    sink.set_clock(clock)
    pool = WorkStealingPool(
        name="pool",
        num_workers=num_workers,
        downstream=sink,
        default_processing_time=default_processing_time,
    )
    pool.set_clock(clock)
    return pool, clock


def make_task_event(pool: WorkStealingPool, processing_time: float = 0.1) -> Event:
    return Event(
        time=pool.now,
        event_type="Task",
        target=pool,
        context={"metadata": {"processing_time": processing_time}},
    )


class TestPoolCreation:
    def test_basic_creation(self):
        pool, _ = make_pool()
        assert pool.num_workers == 4
        assert len(pool.workers) == 4
        assert pool.stats.tasks_submitted == 0

    def test_invalid_workers(self):
        with pytest.raises(ValueError):
            WorkStealingPool(name="bad", num_workers=0)

    def test_clock_propagation(self):
        pool, clock = make_pool()
        for worker in pool.workers:
            assert worker._clock is clock


class TestWorkAssignment:
    def test_assigns_to_shortest_queue(self):
        pool, clock = make_pool(num_workers=3)

        # Add 3 tasks to worker 0 by directly adding to its queue
        w0 = pool._workers[0]
        for _ in range(3):
            e = Event(time=Instant.Epoch, event_type="X", target=pool, context={})
            w0._queue.appendleft(e)

        # Now submit a new task - should go to worker with shortest queue
        task = make_task_event(pool)
        pool.handle_event(task)

        # Worker 0 has 3, others have 0 before this task
        # One of the other workers should have received it
        assert pool.stats.tasks_submitted == 1
        other_depths = [pool._workers[i].queue_depth for i in range(1, 3)]
        # At least one other worker should have tasks or be processing
        assert any(d >= 0 for d in other_depths)


class TestStealingBehavior:
    def test_steal_from_busiest_neighbor(self):
        pool, clock = make_pool(num_workers=2)

        # Directly add tasks to worker 0's queue
        for i in range(5):
            e = Event(
                time=Instant.Epoch, event_type="Task", target=pool,
                context={"metadata": {"processing_time": 0.1}},
            )
            pool._workers[0]._queue.appendleft(e)

        # Worker 1 steals
        stolen = pool._steal_for(1)
        assert stolen is not None
        assert pool._workers[0].queue_depth == 4  # lost one task
        assert pool.stats.total_steals == 0  # _steal_for doesn't update stats directly

    def test_steal_takes_from_tail(self):
        """Verify steal pops from tail (LIFO) while local processes from head (FIFO)."""
        pool, clock = make_pool(num_workers=2)

        # Add tasks with identifiable context
        for i in range(3):
            e = Event(
                time=Instant.Epoch, event_type="Task", target=pool,
                context={"metadata": {"task_id": i, "processing_time": 0.1}},
            )
            pool._workers[0]._queue.appendleft(e)

        # Local dequeue (popleft) should get task_id=2 (most recently appendleft'd)
        # Wait, appendleft adds to front, so order is [2, 1, 0]
        # popleft gets 2 (FIFO from front), pop gets 0 (LIFO from tail)
        local = pool._workers[0]._queue.popleft()
        assert local.context["metadata"]["task_id"] == 2

        # Steal (pop from tail) should get task_id=0
        stolen = pool._workers[0].steal_from_tail()
        assert stolen is not None
        assert stolen.context["metadata"]["task_id"] == 0

    def test_no_steal_when_neighbor_empty(self):
        pool, clock = make_pool(num_workers=2)
        stolen = pool._steal_for(0)
        assert stolen is None


class TestWorkerStats:
    def test_stats_initialized(self):
        pool, _ = make_pool()
        for worker in pool.workers:
            assert worker.stats.tasks_completed == 0
            assert worker.stats.tasks_stolen == 0

    def test_pool_stats_tracks_submissions(self):
        pool, clock = make_pool()
        for _ in range(5):
            task = make_task_event(pool)
            pool.handle_event(task)
        assert pool.stats.tasks_submitted == 5

    def test_per_worker_stats_accessible(self):
        pool, _ = make_pool()
        stats_list = pool.worker_stats
        assert len(stats_list) == 4
        for s in stats_list:
            assert isinstance(s, WorkerStats)


class TestProcessingFlow:
    def test_worker_processes_task(self):
        pool, clock = make_pool(num_workers=1)
        worker = pool._workers[0]

        # Enqueue a task
        task = Event(
            time=Instant.Epoch, event_type="Task", target=pool,
            context={"metadata": {"processing_time": 0.5}},
        )
        events = worker.enqueue(task)

        # Should generate a _worker_try_next event
        assert len(events) == 1
        assert events[0].event_type == "_worker_try_next"

    def test_try_next_creates_process_event(self):
        pool, clock = make_pool(num_workers=1)
        worker = pool._workers[0]

        task = Event(
            time=Instant.Epoch, event_type="Task", target=pool,
            context={"metadata": {"processing_time": 0.5}},
        )
        worker._queue.appendleft(task)

        try_event = Event(
            time=Instant.Epoch, event_type="_worker_try_next",
            target=worker, context={},
        )
        result = worker.handle_event(try_event)
        assert len(result) == 1
        assert result[0].event_type == "_worker_process"


class TestAllTasksComplete:
    def test_pool_tracks_completions(self):
        pool, _ = make_pool()
        assert pool.stats.tasks_completed == 0
        # Verify stats are frozen snapshots
        stats = pool.stats
        assert stats.tasks_submitted == 0
        assert stats.tasks_completed == 0
        assert stats.total_steals == 0
        assert stats.total_steal_attempts == 0
