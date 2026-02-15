"""Work-stealing thread pool for simulating parallel task execution.

Each worker maintains a local deque. New work is assigned to the worker
with the shortest queue. Idle workers steal from the busiest neighbor's
tail, implementing the classic work-stealing property (local=FIFO,
steal=LIFO).

Example:
    from happysimulator.components.scheduling import WorkStealingPool

    pool = WorkStealingPool(
        name="pool", num_workers=4, downstream=sink,
        processing_time_key="processing_time",
    )

    sim = Simulation(entities=[pool, sink, ...])
"""

import logging
from collections import deque
from collections.abc import Generator
from dataclasses import dataclass

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerStats:
    """Frozen snapshot of per-worker statistics."""

    tasks_completed: int = 0
    tasks_stolen: int = 0
    total_processing_time: float = 0.0
    idle_time: float = 0.0


@dataclass(frozen=True)
class WorkStealingPoolStats:
    """Frozen snapshot of aggregate pool statistics."""

    tasks_submitted: int = 0
    tasks_completed: int = 0
    total_steals: int = 0
    total_steal_attempts: int = 0


class _Worker(Entity):
    """Internal per-worker entity with a local task deque."""

    def __init__(self, name: str, pool: "WorkStealingPool", index: int):
        super().__init__(name)
        self._pool = pool
        self._index = index
        self._queue: deque[Event] = deque()
        self._is_processing = False
        self._last_idle_start: Instant | None = None

        # Statistics (private counters → frozen snapshot via @property)
        self._tasks_completed = 0
        self._tasks_stolen = 0
        self._total_processing_time = 0.0
        self._idle_time = 0.0

    @property
    def stats(self) -> WorkerStats:
        """Frozen snapshot of worker statistics."""
        return WorkerStats(
            tasks_completed=self._tasks_completed,
            tasks_stolen=self._tasks_stolen,
            total_processing_time=self._total_processing_time,
            idle_time=self._idle_time,
        )

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    def enqueue(self, event: Event) -> list[Event]:
        """Add a task to the local queue and trigger processing if idle."""
        self._queue.appendleft(event)  # append to front (FIFO popleft)
        if not self._is_processing:
            return [self._make_try_next_event()]
        return []

    def steal_from_tail(self) -> Event | None:
        """Steal a task from the tail of the deque (LIFO steal)."""
        if self._queue:
            return self._queue.pop()
        return None

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]] | list[Event] | None:
        if event.event_type == "_worker_try_next":
            return self._try_next()
        if event.event_type == "_worker_process":
            return self._process_task(event)
        return None

    def _try_next(self) -> list[Event]:
        """Try to process next local task, or steal if empty."""
        if self._queue:
            task = self._queue.popleft()  # FIFO local
            return [self._make_process_event(task)]

        # Try to steal
        self._pool._total_steal_attempts += 1
        stolen = self._pool._steal_for(self._index)
        if stolen is not None:
            self._tasks_stolen += 1
            self._pool._total_steals += 1
            return [self._make_process_event(stolen)]

        # Idle
        self._is_processing = False
        self._last_idle_start = self.now
        return []

    def _process_task(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a task with the configured processing time."""
        self._is_processing = True
        if self._last_idle_start is not None:
            idle_duration = (self.now - self._last_idle_start).to_seconds()
            self._idle_time += idle_duration
            self._last_idle_start = None

        # Extract processing time
        processing_time = self._pool._get_processing_time(event)

        yield processing_time

        self._tasks_completed += 1
        self._total_processing_time += processing_time
        self._pool._tasks_completed += 1

        # Forward to downstream if configured
        result_events: list[Event] = []
        if self._pool._downstream is not None:
            result_events.append(
                Event(
                    time=self.now,
                    event_type="Completed",
                    target=self._pool._downstream,
                    context=event.context,
                )
            )

        # Try next task
        result_events.append(self._make_try_next_event())
        return result_events

    def _make_try_next_event(self) -> Event:
        self._is_processing = True
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_worker_try_next",
            target=self,
            context={},
        )

    def _make_process_event(self, task: Event) -> Event:
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_worker_process",
            target=self,
            context=task.context,
        )


class WorkStealingPool(Entity):
    """Work-stealing pool with N workers, each having a local deque.

    New work goes to the worker with the shortest queue. Idle workers
    steal from the busiest neighbor's tail.

    Attributes:
        name: Pool identifier.
        num_workers: Number of worker entities.
        stats: Frozen statistics snapshot (via property).
    """

    def __init__(
        self,
        name: str,
        num_workers: int = 4,
        downstream: Entity | None = None,
        processing_time_key: str = "processing_time",
        default_processing_time: float = 0.1,
    ):
        """Initialize the work-stealing pool.

        Args:
            name: Pool identifier.
            num_workers: Number of workers.
            downstream: Entity to receive completed events.
            processing_time_key: Metadata key for task processing time.
            default_processing_time: Fallback if key not found.

        Raises:
            ValueError: If num_workers < 1.
        """
        super().__init__(name)

        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self._num_workers = num_workers
        self._downstream = downstream
        self._processing_time_key = processing_time_key
        self._default_processing_time = default_processing_time

        self._workers = [_Worker(f"{name}.worker_{i}", self, i) for i in range(num_workers)]

        # Statistics (private counters → frozen snapshot via @property)
        self._tasks_submitted = 0
        self._tasks_completed = 0
        self._total_steals = 0
        self._total_steal_attempts = 0

        logger.debug(
            "[%s] WorkStealingPool initialized: workers=%d",
            name,
            num_workers,
        )

    @property
    def num_workers(self) -> int:
        """Number of workers in the pool."""
        return self._num_workers

    @property
    def workers(self) -> list[_Worker]:
        """The worker entities (for registration with Simulation)."""
        return list(self._workers)

    @property
    def worker_stats(self) -> list[WorkerStats]:
        """Per-worker statistics (frozen snapshots)."""
        return [w.stats for w in self._workers]

    @property
    def stats(self) -> WorkStealingPoolStats:
        """Frozen snapshot of aggregate pool statistics."""
        return WorkStealingPoolStats(
            tasks_submitted=self._tasks_submitted,
            tasks_completed=self._tasks_completed,
            total_steals=self._total_steals,
            total_steal_attempts=self._total_steal_attempts,
        )

    def set_clock(self, clock: Clock) -> None:
        """Propagate clock to all workers."""
        super().set_clock(clock)
        for worker in self._workers:
            worker.set_clock(clock)

    def handle_event(self, event: Event) -> list[Event] | None:
        """Accept incoming work and assign to shortest queue."""
        self._tasks_submitted += 1

        # Find worker with shortest queue
        target_worker = min(self._workers, key=lambda w: w.queue_depth)

        logger.debug(
            "[%s] Assigning task to %s (depth=%d)",
            self.name,
            target_worker.name,
            target_worker.queue_depth,
        )

        return target_worker.enqueue(event)

    def _steal_for(self, requester_index: int) -> Event | None:
        """Find the busiest neighbor and steal from its tail."""
        busiest = None
        busiest_depth = 0

        for i, worker in enumerate(self._workers):
            if i == requester_index:
                continue
            if worker.queue_depth > busiest_depth:
                busiest = worker
                busiest_depth = worker.queue_depth

        if busiest is not None and busiest_depth > 0:
            return busiest.steal_from_tail()
        return None

    def _get_processing_time(self, event: Event) -> float:
        """Extract processing time from event metadata."""
        metadata = event.context.get("metadata", {})
        return metadata.get(self._processing_time_key, self._default_processing_time)
