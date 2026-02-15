"""Thread pool for task processing simulation.

ThreadPool models a pool of worker threads that process tasks from a queue.
Unlike Server which draws service times from a distribution, ThreadPool
gets processing time from each task's context, allowing variable task durations.

The pool automatically dispatches tasks to idle workers and tracks utilization
statistics for analysis.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Generator

from happysimulator.components.queued_resource import QueuedResource
from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy
from happysimulator.components.server.concurrency import FixedConcurrency
from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThreadPoolStats:
    """Statistics tracked by ThreadPool."""

    tasks_completed: int = 0
    tasks_rejected: int = 0
    total_processing_time: float = 0.0


class ThreadPool(QueuedResource):
    """Simulates a pool of worker threads processing tasks.

    Tasks are events with processing time specified in their context metadata.
    Workers pick tasks from the queue when idle and process them for the
    specified duration.

    Processing time can be specified in task context as:
    - task.context["metadata"]["processing_time"] = 0.05  # 50ms
    - Or via a custom extractor function

    Attributes:
        name: Identifier for logging and debugging.
        num_workers: Number of worker threads in the pool.
    """

    def __init__(
        self,
        name: str,
        num_workers: int,
        queue_policy: QueuePolicy | None = None,
        queue_capacity: int | None = None,
        processing_time_extractor: Callable[[Event], float] | None = None,
        default_processing_time: float = 0.01,
    ):
        """Initialize the thread pool.

        Args:
            name: Pool identifier.
            num_workers: Number of worker threads (must be >= 1).
            queue_policy: Queue ordering policy (default FIFO).
            queue_capacity: Maximum queue size (default unlimited).
            processing_time_extractor: Optional function to extract processing
                time from a task event. If None, looks for
                task.context["metadata"]["processing_time"].
            default_processing_time: Default processing time in seconds when
                not specified in task context (default 10ms).

        Raises:
            ValueError: If num_workers < 1.
        """
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        # Create queue policy with capacity if specified
        if queue_policy is None:
            policy = FIFOQueue(
                capacity=queue_capacity if queue_capacity is not None else float("inf")
            )
        else:
            policy = queue_policy

        super().__init__(name, policy=policy)

        self._num_workers = num_workers
        self._worker_pool = FixedConcurrency(num_workers)
        self._processing_time_extractor = processing_time_extractor
        self._default_processing_time = default_processing_time

        # Statistics
        self._tasks_completed = 0
        self._tasks_rejected = 0
        self._total_processing_time = 0.0

        # Time series for analysis
        self._processing_times: list[float] = []

        logger.debug(
            "[%s] ThreadPool initialized: workers=%d, default_time=%.4fs",
            name,
            num_workers,
            default_processing_time,
        )

    @property
    def num_workers(self) -> int:
        """Total number of worker threads."""
        return self._num_workers

    @property
    def active_workers(self) -> int:
        """Number of workers currently processing tasks."""
        return self._worker_pool.active

    @property
    def idle_workers(self) -> int:
        """Number of workers available to process tasks."""
        return self._worker_pool.available

    @property
    def queued_tasks(self) -> int:
        """Number of tasks waiting in the queue."""
        return self.depth

    @property
    def worker_utilization(self) -> float:
        """Current worker utilization as fraction of total workers.

        Returns:
            Value between 0.0 and 1.0 representing active/total workers.
        """
        if self._num_workers == 0:
            return 0.0
        return self._worker_pool.active / self._num_workers

    @property
    def average_processing_time(self) -> float:
        """Average observed processing time in seconds."""
        if not self._processing_times:
            return 0.0
        return sum(self._processing_times) / len(self._processing_times)

    @property
    def stats(self) -> ThreadPoolStats:
        """Frozen snapshot of current statistics."""
        return ThreadPoolStats(
            tasks_completed=self._tasks_completed,
            tasks_rejected=self._tasks_rejected,
            total_processing_time=self._total_processing_time,
        )

    def has_capacity(self) -> bool:
        """Check if the pool has an idle worker available.

        Returns:
            True if at least one worker is idle.
        """
        return self._worker_pool.has_capacity()

    def submit(self, task: Event) -> Event | None:
        """Submit a task to the pool for processing.

        This is a convenience method that creates a properly formatted
        task event and returns it for scheduling.

        Args:
            task: The task event to process. Processing time should be
                in task.context["metadata"]["processing_time"] or will
                use the default.

        Returns:
            The task event targeted at this pool, ready for scheduling.
        """
        task.target = self
        return task

    def _get_processing_time(self, task: Event) -> float:
        """Extract processing time from a task event.

        Args:
            task: The task to get processing time for.

        Returns:
            Processing time in seconds.
        """
        # Use custom extractor if provided
        if self._processing_time_extractor is not None:
            return self._processing_time_extractor(task)

        # Try to get from context metadata
        try:
            metadata = task.context.get("metadata", {})
            if "processing_time" in metadata:
                return float(metadata["processing_time"])
        except (AttributeError, TypeError, ValueError):
            pass

        return self._default_processing_time

    def handle_queued_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Process a task from the queue.

        Acquires a worker, processes for the task's specified time,
        then releases the worker.

        Args:
            event: The task event to process.

        Yields:
            Processing time delay.

        Returns:
            Any completion events (typically None).
        """
        # Acquire a worker slot
        acquired = self._worker_pool.acquire()
        if not acquired:
            # This shouldn't happen if queue driver checks has_capacity,
            # but handle gracefully
            logger.warning(
                "[%s] Failed to acquire worker for task: type=%s",
                self.name,
                event.event_type,
            )
            self._tasks_rejected += 1
            return None

        # Get processing time for this task
        processing_time = self._get_processing_time(event)

        logger.debug(
            "[%s] Worker processing task: type=%s time=%.4fs active=%d/%d",
            self.name,
            event.event_type,
            processing_time,
            self._worker_pool.active,
            self._num_workers,
        )

        # Record for statistics
        self._processing_times.append(processing_time)

        # Simulate processing
        yield processing_time

        # Release the worker
        self._worker_pool.release()

        # Update statistics
        self._tasks_completed += 1
        self._total_processing_time += processing_time

        logger.debug(
            "[%s] Task completed: type=%s processing_time=%.4fs active=%d/%d",
            self.name,
            event.event_type,
            processing_time,
            self._worker_pool.active,
            self._num_workers,
        )

        return None

    def get_processing_time_percentile(self, percentile: float) -> float:
        """Calculate a percentile of observed processing times.

        Args:
            percentile: Percentile value between 0 and 1 (e.g., 0.99 for p99).

        Returns:
            Processing time at the given percentile, or 0 if no data.
        """
        if not self._processing_times:
            return 0.0

        sorted_times = sorted(self._processing_times)
        n = len(sorted_times)

        if percentile <= 0:
            return sorted_times[0]
        if percentile >= 1:
            return sorted_times[-1]

        pos = percentile * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo

        return sorted_times[lo] * (1.0 - frac) + sorted_times[hi] * frac
