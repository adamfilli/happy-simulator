"""CPU scheduler model with time-slicing policies.

Models CPU scheduling with configurable policies for distributing
processing time across competing tasks. Context switch overhead
models the real cost of switching between tasks.

Policies:
- FairShare: Equal time slices across all registered tasks.
- PriorityPreemptive: Higher priority tasks preempt lower priority,
  with configurable time quantum.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.core.event import Event

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scheduling policies
# ---------------------------------------------------------------------------


@dataclass
class CPUTask:
    """A task registered with the CPU scheduler.

    Attributes:
        task_id: Unique identifier for the task.
        priority: Priority level (higher = more important).
        remaining_s: Remaining CPU time needed.
        wait_time_s: Total time spent waiting for CPU.
    """

    task_id: str
    priority: int = 0
    remaining_s: float = 0.0
    wait_time_s: float = 0.0


class SchedulingPolicy(ABC):
    """Strategy for ordering tasks on the CPU."""

    @abstractmethod
    def select_next(self, tasks: list[CPUTask]) -> CPUTask | None:
        """Select the next task to run from the ready queue.

        Args:
            tasks: List of ready tasks (non-empty).

        Returns:
            The selected task, or None if no tasks.
        """
        ...

    @abstractmethod
    def time_quantum_s(self, task: CPUTask) -> float:
        """Return the time slice for the selected task."""
        ...


class FairShare(SchedulingPolicy):
    """Equal time slices across all tasks (round-robin).

    Args:
        quantum_s: Time slice per task in seconds (default 10ms).
    """

    def __init__(self, quantum_s: float = 0.01) -> None:
        if quantum_s <= 0:
            raise ValueError(f"quantum_s must be > 0, got {quantum_s}")
        self._quantum_s = quantum_s

    def select_next(self, tasks: list[CPUTask]) -> CPUTask | None:
        if not tasks:
            return None
        return tasks[0]

    def time_quantum_s(self, task: CPUTask) -> float:
        return self._quantum_s


class PriorityPreemptive(SchedulingPolicy):
    """Highest priority task runs first, preempts lower priority.

    Tasks with higher priority values are selected first. Equal-priority
    tasks are served in FIFO order.

    Args:
        quantum_s: Time slice in seconds (default 10ms).
    """

    def __init__(self, quantum_s: float = 0.01) -> None:
        if quantum_s <= 0:
            raise ValueError(f"quantum_s must be > 0, got {quantum_s}")
        self._quantum_s = quantum_s

    def select_next(self, tasks: list[CPUTask]) -> CPUTask | None:
        if not tasks:
            return None
        return max(tasks, key=lambda t: t.priority)

    def time_quantum_s(self, task: CPUTask) -> float:
        return self._quantum_s


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CPUSchedulerStats:
    """Frozen snapshot of CPU scheduler statistics.

    Attributes:
        tasks_completed: Total tasks that ran to completion.
        context_switches: Total context switches performed.
        total_cpu_time_s: Total CPU time consumed by tasks.
        total_context_switch_overhead_s: Total time spent in context switches.
        total_wait_time_s: Total time tasks spent waiting in the ready queue.
        ready_queue_depth: Current number of tasks in the ready queue.
        peak_queue_depth: Maximum concurrent tasks observed.
    """

    tasks_completed: int = 0
    context_switches: int = 0
    total_cpu_time_s: float = 0.0
    total_context_switch_overhead_s: float = 0.0
    total_wait_time_s: float = 0.0
    ready_queue_depth: int = 0
    peak_queue_depth: int = 0

    @property
    def overhead_fraction(self) -> float:
        """Fraction of total time spent in context switches."""
        total = self.total_cpu_time_s + self.total_context_switch_overhead_s
        return self.total_context_switch_overhead_s / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# CPUScheduler entity
# ---------------------------------------------------------------------------


class CPUScheduler(Entity):
    """CPU scheduler with time-slicing and context switch overhead.

    Provides ``execute()`` to submit work that is scheduled according
    to the configured policy. Multiple concurrent ``execute()`` calls
    compete for CPU time through the scheduler.

    Args:
        name: Entity name.
        policy: Scheduling policy. Defaults to FairShare.
        context_switch_s: Time cost per context switch (default 5us).

    Example::

        cpu = CPUScheduler("cpu", policy=PriorityPreemptive())
        sim = Simulation(entities=[cpu, ...], ...)

        # In another entity's handle_event:
        yield from cpu.execute("task-1", cpu_time_s=0.05, priority=1)
    """

    def __init__(
        self,
        name: str,
        *,
        policy: SchedulingPolicy | None = None,
        context_switch_s: float = 0.000005,
    ) -> None:
        super().__init__(name)
        self._policy = policy or FairShare()
        self._context_switch_s = context_switch_s
        self._ready_queue: deque[CPUTask] = deque()
        self._running: CPUTask | None = None

        # Stats
        self._tasks_completed: int = 0
        self._context_switches: int = 0
        self._total_cpu_time_s: float = 0.0
        self._total_cs_overhead_s: float = 0.0
        self._total_wait_time_s: float = 0.0
        self._peak_queue_depth: int = 0

    @property
    def ready_queue_depth(self) -> int:
        """Number of tasks waiting in the ready queue."""
        return len(self._ready_queue)

    @property
    def stats(self) -> CPUSchedulerStats:
        """Frozen snapshot of CPU scheduler statistics."""
        return CPUSchedulerStats(
            tasks_completed=self._tasks_completed,
            context_switches=self._context_switches,
            total_cpu_time_s=self._total_cpu_time_s,
            total_context_switch_overhead_s=self._total_cs_overhead_s,
            total_wait_time_s=self._total_wait_time_s,
            ready_queue_depth=len(self._ready_queue),
            peak_queue_depth=self._peak_queue_depth,
        )

    def execute(
        self,
        task_id: str,
        cpu_time_s: float,
        priority: int = 0,
    ) -> Generator[float]:
        """Submit a task for CPU execution, yielding until complete.

        The task will be time-sliced according to the scheduling policy
        and may be preempted by higher-priority tasks.

        Args:
            task_id: Unique task identifier.
            cpu_time_s: Total CPU time required.
            priority: Task priority (higher = more important).
        """
        task = CPUTask(task_id=task_id, priority=priority, remaining_s=cpu_time_s)
        self._ready_queue.append(task)

        if len(self._ready_queue) > self._peak_queue_depth:
            self._peak_queue_depth = len(self._ready_queue)

        # Wait until this task completes
        while task.remaining_s > 0:
            # Context switch overhead
            if self._running is not None and self._running is not task:
                yield self._context_switch_s
                self._context_switches += 1
                self._total_cs_overhead_s += self._context_switch_s

            # Select next task to run
            ready = list(self._ready_queue)
            selected = self._policy.select_next(ready)

            if selected is None or selected is not task:
                # Not our turn — yield a small wait and try again
                yield self._policy.time_quantum_s(task) if selected else 0.001
                task.wait_time_s += self._policy.time_quantum_s(task) if selected else 0.001
                continue

            self._running = task

            # Run for a time quantum
            quantum = self._policy.time_quantum_s(task)
            run_time = min(quantum, task.remaining_s)
            yield run_time
            task.remaining_s -= run_time
            self._total_cpu_time_s += run_time

        # Task complete
        if task in self._ready_queue:
            self._ready_queue.remove(task)
        self._tasks_completed += 1
        self._total_wait_time_s += task.wait_time_s
        if self._running is task:
            self._running = None

    def handle_event(self, event: Event) -> None:
        """CPUScheduler does not process events directly."""

    def __repr__(self) -> str:
        return (
            f"CPUScheduler('{self.name}', ready={len(self._ready_queue)}, "
            f"completed={self._tasks_completed})"
        )
