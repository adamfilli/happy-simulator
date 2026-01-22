"""Composite entity combining queue buffering with worker logic.

QueuedResource is a convenience base class that wires together the
Queue + QueueDriver + Worker pattern into a single entity. Users
subclass QueuedResource and implement handle_queued_event() to define
the work processing logic.

Internal structure:
- Queue: Buffers incoming events according to the configured policy
- QueueDriver: Polls the queue and feeds work to the worker
- Worker Adapter: Internal entity that delegates to handle_queued_event()

This composition is transparent to external code. Events sent to the
QueuedResource are automatically buffered and processed in order.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.clock import Clock
from happysimulator.components.queue import Queue
from happysimulator.components.queue_driver import QueueDriver
from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy

logger = logging.getLogger(__name__)


@dataclass
class _QueuedResourceWorkerAdapter(Entity):
    """Internal entity that bridges the queue driver to the resource's logic."""

    name: str
    _resource: "QueuedResource"

    def handle_event(self, event: Event):
        return self._resource.handle_queued_event(event)

    def has_capacity(self) -> bool:
        return self._resource.has_capacity()


class QueuedResource(Entity, ABC):
    """An entity fronted by a queue.

    External events should target the resource instance directly; the resource
    enqueues them and processes them via its internal queue/driver.

    Subclasses implement :meth:`handle_queued_event` as the resource's work logic.
    """

    def __init__(
        self,
        name: str,
        *,
        policy: QueuePolicy | None = None,
        queue_name: str | None = None,
        driver_name: str | None = None,
        worker_name: str | None = None,
    ):
        super().__init__(name)

        self._queue = Queue(
            name=queue_name or f"{name}.queue",
            egress=None,
            policy=policy or FIFOQueue(),
        )
        self._worker = _QueuedResourceWorkerAdapter(
            name=worker_name or f"{name}.worker",
            _resource=self,
        )
        self._driver = QueueDriver(
            name=driver_name or f"{name}.driver",
            queue=self._queue,
            target=self._worker,
        )
        self._queue.egress = self._driver
        logger.debug(
            "[%s] QueuedResource initialized with queue=%s driver=%s",
            name, self._queue.name, self._driver.name
        )

    @property
    def queue(self) -> Queue:
        """Expose the underlying queue for advanced use-cases."""

        return self._queue

    @property
    def driver(self) -> QueueDriver:
        """Expose the underlying queue driver for advanced use-cases."""

        return self._driver

    @property
    def worker(self) -> Entity:
        """The internal worker entity receiving dequeued work."""

        return self._worker

    @property
    def depth(self) -> int:
        """Current queue depth (useful for :class:`~happysimulator.data.probe.Probe`)."""

        return self._queue.depth

    @property
    def stats_accepted(self) -> int:
        return self._queue.stats_accepted

    @property
    def stats_dropped(self) -> int:
        return self._queue.stats_dropped

    def set_clock(self, clock: Clock):
        """Attach a simulation clock to this resource and its internal components."""

        super().set_clock(clock)

        # Simulation only sets clocks for entities it is given. To make the
        # composition transparent, propagate the clock into the internal queue,
        # driver, and worker adapter.
        self._queue.set_clock(clock)
        self._driver.set_clock(clock)
        self._worker.set_clock(clock)

    def handle_event(self, event: Event):
        """Enqueue incoming work and notify the driver when needed."""

        # The queue stores the event as a payload (even though its `target` is
        # this resource). The driver later retargets it to the worker adapter.
        return self._queue.handle_event(event)

    @abstractmethod
    def handle_queued_event(
        self, event: Event
    ) -> Generator[float | tuple[float, list[Event] | Event | None], None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle a dequeued work item.

        Subclasses implement their server behavior here.
        """

        raise NotImplementedError
