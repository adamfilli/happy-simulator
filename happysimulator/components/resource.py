"""Shared resource with contended capacity.

Provides a ``Resource`` class for modeling shared, contended capacity such as
CPU cores, memory bandwidth, disk I/O, or network bandwidth. Multiple entities
can acquire portions of a resource's capacity, blocking via ``SimFuture`` when
capacity is exhausted.

Like other simulation primitives (Mutex, Semaphore), Resource is an Entity
and must be registered with ``Simulation(entities=[...])``.

Example::

    resource = Resource("cpu_cores", capacity=8)

    class Worker(Entity):
        def handle_event(self, event):
            grant = yield resource.acquire(amount=2)  # blocks if unavailable
            yield 0.1  # do work
            grant.release()  # return capacity, wake waiters
            return []

    sim = Simulation(entities=[resource, worker], ...)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResourceStats:
    """Frozen snapshot of resource statistics.

    Attributes:
        name: Resource name.
        capacity: Total capacity of the resource.
        available: Currently available capacity.
        utilization: Fraction of capacity in use (0.0 to 1.0).
        acquisitions: Total successful acquisitions.
        releases: Total releases.
        contentions: Times a waiter had to queue (capacity was insufficient).
        waiters: Current number of waiting acquirers.
        total_wait_time_ns: Total nanoseconds spent waiting across all waiters.
        peak_utilization: Highest utilization observed.
        peak_waiters: Maximum concurrent waiters observed.
    """

    name: str
    capacity: int | float
    available: int | float
    utilization: float
    acquisitions: int
    releases: int
    contentions: int
    waiters: int
    total_wait_time_ns: int
    peak_utilization: float
    peak_waiters: int


class Grant:
    """Handle to acquired resource capacity.

    A Grant represents capacity that has been acquired from a Resource.
    Call ``release()`` to return the capacity. Release is idempotent —
    calling it multiple times has no effect after the first call.

    Attributes:
        amount: The amount of capacity held by this grant.
        released: Whether this grant has been released.
    """

    __slots__ = ("_resource", "_amount", "_released")

    def __init__(self, resource: Resource, amount: int | float) -> None:
        self._resource = resource
        self._amount = amount
        self._released = False

    @property
    def amount(self) -> int | float:
        """The amount of capacity held by this grant."""
        return self._amount

    @property
    def released(self) -> bool:
        """Whether this grant has been released."""
        return self._released

    def release(self) -> None:
        """Return capacity to the resource and wake waiting acquirers.

        Idempotent — calling release() on an already-released grant is a no-op.
        """
        if self._released:
            return
        self._released = True
        self._resource._do_release(self._amount)

    def __repr__(self) -> str:
        state = "released" if self._released else f"{self._amount}"
        return f"Grant({state})"

    def __del__(self) -> None:
        if not self._released:
            logger.warning(
                "Grant for %s of '%s' was garbage-collected without release()",
                self._amount,
                self._resource.name,
            )


@dataclass
class _Waiter:
    """Internal: a queued acquire request."""

    amount: int | float
    future: SimFuture
    enqueue_time_ns: int


class Resource(Entity):
    """Shared capacity pool that multiple entities can acquire from.

    Resource models shared, contended capacity such as CPU cores, memory,
    disk I/O bandwidth, or network bandwidth. Entities acquire capacity
    via ``yield resource.acquire(amount)`` which returns a ``SimFuture``
    that resolves with a ``Grant`` object.

    Like other simulation primitives (Mutex, Semaphore), Resource is an
    Entity and must be registered with ``Simulation(entities=[...])``.

    Waiter satisfaction is strict FIFO: when capacity is released, only
    the head-of-line waiter is checked. If it needs more capacity than
    is available, no subsequent waiters are served (prevents starvation
    of large requests).

    Args:
        name: Identifier for logging and stats.
        capacity: Total capacity of the resource (must be > 0).

    Raises:
        ValueError: If capacity is not positive.
    """

    def __init__(self, name: str, capacity: int | float) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        super().__init__(name)
        self._capacity = capacity
        self._available = capacity
        self._waiters: deque[_Waiter] = deque()

        # Stats counters
        self._acquisitions = 0
        self._releases = 0
        self._contentions = 0
        self._total_wait_time_ns = 0
        self._peak_utilization = 0.0
        self._peak_waiters = 0

    @property
    def capacity(self) -> int | float:
        """Total capacity of the resource."""
        return self._capacity

    @property
    def available(self) -> int | float:
        """Currently available capacity."""
        return self._available

    @property
    def utilization(self) -> float:
        """Fraction of capacity currently in use (0.0 to 1.0)."""
        return (self._capacity - self._available) / self._capacity

    @property
    def waiters(self) -> int:
        """Number of entities waiting to acquire capacity."""
        return len(self._waiters)

    @property
    def stats(self) -> ResourceStats:
        """Frozen snapshot of current resource statistics."""
        return ResourceStats(
            name=self.name,
            capacity=self._capacity,
            available=self._available,
            utilization=self.utilization,
            acquisitions=self._acquisitions,
            releases=self._releases,
            contentions=self._contentions,
            waiters=len(self._waiters),
            total_wait_time_ns=self._total_wait_time_ns,
            peak_utilization=self._peak_utilization,
            peak_waiters=self._peak_waiters,
        )

    def acquire(self, amount: int | float = 1) -> SimFuture:
        """Acquire capacity, returning a SimFuture that resolves with a Grant.

        If sufficient capacity is available, the returned future is
        pre-resolved (yielding it resumes immediately). Otherwise the
        caller is queued and the future resolves when capacity becomes
        available via FIFO ordering.

        Args:
            amount: Amount of capacity to acquire (default 1).

        Returns:
            A SimFuture that resolves with a Grant object.

        Raises:
            ValueError: If amount is not positive or exceeds total capacity.
        """
        if amount <= 0:
            raise ValueError(f"amount must be > 0, got {amount}")
        if amount > self._capacity:
            raise ValueError(
                f"cannot acquire {amount} from resource '{self.name}' "
                f"with capacity {self._capacity}"
            )

        future = SimFuture()

        if self._available >= amount:
            # Immediate grant
            self._available -= amount
            self._acquisitions += 1
            self._update_peak_utilization()
            grant = Grant(self, amount)
            future.resolve(grant)
            logger.debug(
                "[%s] Immediate acquire(%s), available=%s",
                self.name, amount, self._available,
            )
        else:
            # Must wait — enqueue
            self._contentions += 1
            enqueue_ns = self._current_time_ns()
            waiter = _Waiter(amount=amount, future=future, enqueue_time_ns=enqueue_ns)
            self._waiters.append(waiter)

            if len(self._waiters) > self._peak_waiters:
                self._peak_waiters = len(self._waiters)

            logger.debug(
                "[%s] Queued acquire(%s), waiters=%d, available=%s",
                self.name, amount, len(self._waiters), self._available,
            )

        return future

    def try_acquire(self, amount: int | float = 1) -> Grant | None:
        """Try to acquire capacity without blocking.

        Args:
            amount: Amount of capacity to acquire (default 1).

        Returns:
            A Grant if capacity was available, None otherwise.

        Raises:
            ValueError: If amount is not positive or exceeds total capacity.
        """
        if amount <= 0:
            raise ValueError(f"amount must be > 0, got {amount}")
        if amount > self._capacity:
            raise ValueError(
                f"cannot acquire {amount} from resource '{self.name}' "
                f"with capacity {self._capacity}"
            )

        if self._available >= amount:
            self._available -= amount
            self._acquisitions += 1
            self._update_peak_utilization()
            return Grant(self, amount)

        return None

    def _do_release(self, amount: int | float) -> None:
        """Internal: return capacity and wake eligible waiters.

        Called by Grant.release(). Raises if release would exceed capacity.
        """
        future_available = self._available + amount
        if future_available > self._capacity:
            raise ValueError(
                f"releasing {amount} would exceed capacity "
                f"({self._available} + {amount} > {self._capacity})"
            )

        self._available += amount
        self._releases += 1

        logger.debug(
            "[%s] Released %s, available=%s",
            self.name, amount, self._available,
        )

        self._wake_waiters()

    def _wake_waiters(self) -> None:
        """Satisfy queued waiters in FIFO order.

        Strict FIFO: if the head-of-line waiter cannot be satisfied,
        no subsequent waiters are checked (prevents starvation of
        large requests).
        """
        now_ns = self._current_time_ns()

        while self._waiters:
            waiter = self._waiters[0]

            if self._available >= waiter.amount:
                self._waiters.popleft()
                self._available -= waiter.amount
                self._acquisitions += 1
                self._update_peak_utilization()

                # Track wait time
                if waiter.enqueue_time_ns >= 0 and now_ns >= 0:
                    self._total_wait_time_ns += now_ns - waiter.enqueue_time_ns

                grant = Grant(self, waiter.amount)
                waiter.future.resolve(grant)

                logger.debug(
                    "[%s] Woke waiter for %s, available=%s",
                    self.name, waiter.amount, self._available,
                )
            else:
                # Not enough capacity for head-of-line waiter — stop
                break

    def _update_peak_utilization(self) -> None:
        """Update peak utilization tracking."""
        util = self.utilization
        if util > self._peak_utilization:
            self._peak_utilization = util

    def handle_event(self, event: Event) -> None:
        """Resource does not process events directly."""
        pass

    def _current_time_ns(self) -> int:
        """Get current simulation time in nanoseconds, or -1 if unavailable."""
        if self._clock is not None:
            return self._clock.now.nanoseconds
        return -1

    def __repr__(self) -> str:
        return (
            f"Resource('{self.name}', capacity={self._capacity}, "
            f"available={self._available}, waiters={len(self._waiters)})"
        )
