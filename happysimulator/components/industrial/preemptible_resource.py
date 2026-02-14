"""Resource where higher-priority requests can preempt lower-priority holders.

PreemptibleResource extends the resource concept with priority-based
preemption. Higher-priority requests can evict lower-priority grant
holders, notifying them via an ``on_preempt`` callback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable
import heapq

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreemptibleResourceStats:
    """Snapshot of preemptible resource statistics."""

    capacity: int
    available: int
    acquisitions: int
    releases: int
    preemptions: int
    contentions: int


class PreemptibleGrant:
    """Handle to acquired preemptible resource capacity.

    Like ``Grant`` but with preemption support. When preempted, the
    ``on_preempt`` callback fires and ``preempted`` becomes True.

    Attributes:
        amount: The amount of capacity held.
        priority: The priority of this grant (lower = higher priority).
        released: Whether this grant has been released.
        preempted: Whether this grant was preempted.
    """

    __slots__ = (
        "_resource", "_amount", "_priority", "_released",
        "_preempted", "_on_preempt",
    )

    def __init__(
        self,
        resource: PreemptibleResource,
        amount: int,
        priority: float,
        on_preempt: Callable[[], None] | None = None,
    ):
        self._resource = resource
        self._amount = amount
        self._priority = priority
        self._released = False
        self._preempted = False
        self._on_preempt = on_preempt

    @property
    def amount(self) -> int:
        return self._amount

    @property
    def priority(self) -> float:
        return self._priority

    @property
    def released(self) -> bool:
        return self._released

    @property
    def preempted(self) -> bool:
        return self._preempted

    def release(self) -> None:
        """Return capacity to the resource. Idempotent."""
        if self._released:
            return
        self._released = True
        self._resource._do_release(self._amount)

    def _do_preempt(self) -> None:
        """Called by the resource when this grant is preempted."""
        self._preempted = True
        self._released = True
        if self._on_preempt is not None:
            self._on_preempt()

    def __repr__(self) -> str:
        if self._preempted:
            return f"PreemptibleGrant(preempted, priority={self._priority})"
        if self._released:
            return f"PreemptibleGrant(released)"
        return f"PreemptibleGrant({self._amount}, priority={self._priority})"


@dataclass(order=True)
class _PriorityWaiter:
    """Internal: a queued acquire request ordered by priority."""

    priority: float
    insert_order: int
    amount: int = field(compare=False)
    future: SimFuture = field(compare=False)
    on_preempt: Callable[[], None] | None = field(compare=False, default=None)


class PreemptibleResource(Entity):
    """Resource where higher-priority requests can preempt lower-priority holders.

    Capacity is allocated by priority (lower value = higher priority).
    When capacity is insufficient, a higher-priority request can evict
    the lowest-priority holder if ``preempt=True``.

    Args:
        name: Identifier for logging.
        capacity: Total capacity of the resource (integer units).
    """

    def __init__(self, name: str, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")

        super().__init__(name)
        self._capacity = capacity
        self._available = capacity

        # Active grants tracked for preemption (highest priority value = evict first)
        self._active_grants: list[PreemptibleGrant] = []

        # Priority queue of waiters
        self._waiters: list[_PriorityWaiter] = []
        self._insert_counter = 0

        self._acquisitions = 0
        self._releases = 0
        self._preemptions = 0
        self._contentions = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available(self) -> int:
        return self._available

    @property
    def stats(self) -> PreemptibleResourceStats:
        return PreemptibleResourceStats(
            capacity=self._capacity,
            available=self._available,
            acquisitions=self._acquisitions,
            releases=self._releases,
            preemptions=self._preemptions,
            contentions=self._contentions,
        )

    def acquire(
        self,
        amount: int = 1,
        priority: float = 0.0,
        preempt: bool = True,
        on_preempt: Callable[[], None] | None = None,
    ) -> SimFuture:
        """Acquire capacity, returning a SimFuture resolving with a PreemptibleGrant.

        Args:
            amount: Amount of capacity to acquire.
            priority: Priority level (lower = higher priority).
            preempt: If True, try to preempt lower-priority holders.
            on_preempt: Callback fired if this grant is later preempted.

        Returns:
            SimFuture resolving with a PreemptibleGrant.
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
            self._grant_immediate(future, amount, priority, on_preempt)
            return future

        # Try preemption
        if preempt:
            freed = self._try_preempt(amount, priority)
            if self._available >= amount:
                self._grant_immediate(future, amount, priority, on_preempt)
                return future

        # Must wait
        self._contentions += 1
        waiter = _PriorityWaiter(
            priority=priority,
            insert_order=self._insert_counter,
            amount=amount,
            future=future,
            on_preempt=on_preempt,
        )
        self._insert_counter += 1
        heapq.heappush(self._waiters, waiter)

        logger.debug(
            "[%s] Queued acquire(%d, priority=%.1f), waiters=%d",
            self.name, amount, priority, len(self._waiters),
        )

        return future

    def _grant_immediate(
        self,
        future: SimFuture,
        amount: int,
        priority: float,
        on_preempt: Callable[[], None] | None,
    ) -> None:
        self._available -= amount
        self._acquisitions += 1
        grant = PreemptibleGrant(self, amount, priority, on_preempt)
        self._active_grants.append(grant)
        future.resolve(grant)

    def _try_preempt(self, needed: int, requester_priority: float) -> int:
        """Try to preempt lower-priority grants to free capacity."""
        # Sort active grants by priority descending (lowest priority = evict first)
        candidates = sorted(
            [g for g in self._active_grants if not g.released and g.priority > requester_priority],
            key=lambda g: -g.priority,
        )

        freed = 0
        for grant in candidates:
            if self._available >= needed:
                break
            grant._do_preempt()
            self._active_grants.remove(grant)
            self._available += grant.amount
            self._preemptions += 1
            freed += grant.amount
            logger.debug(
                "[%s] Preempted grant (priority=%.1f, amount=%d)",
                self.name, grant.priority, grant.amount,
            )

        return freed

    def _do_release(self, amount: int) -> None:
        """Internal: return capacity and wake eligible waiters."""
        self._available += amount
        self._releases += 1

        # Remove released grants from active list
        self._active_grants = [g for g in self._active_grants if not g.released]

        self._wake_waiters()

    def _wake_waiters(self) -> None:
        """Satisfy queued waiters in priority order."""
        while self._waiters:
            waiter = self._waiters[0]
            if self._available >= waiter.amount:
                heapq.heappop(self._waiters)
                self._grant_immediate(
                    waiter.future, waiter.amount, waiter.priority, waiter.on_preempt,
                )
            else:
                break

    def handle_event(self, event: Event) -> None:
        """PreemptibleResource does not process events directly."""
        pass
