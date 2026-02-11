"""Distributed lock with fencing tokens and lease-based expiry.

Provides a distributed locking mechanism with monotonically increasing
fencing tokens, lease-based automatic expiry, and queued waiters.
Can operate standalone or backed by a consensus protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LockGrant:
    """Represents a granted lock with fencing token.

    Attributes:
        lock_name: Name of the lock.
        fencing_token: Monotonically increasing token for fencing.
        holder: Name of the lock holder.
        granted_at: Simulation time when lock was granted (seconds).
        lease_duration: Duration of the lease in seconds.
    """
    lock_name: str
    fencing_token: int
    holder: str
    granted_at: float
    lease_duration: float

    @property
    def expires_at(self) -> float:
        """When this lease expires (seconds)."""
        return self.granted_at + self.lease_duration


@dataclass(frozen=True)
class DistributedLockStats:
    """Statistics snapshot from a DistributedLock.

    Attributes:
        total_acquires: Total successful acquisitions.
        total_releases: Total explicit releases.
        total_expirations: Total lease expirations.
        total_rejections: Total rejected acquisitions (queue full).
        active_locks: Number of currently held locks.
        total_waiters: Total queued waiters across all locks.
    """
    total_acquires: int
    total_releases: int
    total_expirations: int
    total_rejections: int
    active_locks: int
    total_waiters: int


@dataclass
class _LockState:
    """Internal state for a single named lock."""
    holder: str | None = None
    fencing_token: int = 0
    granted_at: float = 0.0
    lease_duration: float = 0.0
    lease_event: Event | None = None
    waiters: list[tuple[str, SimFuture]] = field(default_factory=list)


class DistributedLock(Entity):
    """Distributed lock manager with fencing tokens and leases.

    Args:
        name: Entity identifier.
        lease_duration: Default lease duration in seconds.
        max_waiters: Maximum queued waiters per lock (0 = unlimited).
    """

    def __init__(
        self,
        name: str,
        lease_duration: float = 10.0,
        max_waiters: int = 0,
    ) -> None:
        super().__init__(name)
        self._lease_duration = lease_duration
        self._max_waiters = max_waiters
        self._locks: dict[str, _LockState] = {}
        self._next_token: int = 1

        # Stats
        self._total_acquires: int = 0
        self._total_releases: int = 0
        self._total_expirations: int = 0
        self._total_rejections: int = 0

    def acquire(self, lock_name: str, requester: str) -> SimFuture:
        """Acquire a lock, blocking if held by another.

        Args:
            lock_name: Name of the lock.
            requester: Name of the requesting entity.

        Returns:
            SimFuture resolving with LockGrant when acquired.
        """
        future = SimFuture()
        state = self._get_or_create(lock_name)

        if state.holder is None:
            # Lock is free — grant immediately
            grant = self._grant_lock(state, lock_name, requester)
            future.resolve(grant)
        elif state.holder == requester:
            # Reentrant — return existing grant
            grant = LockGrant(
                lock_name=lock_name,
                fencing_token=state.fencing_token,
                holder=requester,
                granted_at=state.granted_at,
                lease_duration=state.lease_duration,
            )
            future.resolve(grant)
        else:
            # Queued
            if self._max_waiters > 0 and len(state.waiters) >= self._max_waiters:
                self._total_rejections += 1
                future.resolve(None)  # rejected
            else:
                state.waiters.append((requester, future))

        return future

    def try_acquire(self, lock_name: str, requester: str) -> LockGrant | None:
        """Non-blocking lock acquisition.

        Args:
            lock_name: Name of the lock.
            requester: Name of the requesting entity.

        Returns:
            LockGrant if acquired, None if lock is held.
        """
        state = self._get_or_create(lock_name)

        if state.holder is None:
            return self._grant_lock(state, lock_name, requester)
        if state.holder == requester:
            return LockGrant(
                lock_name=lock_name,
                fencing_token=state.fencing_token,
                holder=requester,
                granted_at=state.granted_at,
                lease_duration=state.lease_duration,
            )
        return None

    def release(self, lock_name: str, fencing_token: int) -> bool:
        """Release a lock by fencing token.

        Args:
            lock_name: Name of the lock.
            fencing_token: The fencing token from the LockGrant.

        Returns:
            True if the lock was released, False if token doesn't match.
        """
        state = self._locks.get(lock_name)
        if state is None or state.holder is None:
            return False

        if state.fencing_token != fencing_token:
            return False

        self._release_lock(state, lock_name)
        return True

    def handle_event(self, event: Event):
        if event.event_type == "LockLeaseExpiry":
            return self._handle_lease_expiry(event)
        if event.event_type == "LockAcquireRequest":
            return self._handle_acquire_request(event)
        if event.event_type == "LockReleaseRequest":
            return self._handle_release_request(event)
        return None

    def _handle_lease_expiry(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        lock_name = metadata.get("lock_name")
        expected_token = metadata.get("fencing_token")

        if lock_name is None:
            return None

        state = self._locks.get(lock_name)
        if state is None or state.holder is None:
            return None

        # Only expire if the token matches (hasn't been re-acquired)
        if state.fencing_token != expected_token:
            return None

        logger.debug(
            "[%s] Lock '%s' expired (holder=%s, token=%d)",
            self.name, lock_name, state.holder, state.fencing_token,
        )
        self._total_expirations += 1
        state.holder = None
        state.lease_event = None

        # Wake next waiter
        self._wake_next_waiter(state, lock_name)
        return None

    def _handle_acquire_request(self, event: Event) -> list[Event] | None:
        metadata = event.context.get("metadata", {})
        lock_name = metadata.get("lock_name")
        requester = metadata.get("requester")
        reply_future = event.context.get("reply_future")

        if lock_name is None or requester is None:
            return None

        future = self.acquire(lock_name, requester)
        if reply_future and isinstance(reply_future, SimFuture):
            # Chain the futures
            future._add_settle_callback(lambda f: reply_future.resolve(f.value))
        return None

    def _handle_release_request(self, event: Event) -> None:
        metadata = event.context.get("metadata", {})
        lock_name = metadata.get("lock_name")
        fencing_token = metadata.get("fencing_token")

        if lock_name is not None and fencing_token is not None:
            self.release(lock_name, fencing_token)
        return None

    def _get_or_create(self, lock_name: str) -> _LockState:
        if lock_name not in self._locks:
            self._locks[lock_name] = _LockState()
        return self._locks[lock_name]

    def _grant_lock(self, state: _LockState, lock_name: str, requester: str) -> LockGrant:
        token = self._next_token
        self._next_token += 1
        now_s = self.now.to_seconds() if self._clock else 0.0

        state.holder = requester
        state.fencing_token = token
        state.granted_at = now_s
        state.lease_duration = self._lease_duration
        self._total_acquires += 1

        # Schedule lease expiry
        if self._clock:
            expiry_event = Event(
                time=self.now + self._lease_duration,
                event_type="LockLeaseExpiry",
                target=self,
                daemon=True,
                context={"metadata": {
                    "lock_name": lock_name,
                    "fencing_token": token,
                }},
            )
            if state.lease_event:
                state.lease_event.cancel()
            state.lease_event = expiry_event
            # We can't directly push to heap — return it for scheduling
            # The caller should schedule this event
            # For direct API usage (non-event-driven), we store it
            self._pending_expiry = expiry_event

        grant = LockGrant(
            lock_name=lock_name,
            fencing_token=token,
            holder=requester,
            granted_at=now_s,
            lease_duration=self._lease_duration,
        )

        logger.debug(
            "[%s] Lock '%s' granted to %s (token=%d)",
            self.name, lock_name, requester, token,
        )
        return grant

    def _release_lock(self, state: _LockState, lock_name: str) -> None:
        logger.debug(
            "[%s] Lock '%s' released by %s (token=%d)",
            self.name, lock_name, state.holder, state.fencing_token,
        )
        self._total_releases += 1
        state.holder = None
        if state.lease_event:
            state.lease_event.cancel()
            state.lease_event = None

        self._wake_next_waiter(state, lock_name)

    def _wake_next_waiter(self, state: _LockState, lock_name: str) -> None:
        while state.waiters:
            requester, future = state.waiters.pop(0)
            if not future.is_resolved:
                grant = self._grant_lock(state, lock_name, requester)
                future.resolve(grant)
                break

    @property
    def active_locks(self) -> int:
        return sum(1 for s in self._locks.values() if s.holder is not None)

    @property
    def total_waiters(self) -> int:
        return sum(len(s.waiters) for s in self._locks.values())

    def get_holder(self, lock_name: str) -> str | None:
        state = self._locks.get(lock_name)
        return state.holder if state else None

    def get_fencing_token(self, lock_name: str) -> int | None:
        state = self._locks.get(lock_name)
        return state.fencing_token if state and state.holder else None

    @property
    def stats(self) -> DistributedLockStats:
        return DistributedLockStats(
            total_acquires=self._total_acquires,
            total_releases=self._total_releases,
            total_expirations=self._total_expirations,
            total_rejections=self._total_rejections,
            active_locks=self.active_locks,
            total_waiters=self.total_waiters,
        )

    def __repr__(self) -> str:
        return (
            f"DistributedLock({self.name}, "
            f"active={self.active_locks}, "
            f"waiters={self.total_waiters})"
        )
