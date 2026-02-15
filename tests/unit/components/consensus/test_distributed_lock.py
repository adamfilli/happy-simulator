"""Tests for DistributedLock with fencing tokens and leases."""

from happysimulator.components.consensus.distributed_lock import (
    DistributedLock,
    DistributedLockStats,
    LockGrant,
)
from happysimulator.core.clock import Clock
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


def make_clock(t=0.0):
    return Clock(Instant.from_seconds(t))


def make_lock(name="lock-mgr", lease_duration=10.0, max_waiters=0, clock=None):
    """Create a DistributedLock wired to a clock."""
    clock = clock or make_clock(0.0)
    lock = DistributedLock(name=name, lease_duration=lease_duration, max_waiters=max_waiters)
    lock.set_clock(clock)
    return lock, clock


class TestDistributedLockAcquireFree:
    """Tests for acquiring a free lock."""

    def test_acquire_free_lock(self):
        """Acquiring a free lock succeeds immediately."""
        lock, _ = make_lock()

        future = lock.acquire("my-lock", "client-1")

        assert future.is_resolved is True
        grant = future.value
        assert isinstance(grant, LockGrant)
        assert grant.holder == "client-1"
        assert grant.lock_name == "my-lock"
        assert grant.fencing_token == 1


class TestDistributedLockAcquireHeld:
    """Tests for acquiring a held lock."""

    def test_acquire_held_lock_queues(self):
        """Acquiring a held lock queues the requester."""
        lock, _ = make_lock()

        lock.acquire("my-lock", "client-1")
        future2 = lock.acquire("my-lock", "client-2")

        # Should be queued, not resolved yet
        assert future2.is_resolved is False
        assert lock.total_waiters == 1


class TestDistributedLockTryAcquire:
    """Tests for non-blocking try_acquire."""

    def test_try_acquire_free(self):
        """try_acquire on a free lock returns a LockGrant."""
        lock, _ = make_lock()

        grant = lock.try_acquire("my-lock", "client-1")

        assert grant is not None
        assert isinstance(grant, LockGrant)
        assert grant.holder == "client-1"

    def test_try_acquire_held_returns_none(self):
        """try_acquire on a held lock returns None."""
        lock, _ = make_lock()
        lock.try_acquire("my-lock", "client-1")

        result = lock.try_acquire("my-lock", "client-2")

        assert result is None


class TestDistributedLockRelease:
    """Tests for releasing locks."""

    def test_release_by_token(self):
        """Releasing with the correct fencing token succeeds."""
        lock, _ = make_lock()
        grant = lock.try_acquire("my-lock", "client-1")

        result = lock.release("my-lock", grant.fencing_token)

        assert result is True
        assert lock.get_holder("my-lock") is None

    def test_release_wrong_token_fails(self):
        """Releasing with the wrong fencing token fails."""
        lock, _ = make_lock()
        lock.try_acquire("my-lock", "client-1")

        result = lock.release("my-lock", 9999)

        assert result is False
        assert lock.get_holder("my-lock") == "client-1"


class TestDistributedLockFencingTokens:
    """Tests for fencing token behavior."""

    def test_fencing_tokens_monotonic(self):
        """Fencing tokens are monotonically increasing across acquisitions."""
        lock, _ = make_lock()

        grant1 = lock.try_acquire("lock-a", "client-1")
        lock.release("lock-a", grant1.fencing_token)

        grant2 = lock.try_acquire("lock-a", "client-2")
        lock.release("lock-a", grant2.fencing_token)

        grant3 = lock.try_acquire("lock-a", "client-3")

        assert grant1.fencing_token < grant2.fencing_token < grant3.fencing_token


class TestDistributedLockReentrant:
    """Tests for reentrant lock behavior."""

    def test_reentrant_acquire(self):
        """The same requester can re-acquire a lock it already holds."""
        lock, _ = make_lock()
        lock.acquire("my-lock", "client-1")

        future2 = lock.acquire("my-lock", "client-1")

        assert future2.is_resolved is True
        grant2 = future2.value
        assert grant2.holder == "client-1"


class TestDistributedLockMaxWaiters:
    """Tests for max waiters limit."""

    def test_max_waiters_rejects(self):
        """Exceeding max_waiters resolves the future with None (rejected)."""
        lock, _ = make_lock(max_waiters=1)
        lock.acquire("my-lock", "client-1")

        # First waiter is queued
        f2 = lock.acquire("my-lock", "client-2")
        assert f2.is_resolved is False

        # Second waiter is rejected
        f3 = lock.acquire("my-lock", "client-3")
        assert f3.is_resolved is True
        assert f3.value is None


class TestDistributedLockLeaseExpiry:
    """Tests for lease expiration."""

    def test_lease_expiry_event(self):
        """Handling a lease expiry event releases the lock."""
        lock, clock = make_lock(lease_duration=10.0)
        grant = lock.try_acquire("my-lock", "client-1")

        # Simulate lease expiry event
        expiry_event = Event(
            time=Instant.from_seconds(10.0),
            event_type="LockLeaseExpiry",
            target=lock,
            context={
                "metadata": {
                    "lock_name": "my-lock",
                    "fencing_token": grant.fencing_token,
                }
            },
        )
        clock.update(Instant.from_seconds(10.0))
        lock.handle_event(expiry_event)

        assert lock.get_holder("my-lock") is None
        assert lock.stats.total_expirations == 1


class TestDistributedLockStats:
    """Tests for DistributedLockStats."""

    def test_stats(self):
        """stats reflects current lock state."""
        lock, _ = make_lock()
        lock.try_acquire("my-lock", "client-1")

        stats = lock.stats

        assert isinstance(stats, DistributedLockStats)
        assert stats.total_acquires == 1
        assert stats.total_releases == 0
        assert stats.total_expirations == 0
        assert stats.total_rejections == 0
        assert stats.active_locks == 1
        assert stats.total_waiters == 0


class TestDistributedLockGetHolder:
    """Tests for get_holder."""

    def test_get_holder(self):
        """get_holder returns the current holder or None."""
        lock, _ = make_lock()

        assert lock.get_holder("my-lock") is None

        lock.try_acquire("my-lock", "client-1")
        assert lock.get_holder("my-lock") == "client-1"


class TestDistributedLockGetFencingToken:
    """Tests for get_fencing_token."""

    def test_get_fencing_token(self):
        """get_fencing_token returns the token for a held lock."""
        lock, _ = make_lock()

        assert lock.get_fencing_token("my-lock") is None

        grant = lock.try_acquire("my-lock", "client-1")
        assert lock.get_fencing_token("my-lock") == grant.fencing_token


class TestDistributedLockActiveCount:
    """Tests for active_locks count."""

    def test_active_locks_count(self):
        """active_locks counts the number of currently held locks."""
        lock, _ = make_lock()

        assert lock.active_locks == 0

        lock.try_acquire("lock-a", "client-1")
        assert lock.active_locks == 1

        lock.try_acquire("lock-b", "client-2")
        assert lock.active_locks == 2

        lock.release("lock-a", 1)
        assert lock.active_locks == 1


class TestDistributedLockRepr:
    """Tests for __repr__."""

    def test_repr(self):
        """repr includes name, active count, and waiter count."""
        lock, _ = make_lock(name="dl-1")

        r = repr(lock)

        assert "DistributedLock" in r
        assert "dl-1" in r
        assert "active=" in r
        assert "waiters=" in r
