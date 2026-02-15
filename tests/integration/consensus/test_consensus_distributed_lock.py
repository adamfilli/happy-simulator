"""Integration tests for DistributedLock with full Simulation.

Scenario:
- DistributedLock entity manages named locks with fencing tokens and leases
- Tests run within a Simulation to validate lease expiry via timed events
- Fencing tokens are monotonically increasing for safety
- Lease expiry is tested by scheduling events at specific times
"""

from __future__ import annotations

import random

import pytest

from happysimulator.components.consensus.distributed_lock import (
    DistributedLock,
    LockGrant,
)
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class TestDistributedLock:
    """Integration tests for distributed lock with lease-based expiry."""

    def test_basic_acquire_release(self):
        """Acquire a lock, verify the grant, then release it."""
        random.seed(42)
        lock = DistributedLock(
            name="LockManager",
            lease_duration=5.0,
        )

        sim = Simulation(
            duration=10.0,
            entities=[lock],
        )

        results: dict[str, object] = {}

        def do_acquire(event):
            future = lock.acquire("resource-1", "client-A")
            # Lock is free, so future should resolve immediately
            grant = future.value
            results["grant"] = grant
            # Schedule the expiry event if stored
            if hasattr(lock, "_pending_expiry") and lock._pending_expiry is not None:
                return [lock._pending_expiry]
            return None

        def do_release(event):
            grant = results.get("grant")
            if grant and isinstance(grant, LockGrant):
                released = lock.release("resource-1", grant.fencing_token)
                results["released"] = released
            return None

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="Acquire",
            fn=do_acquire,
        ))
        sim.schedule(Event.once(
            time=Instant.from_seconds(1.0),
            event_type="Release",
            fn=do_release,
        ))
        sim.run()

        # Verify the grant
        grant = results.get("grant")
        assert grant is not None, "Lock was not acquired"
        assert isinstance(grant, LockGrant)
        assert grant.lock_name == "resource-1"
        assert grant.holder == "client-A"
        assert grant.fencing_token >= 1

        # Verify release
        assert results.get("released") is True, "Lock was not released"
        assert lock.get_holder("resource-1") is None
        assert lock.active_locks == 0
        assert lock.stats.total_acquires == 1
        assert lock.stats.total_releases == 1

    def test_lock_contention(self):
        """Multiple requesters contend for the same lock; only one wins."""
        random.seed(42)
        lock = DistributedLock(
            name="LockManager",
            lease_duration=10.0,
        )

        sim = Simulation(
            duration=15.0,
            entities=[lock],
        )

        grants: dict[str, LockGrant | None] = {}

        def acquire_a(event):
            future = lock.acquire("shared-lock", "client-A")
            grants["A"] = future.value
            if hasattr(lock, "_pending_expiry") and lock._pending_expiry is not None:
                return [lock._pending_expiry]
            return None

        def acquire_b(event):
            future = lock.acquire("shared-lock", "client-B")
            # B should be queued (future not yet resolved since A holds the lock)
            grants["B_future"] = future
            grants["B_was_queued"] = not future.is_resolved
            return None

        def release_a(event):
            grant_a = grants.get("A")
            if grant_a:
                lock.release("shared-lock", grant_a.fencing_token)
                # After release, B's future should get resolved synchronously
                b_future = grants.get("B_future")
                if b_future and b_future.is_resolved:
                    grants["B_resolved"] = b_future.value
            return None

        # A acquires first
        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="AcquireA",
            fn=acquire_a,
        ))
        # B tries to acquire at the same time (will be queued)
        sim.schedule(Event.once(
            time=Instant.from_seconds(0.2),
            event_type="AcquireB",
            fn=acquire_b,
        ))
        # A releases at t=2.0, B should get the lock
        sim.schedule(Event.once(
            time=Instant.from_seconds(2.0),
            event_type="ReleaseA",
            fn=release_a,
        ))
        sim.run()

        # A should have gotten the lock first
        grant_a = grants.get("A")
        assert grant_a is not None, "Client A did not get the lock"
        assert grant_a.holder == "client-A"

        # B should have been queued (not immediately resolved)
        assert grants.get("B_was_queued") is True, (
            "Client B should have been queued while A held the lock"
        )

        # B should have been granted the lock after A released
        b_resolved = grants.get("B_resolved")
        assert b_resolved is not None, "Client B's lock was not granted after A released"
        assert isinstance(b_resolved, LockGrant)
        assert b_resolved.holder == "client-B"

        # B's fencing token must be higher than A's
        assert b_resolved.fencing_token > grant_a.fencing_token, (
            f"B's token ({b_resolved.fencing_token}) should be > A's ({grant_a.fencing_token})"
        )

    def test_fencing_token_monotonic(self):
        """Fencing tokens strictly increase across multiple acquire/release cycles."""
        random.seed(42)
        lock = DistributedLock(
            name="LockManager",
            lease_duration=10.0,
        )

        sim = Simulation(
            duration=10.0,
            entities=[lock],
        )

        tokens: list[int] = []

        def cycle(event):
            """Acquire, record token, release -- three consecutive cycles."""
            for i in range(3):
                future = lock.acquire("token-test", f"client-{i}")
                grant = future.value
                assert grant is not None, f"Cycle {i}: lock not acquired"
                tokens.append(grant.fencing_token)
                lock.release("token-test", grant.fencing_token)
            return None

        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="TokenCycles",
            fn=cycle,
        ))
        sim.run()

        assert len(tokens) == 3, f"Expected 3 tokens, got {len(tokens)}"

        # Tokens must be strictly increasing
        for i in range(1, len(tokens)):
            assert tokens[i] > tokens[i - 1], (
                f"Token {tokens[i]} at position {i} is not > "
                f"previous token {tokens[i - 1]}"
            )

        # Verify stats
        assert lock.stats.total_acquires == 3
        assert lock.stats.total_releases == 3

    def test_lease_expiry(self):
        """A lock automatically expires after its lease duration."""
        random.seed(42)
        lock = DistributedLock(
            name="LockManager",
            lease_duration=2.0,  # Short lease for testing
        )

        sim = Simulation(
            duration=10.0,
            entities=[lock],
        )

        results: dict[str, object] = {}

        def do_acquire(event):
            future = lock.acquire("expiring-lock", "client-X")
            grant = future.value
            results["grant"] = grant
            # Schedule the expiry event that the lock entity created
            if hasattr(lock, "_pending_expiry") and lock._pending_expiry is not None:
                return [lock._pending_expiry]
            return None

        def check_after_expiry(event):
            results["holder_after_expiry"] = lock.get_holder("expiring-lock")
            results["active_locks_after"] = lock.active_locks
            return None

        # Acquire at t=0.1 (lease expires at t=2.1)
        sim.schedule(Event.once(
            time=Instant.from_seconds(0.1),
            event_type="AcquireLease",
            fn=do_acquire,
        ))
        # Check at t=3.0 (after lease should have expired)
        sim.schedule(Event.once(
            time=Instant.from_seconds(3.0),
            event_type="CheckExpiry",
            fn=check_after_expiry,
        ))
        sim.run()

        # Verify the lock was acquired
        grant = results.get("grant")
        assert grant is not None, "Lock was not acquired"
        assert isinstance(grant, LockGrant)
        assert grant.lease_duration == 2.0

        # After expiry, the lock should be free
        assert results.get("holder_after_expiry") is None, (
            f"Lock should have expired, but holder is {results.get('holder_after_expiry')}"
        )
        assert results.get("active_locks_after") == 0, (
            "Lock should not be active after expiry"
        )
        assert lock.stats.total_expirations == 1, (
            f"Expected 1 expiration, got {lock.stats.total_expirations}"
        )
