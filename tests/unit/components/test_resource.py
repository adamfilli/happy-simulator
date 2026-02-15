"""Unit tests for Resource, Grant, and ResourceStats."""

import pytest

from happysimulator.components.resource import Grant, Resource, ResourceStats


class TestResourceCreation:
    """Tests for Resource construction and validation."""

    def test_create_with_int_capacity(self):
        r = Resource("cpu", capacity=8)
        assert r.name == "cpu"
        assert r.capacity == 8
        assert r.available == 8

    def test_create_with_float_capacity(self):
        r = Resource("bandwidth", capacity=10.5)
        assert r.capacity == 10.5
        assert r.available == 10.5

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            Resource("bad", capacity=0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError, match="capacity must be > 0"):
            Resource("bad", capacity=-1)

    def test_initial_utilization_is_zero(self):
        r = Resource("cpu", capacity=4)
        assert r.utilization == 0.0

    def test_initial_waiters_is_zero(self):
        r = Resource("cpu", capacity=4)
        assert r.waiters == 0

    def test_repr(self):
        r = Resource("cpu", capacity=4)
        assert "cpu" in repr(r)
        assert "capacity=4" in repr(r)


class TestTryAcquire:
    """Tests for non-blocking try_acquire."""

    def test_success_when_available(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire(2)
        assert grant is not None
        assert grant.amount == 2
        assert not grant.released
        assert r.available == 2

    def test_returns_none_when_insufficient(self):
        r = Resource("cpu", capacity=4)
        r.try_acquire(3)
        grant = r.try_acquire(3)
        assert grant is None

    def test_default_amount_is_one(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire()
        assert grant is not None
        assert grant.amount == 1
        assert r.available == 3

    def test_exact_capacity_succeeds(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire(4)
        assert grant is not None
        assert r.available == 0

    def test_zero_amount_raises(self):
        r = Resource("cpu", capacity=4)
        with pytest.raises(ValueError, match="amount must be > 0"):
            r.try_acquire(0)

    def test_negative_amount_raises(self):
        r = Resource("cpu", capacity=4)
        with pytest.raises(ValueError, match="amount must be > 0"):
            r.try_acquire(-1)

    def test_exceeds_capacity_raises(self):
        r = Resource("cpu", capacity=4)
        with pytest.raises(ValueError, match="cannot acquire 5"):
            r.try_acquire(5)


class TestAcquire:
    """Tests for blocking acquire (returns SimFuture)."""

    def test_immediate_when_available(self):
        r = Resource("cpu", capacity=4)
        future = r.acquire(2)
        assert future.is_resolved
        grant = future.value
        assert isinstance(grant, Grant)
        assert grant.amount == 2
        assert r.available == 2

    def test_default_amount_is_one(self):
        r = Resource("cpu", capacity=4)
        future = r.acquire()
        assert future.is_resolved
        assert future.value.amount == 1

    def test_queues_when_insufficient(self):
        r = Resource("cpu", capacity=2)
        f1 = r.acquire(2)
        assert f1.is_resolved
        f2 = r.acquire(1)
        assert not f2.is_resolved
        assert r.waiters == 1

    def test_zero_amount_raises(self):
        r = Resource("cpu", capacity=4)
        with pytest.raises(ValueError, match="amount must be > 0"):
            r.acquire(0)

    def test_exceeds_capacity_raises(self):
        r = Resource("cpu", capacity=4)
        with pytest.raises(ValueError, match="cannot acquire 5"):
            r.acquire(5)


class TestGrantRelease:
    """Tests for Grant.release() behavior."""

    def test_release_returns_capacity(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire(3)
        assert r.available == 1
        grant.release()
        assert r.available == 4
        assert grant.released

    def test_release_is_idempotent(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire(2)
        grant.release()
        grant.release()  # No error
        assert r.available == 4

    def test_grant_repr(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire(2)
        assert "2" in repr(grant)
        grant.release()
        assert "released" in repr(grant)


class TestFIFOWaiterSatisfaction:
    """Tests for FIFO waiter ordering on release."""

    def test_release_wakes_first_waiter(self):
        r = Resource("cpu", capacity=2)
        g1 = r.try_acquire(2)

        f2 = r.acquire(1)
        f3 = r.acquire(1)
        assert not f2.is_resolved
        assert not f3.is_resolved

        g1.release()
        assert f2.is_resolved
        assert f3.is_resolved

    def test_strict_fifo_blocks_if_head_cannot_be_satisfied(self):
        r = Resource("cpu", capacity=3)
        r.try_acquire(3)

        # Waiter 1 wants 2, waiter 2 wants 1
        f_big = r.acquire(2)
        f_small = r.acquire(1)

        # Release 1 — not enough for big waiter, small waiter also blocked
        r._do_release(1)
        assert not f_big.is_resolved
        assert not f_small.is_resolved  # Strict FIFO: blocked behind big

        # Release 1 more — now big waiter can be satisfied
        r._do_release(1)
        assert f_big.is_resolved
        assert not f_small.is_resolved  # Only 0 left after big takes 2

        # Release the big grant — now small can go
        f_big.value.release()
        assert f_small.is_resolved

    def test_multiple_waiters_satisfied_in_order(self):
        r = Resource("cpu", capacity=4)
        g = r.try_acquire(4)

        futures = [r.acquire(1) for _ in range(4)]
        for f in futures:
            assert not f.is_resolved

        g.release()  # Releases 4, should satisfy all 4 in order
        for f in futures:
            assert f.is_resolved


class TestUtilization:
    """Tests for utilization tracking."""

    def test_utilization_increases_on_acquire(self):
        r = Resource("cpu", capacity=4)
        r.try_acquire(2)
        assert r.utilization == pytest.approx(0.5)

    def test_utilization_decreases_on_release(self):
        r = Resource("cpu", capacity=4)
        grant = r.try_acquire(4)
        assert r.utilization == pytest.approx(1.0)
        grant.release()
        assert r.utilization == pytest.approx(0.0)

    def test_full_utilization(self):
        r = Resource("cpu", capacity=4)
        r.try_acquire(4)
        assert r.utilization == pytest.approx(1.0)


class TestStats:
    """Tests for ResourceStats tracking."""

    def test_initial_stats(self):
        r = Resource("cpu", capacity=4)
        s = r.stats
        assert isinstance(s, ResourceStats)
        assert s.name == "cpu"
        assert s.capacity == 4
        assert s.available == 4
        assert s.acquisitions == 0
        assert s.releases == 0
        assert s.contentions == 0
        assert s.peak_utilization == 0.0
        assert s.peak_waiters == 0

    def test_acquisitions_counted(self):
        r = Resource("cpu", capacity=4)
        r.try_acquire(1)
        r.try_acquire(2)
        assert r.stats.acquisitions == 2

    def test_releases_counted(self):
        r = Resource("cpu", capacity=4)
        g1 = r.try_acquire(2)
        g2 = r.try_acquire(1)
        g1.release()
        g2.release()
        assert r.stats.releases == 2

    def test_contentions_counted(self):
        r = Resource("cpu", capacity=1)
        r.try_acquire(1)
        r.acquire(1)  # Will contend
        r.acquire(1)  # Will contend
        assert r.stats.contentions == 2

    def test_peak_utilization_tracked(self):
        r = Resource("cpu", capacity=4)
        g = r.try_acquire(3)
        assert r.stats.peak_utilization == pytest.approx(0.75)
        g.release()
        # Peak should remain at 0.75 even after release
        assert r.stats.peak_utilization == pytest.approx(0.75)

    def test_peak_waiters_tracked(self):
        r = Resource("cpu", capacity=1)
        r.try_acquire(1)
        r.acquire(1)
        r.acquire(1)
        assert r.stats.peak_waiters == 2

    def test_stats_is_frozen(self):
        r = Resource("cpu", capacity=4)
        s = r.stats
        with pytest.raises(AttributeError):
            s.acquisitions = 99


class TestReleaseOverCapacity:
    """Tests for releasing more than capacity."""

    def test_release_over_capacity_raises(self):
        r = Resource("cpu", capacity=2)
        grant = r.try_acquire(1)
        grant.release()
        # Now available=2 (full capacity). A second release would exceed.
        # But Grant.release() is idempotent, so we test _do_release directly.
        with pytest.raises(ValueError, match="would exceed capacity"):
            r._do_release(1)


class TestMultiplePartialAcquires:
    """Tests for multiple acquires exhausting capacity."""

    def test_partial_acquires_exhaust_capacity(self):
        r = Resource("cpu", capacity=4)
        r.try_acquire(1)
        g2 = r.try_acquire(1)
        r.try_acquire(1)
        r.try_acquire(1)
        assert r.available == 0
        assert r.try_acquire(1) is None

        g2.release()
        assert r.available == 1
        g5 = r.try_acquire(1)
        assert g5 is not None
        assert r.available == 0

    def test_float_capacity_partial_acquires(self):
        r = Resource("bw", capacity=10.0)
        r.try_acquire(3.5)
        r.try_acquire(3.5)
        assert r.available == pytest.approx(3.0)
        r.try_acquire(3.0)
        assert r.available == pytest.approx(0.0)
        assert r.try_acquire(0.1) is None
