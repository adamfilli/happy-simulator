"""Tests for Semaphore."""

import pytest

from happysimulator.components.sync import Semaphore


class TestSemaphoreCreation:
    """Tests for Semaphore creation."""

    def test_creates_with_initial_count(self):
        """Semaphore starts with specified permits."""
        sem = Semaphore(name="test", initial_count=5)

        assert sem.available == 5
        assert sem.capacity == 5
        assert sem.waiters == 0

    def test_rejects_invalid_count(self):
        """Semaphore rejects count < 1."""
        with pytest.raises(ValueError):
            Semaphore(name="test", initial_count=0)

        with pytest.raises(ValueError):
            Semaphore(name="test", initial_count=-1)

    def test_has_name(self):
        """Semaphore has a name."""
        sem = Semaphore(name="db_pool", initial_count=10)

        assert sem.name == "db_pool"


class TestSemaphoreTryAcquire:
    """Tests for Semaphore.try_acquire()."""

    def test_succeeds_when_available(self):
        """try_acquire succeeds when permits available."""
        sem = Semaphore(name="test", initial_count=3)

        result = sem.try_acquire()

        assert result is True
        assert sem.available == 2

    def test_fails_when_exhausted(self):
        """try_acquire fails when no permits available."""
        sem = Semaphore(name="test", initial_count=1)
        sem.try_acquire()

        result = sem.try_acquire()

        assert result is False
        assert sem.available == 0

    def test_acquires_multiple(self):
        """try_acquire can acquire multiple permits."""
        sem = Semaphore(name="test", initial_count=5)

        result = sem.try_acquire(count=3)

        assert result is True
        assert sem.available == 2

    def test_fails_insufficient_permits(self):
        """try_acquire fails when not enough permits."""
        sem = Semaphore(name="test", initial_count=3)

        result = sem.try_acquire(count=5)

        assert result is False
        assert sem.available == 3  # Unchanged

    def test_rejects_invalid_count(self):
        """try_acquire rejects invalid count."""
        sem = Semaphore(name="test", initial_count=5)

        with pytest.raises(ValueError):
            sem.try_acquire(count=0)

    def test_tracks_acquisitions(self):
        """Statistics track acquired permits."""
        sem = Semaphore(name="test", initial_count=5)

        sem.try_acquire(count=2)

        assert sem.stats.acquisitions == 2


class TestSemaphoreRelease:
    """Tests for Semaphore.release()."""

    def test_releases_permit(self):
        """release() returns permit."""
        sem = Semaphore(name="test", initial_count=3)
        sem.try_acquire()

        sem.release()

        assert sem.available == 3

    def test_releases_multiple(self):
        """release() can return multiple permits."""
        sem = Semaphore(name="test", initial_count=5)
        sem.try_acquire(count=3)

        sem.release(count=3)

        assert sem.available == 5

    def test_rejects_over_capacity(self):
        """release() rejects if would exceed capacity."""
        sem = Semaphore(name="test", initial_count=3)

        with pytest.raises(ValueError):
            sem.release()  # Already at capacity

    def test_rejects_invalid_count(self):
        """release() rejects invalid count."""
        sem = Semaphore(name="test", initial_count=5)
        sem.try_acquire()

        with pytest.raises(ValueError):
            sem.release(count=0)

    def test_tracks_releases(self):
        """Statistics track released permits."""
        sem = Semaphore(name="test", initial_count=5)
        sem.try_acquire(count=2)

        sem.release(count=2)

        assert sem.stats.releases == 2


class TestSemaphoreAcquireGenerator:
    """Tests for Semaphore.acquire() generator."""

    def test_acquires_immediately_when_available(self):
        """acquire() yields immediately when permits available."""
        sem = Semaphore(name="test", initial_count=3)

        gen = sem.acquire()
        result = list(gen)

        assert sem.available == 2
        assert len(result) == 1

    def test_rejects_count_over_capacity(self):
        """acquire() rejects count > capacity."""
        sem = Semaphore(name="test", initial_count=3)

        with pytest.raises(ValueError):
            list(sem.acquire(count=5))


class TestSemaphoreContention:
    """Tests for contended semaphore behavior."""

    def test_waiter_queued_when_exhausted(self):
        """Waiters are queued when no permits available."""
        sem = Semaphore(name="test", initial_count=1)
        sem.try_acquire()

        gen = sem.acquire()
        next(gen)

        assert sem.waiters == 1
        assert sem.stats.contentions == 1

    def test_waiter_woken_on_release(self):
        """Releasing permits wakes waiters."""
        sem = Semaphore(name="test", initial_count=1)
        sem.try_acquire()

        gen = sem.acquire()
        next(gen)

        # Release - should wake waiter
        sem.release()

        # Continue - should be acquired now
        try:
            next(gen)
        except StopIteration:
            pass

        assert sem.available == 0  # Waiter took the permit

    def test_fifo_wakeup(self):
        """Waiters are woken in FIFO order."""
        sem = Semaphore(name="test", initial_count=1)
        sem.try_acquire()

        # Queue two waiters
        gen1 = sem.acquire()
        next(gen1)

        gen2 = sem.acquire()
        next(gen2)

        assert sem.waiters == 2

        # Release - first waiter should get permit
        sem.release()
        try:
            next(gen1)
        except StopIteration:
            pass

        assert sem.waiters == 1
        assert sem.available == 0

    def test_waits_for_enough_permits(self):
        """Waiter needing multiple permits waits for all."""
        sem = Semaphore(name="test", initial_count=2)
        sem.try_acquire(count=2)

        gen = sem.acquire(count=2)
        next(gen)

        # Release one - not enough
        sem.release(count=1)
        assert sem.waiters == 1  # Still waiting

        # Release another - now enough
        sem.release(count=1)
        try:
            next(gen)
        except StopIteration:
            pass

        assert sem.available == 0

    def test_tracks_peak_waiters(self):
        """Statistics track peak waiters."""
        sem = Semaphore(name="test", initial_count=1)
        sem.try_acquire()

        gen1 = sem.acquire()
        next(gen1)
        gen2 = sem.acquire()
        next(gen2)
        gen3 = sem.acquire()
        next(gen3)

        assert sem.stats.peak_waiters == 3


class TestSemaphoreStatistics:
    """Tests for Semaphore statistics."""

    def test_tracks_all_stats(self):
        """Semaphore tracks all statistics."""
        sem = Semaphore(name="test", initial_count=2)

        # Acquire all
        sem.try_acquire()
        sem.try_acquire()

        # Contention
        gen = sem.acquire()
        next(gen)

        # Release and let waiter through
        sem.release()
        try:
            next(gen)
        except StopIteration:
            pass

        sem.release()
        sem.release()

        assert sem.stats.acquisitions == 3
        assert sem.stats.releases == 3
        assert sem.stats.contentions == 1
