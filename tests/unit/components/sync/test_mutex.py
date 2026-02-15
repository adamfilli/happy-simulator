"""Tests for Mutex."""

import contextlib

import pytest

from happysimulator.components.sync import Mutex


class TestMutexCreation:
    """Tests for Mutex creation."""

    def test_creates_unlocked(self):
        """Mutex starts in unlocked state."""
        mutex = Mutex(name="test")

        assert mutex.is_locked is False
        assert mutex.waiters == 0
        assert mutex.owner is None

    def test_has_name(self):
        """Mutex has a name."""
        mutex = Mutex(name="my_lock")

        assert mutex.name == "my_lock"


class TestMutexTryAcquire:
    """Tests for Mutex.try_acquire()."""

    def test_succeeds_when_unlocked(self):
        """try_acquire succeeds when mutex is unlocked."""
        mutex = Mutex(name="test")

        result = mutex.try_acquire()

        assert result is True
        assert mutex.is_locked is True

    def test_fails_when_locked(self):
        """try_acquire fails when mutex is already locked."""
        mutex = Mutex(name="test")
        mutex.try_acquire()

        result = mutex.try_acquire()

        assert result is False

    def test_sets_owner(self):
        """try_acquire sets owner when provided."""
        mutex = Mutex(name="test")

        mutex.try_acquire(owner="thread-1")

        assert mutex.owner == "thread-1"

    def test_tracks_acquisitions(self):
        """Statistics track successful acquisitions."""
        mutex = Mutex(name="test")

        mutex.try_acquire()

        assert mutex.stats.acquisitions == 1


class TestMutexRelease:
    """Tests for Mutex.release()."""

    def test_unlocks_mutex(self):
        """release() unlocks the mutex."""
        mutex = Mutex(name="test")
        mutex.try_acquire()

        mutex.release()

        assert mutex.is_locked is False
        assert mutex.owner is None

    def test_raises_when_not_locked(self):
        """release() raises when mutex not locked."""
        mutex = Mutex(name="test")

        with pytest.raises(RuntimeError):
            mutex.release()

    def test_tracks_releases(self):
        """Statistics track releases."""
        mutex = Mutex(name="test")
        mutex.try_acquire()

        mutex.release()

        assert mutex.stats.releases == 1


class TestMutexAcquireGenerator:
    """Tests for Mutex.acquire() generator."""

    def test_acquires_immediately_when_unlocked(self):
        """acquire() yields immediately when unlocked."""
        mutex = Mutex(name="test")

        gen = mutex.acquire()
        result = list(gen)

        assert mutex.is_locked is True
        assert len(result) == 1
        assert result[0] == 0.0

    def test_sets_owner_on_immediate_acquire(self):
        """acquire() sets owner on immediate acquisition."""
        mutex = Mutex(name="test")

        list(mutex.acquire(owner="thread-1"))

        assert mutex.owner == "thread-1"


class TestMutexContention:
    """Tests for contended mutex behavior."""

    def test_waiter_queued_when_locked(self):
        """Waiters are queued when mutex is locked."""
        mutex = Mutex(name="test")
        mutex.try_acquire(owner="thread-1")

        # Start waiting
        gen = mutex.acquire(owner="thread-2")
        next(gen)  # First yield - waiter is now queued

        assert mutex.waiters == 1
        assert mutex.stats.contentions == 1

    def test_waiter_woken_on_release(self):
        """Releasing mutex wakes next waiter."""
        mutex = Mutex(name="test")
        mutex.try_acquire(owner="thread-1")

        # Start waiting
        gen = mutex.acquire(owner="thread-2")
        next(gen)

        # Release - should wake waiter
        mutex.release()

        # Continue generator - should now be acquired
        with contextlib.suppress(StopIteration):
            next(gen)

        assert mutex.is_locked is True
        assert mutex.owner == "thread-2"

    def test_fifo_order(self):
        """Waiters are woken in FIFO order."""
        mutex = Mutex(name="test")
        mutex.try_acquire(owner="thread-1")

        # Queue two waiters
        gen2 = mutex.acquire(owner="thread-2")
        next(gen2)

        gen3 = mutex.acquire(owner="thread-3")
        next(gen3)

        assert mutex.waiters == 2

        # Release - thread-2 should get lock
        mutex.release()
        with contextlib.suppress(StopIteration):
            next(gen2)

        assert mutex.owner == "thread-2"

        # Release again - thread-3 should get lock
        mutex.release()
        with contextlib.suppress(StopIteration):
            next(gen3)

        assert mutex.owner == "thread-3"


class TestMutexStatistics:
    """Tests for Mutex statistics."""

    def test_tracks_all_stats(self):
        """Mutex tracks all statistics."""
        mutex = Mutex(name="test")

        # First acquisition
        mutex.try_acquire(owner="thread-1")

        # Contention
        gen = mutex.acquire(owner="thread-2")
        next(gen)

        # Release and let thread-2 acquire
        mutex.release()
        with contextlib.suppress(StopIteration):
            next(gen)

        # Final release
        mutex.release()

        assert mutex.stats.acquisitions == 2
        assert mutex.stats.releases == 2
        assert mutex.stats.contentions == 1
