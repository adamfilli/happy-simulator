"""Tests for RWLock (Read-Write Lock)."""

import contextlib

import pytest

from happysimulator.components.sync import RWLock


class TestRWLockCreation:
    """Tests for RWLock creation."""

    def test_creates_unlocked(self):
        """RWLock starts with no readers or writers."""
        lock = RWLock(name="test")

        assert lock.active_readers == 0
        assert lock.is_write_locked is False
        assert lock.waiters == 0

    def test_creates_with_max_readers(self):
        """RWLock can limit concurrent readers."""
        lock = RWLock(name="test", max_readers=5)

        assert lock.max_readers == 5

    def test_rejects_invalid_max_readers(self):
        """RWLock rejects max_readers < 1."""
        with pytest.raises(ValueError):
            RWLock(name="test", max_readers=0)

    def test_has_name(self):
        """RWLock has a name."""
        lock = RWLock(name="cache_lock")

        assert lock.name == "cache_lock"


class TestRWLockTryAcquireRead:
    """Tests for RWLock.try_acquire_read()."""

    def test_succeeds_when_unlocked(self):
        """try_acquire_read succeeds when lock is free."""
        lock = RWLock(name="test")

        result = lock.try_acquire_read()

        assert result is True
        assert lock.active_readers == 1

    def test_succeeds_with_other_readers(self):
        """try_acquire_read succeeds when other readers present."""
        lock = RWLock(name="test")
        lock.try_acquire_read()

        result = lock.try_acquire_read()

        assert result is True
        assert lock.active_readers == 2

    def test_fails_when_write_locked(self):
        """try_acquire_read fails when writer holds lock."""
        lock = RWLock(name="test")
        lock.try_acquire_write()

        result = lock.try_acquire_read()

        assert result is False
        assert lock.active_readers == 0

    def test_respects_max_readers(self):
        """try_acquire_read respects max_readers limit."""
        lock = RWLock(name="test", max_readers=2)
        lock.try_acquire_read()
        lock.try_acquire_read()

        result = lock.try_acquire_read()

        assert result is False
        assert lock.active_readers == 2

    def test_tracks_acquisitions(self):
        """Statistics track read acquisitions."""
        lock = RWLock(name="test")

        lock.try_acquire_read()
        lock.try_acquire_read()

        assert lock.stats.read_acquisitions == 2


class TestRWLockTryAcquireWrite:
    """Tests for RWLock.try_acquire_write()."""

    def test_succeeds_when_unlocked(self):
        """try_acquire_write succeeds when lock is free."""
        lock = RWLock(name="test")

        result = lock.try_acquire_write()

        assert result is True
        assert lock.is_write_locked is True

    def test_fails_with_readers(self):
        """try_acquire_write fails when readers present."""
        lock = RWLock(name="test")
        lock.try_acquire_read()

        result = lock.try_acquire_write()

        assert result is False
        assert lock.is_write_locked is False

    def test_fails_when_write_locked(self):
        """try_acquire_write fails when writer holds lock."""
        lock = RWLock(name="test")
        lock.try_acquire_write()

        result = lock.try_acquire_write()

        assert result is False

    def test_tracks_acquisitions(self):
        """Statistics track write acquisitions."""
        lock = RWLock(name="test")

        lock.try_acquire_write()

        assert lock.stats.write_acquisitions == 1


class TestRWLockRelease:
    """Tests for RWLock release methods."""

    def test_release_read(self):
        """release_read decrements reader count."""
        lock = RWLock(name="test")
        lock.try_acquire_read()
        lock.try_acquire_read()

        lock.release_read()

        assert lock.active_readers == 1

    def test_release_read_raises_when_no_readers(self):
        """release_read raises when no readers."""
        lock = RWLock(name="test")

        with pytest.raises(RuntimeError):
            lock.release_read()

    def test_release_write(self):
        """release_write unlocks write lock."""
        lock = RWLock(name="test")
        lock.try_acquire_write()

        lock.release_write()

        assert lock.is_write_locked is False

    def test_release_write_raises_when_not_locked(self):
        """release_write raises when not write-locked."""
        lock = RWLock(name="test")

        with pytest.raises(RuntimeError):
            lock.release_write()

    def test_tracks_releases(self):
        """Statistics track releases."""
        lock = RWLock(name="test")
        lock.try_acquire_read()
        lock.try_acquire_write()  # Will fail, but let's set up properly
        lock.release_read()

        lock.try_acquire_write()
        lock.release_write()

        assert lock.stats.read_releases == 1
        assert lock.stats.write_releases == 1


class TestRWLockAcquireReadGenerator:
    """Tests for RWLock.acquire_read() generator."""

    def test_acquires_immediately_when_free(self):
        """acquire_read yields immediately when lock is free."""
        lock = RWLock(name="test")

        gen = lock.acquire_read()
        result = list(gen)

        assert lock.active_readers == 1
        assert len(result) == 1


class TestRWLockAcquireWriteGenerator:
    """Tests for RWLock.acquire_write() generator."""

    def test_acquires_immediately_when_free(self):
        """acquire_write yields immediately when lock is free."""
        lock = RWLock(name="test")

        gen = lock.acquire_write()
        result = list(gen)

        assert lock.is_write_locked is True
        assert len(result) == 1


class TestRWLockContention:
    """Tests for RWLock contention behavior."""

    def test_reader_waits_for_writer(self):
        """Reader waits when writer holds lock."""
        lock = RWLock(name="test")
        lock.try_acquire_write()

        gen = lock.acquire_read()
        next(gen)

        assert lock.waiters == 1
        assert lock.stats.read_contentions == 1

    def test_writer_waits_for_readers(self):
        """Writer waits when readers hold lock."""
        lock = RWLock(name="test")
        lock.try_acquire_read()

        gen = lock.acquire_write()
        next(gen)

        assert lock.waiters == 1
        assert lock.stats.write_contentions == 1

    def test_reader_woken_after_writer_releases(self):
        """Reader is woken when writer releases."""
        lock = RWLock(name="test")
        lock.try_acquire_write()

        gen = lock.acquire_read()
        next(gen)

        lock.release_write()

        with contextlib.suppress(StopIteration):
            next(gen)

        assert lock.active_readers == 1
        assert lock.is_write_locked is False

    def test_writer_woken_after_readers_release(self):
        """Writer is woken when all readers release."""
        lock = RWLock(name="test")
        lock.try_acquire_read()
        lock.try_acquire_read()

        gen = lock.acquire_write()
        next(gen)

        # Release first reader - writer still waiting
        lock.release_read()
        assert lock.waiters == 1

        # Release second reader - writer should wake
        lock.release_read()

        with contextlib.suppress(StopIteration):
            next(gen)

        assert lock.is_write_locked is True

    def test_multiple_readers_wake_together(self):
        """Multiple waiting readers wake when writer releases."""
        lock = RWLock(name="test")
        lock.try_acquire_write()

        gen1 = lock.acquire_read()
        next(gen1)
        gen2 = lock.acquire_read()
        next(gen2)

        assert lock.waiters == 2

        # Release writer - both readers should wake
        lock.release_write()

        with contextlib.suppress(StopIteration):
            next(gen1)
        with contextlib.suppress(StopIteration):
            next(gen2)

        assert lock.active_readers == 2

    def test_writer_priority_over_new_readers(self):
        """Waiting writer blocks new readers (prevents starvation)."""
        lock = RWLock(name="test")
        lock.try_acquire_read()

        # Writer starts waiting
        gen_writer = lock.acquire_write()
        next(gen_writer)

        # New reader should be blocked (writer waiting)
        result = lock.try_acquire_read()

        assert result is False


class TestRWLockStatistics:
    """Tests for RWLock statistics."""

    def test_tracks_peak_readers(self):
        """Statistics track peak concurrent readers."""
        lock = RWLock(name="test")

        lock.try_acquire_read()
        lock.try_acquire_read()
        lock.try_acquire_read()
        lock.release_read()

        assert lock.stats.peak_readers == 3
