"""Tests for Condition Variable."""

import pytest

from happysimulator.components.sync import Condition, Mutex


class TestConditionCreation:
    """Tests for Condition creation."""

    def test_creates_with_lock(self):
        """Condition is created with associated lock."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)

        assert cond.lock is mutex
        assert cond.waiters == 0

    def test_has_name(self):
        """Condition has a name."""
        mutex = Mutex(name="lock")
        cond = Condition(name="not_empty", lock=mutex)

        assert cond.name == "not_empty"


class TestConditionWait:
    """Tests for Condition.wait()."""

    def test_raises_without_lock(self):
        """wait() raises if mutex not held."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)

        with pytest.raises(RuntimeError):
            list(cond.wait())

    def test_adds_waiter(self):
        """wait() adds to waiter count."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        gen = cond.wait()
        next(gen)

        assert cond.waiters == 1

    def test_releases_lock_while_waiting(self):
        """wait() releases mutex while waiting."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        gen = cond.wait()
        next(gen)

        # Lock should be released
        assert mutex.is_locked is False

    def test_tracks_waits(self):
        """Statistics track wait calls."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        gen = cond.wait()
        next(gen)

        assert cond.stats.waits == 1


class TestConditionNotify:
    """Tests for Condition.notify()."""

    def test_wakes_one_waiter(self):
        """notify() wakes one waiting thread."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        # Start waiting
        gen = cond.wait()
        next(gen)

        # Notify
        cond.notify()

        assert cond.waiters == 0
        assert cond.stats.wakeups == 1

    def test_wakes_n_waiters(self):
        """notify(n) wakes n waiting threads."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        # Start three waiters (we need to manually handle the lock release)
        # This is a simplified test - in real usage each would be in separate entities

        gen1 = cond.wait()
        next(gen1)
        # Lock is now released, acquire again for next waiter
        mutex.try_acquire()

        gen2 = cond.wait()
        next(gen2)
        mutex.try_acquire()

        gen3 = cond.wait()
        next(gen3)

        assert cond.waiters == 3

        # Notify 2
        cond.notify(n=2)

        assert cond.waiters == 1
        assert cond.stats.wakeups == 2

    def test_notify_empty_does_nothing(self):
        """notify() on empty queue does nothing."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)

        cond.notify()

        assert cond.stats.notifies == 1
        assert cond.stats.wakeups == 0

    def test_tracks_notifies(self):
        """Statistics track notify calls."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)

        cond.notify()
        cond.notify()

        assert cond.stats.notifies == 2


class TestConditionNotifyAll:
    """Tests for Condition.notify_all()."""

    def test_wakes_all_waiters(self):
        """notify_all() wakes all waiting threads."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        # Start three waiters
        gen1 = cond.wait()
        next(gen1)
        mutex.try_acquire()

        gen2 = cond.wait()
        next(gen2)
        mutex.try_acquire()

        gen3 = cond.wait()
        next(gen3)

        assert cond.waiters == 3

        # Notify all
        cond.notify_all()

        assert cond.waiters == 0
        assert cond.stats.wakeups == 3

    def test_tracks_notify_alls(self):
        """Statistics track notify_all calls."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)

        cond.notify_all()

        assert cond.stats.notify_alls == 1


class TestConditionWakeupBehavior:
    """Tests for condition wakeup and lock reacquisition."""

    def test_waiter_reacquires_lock_after_notify(self):
        """Woken waiter reacquires the mutex."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        gen = cond.wait()
        next(gen)

        # Lock should be released
        assert mutex.is_locked is False

        # Notify the waiter
        cond.notify()

        # Drive the generator forward - it will try to reacquire lock
        # The lock is free, so it should succeed
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Lock should be held again (by the woken waiter)
        assert mutex.is_locked is True

    def test_fifo_wakeup_order(self):
        """Waiters are woken in FIFO order."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)

        mutex.try_acquire()
        gen1 = cond.wait()
        next(gen1)
        mutex.try_acquire()

        gen2 = cond.wait()
        next(gen2)

        # Notify first - should wake gen1
        cond.notify()
        assert cond.waiters == 1

        # Notify second - should wake gen2
        cond.notify()
        assert cond.waiters == 0


class TestConditionStatistics:
    """Tests for Condition statistics."""

    def test_tracks_all_stats(self):
        """Condition tracks all statistics."""
        mutex = Mutex(name="lock")
        cond = Condition(name="cond", lock=mutex)
        mutex.try_acquire()

        # Wait
        gen = cond.wait()
        next(gen)

        # Notify
        cond.notify()

        # Notify all (empty)
        cond.notify_all()

        assert cond.stats.waits == 1
        assert cond.stats.notifies == 1
        assert cond.stats.notify_alls == 1
        assert cond.stats.wakeups == 1
