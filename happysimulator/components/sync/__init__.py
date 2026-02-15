"""Synchronization primitives for concurrent resource access.

This module provides simulation-aware synchronization primitives that model
concurrency control mechanisms like mutexes, semaphores, and read-write locks.

Example:
    from happysimulator.components.sync import Mutex, Semaphore, RWLock

    # Protect a critical section
    mutex = Mutex(name="db_lock")

    def handle_event(self, event):
        yield from mutex.acquire()
        try:
            yield 0.01  # Critical section
        finally:
            return mutex.release()
"""

from happysimulator.components.sync.barrier import Barrier, BarrierStats
from happysimulator.components.sync.condition import Condition, ConditionStats
from happysimulator.components.sync.mutex import Mutex, MutexStats
from happysimulator.components.sync.rwlock import RWLock, RWLockStats
from happysimulator.components.sync.semaphore import Semaphore, SemaphoreStats

__all__ = [
    "Barrier",
    "BarrierStats",
    "Condition",
    "ConditionStats",
    "Mutex",
    "MutexStats",
    "RWLock",
    "RWLockStats",
    "Semaphore",
    "SemaphoreStats",
]
