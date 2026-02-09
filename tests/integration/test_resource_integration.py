"""Integration tests for Resource with full simulation execution.

Tests cover: basic contention, grant release waking waiters, multiple
resources, utilization tracking, concurrent workers, and try_acquire.
"""

from typing import Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.components.resource import Resource


# ---------------------------------------------------------------------------
# Helper entities
# ---------------------------------------------------------------------------

class Worker(Entity):
    """Worker that acquires a resource, does work, then releases."""

    def __init__(self, name: str, resource: Resource, amount: int | float = 1,
                 work_time: float = 0.1):
        super().__init__(name)
        self.resource = resource
        self.amount = amount
        self.work_time = work_time
        self.completed = 0
        self.acquired_at: list[float] = []
        self.released_at: list[float] = []

    def handle_event(self, event: Event) -> Generator:
        grant = yield self.resource.acquire(self.amount)
        self.acquired_at.append(self.now.to_seconds())
        yield self.work_time
        grant.release()
        self.released_at.append(self.now.to_seconds())
        self.completed += 1
        return []


class MultiResourceWorker(Entity):
    """Worker that acquires multiple resources before doing work."""

    def __init__(self, name: str, cpu: Resource, memory: Resource,
                 cpu_amount: int = 1, mem_amount: int = 1):
        super().__init__(name)
        self.cpu = cpu
        self.memory = memory
        self.cpu_amount = cpu_amount
        self.mem_amount = mem_amount
        self.completed = 0

    def handle_event(self, event: Event) -> Generator:
        cpu_grant = yield self.cpu.acquire(self.cpu_amount)
        mem_grant = yield self.memory.acquire(self.mem_amount)
        yield 0.1  # Work
        cpu_grant.release()
        mem_grant.release()
        self.completed += 1
        return []


class TryAcquireWorker(Entity):
    """Worker that uses try_acquire and tracks success/failure."""

    def __init__(self, name: str, resource: Resource, amount: int = 1):
        super().__init__(name)
        self.resource = resource
        self.amount = amount
        self.successes = 0
        self.failures = 0

    def handle_event(self, event: Event) -> Generator:
        grant = self.resource.try_acquire(self.amount)
        if grant is not None:
            self.successes += 1
            yield 0.1
            grant.release()
        else:
            self.failures += 1
        return []


def _make_sim(*entities, end_time_s=60.0):
    """Create a Simulation with clock injection for the given entities."""
    return Simulation(
        end_time=Instant.from_seconds(end_time_s),
        entities=list(entities),
    )


def _trigger(sim, target, event_type="Go", time_s=0.0, **extra_context):
    """Schedule a trigger event into the simulation."""
    sim.schedule(Event(
        time=Instant.from_seconds(time_s),
        event_type=event_type,
        target=target,
        context=extra_context,
    ))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicContention:
    """Workers sharing a CPU resource with basic contention."""

    def test_two_workers_share_one_core(self):
        """Two workers need 1 core each from a 1-core resource.
        Second worker must wait until first finishes."""
        resource = Resource("cpu", capacity=1)
        w1 = Worker("w1", resource, amount=1, work_time=0.1)
        w2 = Worker("w2", resource, amount=1, work_time=0.1)
        sim = _make_sim(w1, w2)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        sim.run()

        assert w1.completed == 1
        assert w2.completed == 1
        # w1 acquires at t=0, releases at t=0.1
        # w2 acquires at t=0.1, releases at t=0.2
        assert w1.acquired_at[0] < w2.acquired_at[0]
        assert w2.acquired_at[0] >= w1.released_at[0]

    def test_three_workers_four_cores(self):
        """Three workers need 2 cores each from a 4-core resource.
        First two can run concurrently, third must wait."""
        resource = Resource("cpu", capacity=4)
        w1 = Worker("w1", resource, amount=2, work_time=0.1)
        w2 = Worker("w2", resource, amount=2, work_time=0.1)
        w3 = Worker("w3", resource, amount=2, work_time=0.1)
        sim = _make_sim(w1, w2, w3)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        _trigger(sim, w3, time_s=0.0)
        sim.run()

        assert w1.completed == 1
        assert w2.completed == 1
        assert w3.completed == 1
        # w1 and w2 should acquire at the same time
        assert w1.acquired_at[0] == w2.acquired_at[0]
        # w3 should wait until one of them releases
        assert w3.acquired_at[0] >= min(w1.released_at[0], w2.released_at[0])


class TestGrantReleaseWakesWaiters:
    """Grant.release() wakes the next queued waiter in simulation context."""

    def test_release_wakes_waiter_at_current_time(self):
        """When a grant is released at t=0.1, the waiter should
        acquire at t=0.1 (not at some future time)."""
        resource = Resource("cpu", capacity=1)
        w1 = Worker("w1", resource, amount=1, work_time=0.1)
        w2 = Worker("w2", resource, amount=1, work_time=0.1)
        sim = _make_sim(w1, w2)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        sim.run()

        # w2 should acquire exactly when w1 releases
        assert w2.acquired_at[0] == w1.released_at[0]


class TestMultipleResources:
    """Workers acquiring CPU + memory simultaneously."""

    def test_acquire_two_resources(self):
        cpu = Resource("cpu", capacity=4)
        memory = Resource("memory", capacity=1024)
        worker = MultiResourceWorker("w1", cpu, memory, cpu_amount=2, mem_amount=512)
        sim = _make_sim(worker)
        _trigger(sim, worker, time_s=0.0)
        sim.run()

        assert worker.completed == 1
        assert cpu.available == 4
        assert memory.available == 1024

    def test_contention_on_second_resource(self):
        """Two workers: first holds all memory, second blocks on memory
        even though CPU is available."""
        cpu = Resource("cpu", capacity=4)
        memory = Resource("memory", capacity=512)
        w1 = MultiResourceWorker("w1", cpu, memory, cpu_amount=1, mem_amount=512)
        w2 = MultiResourceWorker("w2", cpu, memory, cpu_amount=1, mem_amount=512)
        sim = _make_sim(w1, w2)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        sim.run()

        assert w1.completed == 1
        assert w2.completed == 1
        # Both should complete; w2 had to wait on memory


class TestUtilizationTracking:
    """Utilization tracks correctly over simulation lifetime."""

    def test_utilization_returns_to_zero(self):
        resource = Resource("cpu", capacity=4)
        w = Worker("w1", resource, amount=4, work_time=0.1)
        sim = _make_sim(w)
        _trigger(sim, w, time_s=0.0)
        sim.run()

        assert w.completed == 1
        assert resource.utilization == 0.0
        assert resource.stats.peak_utilization == 1.0

    def test_stats_count_contentions(self):
        resource = Resource("cpu", capacity=1)
        w1 = Worker("w1", resource, amount=1, work_time=0.1)
        w2 = Worker("w2", resource, amount=1, work_time=0.1)
        sim = _make_sim(w1, w2)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        sim.run()

        s = resource.stats
        assert s.acquisitions == 2
        assert s.releases == 2
        assert s.contentions == 1  # w2 had to wait


class TestConcurrentWorkers:
    """Multiple workers with staggered acquire/release."""

    def test_staggered_arrivals(self):
        """Workers arrive at different times. Resource with capacity=2,
        so 3rd worker must wait until one finishes."""
        resource = Resource("cpu", capacity=2)
        workers = [Worker(f"w{i}", resource, amount=1, work_time=0.2) for i in range(3)]
        sim = _make_sim(*workers)
        _trigger(sim, workers[0], time_s=0.0)
        _trigger(sim, workers[1], time_s=0.0)
        _trigger(sim, workers[2], time_s=0.05)  # Arrives while first two are working
        sim.run()

        for w in workers:
            assert w.completed == 1

        # Worker 2 should start at or after one of the first two releases
        assert workers[2].acquired_at[0] >= 0.05  # Can't start before arrival


class TestTryAcquireInSimulation:
    """try_acquire in simulation context (non-blocking path)."""

    def test_try_acquire_succeeds(self):
        resource = Resource("cpu", capacity=2)
        w = TryAcquireWorker("w", resource, amount=1)
        sim = _make_sim(w)
        _trigger(sim, w, time_s=0.0)
        sim.run()

        assert w.successes == 1
        assert w.failures == 0

    def test_try_acquire_fails_when_exhausted(self):
        """First worker holds all capacity, second try_acquire fails."""
        resource = Resource("cpu", capacity=1)
        holder = Worker("holder", resource, amount=1, work_time=1.0)
        trier = TryAcquireWorker("trier", resource, amount=1)
        sim = _make_sim(holder, trier)
        _trigger(sim, holder, time_s=0.0)
        _trigger(sim, trier, time_s=0.05)  # While holder is working
        sim.run()

        assert holder.completed == 1
        assert trier.failures == 1
        assert trier.successes == 0


class TestWaitTimeTracking:
    """Wait time is recorded in stats."""

    def test_wait_time_is_positive_for_contended_acquire(self):
        resource = Resource("cpu", capacity=1)
        w1 = Worker("w1", resource, amount=1, work_time=0.5)
        w2 = Worker("w2", resource, amount=1, work_time=0.1)
        sim = _make_sim(w1, w2)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        sim.run()

        assert resource.stats.total_wait_time_ns > 0
