"""Integration tests for Resource with full simulation execution.

Tests cover: basic contention, grant release waking waiters, multiple
resources, utilization tracking, concurrent workers, try_acquire,
and a visualization test with 2N workers contending for N capacity.
"""

import json
from pathlib import Path
from typing import Generator

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.components.resource import Resource
from happysimulator.instrumentation.data import Data


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
        duration=end_time_s,
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
        sim = _make_sim(resource, w1, w2)
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
        sim = _make_sim(resource, w1, w2, w3)
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
        sim = _make_sim(resource, w1, w2)
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
        sim = _make_sim(cpu, memory, worker)
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
        sim = _make_sim(cpu, memory, w1, w2)
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
        sim = _make_sim(resource, w)
        _trigger(sim, w, time_s=0.0)
        sim.run()

        assert w.completed == 1
        assert resource.utilization == 0.0
        assert resource.stats.peak_utilization == 1.0

    def test_stats_count_contentions(self):
        resource = Resource("cpu", capacity=1)
        w1 = Worker("w1", resource, amount=1, work_time=0.1)
        w2 = Worker("w2", resource, amount=1, work_time=0.1)
        sim = _make_sim(resource, w1, w2)
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
        sim = _make_sim(resource, *workers)
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
        sim = _make_sim(resource, w)
        _trigger(sim, w, time_s=0.0)
        sim.run()

        assert w.successes == 1
        assert w.failures == 0

    def test_try_acquire_fails_when_exhausted(self):
        """First worker holds all capacity, second try_acquire fails."""
        resource = Resource("cpu", capacity=1)
        holder = Worker("holder", resource, amount=1, work_time=1.0)
        trier = TryAcquireWorker("trier", resource, amount=1)
        sim = _make_sim(resource, holder, trier)
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
        sim = _make_sim(resource, w1, w2)
        _trigger(sim, w1, time_s=0.0)
        _trigger(sim, w2, time_s=0.0)
        sim.run()

        assert resource.stats.total_wait_time_ns > 0


# ---------------------------------------------------------------------------
# Visualization test: 2N workers contending for N-capacity resource
# ---------------------------------------------------------------------------

class RepeatingWorker(Entity):
    """Worker that repeatedly acquires a resource, works, and releases.

    Each incoming event triggers one acquire-work-release cycle.
    Records acquire and release timestamps for analysis.
    """

    def __init__(self, name: str, resource: Resource, amount: int = 1,
                 work_time: float = 0.1):
        super().__init__(name)
        self.resource = resource
        self.amount = amount
        self.work_time = work_time
        self.completed = 0
        self.acquired_at: list[float] = []
        self.released_at: list[float] = []
        self.waited: list[float] = []

    def handle_event(self, event: Event) -> Generator:
        request_time = self.now.to_seconds()
        grant = yield self.resource.acquire(self.amount)
        acquire_time = self.now.to_seconds()
        self.acquired_at.append(acquire_time)
        self.waited.append(acquire_time - request_time)
        yield self.work_time
        grant.release()
        release_time = self.now.to_seconds()
        self.released_at.append(release_time)
        self.completed += 1
        return []


class ResourceSampler(Entity):
    """Periodically samples resource utilization and waiter count.

    Self-schedules at a fixed interval to record snapshots of a Resource.
    Runs as a daemon-like entity — triggered by its own events.
    """

    def __init__(self, name: str, resource: Resource,
                 utilization_data: Data, waiters_data: Data,
                 interval: float = 0.1, duration: float = 10.0):
        super().__init__(name)
        self.resource = resource
        self.utilization_data = utilization_data
        self.waiters_data = waiters_data
        self.interval = interval
        self.duration = duration

    def handle_event(self, event: Event) -> list[Event] | None:
        t = self.now
        self.utilization_data.add_stat(self.resource.utilization, t)
        self.waiters_data.add_stat(self.resource.waiters, t)

        # Schedule next sample if within duration
        next_t = self.now.to_seconds() + self.interval
        if next_t <= self.duration:
            return [Event(
                time=Instant.from_seconds(next_t),
                event_type="Sample",
                target=self,
                daemon=True,
            )]
        return []


class TestResourceContentionVisualization:
    """2N workers contend for a resource with capacity N.

    Generates plots showing utilization and waiter count over time.
    """

    def test_contention_visualization(self, test_output_dir: Path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── Configuration ──
        N = 4              # resource capacity
        NUM_WORKERS = 2 * N  # 2N = 8 workers
        WORK_TIME = 0.3    # each worker holds the resource for 300ms
        RATE = 2.0         # each worker gets a new job every 0.5s
        DURATION = 10.0    # simulation duration
        SAMPLE_INTERVAL = 0.05

        # ── Setup ──
        resource = Resource("shared_cpu", capacity=N)

        utilization_data = Data()
        waiters_data = Data()
        sampler = ResourceSampler(
            "sampler", resource, utilization_data, waiters_data,
            interval=SAMPLE_INTERVAL, duration=DURATION,
        )

        # Create 2N workers
        workers = [
            RepeatingWorker(f"worker_{i}", resource, amount=1, work_time=WORK_TIME)
            for i in range(NUM_WORKERS)
        ]

        sim = Simulation(
            duration=DURATION,
            entities=[resource, sampler, *workers],
        )

        # Kick off the sampler
        sim.schedule(Event(
            time=Instant.Epoch,
            event_type="Sample",
            target=sampler,
            daemon=True,
        ))

        # Each worker gets jobs at a constant rate, staggered slightly
        for i, w in enumerate(workers):
            offset = i * (1.0 / RATE / NUM_WORKERS)  # stagger starts
            t = offset
            while t < DURATION:
                sim.schedule(Event(
                    time=Instant.from_seconds(t),
                    event_type="Job",
                    target=w,
                ))
                t += 1.0 / RATE

        summary = sim.run()

        # ── Assertions ──
        total_completed = sum(w.completed for w in workers)
        assert total_completed > 0, "No workers completed"
        assert resource.stats.contentions > 0, "Expected contention with 2N workers"
        assert resource.stats.peak_utilization > 0.5, "Expected significant utilization"

        # ── Collect per-worker wait times ──
        all_waits = []
        for w in workers:
            all_waits.extend(w.waited)
        avg_wait = sum(all_waits) / len(all_waits) if all_waits else 0

        # ── Plot 1: Utilization & Waiters over time ──
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Utilization
        util_times = utilization_data.times()
        util_values = utilization_data.raw_values()
        axes[0].plot(util_times, util_values, "b-", linewidth=0.8)
        axes[0].axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="100%")
        axes[0].set_ylabel("Utilization")
        axes[0].set_title(
            f"Resource Contention: {NUM_WORKERS} workers, "
            f"capacity={N}, work_time={WORK_TIME}s"
        )
        axes[0].set_ylim(-0.05, 1.15)
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Waiters
        wait_times = waiters_data.times()
        wait_values = waiters_data.raw_values()
        axes[1].plot(wait_times, wait_values, "r-", linewidth=0.8)
        axes[1].set_ylabel("Queued Waiters")
        axes[1].grid(True, alpha=0.3)

        # Per-worker Gantt chart showing acquire/release intervals
        colors = plt.cm.tab10.colors
        for i, w in enumerate(workers):
            for acq, rel in zip(w.acquired_at, w.released_at):
                axes[2].barh(
                    i, rel - acq, left=acq, height=0.6,
                    color=colors[i % len(colors)], alpha=0.7,
                )
        axes[2].set_ylabel("Worker")
        axes[2].set_xlabel("Simulation Time (s)")
        axes[2].set_yticks(range(NUM_WORKERS))
        axes[2].set_yticklabels([f"w{i}" for i in range(NUM_WORKERS)])
        axes[2].grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        plot_path = test_output_dir / "resource_contention.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

        assert plot_path.exists()
        assert plot_path.stat().st_size > 1000

        # ── Save summary JSON ──
        stats = resource.stats
        summary_data = {
            "config": {
                "capacity": N,
                "num_workers": NUM_WORKERS,
                "work_time_s": WORK_TIME,
                "job_rate_per_worker": RATE,
                "duration_s": DURATION,
            },
            "results": {
                "total_completed": total_completed,
                "acquisitions": stats.acquisitions,
                "releases": stats.releases,
                "contentions": stats.contentions,
                "peak_utilization": round(stats.peak_utilization, 4),
                "peak_waiters": stats.peak_waiters,
                "avg_wait_time_s": round(avg_wait, 6),
                "total_wait_time_ns": stats.total_wait_time_ns,
            },
            "per_worker": [
                {
                    "name": w.name,
                    "completed": w.completed,
                    "avg_wait_s": round(
                        sum(w.waited) / len(w.waited), 6
                    ) if w.waited else 0,
                }
                for w in workers
            ],
        }
        json_path = test_output_dir / "resource_contention_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        assert json_path.exists()
