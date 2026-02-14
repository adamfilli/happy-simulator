"""Integration tests for ThreadPool with visualization.

Scenarios:
- Thread pool under varying load conditions
- Mixed task durations (CPU-bound vs I/O-bound)
- Queue buildup and drain behavior
- Worker utilization patterns

Output includes:
- Processing time distributions
- Queue depth over time
- Worker utilization over time
- Throughput analysis

Run:
    pytest tests/integration/test_thread_pool_visualization.py -v

Output:
    test_output/test_thread_pool_visualization/<test_name>/...
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import pytest

from happysimulator.components.server.thread_pool import ThreadPool, ThreadPoolStats
from happysimulator.components.queue_policy import FIFOQueue, LIFOQueue
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Duration, Instant
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


# --- Profiles ---


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


@dataclass(frozen=True)
class BurstProfile(Profile):
    """Profile with periodic burst patterns."""
    base_rate: float
    burst_rate: float
    burst_duration_s: float
    burst_interval_s: float

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()
        cycle_pos = t % self.burst_interval_s
        if cycle_pos < self.burst_duration_s:
            return float(self.burst_rate)
        return float(self.base_rate)


@dataclass(frozen=True)
class RampProfile(Profile):
    """Linear ramp from start to end rate."""
    start_rate: float
    end_rate: float
    ramp_duration_s: float

    def get_rate(self, time: Instant) -> float:
        t = min(time.to_seconds(), self.ramp_duration_s)
        if self.ramp_duration_s <= 0:
            return float(self.end_rate)
        frac = t / self.ramp_duration_s
        return float(self.start_rate + frac * (self.end_rate - self.start_rate))


# --- Metrics Collector ---


@dataclass
class MetricsCollector(Entity):
    """Collects time-series metrics from a thread pool."""
    name: str
    pool: ThreadPool
    sample_interval_s: float = 0.1

    # Time series data
    timestamps: list[float] = field(default_factory=list, init=False)
    queue_depths: list[int] = field(default_factory=list, init=False)
    active_workers: list[int] = field(default_factory=list, init=False)
    utilizations: list[float] = field(default_factory=list, init=False)
    tasks_completed: list[int] = field(default_factory=list, init=False)

    _last_sample_time: float = field(default=0.0, init=False)
    _last_tasks_completed: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> list[Event] | None:
        """Record metrics and schedule next sample."""
        current_time = self.now.to_seconds()

        # Record current state
        self.timestamps.append(current_time)
        self.queue_depths.append(self.pool.queued_tasks)
        self.active_workers.append(self.pool.active_workers)
        self.utilizations.append(self.pool.worker_utilization)
        self.tasks_completed.append(self.pool.stats.tasks_completed)

        # Schedule next sample
        next_time = self.now + Duration.from_seconds(self.sample_interval_s)
        return [
            Event(
                time=next_time,
                event_type="metrics_sample",
                target=self,
            )
        ]

    def get_throughput_per_interval(self) -> list[float]:
        """Calculate tasks completed per interval."""
        throughput = []
        for i in range(1, len(self.tasks_completed)):
            delta = self.tasks_completed[i] - self.tasks_completed[i - 1]
            throughput.append(delta / self.sample_interval_s)
        return throughput


# --- Task Completion Tracker ---


@dataclass
class TaskCompletionTracker(Entity):
    """Tracks individual task completion times."""
    name: str

    completion_times: list[float] = field(default_factory=list, init=False)
    processing_times: list[float] = field(default_factory=list, init=False)
    wait_times: list[float] = field(default_factory=list, init=False)

    def handle_event(self, event: Event) -> None:
        """Record task completion."""
        self.completion_times.append(self.now.to_seconds())

        # Extract timing from context if available
        created_at = event.context.get("created_at")
        if isinstance(created_at, Instant):
            total_time = (self.now - created_at).to_seconds()
            self.wait_times.append(total_time)

        proc_time = event.context.get("metadata", {}).get("processing_time", 0)
        self.processing_times.append(proc_time)


# --- Task Providers ---


class VariableTaskProvider(EventProvider):
    """Generates tasks with variable processing times."""

    def __init__(
        self,
        pool: ThreadPool,
        processing_time_distribution: str = "uniform",
        min_time: float = 0.010,
        max_time: float = 0.100,
        stop_after: Instant | None = None,
        completion_tracker: TaskCompletionTracker | None = None,
    ):
        self.pool = pool
        self.distribution = processing_time_distribution
        self.min_time = min_time
        self.max_time = max_time
        self.stop_after = stop_after
        self.completion_tracker = completion_tracker
        self.generated = 0

    def _get_processing_time(self) -> float:
        """Generate a random processing time based on distribution."""
        if self.distribution == "uniform":
            return random.uniform(self.min_time, self.max_time)
        elif self.distribution == "bimodal":
            # 70% fast tasks, 30% slow tasks
            if random.random() < 0.7:
                return random.uniform(self.min_time, self.min_time * 2)
            else:
                return random.uniform(self.max_time * 0.5, self.max_time)
        elif self.distribution == "exponential":
            mean = (self.min_time + self.max_time) / 2
            return max(self.min_time, min(self.max_time, random.expovariate(1 / mean)))
        else:
            return (self.min_time + self.max_time) / 2

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        proc_time = self._get_processing_time()

        event = Event(
            time=time,
            event_type=f"Task-{self.generated}",
            target=self.pool,
            context={"created_at": time},
        )
        event.context["metadata"]["processing_time"] = proc_time

        # Add completion hook if tracker is configured
        if self.completion_tracker is not None:
            def on_complete(finish_time: Instant):
                return Event(
                    time=finish_time,
                    event_type="task_complete",
                    target=self.completion_tracker,
                    context={
                        "metadata": {"processing_time": proc_time},
                        "created_at": time,
                    },
                )
            event.add_completion_hook(on_complete)

        return [event]


# --- Helper Functions ---


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write data to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _percentile_sorted(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from sorted values."""
    if not sorted_values:
        return 0.0
    if p <= 0:
        return float(sorted_values[0])
    if p >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _bin_data(times: list[float], values: list[float], bin_size: float) -> tuple[list[float], list[float]]:
    """Bin time series data."""
    if not times or not values:
        return [], []

    max_time = max(times)
    n_bins = int(max_time / bin_size) + 1

    bins = defaultdict(list)
    for t, v in zip(times, values):
        idx = int(t / bin_size)
        bins[idx].append(v)

    centers = []
    averages = []
    for i in range(n_bins):
        if bins[i]:
            centers.append((i + 0.5) * bin_size)
            averages.append(sum(bins[i]) / len(bins[i]))

    return centers, averages


# --- Scenario Runner ---


@dataclass
class ThreadPoolScenarioResult:
    """Results from a thread pool simulation scenario."""
    pool: ThreadPool
    metrics: MetricsCollector
    tracker: TaskCompletionTracker
    duration_s: float
    tasks_generated: int


def run_thread_pool_scenario(
    *,
    num_workers: int = 4,
    duration_s: float = 30.0,
    profile: Profile,
    task_distribution: str = "uniform",
    min_task_time: float = 0.010,
    max_task_time: float = 0.100,
    queue_policy: str = "fifo",
    sample_interval_s: float = 0.1,
    random_seed: int = 42,
) -> ThreadPoolScenarioResult:
    """Run a thread pool simulation scenario."""
    random.seed(random_seed)

    # Create thread pool
    policy = LIFOQueue() if queue_policy == "lifo" else FIFOQueue()
    pool = ThreadPool(
        name="thread_pool",
        num_workers=num_workers,
        queue_policy=policy,
    )

    # Create metrics collector
    metrics = MetricsCollector(
        name="metrics_collector",
        pool=pool,
        sample_interval_s=sample_interval_s,
    )

    # Create completion tracker
    tracker = TaskCompletionTracker(name="completion_tracker")

    # Create task provider
    provider = VariableTaskProvider(
        pool=pool,
        processing_time_distribution=task_distribution,
        min_time=min_task_time,
        max_time=max_task_time,
        stop_after=Instant.from_seconds(duration_s),
        completion_tracker=tracker,
    )

    arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source("task_source", provider, arrival)

    # Initial metrics event
    initial_metrics = Event(
        time=Instant.Epoch,
        event_type="metrics_sample",
        target=metrics,
    )

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 5.0),  # Extra time to drain
        sources=[source],
        entities=[pool, metrics, tracker],
    )
    sim.schedule(initial_metrics)
    sim.run()

    return ThreadPoolScenarioResult(
        pool=pool,
        metrics=metrics,
        tracker=tracker,
        duration_s=duration_s,
        tasks_generated=provider.generated,
    )


def generate_visualizations(result: ThreadPoolScenarioResult, output_dir: Path) -> None:
    """Generate visualization plots and CSV files."""

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = result.metrics
    pool = result.pool
    tracker = result.tracker

    # --- Write CSV files ---

    # Time series metrics
    _write_csv(
        output_dir / "timeseries_metrics.csv",
        header=["time_s", "queue_depth", "active_workers", "utilization", "tasks_completed"],
        rows=[
            [t, q, a, u, c]
            for t, q, a, u, c in zip(
                metrics.timestamps,
                metrics.queue_depths,
                metrics.active_workers,
                metrics.utilizations,
                metrics.tasks_completed,
            )
        ],
    )

    # Processing times
    proc_times = pool._processing_times
    _write_csv(
        output_dir / "processing_times.csv",
        header=["index", "processing_time_s"],
        rows=[[i, t] for i, t in enumerate(proc_times)],
    )

    # Summary statistics
    sorted_proc = sorted(proc_times) if proc_times else []
    summary_stats = {
        "tasks_generated": result.tasks_generated,
        "tasks_completed": pool.stats.tasks_completed,
        "total_processing_time_s": pool.stats.total_processing_time,
        "avg_processing_time_s": pool.average_processing_time,
        "p50_processing_time_s": _percentile_sorted(sorted_proc, 0.50),
        "p90_processing_time_s": _percentile_sorted(sorted_proc, 0.90),
        "p99_processing_time_s": _percentile_sorted(sorted_proc, 0.99),
        "max_queue_depth": max(metrics.queue_depths) if metrics.queue_depths else 0,
        "avg_queue_depth": sum(metrics.queue_depths) / len(metrics.queue_depths) if metrics.queue_depths else 0,
        "avg_utilization": sum(metrics.utilizations) / len(metrics.utilizations) if metrics.utilizations else 0,
    }

    _write_csv(
        output_dir / "summary_stats.csv",
        header=["metric", "value"],
        rows=[[k, v] for k, v in summary_stats.items()],
    )

    # --- Generate Plots ---

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Processing Time Distribution
    ax = axes[0, 0]
    if proc_times:
        ax.hist(proc_times, bins=40, alpha=0.7, color="steelblue", edgecolor="black")
        ax.axvline(_percentile_sorted(sorted_proc, 0.50), color="orange", linestyle="--", linewidth=2, label=f"p50: {_percentile_sorted(sorted_proc, 0.50)*1000:.1f}ms")
        ax.axvline(_percentile_sorted(sorted_proc, 0.99), color="red", linestyle="--", linewidth=2, label=f"p99: {_percentile_sorted(sorted_proc, 0.99)*1000:.1f}ms")
        ax.legend()
    ax.set_title("Processing Time Distribution")
    ax.set_xlabel("Processing Time (s)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Plot 2: Queue Depth Over Time
    ax = axes[0, 1]
    if metrics.timestamps and metrics.queue_depths:
        ax.fill_between(metrics.timestamps, metrics.queue_depths, alpha=0.5, color="coral")
        ax.plot(metrics.timestamps, metrics.queue_depths, color="coral", linewidth=1)
    ax.set_title("Queue Depth Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Depth")
    ax.grid(True, alpha=0.3)

    # Plot 3: Worker Utilization Over Time
    ax = axes[0, 2]
    if metrics.timestamps and metrics.utilizations:
        ax.fill_between(metrics.timestamps, [u * 100 for u in metrics.utilizations], alpha=0.5, color="forestgreen")
        ax.plot(metrics.timestamps, [u * 100 for u in metrics.utilizations], color="forestgreen", linewidth=1)
        ax.axhline(100, color="red", linestyle="--", alpha=0.5, label="100% utilization")
        ax.legend()
    ax.set_title("Worker Utilization Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3)

    # Plot 4: Throughput Over Time
    ax = axes[1, 0]
    throughput = metrics.get_throughput_per_interval()
    if throughput and metrics.timestamps:
        # Use timestamps starting from index 1 (since throughput is delta)
        ax.plot(metrics.timestamps[1:len(throughput)+1], throughput, color="purple", linewidth=1)
        ax.fill_between(metrics.timestamps[1:len(throughput)+1], throughput, alpha=0.3, color="purple")
    ax.set_title("Throughput Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tasks/second")
    ax.grid(True, alpha=0.3)

    # Plot 5: Active Workers vs Queue Depth
    ax = axes[1, 1]
    if metrics.timestamps:
        ax2 = ax.twinx()
        line1, = ax.plot(metrics.timestamps, metrics.active_workers, color="blue", linewidth=1.5, label="Active Workers")
        line2, = ax2.plot(metrics.timestamps, metrics.queue_depths, color="coral", linewidth=1.5, label="Queue Depth")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Active Workers", color="blue")
        ax2.set_ylabel("Queue Depth", color="coral")
        ax.tick_params(axis="y", labelcolor="blue")
        ax2.tick_params(axis="y", labelcolor="coral")
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right")
    ax.set_title("Workers vs Queue Depth")
    ax.grid(True, alpha=0.3)

    # Plot 6: Cumulative Tasks Completed
    ax = axes[1, 2]
    if metrics.timestamps and metrics.tasks_completed:
        ax.plot(metrics.timestamps, metrics.tasks_completed, color="darkgreen", linewidth=2)
        # Add reference line for theoretical maximum
        if pool.num_workers > 0 and pool.average_processing_time > 0:
            max_rate = pool.num_workers / pool.average_processing_time
            theoretical = [min(t * max_rate, result.tasks_generated) for t in metrics.timestamps]
            ax.plot(metrics.timestamps, theoretical, color="gray", linestyle="--", alpha=0.5, label=f"Theoretical max (~{max_rate:.0f}/s)")
            ax.legend()
    ax.set_title("Cumulative Tasks Completed")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tasks Completed")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Thread Pool Analysis (Workers: {pool.num_workers}, Tasks: {pool.stats.tasks_completed})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "thread_pool_analysis.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved visualization to: {output_dir}")


# --- Test Cases ---


class TestThreadPoolVisualization:
    """Integration tests for thread pool with visualization output."""

    def test_steady_state_load(self, test_output_dir: Path):
        """Thread pool under constant load at ~80% capacity."""
        num_workers = 4
        avg_task_time = 0.050  # 50ms average
        capacity = num_workers / avg_task_time  # 80 tasks/s theoretical max
        target_rate = capacity * 0.8  # 80% utilization

        result = run_thread_pool_scenario(
            num_workers=num_workers,
            duration_s=30.0,
            profile=ConstantRateProfile(rate_per_s=target_rate),
            task_distribution="uniform",
            min_task_time=0.040,
            max_task_time=0.060,
        )

        generate_visualizations(result, test_output_dir)

        # Verify expected behavior
        assert result.pool.stats.tasks_completed > 0
        avg_util = sum(result.metrics.utilizations) / len(result.metrics.utilizations)
        assert avg_util > 0.5  # Should have decent utilization

    def test_overload_scenario(self, test_output_dir: Path):
        """Thread pool under load exceeding capacity - queue buildup."""
        num_workers = 2
        avg_task_time = 0.100  # 100ms average
        capacity = num_workers / avg_task_time  # 20 tasks/s theoretical max

        result = run_thread_pool_scenario(
            num_workers=num_workers,
            duration_s=20.0,
            profile=ConstantRateProfile(rate_per_s=capacity * 1.5),  # 150% of capacity
            task_distribution="uniform",
            min_task_time=0.080,
            max_task_time=0.120,
        )

        generate_visualizations(result, test_output_dir)

        # Verify queue buildup occurred
        max_queue = max(result.metrics.queue_depths)
        assert max_queue > 5, "Queue should have built up under overload"

    def test_burst_traffic_pattern(self, test_output_dir: Path):
        """Thread pool handling burst traffic patterns."""
        result = run_thread_pool_scenario(
            num_workers=4,
            duration_s=30.0,
            profile=BurstProfile(
                base_rate=10.0,  # 10 tasks/s baseline
                burst_rate=100.0,  # 100 tasks/s during bursts
                burst_duration_s=2.0,  # 2s bursts
                burst_interval_s=10.0,  # Every 10s
            ),
            task_distribution="uniform",
            min_task_time=0.020,
            max_task_time=0.040,
        )

        generate_visualizations(result, test_output_dir)

        # Should see periodic queue buildup during bursts
        assert result.pool.stats.tasks_completed > 0

    def test_ramp_up_load(self, test_output_dir: Path):
        """Thread pool with gradually increasing load."""
        result = run_thread_pool_scenario(
            num_workers=4,
            duration_s=30.0,
            profile=RampProfile(
                start_rate=5.0,  # Start slow
                end_rate=100.0,  # End fast (overload)
                ramp_duration_s=25.0,
            ),
            task_distribution="uniform",
            min_task_time=0.040,
            max_task_time=0.060,
        )

        generate_visualizations(result, test_output_dir)

        # Compare early vs mid-simulation (before drain period)
        # With 0.1s sample interval, 30s duration = ~300 samples
        n_samples = len(result.metrics.utilizations)
        early_samples = result.metrics.utilizations[:20]  # First 2 seconds
        # Take mid-simulation samples (around 15-20s mark, during high load)
        mid_start = min(150, n_samples - 30)  # ~15s into simulation
        mid_end = min(200, n_samples - 10)    # ~20s into simulation
        mid_samples = result.metrics.utilizations[mid_start:mid_end] if mid_start < mid_end else []

        if early_samples and mid_samples:
            early_util = sum(early_samples) / len(early_samples)
            mid_util = sum(mid_samples) / len(mid_samples)
            assert mid_util > early_util, "Utilization should increase as load ramps up"

    def test_bimodal_task_distribution(self, test_output_dir: Path):
        """Thread pool with bimodal task durations (fast + slow tasks)."""
        result = run_thread_pool_scenario(
            num_workers=4,
            duration_s=30.0,
            profile=ConstantRateProfile(rate_per_s=40.0),
            task_distribution="bimodal",
            min_task_time=0.010,  # Fast tasks: 10-20ms
            max_task_time=0.200,  # Slow tasks: 100-200ms
        )

        generate_visualizations(result, test_output_dir)

        # Should have variable processing times
        proc_times = result.pool._processing_times
        if len(proc_times) > 10:
            variance = sum((t - result.pool.average_processing_time) ** 2 for t in proc_times) / len(proc_times)
            assert variance > 0.0001, "Should have significant variance in processing times"

    def test_fifo_vs_lifo_comparison(self, test_output_dir: Path):
        """Compare FIFO and LIFO queue policies under identical load."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Common parameters
        params = dict(
            num_workers=2,
            duration_s=20.0,
            profile=ConstantRateProfile(rate_per_s=30.0),  # Slight overload
            task_distribution="uniform",
            min_task_time=0.050,
            max_task_time=0.100,
            random_seed=42,
        )

        # Run both scenarios
        result_fifo = run_thread_pool_scenario(**params, queue_policy="fifo")
        result_lifo = run_thread_pool_scenario(**params, queue_policy="lifo")

        # Generate comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Queue Depth Comparison
        ax = axes[0, 0]
        ax.plot(result_fifo.metrics.timestamps, result_fifo.metrics.queue_depths, label="FIFO", color="blue", alpha=0.7)
        ax.plot(result_lifo.metrics.timestamps, result_lifo.metrics.queue_depths, label="LIFO", color="red", alpha=0.7)
        ax.set_title("Queue Depth: FIFO vs LIFO")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Queue Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Processing Time Distribution Comparison
        ax = axes[0, 1]
        if result_fifo.pool._processing_times:
            ax.hist(result_fifo.pool._processing_times, bins=30, alpha=0.5, label="FIFO", color="blue")
        if result_lifo.pool._processing_times:
            ax.hist(result_lifo.pool._processing_times, bins=30, alpha=0.5, label="LIFO", color="red")
        ax.set_title("Processing Time Distribution")
        ax.set_xlabel("Processing Time (s)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Throughput Comparison
        ax = axes[1, 0]
        fifo_throughput = result_fifo.metrics.get_throughput_per_interval()
        lifo_throughput = result_lifo.metrics.get_throughput_per_interval()
        if fifo_throughput:
            ax.plot(result_fifo.metrics.timestamps[1:len(fifo_throughput)+1], fifo_throughput, label="FIFO", color="blue", alpha=0.7)
        if lifo_throughput:
            ax.plot(result_lifo.metrics.timestamps[1:len(lifo_throughput)+1], lifo_throughput, label="LIFO", color="red", alpha=0.7)
        ax.set_title("Throughput Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tasks/second")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Summary Statistics
        ax = axes[1, 1]
        fifo_sorted = sorted(result_fifo.pool._processing_times)
        lifo_sorted = sorted(result_lifo.pool._processing_times)

        categories = ["Tasks\nCompleted", "Avg Queue\nDepth", "p50 Time\n(ms)", "p99 Time\n(ms)"]
        fifo_values = [
            result_fifo.pool.stats.tasks_completed,
            sum(result_fifo.metrics.queue_depths) / len(result_fifo.metrics.queue_depths),
            _percentile_sorted(fifo_sorted, 0.50) * 1000,
            _percentile_sorted(fifo_sorted, 0.99) * 1000,
        ]
        lifo_values = [
            result_lifo.pool.stats.tasks_completed,
            sum(result_lifo.metrics.queue_depths) / len(result_lifo.metrics.queue_depths),
            _percentile_sorted(lifo_sorted, 0.50) * 1000,
            _percentile_sorted(lifo_sorted, 0.99) * 1000,
        ]

        x = range(len(categories))
        width = 0.35
        ax.bar([i - width/2 for i in x], fifo_values, width, label="FIFO", color="blue", alpha=0.7)
        ax.bar([i + width/2 for i in x], lifo_values, width, label="LIFO", color="red", alpha=0.7)
        ax.set_title("Summary Statistics Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Thread Pool: FIFO vs LIFO Queue Policy Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(test_output_dir / "fifo_vs_lifo_comparison.png", dpi=150)
        plt.close(fig)

        # Write comparison CSV
        _write_csv(
            test_output_dir / "fifo_vs_lifo_summary.csv",
            header=["metric", "fifo", "lifo"],
            rows=[
                ["tasks_completed", result_fifo.pool.stats.tasks_completed, result_lifo.pool.stats.tasks_completed],
                ["avg_queue_depth", fifo_values[1], lifo_values[1]],
                ["p50_processing_ms", fifo_values[2], lifo_values[2]],
                ["p99_processing_ms", fifo_values[3], lifo_values[3]],
            ],
        )

        print(f"\nSaved FIFO vs LIFO comparison to: {test_output_dir}")

    def test_worker_scaling_comparison(self, test_output_dir: Path):
        """Compare thread pool performance with different worker counts."""
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        worker_counts = [1, 2, 4, 8]
        results = []

        for num_workers in worker_counts:
            result = run_thread_pool_scenario(
                num_workers=num_workers,
                duration_s=15.0,
                profile=ConstantRateProfile(rate_per_s=50.0),  # Fixed load
                task_distribution="uniform",
                min_task_time=0.040,
                max_task_time=0.060,
                random_seed=42,
            )
            results.append((num_workers, result))

        # Generate comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Tasks Completed vs Workers
        ax = axes[0, 0]
        workers = [r[0] for r in results]
        completed = [r[1].pool.stats.tasks_completed for r in results]
        ax.bar(workers, completed, color="steelblue", alpha=0.7)
        ax.set_title("Tasks Completed vs Worker Count")
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Tasks Completed")
        ax.set_xticks(workers)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 2: Average Queue Depth vs Workers
        ax = axes[0, 1]
        avg_queues = [sum(r[1].metrics.queue_depths) / len(r[1].metrics.queue_depths) for r in results]
        ax.bar(workers, avg_queues, color="coral", alpha=0.7)
        ax.set_title("Average Queue Depth vs Worker Count")
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Avg Queue Depth")
        ax.set_xticks(workers)
        ax.grid(True, alpha=0.3, axis="y")

        # Plot 3: Queue Depth Over Time for All Configurations
        ax = axes[1, 0]
        colors = ["red", "orange", "green", "blue"]
        for (num_w, result), color in zip(results, colors):
            ax.plot(result.metrics.timestamps, result.metrics.queue_depths, label=f"{num_w} workers", color=color, alpha=0.7)
        ax.set_title("Queue Depth Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Queue Depth")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Average Utilization vs Workers
        ax = axes[1, 1]
        avg_utils = [sum(r[1].metrics.utilizations) / len(r[1].metrics.utilizations) * 100 for r in results]
        ax.bar(workers, avg_utils, color="forestgreen", alpha=0.7)
        ax.axhline(100, color="red", linestyle="--", alpha=0.5)
        ax.set_title("Average Utilization vs Worker Count")
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Utilization (%)")
        ax.set_xticks(workers)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Thread Pool: Worker Scaling Analysis", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(test_output_dir / "worker_scaling_comparison.png", dpi=150)
        plt.close(fig)

        # Write comparison CSV
        _write_csv(
            test_output_dir / "worker_scaling_summary.csv",
            header=["workers", "tasks_completed", "avg_queue_depth", "avg_utilization_pct"],
            rows=[
                [w, c, q, u]
                for (w, _), c, q, u in zip(results, completed, avg_queues, avg_utils)
            ],
        )

        print(f"\nSaved worker scaling comparison to: {test_output_dir}")
