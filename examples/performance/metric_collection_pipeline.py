"""Metric collection pipeline with chained ThreadPoolExecutors.

This example simulates a metric collector that must complete N tasks within
a configurable deadline (default: 60 seconds). The pipeline consists of:

1. JobSource - Generates batches of N tasks at regular intervals
2. Executor1 - First stage processing (e.g., task mapping)
3. Executor2 - Second stage processing (e.g., get statistics)
4. MetricSink - Records task completion and batch timing

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    METRIC COLLECTION PIPELINE SIMULATION                      │
└──────────────────────────────────────────────────────────────────────────────┘

    JOB TRIGGER (configurable interval, default 60s)
    ┌────────────────────────────────────────────────────────────────────────┐
    │  ┌──────────────┐                                                      │
    │  │  JobSource   │  Generates N tasks per batch with batch_id           │
    │  │  (periodic)  │  Records batch start time in BatchTracker            │
    │  └──────┬───────┘                                                      │
    └─────────┼──────────────────────────────────────────────────────────────┘
              │ N tasks (batch_id, task_id, created_at)
              ▼
    ┌─────────────────────┐
    │ RateLimiter1        │ (optional, NullRateLimiter by default)
    │ (pass-through)      │
    └─────────┬───────────┘
              ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 1: Executor1 (Microservice A - task mapping)                    │
    │  ┌─────────────┐   ┌─────────────┐                                     │
    │  │ FIFO Queue  │──►│ Workers (W1)│  Latency: configurable distribution │
    │  └─────────────┘   └──────┬──────┘                                     │
    │         ▲depth            │                                            │
    └─────────┼─────────────────┼────────────────────────────────────────────┘
              │                 │ 1:1 (N outputs)
         ┌────┴────┐            ▼
         │ Probe1  │  ┌─────────────────────┐
         └─────────┘  │ RateLimiter2        │ (optional, NullRateLimiter by default)
                      │ (pass-through)      │
                      └─────────┬───────────┘
                                ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  STAGE 2: Executor2 (Microservice B - get statistics)                  │
    │  ┌─────────────┐   ┌─────────────┐                                     │
    │  │ FIFO Queue  │──►│ Workers (W2)│  Latency: configurable distribution │
    │  └─────────────┘   └──────┬──────┘                                     │
    │         ▲depth            │                                            │
    └─────────┼─────────────────┼────────────────────────────────────────────┘
              │                 │ 1:1 (N outputs with statistics)
         ┌────┴────┐            ▼
         │ Probe2  │  ┌──────────────────────────────────────────────────────┐
         └─────────┘  │  STAGE 3: MetricSink (Microservice C - emit stats)   │
                      │  ┌─────────────────────────────────────────────────┐ │
                      │  │ Records task completion in BatchTracker         │ │
                      │  │ Tracks per-task latency                         │ │
                      │  │ When batch completes: records batch duration    │ │
                      │  └─────────────────────────────────────────────────┘ │
                      └──────────────────────────────────────────────────────┘
```

Expected results:
- Each batch of N tasks flows through the two-stage pipeline
- Batch completion times should be under the deadline (60s by default)
- Queue depths show backpressure when workers are saturated

## Running the Examples

The LeakyBucketRateLimiter used here **queues and delays** excess
requests rather than dropping them. When a batch of 100 tasks arrives
simultaneously, they enter the bucket and leak out at the configured
rate (requests/second), smoothing the burst into a steady stream. Tasks
are only dropped if the bucket capacity is exceeded — with a capacity
equal to the batch size, all tasks are preserved.

Use `--output` to write each scenario to a separate directory so results
are not overwritten between runs.

### 1. Baseline (no rate limiting)

All 100 tasks arrive at each executor simultaneously, causing sharp bursts
in queue depth and full worker saturation. All tasks complete successfully:

```bash
python examples/metric_collection_pipeline.py \
    --output output/metric_pipeline/baseline
```

### 2. Rate limit Stage 1 only (smoothing ingestion)

A leaky bucket before Stage 1 queues all 100 tasks and releases them at
20 req/s, spreading the batch over ~5 seconds. Executor1 sees a steady
trickle instead of a burst — queue depth stays low and workers sustain
moderate utilization. All 500 tasks complete (none dropped):

```bash
python examples/metric_collection_pipeline.py \
    --rate-limit-stage1 20 \
    --output output/metric_pipeline/rate_limit_stage1
```

### 3. Rate limit both stages (cascaded smoothing)

Adding a second leaky bucket between Stage 1 and Stage 2 (15 req/s)
further smooths traffic. Tasks trickle out of Stage 1 at 20 req/s,
then into Stage 2 at 15 req/s. Both queue depths show gradual draining
instead of sharp spikes, and all 500 tasks complete end-to-end:

```bash
python examples/metric_collection_pipeline.py \
    --rate-limit-stage1 20 \
    --rate-limit-stage2 15 \
    --output output/metric_pipeline/rate_limit_both
```

### 4. Aggressive rate limiting (slow drain)

Setting the leak rate low (10 req/s) spreads each batch of 100 over
~10 seconds. All tasks still complete, but batch durations increase
and may approach or exceed the batch interval deadline:

```bash
python examples/metric_collection_pipeline.py \
    --rate-limit-stage1 10 \
    --output output/metric_pipeline/rate_limit_aggressive
```

### Comparing results

Run all scenarios above, then compare the `queue_depths.png` and
`worker_utilization.png` charts across output directories to see how
rate limiting smooths bursty traffic through the pipeline while
preserving all tasks. Key things to observe:

- **Baseline**: Executor queue spikes to ~90, workers saturate at 10/10
- **Stage 1 limited**: Queue drains gradually, workers sustain moderate load
- **Both limited**: Both stages show smooth, controlled utilization
- **Aggressive**: Slow draining, longer batch durations, possible deadline overlap
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantLatency,
    ConstantRateProfile,
    Data,
    Entity,
    Event,
    EventProvider,
    ExponentialLatency,
    FIFOQueue,
    Instant,
    Probe,
    QueuedResource,
    Simulation,
    Source,
)
from happysimulator.components.rate_limiter import (
    LeakyBucketPolicy,
    NullRateLimiter,
    RateLimitedEntity,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.distributions.latency_distribution import LatencyDistribution

# =============================================================================
# Batch Tracking
# =============================================================================


@dataclass
class BatchInfo:
    """Information about a single batch."""

    batch_id: int
    start_time: Instant
    end_time: Instant | None = None
    tasks_total: int = 0
    tasks_completed: int = 0


class BatchTracker:
    """Tracks batch start/completion times across the pipeline.

    Thread-safe tracking of:
    - When each batch started
    - How many tasks are in each batch
    - When the last task of each batch completed
    """

    def __init__(self):
        self._batches: dict[int, BatchInfo] = {}
        self._completed_batches: list[BatchInfo] = []

    def start_batch(self, batch_id: int, start_time: Instant, num_tasks: int) -> None:
        """Record the start of a new batch."""
        self._batches[batch_id] = BatchInfo(
            batch_id=batch_id,
            start_time=start_time,
            tasks_total=num_tasks,
        )

    def task_completed(self, batch_id: int, completion_time: Instant) -> bool:
        """Record a task completion. Returns True if batch is now complete."""
        if batch_id not in self._batches:
            return False

        batch = self._batches[batch_id]
        batch.tasks_completed += 1

        if batch.tasks_completed >= batch.tasks_total:
            batch.end_time = completion_time
            self._completed_batches.append(batch)
            return True
        return False

    def get_batch(self, batch_id: int) -> BatchInfo | None:
        """Get batch info by ID."""
        return self._batches.get(batch_id)

    @property
    def completed_batches(self) -> list[BatchInfo]:
        """All completed batches in order."""
        return list(self._completed_batches)

    def batch_durations_seconds(self) -> list[tuple[int, float]]:
        """Return (batch_id, duration_seconds) for completed batches."""
        result = []
        for batch in self._completed_batches:
            if batch.end_time is not None:
                duration = (batch.end_time - batch.start_time).to_seconds()
                result.append((batch.batch_id, duration))
        return result


# =============================================================================
# Event Provider: Generates batch of N tasks
# =============================================================================


class BatchJobProvider(EventProvider):
    """Generates N tasks per batch trigger.

    Each task includes:
    - batch_id: Identifies which batch this task belongs to
    - task_id: Task number within the batch (0 to N-1)
    - created_at: Timestamp for latency tracking
    """

    def __init__(
        self,
        target: Entity,
        tasks_per_batch: int,
        batch_tracker: BatchTracker,
        *,
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._tasks_per_batch = tasks_per_batch
        self._batch_tracker = batch_tracker
        self._stop_after = stop_after
        self._current_batch_id = 0
        self.total_tasks_generated = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        batch_id = self._current_batch_id
        self._current_batch_id += 1

        # Register batch with tracker
        self._batch_tracker.start_batch(batch_id, time, self._tasks_per_batch)

        events = []
        for task_id in range(self._tasks_per_batch):
            self.total_tasks_generated += 1
            events.append(
                Event(
                    time=time,
                    event_type="Task",
                    target=self._target,
                    context={
                        "batch_id": batch_id,
                        "task_id": task_id,
                        "created_at": time,
                    },
                )
            )

        return events


# =============================================================================
# MetricExecutor: ThreadPool-like executor with downstream forwarding
# =============================================================================


class MetricExecutor(QueuedResource):
    """A queued executor with configurable worker count and latency.

    Models a ThreadPoolExecutor-like service that:
    - Queues incoming tasks
    - Processes them with configurable concurrency
    - Applies latency from a configurable distribution
    - Forwards completed tasks downstream
    """

    def __init__(
        self,
        name: str,
        *,
        downstream: Entity,
        concurrency: int = 10,
        latency: LatencyDistribution,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.downstream = downstream
        self.concurrency = concurrency
        self.latency = latency
        self._in_flight: int = 0
        self.stats_processed: int = 0

    def has_capacity(self) -> bool:
        """Return True if executor can accept more work."""
        return self._in_flight < self.concurrency

    @property
    def in_flight(self) -> int:
        """Number of tasks currently being processed."""
        return self._in_flight

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a task: apply latency, then forward downstream."""
        self._in_flight += 1

        # Sample latency from distribution
        latency_duration = self.latency.get_latency(self.now)
        yield latency_duration.to_seconds(), None

        self._in_flight -= 1
        self.stats_processed += 1

        # Forward to downstream with preserved context
        completed = self.forward(event, self.downstream)
        return [completed]


# =============================================================================
# MetricSink: Records completions and batch timing
# =============================================================================


class MetricSink(Entity):
    """Sink that records task completions and notifies BatchTracker.

    Tracks:
    - Per-task end-to-end latency
    - Batch completion events
    - Throughput over time
    """

    def __init__(self, name: str, batch_tracker: BatchTracker):
        super().__init__(name)
        self._batch_tracker = batch_tracker
        self.tasks_received: int = 0
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []
        self.batch_ids: list[int] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.tasks_received += 1

        batch_id: int = event.context.get("batch_id", -1)
        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)
        self.batch_ids.append(batch_id)

        # Notify batch tracker
        self._batch_tracker.task_completed(batch_id, event.time)

        return []

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)


# =============================================================================
# Helper Functions
# =============================================================================


def percentile_sorted(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from sorted values (p in [0, 1])."""
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


def create_latency_distribution(dist_type: str, mean_latency: float) -> LatencyDistribution:
    """Create a latency distribution by type name."""
    if dist_type == "constant":
        return ConstantLatency(mean_latency)
    if dist_type == "exponential":
        return ExponentialLatency(mean_latency)
    raise ValueError(f"Unknown distribution type: {dist_type}")


# =============================================================================
# Simulation Result
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the metric collection pipeline simulation."""

    batch_tracker: BatchTracker
    sink: MetricSink
    executor1: MetricExecutor
    executor2: MetricExecutor
    queue1_depth_data: Data
    queue2_depth_data: Data
    executor1_utilization_data: Data
    executor2_utilization_data: Data
    total_tasks_generated: int
    batch_interval: float
    tasks_per_batch: int


# =============================================================================
# Main Simulation
# =============================================================================


def run_simulation(
    *,
    duration_s: float = 300.0,
    batch_interval_s: float = 60.0,
    tasks_per_batch: int = 100,
    executor1_workers: int = 10,
    executor1_latency_s: float = 0.05,
    executor1_dist: str = "exponential",
    executor2_workers: int = 10,
    executor2_latency_s: float = 0.1,
    executor2_dist: str = "exponential",
    rate_limit_stage1: float | None = None,
    rate_limit_stage2: float | None = None,
    probe_interval_s: float = 0.1,
    seed: int = 42,
) -> SimulationResult:
    """Run the metric collection pipeline simulation.

    Args:
        duration_s: Total simulation duration in seconds.
        batch_interval_s: Time between batch triggers.
        tasks_per_batch: Number of tasks per batch.
        executor1_workers: Worker count for stage 1.
        executor1_latency_s: Mean latency for stage 1.
        executor1_dist: Distribution type for stage 1 ("constant" or "exponential").
        executor2_workers: Worker count for stage 2.
        executor2_latency_s: Mean latency for stage 2.
        executor2_dist: Distribution type for stage 2 ("constant" or "exponential").
        rate_limit_stage1: Optional rate limit for stage 1 (requests/second).
        rate_limit_stage2: Optional rate limit for stage 2 (requests/second).
        probe_interval_s: Interval for queue depth probes.
        seed: Random seed for reproducibility.

    Returns:
        SimulationResult with all collected data.
    """
    random.seed(seed)

    # Create shared batch tracker
    batch_tracker = BatchTracker()

    # Create sink (end of pipeline)
    sink = MetricSink(name="MetricSink", batch_tracker=batch_tracker)

    # Create executor 2 (stage 2)
    executor2 = MetricExecutor(
        name="Executor2",
        downstream=sink,
        concurrency=executor2_workers,
        latency=create_latency_distribution(executor2_dist, executor2_latency_s),
    )

    # Create rate limiter 2 (between executor 1 and 2)
    if rate_limit_stage2 is not None:
        rate_limiter2: Entity = RateLimitedEntity(
            name="RateLimiter2",
            downstream=executor2,
            policy=LeakyBucketPolicy(leak_rate=rate_limit_stage2),
        )
    else:
        rate_limiter2 = NullRateLimiter(name="RateLimiter2", downstream=executor2)

    # Create executor 1 (stage 1)
    executor1 = MetricExecutor(
        name="Executor1",
        downstream=rate_limiter2,
        concurrency=executor1_workers,
        latency=create_latency_distribution(executor1_dist, executor1_latency_s),
    )

    # Create rate limiter 1 (between source and executor 1)
    if rate_limit_stage1 is not None:
        rate_limiter1: Entity = RateLimitedEntity(
            name="RateLimiter1",
            downstream=executor1,
            policy=LeakyBucketPolicy(leak_rate=rate_limit_stage1),
        )
    else:
        rate_limiter1 = NullRateLimiter(name="RateLimiter1", downstream=executor1)

    # Create batch job provider and source
    stop_after = Instant.from_seconds(duration_s)
    batch_provider = BatchJobProvider(
        target=rate_limiter1,
        tasks_per_batch=tasks_per_batch,
        batch_tracker=batch_tracker,
        stop_after=stop_after,
    )

    # Rate for batch triggers: 1 batch per batch_interval
    batch_rate = 1.0 / batch_interval_s
    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate=batch_rate),
        start_time=Instant.Epoch,
    )
    source = Source(
        name="JobSource",
        event_provider=batch_provider,
        arrival_time_provider=arrival,
    )

    # Create probes for queue depths and worker utilization
    exec1_probes, exec1_data = Probe.on_many(
        executor1, ["depth", "in_flight"], interval=probe_interval_s
    )
    exec2_probes, exec2_data = Probe.on_many(
        executor2, ["depth", "in_flight"], interval=probe_interval_s
    )
    queue1_depth_data = exec1_data["depth"]
    queue2_depth_data = exec2_data["depth"]
    executor1_utilization_data = exec1_data["in_flight"]
    executor2_utilization_data = exec2_data["in_flight"]

    # Collect entities — always include rate limiters (LeakyBucket
    # self-schedules leak events that need entity registration)
    entities: list[Entity] = [
        executor1,
        executor2,
        sink,
        rate_limiter1,
        rate_limiter2,
    ]

    # Run simulation with drain time
    drain_s = batch_interval_s  # Allow time for last batch to complete
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=entities,
        probes=exec1_probes + exec2_probes,
    )
    sim.run()

    return SimulationResult(
        batch_tracker=batch_tracker,
        sink=sink,
        executor1=executor1,
        executor2=executor2,
        queue1_depth_data=queue1_depth_data,
        queue2_depth_data=queue2_depth_data,
        executor1_utilization_data=executor1_utilization_data,
        executor2_utilization_data=executor2_utilization_data,
        total_tasks_generated=batch_provider.total_tasks_generated,
        batch_interval=batch_interval_s,
        tasks_per_batch=tasks_per_batch,
    )


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations of the simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Batch Completion Times with deadline
    batch_durations = result.batch_tracker.batch_durations_seconds()
    if batch_durations:
        batch_ids = [b[0] for b in batch_durations]
        durations = [b[1] for b in batch_durations]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(batch_ids, durations, color="steelblue", alpha=0.8)

        # Color bars based on deadline
        deadline = result.batch_interval
        for bar, duration in zip(bars, durations, strict=False):
            if duration > deadline:
                bar.set_color("red")
            else:
                bar.set_color("green")

        ax.axhline(
            y=deadline,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Deadline ({deadline}s)",
        )
        ax.set_title("Batch Completion Times")
        ax.set_xlabel("Batch ID")
        ax.set_ylabel("Duration (seconds)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / "batch_completion_times.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'batch_completion_times.png'}")

    # Figure 2: Queue Depths Over Time
    q1_times = [t for (t, _) in result.queue1_depth_data.values]
    q1_depths = [v for (_, v) in result.queue1_depth_data.values]
    q2_times = [t for (t, _) in result.queue2_depth_data.values]
    q2_depths = [v for (_, v) in result.queue2_depth_data.values]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(q1_times, q1_depths, label="Executor1 Queue", color="blue")
    axes[0].set_ylabel("Queue Depth")
    axes[0].set_title("Executor1 Queue Depth (Stage 1)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(q2_times, q2_depths, label="Executor2 Queue", color="green")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Queue Depth")
    axes[1].set_title("Executor2 Queue Depth (Stage 2)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "queue_depths.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'queue_depths.png'}")

    # Figure 3: Worker Utilization Over Time
    u1_times = [t for (t, _) in result.executor1_utilization_data.values]
    u1_values = [v for (_, v) in result.executor1_utilization_data.values]
    u2_times = [t for (t, _) in result.executor2_utilization_data.values]
    u2_values = [v for (_, v) in result.executor2_utilization_data.values]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(u1_times, u1_values, label="Executor1 Workers", color="blue")
    axes[0].axhline(
        y=result.executor1.concurrency,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Max Workers ({result.executor1.concurrency})",
    )
    axes[0].set_ylabel("Active Workers")
    axes[0].set_title("Executor1 Worker Utilization")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(u2_times, u2_values, label="Executor2 Workers", color="green")
    axes[1].axhline(
        y=result.executor2.concurrency,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Max Workers ({result.executor2.concurrency})",
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Active Workers")
    axes[1].set_title("Executor2 Worker Utilization")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "worker_utilization.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'worker_utilization.png'}")

    # Figure 4: Per-Task Latency Distribution
    latencies = result.sink.latencies_s
    if latencies:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(latencies, bins=50, color="steelblue", alpha=0.7, edgecolor="black")

        # Add percentile lines
        sorted_latencies = sorted(latencies)
        p50 = percentile_sorted(sorted_latencies, 0.50)
        p95 = percentile_sorted(sorted_latencies, 0.95)
        p99 = percentile_sorted(sorted_latencies, 0.99)

        ax.axvline(x=p50, color="green", linestyle="--", label=f"p50: {p50:.3f}s")
        ax.axvline(x=p95, color="orange", linestyle="--", label=f"p95: {p95:.3f}s")
        ax.axvline(x=p99, color="red", linestyle="--", label=f"p99: {p99:.3f}s")

        ax.set_title("Per-Task End-to-End Latency Distribution")
        ax.set_xlabel("Latency (seconds)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / "latency_distribution.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'latency_distribution.png'}")

    # Figure 5: Pipeline Throughput Over Time
    times_s, _ = result.sink.latency_time_series_seconds()
    if times_s:
        # Bucket completions per second
        completion_buckets: dict[int, int] = defaultdict(int)
        for t in times_s:
            bucket = math.floor(t)
            completion_buckets[bucket] += 1

        throughput_times = sorted(completion_buckets.keys())
        throughput_values = [completion_buckets[t] for t in throughput_times]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(throughput_times, throughput_values, width=0.8, alpha=0.7, color="steelblue")
        ax.set_title("Pipeline Throughput (Tasks Completed per Second)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Tasks/second")
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(output_dir / "throughput.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'throughput.png'}")


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("METRIC COLLECTION PIPELINE RESULTS")
    print("=" * 60)

    print("\nConfiguration:")
    print(f"  Batch interval: {result.batch_interval}s")
    print(f"  Tasks per batch: {result.tasks_per_batch}")
    print(f"  Executor1 workers: {result.executor1.concurrency}")
    print(f"  Executor2 workers: {result.executor2.concurrency}")

    print("\nTasks:")
    print(f"  Generated: {result.total_tasks_generated}")
    print(f"  Executor1 processed: {result.executor1.stats_processed}")
    print(f"  Executor2 processed: {result.executor2.stats_processed}")
    print(f"  Sink received: {result.sink.tasks_received}")

    # Batch statistics
    batch_durations = result.batch_tracker.batch_durations_seconds()
    if batch_durations:
        print("\nBatches:")
        print(f"  Completed: {len(batch_durations)}")

        durations = [d[1] for d in batch_durations]
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)

        deadline = result.batch_interval
        batches_over_deadline = sum(1 for d in durations if d > deadline)

        print(f"  Avg duration: {avg_duration:.3f}s")
        print(f"  Min duration: {min_duration:.3f}s")
        print(f"  Max duration: {max_duration:.3f}s")
        print(f"  Over deadline ({deadline}s): {batches_over_deadline}")

        print("\n  Per-batch durations:")
        for batch_id, duration in batch_durations:
            status = "OK" if duration <= deadline else "OVER"
            print(f"    Batch {batch_id}: {duration:.3f}s [{status}]")

    # Latency statistics
    latencies = result.sink.latencies_s
    if latencies:
        sorted_latencies = sorted(latencies)
        print("\nPer-Task Latency:")
        print(f"  Average: {sum(latencies) / len(latencies):.4f}s")
        print(f"  p0 (min): {percentile_sorted(sorted_latencies, 0.0):.4f}s")
        print(f"  p50: {percentile_sorted(sorted_latencies, 0.50):.4f}s")
        print(f"  p95: {percentile_sorted(sorted_latencies, 0.95):.4f}s")
        print(f"  p99: {percentile_sorted(sorted_latencies, 0.99):.4f}s")
        print(f"  p100 (max): {percentile_sorted(sorted_latencies, 1.0):.4f}s")

    print("\n" + "=" * 60)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metric collection pipeline simulation")

    # Simulation parameters
    parser.add_argument(
        "--duration",
        type=float,
        default=300.0,
        help="Simulation duration in seconds (default: 300)",
    )
    parser.add_argument(
        "--batch-interval",
        type=float,
        default=60.0,
        help="Time between batches in seconds (default: 60)",
    )
    parser.add_argument(
        "--tasks-per-batch",
        type=int,
        default=100,
        help="Number of tasks per batch (default: 100)",
    )

    # Executor 1 parameters
    parser.add_argument(
        "--executor1-workers",
        type=int,
        default=10,
        help="Workers in Executor1 (default: 10)",
    )
    parser.add_argument(
        "--executor1-latency",
        type=float,
        default=0.05,
        help="Mean latency for Stage 1 in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--executor1-dist",
        choices=["constant", "exponential"],
        default="exponential",
        help="Distribution type for Stage 1 (default: exponential)",
    )

    # Executor 2 parameters
    parser.add_argument(
        "--executor2-workers",
        type=int,
        default=10,
        help="Workers in Executor2 (default: 10)",
    )
    parser.add_argument(
        "--executor2-latency",
        type=float,
        default=0.1,
        help="Mean latency for Stage 2 in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--executor2-dist",
        choices=["constant", "exponential"],
        default="exponential",
        help="Distribution type for Stage 2 (default: exponential)",
    )

    # Rate limiting
    parser.add_argument(
        "--rate-limit-stage1",
        type=float,
        default=None,
        help="Optional rate limit for Stage 1 (requests/second)",
    )
    parser.add_argument(
        "--rate-limit-stage2",
        type=float,
        default=None,
        help="Optional rate limit for Stage 2 (requests/second)",
    )

    # Output parameters
    parser.add_argument(
        "--output",
        type=str,
        default="output/metric_pipeline",
        help="Output directory (default: output/metric_pipeline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation",
    )

    args = parser.parse_args()

    print("Running metric collection pipeline simulation...")
    print(f"  Duration: {args.duration}s")
    print(f"  Batch interval: {args.batch_interval}s")
    print(f"  Tasks per batch: {args.tasks_per_batch}")
    print(
        f"  Executor1: {args.executor1_workers} workers, {args.executor1_latency}s mean ({args.executor1_dist})"
    )
    print(
        f"  Executor2: {args.executor2_workers} workers, {args.executor2_latency}s mean ({args.executor2_dist})"
    )
    if args.rate_limit_stage1:
        print(
            f"  Rate limit Stage 1: {args.rate_limit_stage1} req/s (leaky bucket, capacity={args.tasks_per_batch})"
        )
    if args.rate_limit_stage2:
        print(
            f"  Rate limit Stage 2: {args.rate_limit_stage2} req/s (leaky bucket, capacity={args.tasks_per_batch})"
        )

    result = run_simulation(
        duration_s=args.duration,
        batch_interval_s=args.batch_interval,
        tasks_per_batch=args.tasks_per_batch,
        executor1_workers=args.executor1_workers,
        executor1_latency_s=args.executor1_latency,
        executor1_dist=args.executor1_dist,
        executor2_workers=args.executor2_workers,
        executor2_latency_s=args.executor2_latency,
        executor2_dist=args.executor2_dist,
        rate_limit_stage1=args.rate_limit_stage1,
        rate_limit_stage2=args.rate_limit_stage2,
        seed=args.seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
