"""Dual-path queuing system with stepwise load and latency analysis.

This example demonstrates a queuing system with:
- Two sources generating requests
- Two intermediate queued entities (with different service times)
- A final queued entity that receives from both
- A latency-tracking sink
- Volume probes at each entity
- Stepwise load function (stable, then one source overloads)

Architecture:
    Source1 --> Queue1 --> Entity1 (fast: 0.1s) --\
                                                   --> Queue3 --> Entity3 --> Sink
    Source2 --> Queue2 --> Entity2 (slow: 0.2s) --/

Expected results:
- Entity3 should see twice the volume of Entity1 or Entity2
- Latency distribution at sink should show bimodal pattern from different service times
- When Source1 overloads, queue depth and latency should increase
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

from happysimulator import (
    ConstantArrivalTimeProvider,
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    Probe,
    Profile,
    Queue,
    QueueDriver,
    Simulation,
    Source,
)


# =============================================================================
# Profile: Stepwise load function
# =============================================================================


@dataclass(frozen=True)
class StepwiseProfile(Profile):
    """Profile that changes rate at a specific step time.

    Rate is `rate_before` until `step_time_s`, then switches to `rate_after`.
    """

    step_time_s: float
    rate_before: float
    rate_after: float

    def get_rate(self, time: Instant) -> float:
        t_s = time.to_seconds()
        if t_s < self.step_time_s:
            return self.rate_before
        return self.rate_after


# =============================================================================
# Event Provider: Creates requests with latency tracking metadata
# =============================================================================


class RequestProvider(EventProvider):
    """Generates request events targeting a specific entity.

    Each request includes creation time in context for end-to-end latency tracking.
    Optionally stops emitting after a cutoff time.
    """

    def __init__(
        self,
        target: Entity,
        source_id: str,
        *,
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._source_id = source_id
        self._stop_after = stop_after
        self.generated_requests: int = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated_requests += 1
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={
                    "created_at": time,
                    "source_id": self._source_id,
                    "request_id": self.generated_requests,
                },
            )
        ]


# =============================================================================
# Latency Tracking Sink
# =============================================================================


class LatencyTrackingSink(Entity):
    """Sink that records end-to-end latency using event context.

    Tracks completion times and latencies separately by source for analysis.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.events_received: int = 0
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []
        self.source_ids: list[str] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.events_received += 1

        created_at: Instant = event.context.get("created_at", event.time)
        source_id: str = event.context.get("source_id", "unknown")
        latency_s = (event.time - created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)
        self.source_ids.append(source_id)

        return []

    def average_latency(self) -> float:
        if not self.latencies_s:
            return 0.0
        return sum(self.latencies_s) / len(self.latencies_s)

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)

    def latencies_by_source(self) -> dict[str, list[float]]:
        """Return latencies grouped by source_id."""
        result: dict[str, list[float]] = defaultdict(list)
        for source_id, latency in zip(self.source_ids, self.latencies_s, strict=False):
            result[source_id].append(latency)
        return dict(result)


# =============================================================================
# Queued Server Entity
# =============================================================================


@dataclass
class QueuedServer(Entity):
    """A server with configurable service time and optional downstream target.

    Supports concurrency limiting and tracks processed request count.
    """

    name: str = "Server"
    service_time_s: float = 0.1
    concurrency: int = 1
    downstream: Entity | None = None

    _in_flight: int = field(default=0, init=False)
    stats_processed: int = field(default=0, init=False)

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._in_flight += 1
        yield self.service_time_s, None
        self._in_flight -= 1
        self.stats_processed += 1

        if self.downstream is None:
            return []

        completed = Event(
            time=self.now,
            event_type="Completed",
            target=self.downstream,
            context=event.context,
        )
        return [completed]


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


def bucket_latencies(
    times_s: list[float],
    latencies_s: list[float],
    bucket_size_s: float = 1.0,
) -> dict[str, list[float]]:
    """Bucket latencies by time and compute statistics."""
    buckets: dict[int, list[float]] = defaultdict(list)
    for t_s, latency_s in zip(times_s, latencies_s, strict=False):
        bucket = int(math.floor(t_s / bucket_size_s))
        buckets[bucket].append(latency_s)

    result = {
        "time_s": [],
        "avg": [],
        "p0": [],
        "p50": [],
        "p99": [],
        "p100": [],
    }

    for bucket in sorted(buckets.keys()):
        vals_sorted = sorted(buckets[bucket])
        bucket_start = bucket * bucket_size_s

        result["time_s"].append(bucket_start)
        result["avg"].append(sum(vals_sorted) / len(vals_sorted))
        result["p0"].append(percentile_sorted(vals_sorted, 0.0))
        result["p50"].append(percentile_sorted(vals_sorted, 0.50))
        result["p99"].append(percentile_sorted(vals_sorted, 0.99))
        result["p100"].append(percentile_sorted(vals_sorted, 1.0))

    return result


def create_queued_entity(
    name: str,
    service_time_s: float,
    downstream: Entity,
    concurrency: int = 1,
) -> tuple[Queue, QueueDriver, QueuedServer]:
    """Create a queue + driver + server chain."""
    server = QueuedServer(
        name=f"{name}Server",
        service_time_s=service_time_s,
        concurrency=concurrency,
        downstream=downstream,
    )
    driver = QueueDriver(name=f"{name}Driver", queue=None, target=server)
    queue = Queue(name=f"{name}Queue", egress=driver, policy=FIFOQueue())
    driver.queue = queue
    return queue, driver, server


# =============================================================================
# Main Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the dual-path queue simulation."""
    sink: LatencyTrackingSink
    entity1_processed: int
    entity2_processed: int
    entity3_processed: int
    queue1_depth_data: Data
    queue2_depth_data: Data
    queue3_depth_data: Data
    source1_generated: int
    source2_generated: int


def run_dual_path_simulation(
    *,
    duration_s: float = 60.0,
    drain_s: float = 10.0,
    step_time_s: float = 30.0,
    rate_before: float = 4.0,        # events/sec (below capacity)
    rate_after_source1: float = 12.0, # events/sec (above capacity for path 1)
    rate_source2: float = 4.0,        # events/sec (constant, below capacity)
    entity1_service_time_s: float = 0.1,  # fast entity
    entity2_service_time_s: float = 0.2,  # slow entity (2x latency)
    entity3_service_time_s: float = 0.05, # final entity
    probe_interval_s: float = 0.1,
    output_dir: Path | None = None,
) -> SimulationResult:
    """Run the dual-path queue simulation.

    Architecture:
        Source1 --> Queue1 --> Entity1 (fast) --\
                                                 --> Queue3 --> Entity3 --> Sink
        Source2 --> Queue2 --> Entity2 (slow) --/

    Load profile:
        - Source1: rate_before until step_time_s, then rate_after_source1
        - Source2: constant rate_source2
    """

    # Create sink (end of pipeline)
    sink = LatencyTrackingSink(name="Sink")

    # Create final queued entity (Entity3)
    queue3, driver3, server3 = create_queued_entity(
        name="Entity3",
        service_time_s=entity3_service_time_s,
        downstream=sink,
        concurrency=2,  # Higher concurrency to handle combined load
    )

    # Create Entity1 (fast path)
    queue1, driver1, server1 = create_queued_entity(
        name="Entity1",
        service_time_s=entity1_service_time_s,
        downstream=queue3,
        concurrency=1,
    )

    # Create Entity2 (slow path - 2x latency)
    queue2, driver2, server2 = create_queued_entity(
        name="Entity2",
        service_time_s=entity2_service_time_s,
        downstream=queue3,
        concurrency=1,
    )

    # Create probes for queue depths
    queue1_depth_data = Data()
    queue1_probe = Probe(
        target=queue1,
        metric="depth",
        data=queue1_depth_data,
        interval=probe_interval_s,
        start_time=Instant.Epoch,
    )

    queue2_depth_data = Data()
    queue2_probe = Probe(
        target=queue2,
        metric="depth",
        data=queue2_depth_data,
        interval=probe_interval_s,
        start_time=Instant.Epoch,
    )

    queue3_depth_data = Data()
    queue3_probe = Probe(
        target=queue3,
        metric="depth",
        data=queue3_depth_data,
        interval=probe_interval_s,
        start_time=Instant.Epoch,
    )

    # Create sources with stepwise load profiles
    stop_after = Instant.from_seconds(duration_s)

    # Source1: stepwise profile (stable then overload)
    profile1 = StepwiseProfile(
        step_time_s=step_time_s,
        rate_before=rate_before,
        rate_after=rate_after_source1,
    )
    provider1 = RequestProvider(queue1, source_id="source1", stop_after=stop_after)
    arrival1 = ConstantArrivalTimeProvider(profile1, start_time=Instant.Epoch)
    source1 = Source(name="Source1", event_provider=provider1, arrival_time_provider=arrival1)

    # Source2: constant rate profile
    profile2 = StepwiseProfile(
        step_time_s=duration_s + 1,  # Never steps
        rate_before=rate_source2,
        rate_after=rate_source2,
    )
    provider2 = RequestProvider(queue2, source_id="source2", stop_after=stop_after)
    arrival2 = ConstantArrivalTimeProvider(profile2, start_time=Instant.Epoch)
    source2 = Source(name="Source2", event_provider=provider2, arrival_time_provider=arrival2)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + drain_s),
        sources=[source1, source2],
        entities=[queue1, driver1, server1, queue2, driver2, server2, queue3, driver3, server3, sink],
        probes=[queue1_probe, queue2_probe, queue3_probe],
    )
    sim.run()

    return SimulationResult(
        sink=sink,
        entity1_processed=server1.stats_processed,
        entity2_processed=server2.stats_processed,
        entity3_processed=server3.stats_processed,
        queue1_depth_data=queue1_depth_data,
        queue2_depth_data=queue2_depth_data,
        queue3_depth_data=queue3_depth_data,
        source1_generated=provider1.generated_requests,
        source2_generated=provider2.generated_requests,
    )


def visualize_results(result: SimulationResult, output_dir: Path, step_time_s: float) -> None:
    """Generate visualizations of the simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get latency data
    times_s, latencies_s = result.sink.latency_time_series_seconds()
    latency_buckets = bucket_latencies(times_s, latencies_s, bucket_size_s=1.0)

    # Extract queue depth data
    q1_times = [t for (t, _) in result.queue1_depth_data.values]
    q1_depths = [v for (_, v) in result.queue1_depth_data.values]
    q2_times = [t for (t, _) in result.queue2_depth_data.values]
    q2_depths = [v for (_, v) in result.queue2_depth_data.values]
    q3_times = [t for (t, _) in result.queue3_depth_data.values]
    q3_depths = [v for (_, v) in result.queue3_depth_data.values]

    # Figure 1: Latency time series with percentiles
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(latency_buckets["time_s"], latency_buckets["avg"], label="avg", linewidth=2)
    ax.plot(latency_buckets["time_s"], latency_buckets["p0"], label="p0 (min)", linestyle="--", alpha=0.7)
    ax.plot(latency_buckets["time_s"], latency_buckets["p50"], label="p50", linewidth=2)
    ax.plot(latency_buckets["time_s"], latency_buckets["p99"], label="p99", linewidth=2)
    ax.plot(latency_buckets["time_s"], latency_buckets["p100"], label="p100 (max)", linestyle="--", alpha=0.7)
    ax.axvline(x=step_time_s, color="red", linestyle=":", label=f"Load step at {step_time_s}s")
    ax.set_title("End-to-End Latency Over Time (1s buckets)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "latency_timeseries.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'latency_timeseries.png'}")

    # Figure 1b: Latency time series for first half only (before load step)
    pre_step_times = [t for t in times_s if t < step_time_s]
    pre_step_latencies = [lat for t, lat in zip(times_s, latencies_s, strict=False) if t < step_time_s]

    if pre_step_times:
        pre_step_buckets = bucket_latencies(pre_step_times, pre_step_latencies, bucket_size_s=1.0)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(pre_step_buckets["time_s"], pre_step_buckets["avg"], label="avg", linewidth=2)
        ax.plot(pre_step_buckets["time_s"], pre_step_buckets["p0"], label="p0 (min)", linestyle="--", alpha=0.7)
        ax.plot(pre_step_buckets["time_s"], pre_step_buckets["p50"], label="p50", linewidth=2)
        ax.plot(pre_step_buckets["time_s"], pre_step_buckets["p99"], label="p99", linewidth=2)
        ax.plot(pre_step_buckets["time_s"], pre_step_buckets["p100"], label="p100 (max)", linestyle="--", alpha=0.7)
        ax.set_title(f"End-to-End Latency - First Half (before load step at {step_time_s}s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Latency (s)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "latency_timeseries_pre_step.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'latency_timeseries_pre_step.png'}")

    # Figure 2: Queue depths over time
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(q1_times, q1_depths, label="Entity1 Queue", color="blue")
    axes[0].axvline(x=step_time_s, color="red", linestyle=":", label=f"Load step")
    axes[0].set_ylabel("Queue Depth")
    axes[0].set_title("Entity1 Queue Depth (fast path, overloaded after step)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(q2_times, q2_depths, label="Entity2 Queue", color="green")
    axes[1].axvline(x=step_time_s, color="red", linestyle=":")
    axes[1].set_ylabel("Queue Depth")
    axes[1].set_title("Entity2 Queue Depth (slow path, constant load)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(q3_times, q3_depths, label="Entity3 Queue", color="purple")
    axes[2].axvline(x=step_time_s, color="red", linestyle=":")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Queue Depth")
    axes[2].set_title("Entity3 Queue Depth (final entity, receives from both)")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "queue_depths.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'queue_depths.png'}")

    # Figure 3: Latency distribution by source
    latencies_by_source = result.sink.latencies_by_source()
    fig, ax = plt.subplots(figsize=(10, 5))

    data_to_plot = []
    labels = []
    for source_id in sorted(latencies_by_source.keys()):
        data_to_plot.append(latencies_by_source[source_id])
        labels.append(source_id)

    ax.boxplot(data_to_plot, tick_labels=labels)
    ax.set_title("Latency Distribution by Source")
    ax.set_xlabel("Source")
    ax.set_ylabel("Latency (s)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "latency_by_source.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'latency_by_source.png'}")

    # Figure 4: Throughput over time (requests completed per second)
    completion_buckets: dict[int, int] = defaultdict(int)
    for t in times_s:
        bucket = int(math.floor(t))
        completion_buckets[bucket] += 1

    throughput_times = sorted(completion_buckets.keys())
    throughput_values = [completion_buckets[t] for t in throughput_times]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(throughput_times, throughput_values, width=0.8, alpha=0.7)
    ax.axvline(x=step_time_s, color="red", linestyle=":", label=f"Load step at {step_time_s}s")
    ax.set_title("Throughput Over Time (requests completed per second)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests/second")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "throughput.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'throughput.png'}")


def print_summary(result: SimulationResult, step_time_s: float) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 60)

    print(f"\nLoad Configuration:")
    print(f"  Step time: {step_time_s}s (Source1 rate increases)")

    print(f"\nRequests Generated:")
    print(f"  Source1: {result.source1_generated}")
    print(f"  Source2: {result.source2_generated}")
    print(f"  Total:   {result.source1_generated + result.source2_generated}")

    print(f"\nRequests Processed:")
    print(f"  Entity1 (fast, 0.1s): {result.entity1_processed}")
    print(f"  Entity2 (slow, 0.2s): {result.entity2_processed}")
    print(f"  Entity3 (final):      {result.entity3_processed}")
    print(f"  Sink received:        {result.sink.events_received}")

    # Verify Entity3 sees combined volume
    combined_e1_e2 = result.entity1_processed + result.entity2_processed
    print(f"\n  Entity3 received {result.entity3_processed} (Entity1 + Entity2 = {combined_e1_e2})")

    # Latency statistics
    latencies = result.sink.latencies_s
    if latencies:
        sorted_latencies = sorted(latencies)
        print(f"\nOverall Latency Statistics:")
        print(f"  Average: {sum(latencies) / len(latencies):.4f}s")
        print(f"  p0 (min): {percentile_sorted(sorted_latencies, 0.0):.4f}s")
        print(f"  p50:      {percentile_sorted(sorted_latencies, 0.5):.4f}s")
        print(f"  p99:      {percentile_sorted(sorted_latencies, 0.99):.4f}s")
        print(f"  p100 (max): {percentile_sorted(sorted_latencies, 1.0):.4f}s")

    # Latency by source
    latencies_by_source = result.sink.latencies_by_source()
    print(f"\nLatency by Source:")
    for source_id in sorted(latencies_by_source.keys()):
        source_latencies = sorted(latencies_by_source[source_id])
        if source_latencies:
            print(f"  {source_id}:")
            print(f"    Count:   {len(source_latencies)}")
            print(f"    Average: {sum(source_latencies) / len(source_latencies):.4f}s")
            print(f"    p50:     {percentile_sorted(source_latencies, 0.5):.4f}s")
            print(f"    p99:     {percentile_sorted(source_latencies, 0.99):.4f}s")

    print("\n" + "=" * 60)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dual-path queue latency simulation")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration (s)")
    parser.add_argument("--drain", type=float, default=10.0, help="Drain time after load stops (s)")
    parser.add_argument("--step-time", type=float, default=30.0, help="Time when Source1 load increases (s)")
    parser.add_argument("--output", type=str, default="output/dual_path_queue", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    print("Running dual-path queue latency simulation...")
    print(f"  Duration: {args.duration}s + {args.drain}s drain")
    print(f"  Load step at: {args.step_time}s")

    result = run_dual_path_simulation(
        duration_s=args.duration,
        drain_s=args.drain,
        step_time_s=args.step_time,
        rate_before=4.0,           # 4 req/s (below Entity1 capacity of 10 req/s)
        rate_after_source1=12.0,   # 12 req/s (above Entity1 capacity)
        rate_source2=4.0,          # 4 req/s constant (below Entity2 capacity of 5 req/s)
        entity1_service_time_s=0.1,   # 10 req/s capacity
        entity2_service_time_s=0.2,   # 5 req/s capacity (2x latency)
        entity3_service_time_s=0.05,  # 20 req/s capacity
    )

    print_summary(result, args.step_time)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir, args.step_time)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
