"""Demonstration of queue depth buildup when load exceeds server capacity.

This example shows the fundamental queuing theory principle:
- When arrival rate λ < service rate μ: queue stays bounded
- When arrival rate λ > service rate μ: queue grows linearly at rate (λ - μ)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INCREASING QUEUE DEPTH SIMULATION                         │
└─────────────────────────────────────────────────────────────────────────────┘

    LOAD PROFILE (Linear Ramp)
    ───────────────────────────────────────────────────────────────────────

    Rate (req/s)
    20 │                                              ╱
       │                                           ╱
    15 │                                        ╱
       │                                     ╱
    10 │─────────────────────────────────╱──────── Server Capacity (μ = 10)
       │                              ╱
     5 │                           ╱
       │                        ╱
     2 │                     ╱
       │──────────────────╱
     0 └──────────────────────────────────────────────────────────→ Time(s)
       0                 30                60                90

    Phase 1 (0-30s):   Ramp from 2 to 10 req/s (λ < μ, queue bounded)
    Phase 2 (30-60s):  Ramp from 10 to 18 req/s (λ > μ, queue grows!)
    Phase 3 (60-90s):  Continue ramp, queue grows faster

                            QUEUED SERVER
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   ┌─────────┐      ┌───────────────────┐      ┌─────────┐          │
    │   │ Source  │─────►│      Queue        │─────►│ Server  │──► Sink  │
    │   │ (ramp)  │      │      (FIFO)       │      │ (100ms) │          │
    │   └─────────┘      └───────────────────┘      └─────────┘          │
    │                            ▲                                       │
    │                            │                                       │
    │                       Queue depth                                  │
    │                       monitored                                    │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

```

## Queuing Theory

Server capacity μ = 1 / service_time = 1 / 0.1s = 10 req/s

When λ < μ (stable):
- Average queue length = λ² / (μ(μ-λ))  [for M/M/1]
- Queue is bounded

When λ > μ (unstable):
- Queue grows at rate (λ - μ) per second
- At λ = 15 req/s: queue grows by 5 requests per second
- At λ = 20 req/s: queue grows by 10 requests per second

"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    LinearRampProfile,
    Probe,
    QueuedResource,
    Simulation,
    Source,
)


# =============================================================================
# Queued Server with Exponential Service Time
# =============================================================================


class QueuedServer(QueuedResource):
    """A queued server with exponential service times.

    Sends completed events to a downstream sink for latency tracking.
    """

    def __init__(
        self,
        name: str,
        *,
        mean_service_time_s: float = 0.1,
        concurrency: int = 1,
        downstream: Entity | None = None,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.mean_service_time_s = mean_service_time_s
        self.concurrency = concurrency
        self.downstream = downstream
        self._in_flight: int = 0
        self.stats_processed: int = 0

        # Service time tracking
        self.completion_times: list[Instant] = []
        self.service_times_s: list[float] = []

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process request with exponential service time."""
        self._in_flight += 1
        service_time = random.expovariate(1.0 / self.mean_service_time_s)
        yield service_time, None
        self._in_flight -= 1

        self.stats_processed += 1
        self.completion_times.append(self.now)
        self.service_times_s.append(service_time)

        if self.downstream is None:
            return []

        completed = self.forward(event, self.downstream, event_type="Completed")
        return [completed]

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, service_times_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.service_times_s)


# =============================================================================
# Latency Tracking Sink
# =============================================================================


class LatencyTrackingSink(Entity):
    """Sink that records end-to-end latency using event context."""

    def __init__(self, name: str):
        super().__init__(name)
        self.events_received: int = 0
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.events_received += 1

        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)

        return []

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)


# =============================================================================
# Event Provider
# =============================================================================


class RequestProvider(EventProvider):
    """Generates request events targeting the server."""

    def __init__(self, target: Entity, *, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after
        self.generated_requests: int = 0

    def get_events(self, time: Instant) -> list[Event]:
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
                    "request_id": self.generated_requests,
                },
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the simulation."""
    sink: LatencyTrackingSink
    server: QueuedServer
    queue_depth_data: Data
    requests_generated: int
    profile: LinearRampProfile


def run_simulation(
    *,
    duration_s: float = 90.0,
    drain_s: float = 30.0,
    mean_service_time_s: float = 0.1,
    start_rate: float = 2.0,
    end_rate: float = 20.0,
    probe_interval_s: float = 0.1,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the queue depth simulation.

    Args:
        duration_s: How long to run the load ramp
        drain_s: Extra time for queue to drain after load stops
        mean_service_time_s: Mean server processing time (100ms = 10 req/s capacity)
        start_rate: Initial arrival rate
        end_rate: Final arrival rate
        probe_interval_s: Queue depth sampling interval
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Create pipeline: Source -> Server -> Sink
    sink = LatencyTrackingSink(name="Sink")
    server = QueuedServer(
        name="Server",
        mean_service_time_s=mean_service_time_s,
        downstream=sink,
    )

    # Create queue depth probe

    queue_probe, queue_depth_data = Probe.on(server, "depth", interval=probe_interval_s)

    # Create load profile and source
    profile = LinearRampProfile(
        duration_s=duration_s,
        start_rate=start_rate,
        end_rate=end_rate,
    )
    stop_after = Instant.from_seconds(duration_s)

    provider = RequestProvider(server, stop_after=stop_after)
    arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=[server, sink],
        probes=[queue_probe],
    )
    sim.run()

    return SimulationResult(
        sink=sink,
        server=server,
        queue_depth_data=queue_depth_data,
        requests_generated=provider.generated_requests,
        profile=profile,
    )


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    profile = result.profile
    server_capacity = 1.0 / result.server.mean_service_time_s

    # Calculate when load crosses capacity
    # start + fraction * (end - start) = capacity
    # fraction = (capacity - start) / (end - start)
    if result.profile.end_rate > result.profile.start_rate:
        crossover_fraction = (server_capacity - profile.start_rate) / (profile.end_rate - profile.start_rate)
        crossover_time = crossover_fraction * profile.duration_s
    else:
        crossover_time = None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Load profile with capacity line
    ax = axes[0, 0]
    time_points = list(range(0, int(profile.duration_s) + 10))
    rates = [profile.get_rate(Instant.from_seconds(t)) for t in time_points]
    ax.plot(time_points, rates, 'b-', linewidth=2, label='Arrival Rate (λ)')
    ax.axhline(y=server_capacity, color='r', linestyle='--', linewidth=2,
               label=f'Server Capacity (μ = {server_capacity:.0f} req/s)')
    if crossover_time:
        ax.axvline(x=crossover_time, color='orange', linestyle=':', linewidth=2,
                   label=f'λ = μ at t={crossover_time:.0f}s')
        ax.fill_between(time_points, rates, server_capacity,
                        where=[r > server_capacity for r in rates],
                        alpha=0.3, color='red', label='Overload (λ > μ)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate (req/s)')
    ax.set_title('Load Profile: Linear Ramp')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(rates) * 1.1)

    # 2. Queue depth over time
    ax = axes[0, 1]
    q_times = [t for (t, _) in result.queue_depth_data.values]
    q_depths = [v for (_, v) in result.queue_depth_data.values]
    ax.plot(q_times, q_depths, 'b-', linewidth=1)
    if crossover_time:
        ax.axvline(x=crossover_time, color='orange', linestyle=':', linewidth=2,
                   label=f'λ = μ at t={crossover_time:.0f}s')
    ax.axvline(x=profile.duration_s, color='green', linestyle=':', linewidth=2,
               label=f'Load stops at t={profile.duration_s:.0f}s')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Queue Depth')
    ax.set_title('Queue Depth Over Time')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 3. End-to-end latency over time (binned)
    ax = axes[1, 0]
    times_s, latencies_s = result.sink.latency_time_series_seconds()

    latency_buckets: dict[int, list[float]] = defaultdict(list)
    for t, lat in zip(times_s, latencies_s):
        bucket = int(t)
        latency_buckets[bucket].append(lat)

    bucket_times = sorted(latency_buckets.keys())
    bucket_avg_latencies = [sum(latency_buckets[b]) / len(latency_buckets[b]) * 1000
                           for b in bucket_times]

    ax.plot(bucket_times, bucket_avg_latencies, 'b-', linewidth=1.5, marker='o', markersize=3)
    ax.axhline(y=result.server.mean_service_time_s * 1000, color='g', linestyle='--',
               label=f'Service time ({result.server.mean_service_time_s * 1000:.0f}ms)')
    if crossover_time:
        ax.axvline(x=crossover_time, color='orange', linestyle=':', linewidth=2)
    ax.axvline(x=profile.duration_s, color='green', linestyle=':', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Avg Latency (ms)')
    ax.set_title('End-to-End Latency Over Time (1s avg)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 4. Throughput over time
    ax = axes[1, 1]
    completion_buckets: dict[int, int] = defaultdict(int)
    for t in times_s:
        bucket = int(t)
        completion_buckets[bucket] += 1

    throughput_times = sorted(completion_buckets.keys())
    throughput_values = [completion_buckets[t] for t in throughput_times]

    ax.plot(throughput_times, throughput_values, 'g-', linewidth=1.5, marker='o', markersize=3)
    ax.axhline(y=server_capacity, color='r', linestyle='--',
               label=f'Server Capacity ({server_capacity:.0f} req/s)')
    if crossover_time:
        ax.axvline(x=crossover_time, color='orange', linestyle=':', linewidth=2)
    ax.axvline(x=profile.duration_s, color='green', linestyle=':', linewidth=2,
               label='Load stops')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Completions / second')
    ax.set_title('Throughput Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "increasing_queue_depth.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'increasing_queue_depth.png'}")


def print_summary(result: SimulationResult) -> None:
    """Print summary statistics."""
    profile = result.profile
    server_capacity = 1.0 / result.server.mean_service_time_s

    print("\n" + "=" * 70)
    print("INCREASING QUEUE DEPTH SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Server capacity: {server_capacity:.0f} req/s (mean service time = {result.server.mean_service_time_s * 1000:.0f}ms)")
    print(f"  Load ramp: {profile.start_rate} → {profile.end_rate} req/s over {profile.duration_s}s")

    # Calculate crossover time
    crossover_time = 0.0
    if profile.end_rate > profile.start_rate:
        crossover_fraction = (server_capacity - profile.start_rate) / (profile.end_rate - profile.start_rate)
        crossover_time = crossover_fraction * profile.duration_s
        print(f"  Crossover (λ = μ) at: t = {crossover_time:.1f}s")

    print(f"\nRequests:")
    print(f"  Generated: {result.requests_generated}")
    print(f"  Completed: {result.sink.events_received}")
    print(f"  Server processed: {result.server.stats_processed}")

    # Queue depth analysis
    q_times = [t for (t, _) in result.queue_depth_data.values]
    q_depths = [v for (_, v) in result.queue_depth_data.values]

    max_depth = max(q_depths) if q_depths else 0
    max_depth_time = q_times[q_depths.index(max_depth)] if q_depths else 0

    print(f"\nQueue Depth:")
    print(f"  Maximum: {max_depth:.0f} at t = {max_depth_time:.1f}s")

    # Analyze by phase
    def avg_depth_in_range(start: float, end: float) -> float:
        depths = [d for t, d in zip(q_times, q_depths) if start <= t < end]
        return sum(depths) / len(depths) if depths else 0.0

    print(f"\n  By phase:")
    print(f"    Before crossover (0-{crossover_time:.0f}s): avg = {avg_depth_in_range(0, crossover_time):.1f}")
    print(f"    After crossover ({crossover_time:.0f}-{profile.duration_s:.0f}s): avg = {avg_depth_in_range(crossover_time, profile.duration_s):.1f}")
    print(f"    During drain ({profile.duration_s:.0f}-{profile.duration_s + 30:.0f}s): avg = {avg_depth_in_range(profile.duration_s, profile.duration_s + 30):.1f}")

    # Latency analysis
    times_s, latencies_s = result.sink.latency_time_series_seconds()

    def avg_latency_in_range(start: float, end: float) -> float:
        lats = [lat for t, lat in zip(times_s, latencies_s) if start <= t < end]
        return sum(lats) / len(lats) if lats else 0.0

    print(f"\nLatency:")
    print(f"  Overall average: {sum(latencies_s) / len(latencies_s) * 1000:.1f}ms")
    print(f"\n  By phase:")
    print(f"    Before crossover: avg = {avg_latency_in_range(0, crossover_time) * 1000:.1f}ms")
    print(f"    After crossover: avg = {avg_latency_in_range(crossover_time, profile.duration_s) * 1000:.1f}ms")
    print(f"    During drain: avg = {avg_latency_in_range(profile.duration_s, profile.duration_s + 30) * 1000:.1f}ms")

    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("-" * 70)
    print(f"""
    When arrival rate λ exceeds service rate μ:
    - Queue grows at rate (λ - μ) per second
    - At end of ramp (λ = {profile.end_rate} req/s): growth rate = {profile.end_rate - server_capacity:.0f} req/s
    - Latency increases as queuing delay dominates

    After load stops, queue drains at rate μ = {server_capacity:.0f} req/s
    Time to drain queue of {max_depth:.0f} requests ≈ {max_depth / server_capacity:.1f}s
    """)
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Queue depth buildup simulation")
    parser.add_argument("--duration", type=float, default=90.0, help="Ramp duration (s)")
    parser.add_argument("--drain", type=float, default=30.0, help="Drain time after load stops (s)")
    parser.add_argument("--start-rate", type=float, default=2.0, help="Starting arrival rate (req/s)")
    parser.add_argument("--end-rate", type=float, default=20.0, help="Ending arrival rate (req/s)")
    parser.add_argument("--service-time", type=float, default=0.1, help="Mean service time (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/increasing_queue_depth", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    server_capacity = 1.0 / args.service_time

    print("Running queue depth buildup simulation...")
    print(f"  Load ramp: {args.start_rate} → {args.end_rate} req/s over {args.duration}s")
    print(f"  Server capacity: {server_capacity:.0f} req/s")
    print(f"  Drain time: {args.drain}s")

    result = run_simulation(
        duration_s=args.duration,
        drain_s=args.drain,
        mean_service_time_s=args.service_time,
        start_rate=args.start_rate,
        end_rate=args.end_rate,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
