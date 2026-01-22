"""Metastable failure demonstration with an M/M/1 queue.

This example demonstrates metastable failure behavior in a queuing system:
1. Non-vulnerable state: Low utilization, system recovers quickly from load spikes
2. Vulnerable state: Near saturation (~90% utilization), a spike triggers persistent failure
3. Recovery search: Gradually reduce load to find the recovery threshold

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       METASTABLE FAILURE SIMULATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

    LOAD PROFILE (120 seconds total)
    ─────────────────────────────────────────────────────────────────────────

    Rate (req/s)
    15 │                    ╭───╮              ╭───╮
       │                    │   │              │   │
    10 │────────────────────┤   │──────────────┤   ├──────────────
       │  Service capacity  │   │              │   │     ↓ step-down
     9 │                    │   │    ╭─────────┤   ├─────────────────
       │                    │   │    │         │   │   7   6   5
     5 │   ╭────────────────┤   ├────╯         │   │
       │   │  Low load      │   │  Vulnerable  │   │  Recovery search
       │   │  (stable)      │   │  (near sat.) │   │
     0 └───┴────────────────┴───┴──────────────┴───┴─────────────────→ Time(s)
       0   10          25  30 35          55  60 65    80  90 100 110 120

    Phase 1 (0-10s):   Low load, 5 req/s (50% utilization) - stable baseline
    Phase 2 (10-25s):  Continue low load - observe steady state
    Phase 3 (25-30s):  SPIKE to 15 req/s while in stable state
    Phase 4 (30-35s):  Return to 5 req/s - observe quick recovery
    Phase 5 (35-55s):  Increase to 9 req/s (90% utilization) - vulnerable state
    Phase 6 (55-60s):  SPIKE to 15 req/s while in vulnerable state
    Phase 7 (60-65s):  Return to 9 req/s - observe metastable failure persist
    Phase 8 (65-120s): Step down load: 7, 6, 5, 4, 3 req/s to find recovery

                          QUEUED SERVER (M/M/1)
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   ┌─────────┐      ┌───────────────────┐      ┌─────────┐          │
    │   │ Source  │─────►│      Queue        │─────►│ Server  │──────►   │
    │   │(Poisson)│      │      (FIFO)       │      │(Exp~0.1s)│   Sink  │
    │   └─────────┘      └───────────────────┘      └─────────┘          │
    │       ▲                    ▲                                       │
    │       │                    │                                       │
    │   Rate varies          Queue depth                                 │
    │   per profile          monitored                                   │
    │                                                                    │
    └────────────────────────────────────────────────────────────────────┘

```

## M/M/1 Queue Theory

With mean service time = 100ms, service rate μ = 10 req/s.
For stability, arrival rate λ < μ.
Utilization ρ = λ/μ.

Expected queue length = ρ²/(1-ρ)
At 50% utilization: E[queue] = 0.5 (very short)
At 90% utilization: E[queue] = 8.1 (significant)
At 95% utilization: E[queue] = 18.05 (very long)

When a spike pushes λ > μ temporarily, the queue grows linearly.
In the vulnerable state, even a brief spike creates a large queue
that takes a long time to drain (drain rate = μ - λ).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    PoissonArrivalTimeProvider,
    Probe,
    Profile,
    QueuedResource,
    Simulation,
    Source,
)


# =============================================================================
# Profile: Multi-step load function for metastable failure demonstration
# =============================================================================


@dataclass(frozen=True)
class MetastableLoadProfile(Profile):
    """Load profile demonstrating metastable failure conditions.

    Creates a stepwise function with:
    - Stable low load period
    - Spike during stable state (should recover)
    - Vulnerable high load period (near saturation)
    - Spike during vulnerable state (triggers metastable failure)
    - Step-down recovery search
    """

    # Phase timings (in seconds)
    stable_end: float = 25.0
    first_spike_start: float = 25.0
    first_spike_end: float = 30.0
    recovery_observe_end: float = 35.0
    vulnerable_start: float = 35.0
    second_spike_start: float = 55.0
    second_spike_end: float = 60.0
    step_down_start: float = 65.0

    # Rates (requests per second)
    stable_rate: float = 5.0      # 50% utilization
    vulnerable_rate: float = 9.0  # 90% utilization
    spike_rate: float = 15.0      # 150% - overload

    # Step-down rates for recovery search
    step_down_rates: tuple[float, ...] = (7.0, 6.0, 5.0, 4.0, 3.0)
    step_down_duration: float = 11.0  # seconds per step

    def get_rate(self, time: Instant) -> float:
        t = time.to_seconds()

        # Phase 1-2: Stable low load
        if t < self.first_spike_start:
            return self.stable_rate

        # Phase 3: First spike (during stable state)
        if t < self.first_spike_end:
            return self.spike_rate

        # Phase 4: Return to stable, observe recovery
        if t < self.recovery_observe_end:
            return self.stable_rate

        # Phase 5: Vulnerable state (near saturation)
        if t < self.second_spike_start:
            return self.vulnerable_rate

        # Phase 6: Second spike (during vulnerable state)
        if t < self.second_spike_end:
            return self.spike_rate

        # Phase 7: Return to vulnerable rate briefly
        if t < self.step_down_start:
            return self.vulnerable_rate

        # Phase 8: Step-down recovery search
        time_in_step_down = t - self.step_down_start
        step_index = int(time_in_step_down / self.step_down_duration)
        if step_index < len(self.step_down_rates):
            return self.step_down_rates[step_index]

        # Final: lowest rate
        return self.step_down_rates[-1]


# =============================================================================
# Event Provider: Creates requests with latency tracking metadata
# =============================================================================


class RequestProvider(EventProvider):
    """Generates request events targeting a specific entity.

    Each request includes creation time in context for end-to-end latency tracking.
    """

    def __init__(
        self,
        target: Entity,
        *,
        stop_after: Instant | None = None,
    ):
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
# M/M/1 Server with Exponential Service Times
# =============================================================================


class MM1Server(QueuedResource):
    """An M/M/1 queued server with exponential service times.

    Service times are exponentially distributed with the configured mean.
    This creates the classic M/M/1 queue behavior from queuing theory.
    """

    def __init__(
        self,
        name: str,
        *,
        mean_service_time_s: float = 0.1,
        downstream: Entity | None = None,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.mean_service_time_s = mean_service_time_s
        self.downstream = downstream
        self.stats_processed: int = 0

    def has_capacity(self) -> bool:
        """Single server: only has capacity when not processing."""
        return True  # Queue driver handles concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a request with exponential service time."""
        # Sample from exponential distribution
        service_time = random.expovariate(1.0 / self.mean_service_time_s)
        yield service_time, None

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

    result: dict[str, list[float]] = {
        "time_s": [],
        "avg": [],
        "p50": [],
        "p99": [],
        "p100": [],
        "count": [],
    }

    for bucket in sorted(buckets.keys()):
        vals_sorted = sorted(buckets[bucket])
        bucket_start = bucket * bucket_size_s

        result["time_s"].append(bucket_start)
        result["avg"].append(sum(vals_sorted) / len(vals_sorted))
        result["p50"].append(percentile_sorted(vals_sorted, 0.50))
        result["p99"].append(percentile_sorted(vals_sorted, 0.99))
        result["p100"].append(percentile_sorted(vals_sorted, 1.0))
        result["count"].append(float(len(vals_sorted)))

    return result


# =============================================================================
# Main Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the metastable failure simulation."""
    sink: LatencyTrackingSink
    server: MM1Server
    queue_depth_data: Data
    source_generated: int
    profile: MetastableLoadProfile


def run_metastable_simulation(
    *,
    duration_s: float = 120.0,
    drain_s: float = 10.0,
    mean_service_time_s: float = 0.1,
    probe_interval_s: float = 0.1,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the metastable failure simulation.

    Args:
        duration_s: How long to run the load generation
        drain_s: Extra time after load stops for queue to drain
        mean_service_time_s: Mean service time (100ms = 10 req/s capacity)
        probe_interval_s: How often to sample queue depth
        seed: Random seed for reproducibility (None for random)

    Returns:
        SimulationResult with all metrics for analysis
    """
    if seed is not None:
        random.seed(seed)

    # Create pipeline: Source -> Server -> Sink
    sink = LatencyTrackingSink(name="Sink")
    server = MM1Server(
        name="Server",
        mean_service_time_s=mean_service_time_s,
        downstream=sink,
    )

    # Create queue depth probe
    queue_depth_data = Data()
    queue_probe = Probe(
        target=server,
        metric="depth",
        data=queue_depth_data,
        interval=probe_interval_s,
        start_time=Instant.Epoch,
    )

    # Create load profile and source
    profile = MetastableLoadProfile()
    stop_after = Instant.from_seconds(duration_s)

    provider = RequestProvider(server, stop_after=stop_after)
    arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + drain_s),
        sources=[source],
        entities=[server, sink],
        probes=[queue_probe],
    )
    sim.run()

    return SimulationResult(
        sink=sink,
        server=server,
        queue_depth_data=queue_depth_data,
        source_generated=provider.generated_requests,
        profile=profile,
    )


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations of the simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data
    times_s, latencies_s = result.sink.latency_time_series_seconds()
    latency_buckets = bucket_latencies(times_s, latencies_s, bucket_size_s=1.0)

    q_times = [t for (t, _) in result.queue_depth_data.values]
    q_depths = [v for (_, v) in result.queue_depth_data.values]

    # Profile parameters for vertical lines
    profile = result.profile
    spike1_start = profile.first_spike_start
    spike1_end = profile.first_spike_end
    spike2_start = profile.second_spike_start
    spike2_end = profile.second_spike_end
    vulnerable_start = profile.vulnerable_start
    step_down_start = profile.step_down_start

    # Figure 1: Combined overview (3 subplots)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Top: Load profile
    ax1 = axes[0]
    time_points = list(range(0, 130))
    rates = [profile.get_rate(Instant.from_seconds(t)) for t in time_points]
    ax1.plot(time_points, rates, 'b-', linewidth=2, label='Arrival Rate')
    ax1.axhline(y=10, color='r', linestyle='--', label='Service Capacity (10 req/s)')
    ax1.fill_between([spike1_start, spike1_end], 0, 20, alpha=0.2, color='orange', label='Spike 1 (stable)')
    ax1.fill_between([spike2_start, spike2_end], 0, 20, alpha=0.2, color='red', label='Spike 2 (vulnerable)')
    ax1.axvline(x=vulnerable_start, color='purple', linestyle=':', alpha=0.7)
    ax1.axvline(x=step_down_start, color='green', linestyle=':', alpha=0.7)
    ax1.set_ylabel('Rate (req/s)')
    ax1.set_title('Load Profile: Metastable Failure Demonstration')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 18)
    ax1.grid(True, alpha=0.3)

    # Annotations
    ax1.annotate('Stable\n(50%)', xy=(12, 5), ha='center', fontsize=9)
    ax1.annotate('Vulnerable\n(90%)', xy=(45, 9), ha='center', fontsize=9)
    ax1.annotate('Step-down\nRecovery', xy=(95, 5), ha='center', fontsize=9)

    # Middle: Queue depth
    ax2 = axes[1]
    ax2.plot(q_times, q_depths, 'g-', linewidth=1, alpha=0.8)
    ax2.fill_between([spike1_start, spike1_end], 0, max(q_depths) if q_depths else 10,
                     alpha=0.2, color='orange')
    ax2.fill_between([spike2_start, spike2_end], 0, max(q_depths) if q_depths else 10,
                     alpha=0.2, color='red')
    ax2.axvline(x=vulnerable_start, color='purple', linestyle=':', alpha=0.7, label='Enter vulnerable state')
    ax2.axvline(x=step_down_start, color='green', linestyle=':', alpha=0.7, label='Begin step-down')
    ax2.set_ylabel('Queue Depth')
    ax2.set_title('Queue Depth Over Time')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Bottom: Latency
    ax3 = axes[2]
    ax3.plot(latency_buckets["time_s"], latency_buckets["avg"], 'b-', linewidth=2, label='Avg')
    ax3.plot(latency_buckets["time_s"], latency_buckets["p99"], 'r-', linewidth=1.5, label='p99')
    ax3.fill_between([spike1_start, spike1_end], 0, max(latency_buckets["p99"]) if latency_buckets["p99"] else 1,
                     alpha=0.2, color='orange')
    ax3.fill_between([spike2_start, spike2_end], 0, max(latency_buckets["p99"]) if latency_buckets["p99"] else 1,
                     alpha=0.2, color='red')
    ax3.axvline(x=vulnerable_start, color='purple', linestyle=':', alpha=0.7)
    ax3.axvline(x=step_down_start, color='green', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Latency (s)')
    ax3.set_title('End-to-End Latency Over Time')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "metastable_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'metastable_overview.png'}")

    # Figure 2: Detailed latency comparison (before/after each spike)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Spike 1 analysis
    pre_spike1 = [(t, lat) for t, lat in zip(times_s, latencies_s) if 15 <= t < spike1_start]
    during_spike1 = [(t, lat) for t, lat in zip(times_s, latencies_s) if spike1_start <= t < spike1_end]
    post_spike1 = [(t, lat) for t, lat in zip(times_s, latencies_s) if spike1_end <= t < vulnerable_start]

    ax = axes[0, 0]
    if pre_spike1:
        ax.hist([lat for _, lat in pre_spike1], bins=30, alpha=0.5, label='Before spike', color='blue')
    if post_spike1:
        ax.hist([lat for _, lat in post_spike1], bins=30, alpha=0.5, label='After spike', color='green')
    ax.set_xlabel('Latency (s)')
    ax.set_ylabel('Count')
    ax.set_title('Spike 1 (Stable State): Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spike 2 analysis
    pre_spike2 = [(t, lat) for t, lat in zip(times_s, latencies_s) if 45 <= t < spike2_start]
    during_spike2 = [(t, lat) for t, lat in zip(times_s, latencies_s) if spike2_start <= t < spike2_end]
    post_spike2 = [(t, lat) for t, lat in zip(times_s, latencies_s) if spike2_end <= t < step_down_start]

    ax = axes[0, 1]
    if pre_spike2:
        ax.hist([lat for _, lat in pre_spike2], bins=30, alpha=0.5, label='Before spike', color='blue')
    if post_spike2:
        ax.hist([lat for _, lat in post_spike2], bins=30, alpha=0.5, label='After spike', color='red')
    ax.set_xlabel('Latency (s)')
    ax.set_ylabel('Count')
    ax.set_title('Spike 2 (Vulnerable State): Latency Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recovery analysis
    step_times = [step_down_start + i * profile.step_down_duration for i in range(len(profile.step_down_rates))]

    ax = axes[1, 0]
    for i, (start_t, rate) in enumerate(zip(step_times, profile.step_down_rates)):
        end_t = start_t + profile.step_down_duration
        step_latencies = [lat for t, lat in zip(times_s, latencies_s) if start_t <= t < end_t]
        if step_latencies:
            positions = [i + 1]
            ax.boxplot([step_latencies], positions=positions, widths=0.6)

    ax.set_xticks(range(1, len(profile.step_down_rates) + 1))
    ax.set_xticklabels([f"{r} req/s" for r in profile.step_down_rates])
    ax.set_xlabel('Load Level')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Recovery Search: Latency at Each Step-Down Level')
    ax.grid(True, alpha=0.3, axis='y')

    # Throughput over time
    completion_buckets: dict[int, int] = defaultdict(int)
    for t in times_s:
        bucket = int(math.floor(t))
        completion_buckets[bucket] += 1

    throughput_times = sorted(completion_buckets.keys())
    throughput_values = [completion_buckets[t] for t in throughput_times]

    ax = axes[1, 1]
    ax.bar(throughput_times, throughput_values, width=0.8, alpha=0.7)
    ax.axhline(y=10, color='r', linestyle='--', label='Capacity')
    ax.fill_between([spike1_start, spike1_end], 0, max(throughput_values) if throughput_values else 15,
                    alpha=0.2, color='orange')
    ax.fill_between([spike2_start, spike2_end], 0, max(throughput_values) if throughput_values else 15,
                    alpha=0.2, color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Completions/second')
    ax.set_title('Throughput Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(output_dir / "metastable_analysis.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'metastable_analysis.png'}")


def print_summary(result: SimulationResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("METASTABLE FAILURE SIMULATION RESULTS")
    print("=" * 70)

    profile = result.profile
    times_s, latencies_s = result.sink.latency_time_series_seconds()

    print(f"\nConfiguration:")
    print(f"  Service capacity: 10 req/s (mean service time = 100ms)")
    print(f"  Stable rate: {profile.stable_rate} req/s ({profile.stable_rate/10*100:.0f}% utilization)")
    print(f"  Vulnerable rate: {profile.vulnerable_rate} req/s ({profile.vulnerable_rate/10*100:.0f}% utilization)")
    print(f"  Spike rate: {profile.spike_rate} req/s ({profile.spike_rate/10*100:.0f}% utilization)")

    print(f"\nRequests:")
    print(f"  Generated: {result.source_generated}")
    print(f"  Completed: {result.sink.events_received}")
    print(f"  Processed by server: {result.server.stats_processed}")

    # Analyze queue depth at different phases
    q_times = [t for (t, _) in result.queue_depth_data.values]
    q_depths = [v for (_, v) in result.queue_depth_data.values]

    def avg_depth_in_range(start: float, end: float) -> float:
        depths = [d for t, d in zip(q_times, q_depths) if start <= t < end]
        return sum(depths) / len(depths) if depths else 0.0

    def max_depth_in_range(start: float, end: float) -> float:
        depths = [d for t, d in zip(q_times, q_depths) if start <= t < end]
        return max(depths) if depths else 0.0

    print(f"\nQueue Depth Analysis:")
    print(f"  Stable period (10-25s):     avg={avg_depth_in_range(10, 25):.1f}, max={max_depth_in_range(10, 25):.0f}")
    print(f"  During spike 1 (25-30s):    avg={avg_depth_in_range(25, 30):.1f}, max={max_depth_in_range(25, 30):.0f}")
    print(f"  Post-spike 1 (30-35s):      avg={avg_depth_in_range(30, 35):.1f}, max={max_depth_in_range(30, 35):.0f}")
    print(f"  Vulnerable period (45-55s): avg={avg_depth_in_range(45, 55):.1f}, max={max_depth_in_range(45, 55):.0f}")
    print(f"  During spike 2 (55-60s):    avg={avg_depth_in_range(55, 60):.1f}, max={max_depth_in_range(55, 60):.0f}")
    print(f"  Post-spike 2 (60-65s):      avg={avg_depth_in_range(60, 65):.1f}, max={max_depth_in_range(60, 65):.0f}")

    # Latency analysis
    def avg_latency_in_range(start: float, end: float) -> float:
        lats = [lat for t, lat in zip(times_s, latencies_s) if start <= t < end]
        return sum(lats) / len(lats) if lats else 0.0

    def p99_latency_in_range(start: float, end: float) -> float:
        lats = sorted([lat for t, lat in zip(times_s, latencies_s) if start <= t < end])
        return percentile_sorted(lats, 0.99)

    print(f"\nLatency Analysis (average / p99):")
    print(f"  Stable period (10-25s):     {avg_latency_in_range(10, 25)*1000:.1f}ms / {p99_latency_in_range(10, 25)*1000:.1f}ms")
    print(f"  Post-spike 1 (30-35s):      {avg_latency_in_range(30, 35)*1000:.1f}ms / {p99_latency_in_range(30, 35)*1000:.1f}ms")
    print(f"  Vulnerable period (45-55s): {avg_latency_in_range(45, 55)*1000:.1f}ms / {p99_latency_in_range(45, 55)*1000:.1f}ms")
    print(f"  Post-spike 2 (60-65s):      {avg_latency_in_range(60, 65)*1000:.1f}ms / {p99_latency_in_range(60, 65)*1000:.1f}ms")

    # Recovery search analysis
    print(f"\nRecovery Search (step-down phases):")
    step_start = profile.step_down_start
    for i, rate in enumerate(profile.step_down_rates):
        start_t = step_start + i * profile.step_down_duration
        end_t = start_t + profile.step_down_duration
        avg_lat = avg_latency_in_range(start_t, end_t)
        avg_q = avg_depth_in_range(start_t, end_t)
        print(f"  {rate} req/s ({rate/10*100:.0f}%): avg_latency={avg_lat*1000:.1f}ms, avg_queue={avg_q:.1f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    # Compare recovery behavior
    post_spike1_q = avg_depth_in_range(30, 35)
    post_spike2_q = avg_depth_in_range(60, 65)

    print(f"\n1. SPIKE IN STABLE STATE (50% utilization):")
    print(f"   Queue recovers quickly after spike ends.")
    print(f"   Post-spike queue depth: {post_spike1_q:.1f}")

    print(f"\n2. SPIKE IN VULNERABLE STATE (90% utilization):")
    print(f"   Queue buildup persists after spike ends - METASTABLE FAILURE.")
    print(f"   Post-spike queue depth: {post_spike2_q:.1f}")

    if post_spike2_q > post_spike1_q * 2:
        print(f"\n   The {post_spike2_q/post_spike1_q:.1f}x higher queue depth after spike 2 demonstrates")
        print(f"   why systems near saturation are vulnerable to metastable failure.")

    print("\n" + "=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metastable failure simulation")
    parser.add_argument("--duration", type=float, default=120.0, help="Simulation duration (s)")
    parser.add_argument("--drain", type=float, default=10.0, help="Drain time after load stops (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (use -1 for random)")
    parser.add_argument("--output", type=str, default="output/metastable", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization generation")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running metastable failure simulation...")
    print(f"  Duration: {args.duration}s + {args.drain}s drain")
    print(f"  Random seed: {seed if seed is not None else 'random'}")

    result = run_metastable_simulation(
        duration_s=args.duration,
        drain_s=args.drain,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
