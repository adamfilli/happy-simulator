"""Metastable failure demonstration with an M/M/1 queue.

This example demonstrates metastable failure behavior in a queuing system:
1. Non-vulnerable state: Low utilization, system recovers quickly from load spikes
2. Vulnerable state: Near saturation (~90% utilization), a spike triggers persistent failure
3. Recovery search: Gradually reduce load to find the recovery threshold

## Architecture Diagram

```
+-----------------------------------------------------------------------------+
|                       METASTABLE FAILURE SIMULATION                          |
+-----------------------------------------------------------------------------+

    LOAD PROFILE (120 seconds total)
    ---------------------------------------------------------------------------

    Rate (req/s)
    15 |                    +---+              +---+
       |                    |   |              |   |
    10 |--------------------+   |--------------+   +----------
       |  Service capacity  |   |              |   |     : step-down
     9 |                    |   |    +---------+   +-----------
       |                    |   |    |         |   |   7   6   5
     5 |   +----------------+   +----+         |   |
       |   |  Low load      |   |  Vulnerable  |   |  Recovery search
       |   |  (stable)      |   |  (near sat.) |   |
     0 +---+----------------+---+--------------+---+------------> Time(s)
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
    +---------------------------------------------------------------------+
    |                                                                     |
    |   +---------+      +-------------------+      +---------+          |
    |   | Source  |----->|      Queue        |----->| Server  |------>   |
    |   |(Poisson)|      |      (FIFO)       |      |(Exp~0.1s)|   Sink  |
    |   +---------+      +-------------------+      +---------+          |
    |       ^                    ^                                       |
    |       |                    |                                       |
    |   Rate varies          Queue depth                                 |
    |   per profile          monitored                                   |
    |                                                                    |
    +--------------------------------------------------------------------+

```

## M/M/1 Queue Theory

With mean service time = 100ms, service rate mu = 10 req/s.
For stability, arrival rate lambda < mu.
Utilization rho = lambda/mu.

Expected queue length = rho^2/(1-rho)
At 50% utilization: E[queue] = 0.5 (very short)
At 90% utilization: E[queue] = 8.1 (significant)
At 95% utilization: E[queue] = 18.05 (very long)

When a spike pushes lambda > mu temporarily, the queue grows linearly.
In the vulnerable state, even a brief spike creates a large queue
that takes a long time to drain (drain rate = mu - lambda).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from happysimulator import (
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    LatencyTracker,
    PoissonArrivalTimeProvider,
    Probe,
    Profile,
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)

if TYPE_CHECKING:
    from collections.abc import Generator

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
    stable_rate: float = 5.0  # 50% utilization
    vulnerable_rate: float = 9.0  # 90% utilization
    spike_rate: float = 15.0  # 150% - overload

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

        completed = self.forward(event, self.downstream, event_type="Completed")
        return [completed]


# =============================================================================
# Main Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the metastable failure simulation."""

    sink: LatencyTracker
    server: MM1Server
    queue_depth_data: Data
    source_generated: int
    profile: MetastableLoadProfile
    summary: SimulationSummary


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
    # Using built-in LatencyTracker instead of custom LatencyTrackingSink
    sink = LatencyTracker(name="Sink")
    server = MM1Server(
        name="Server",
        mean_service_time_s=mean_service_time_s,
        downstream=sink,
    )

    # Create queue depth probe

    queue_probe, queue_depth_data = Probe.on(server, "depth", interval=probe_interval_s)

    # Create load profile and source
    profile = MetastableLoadProfile()
    stop_after = Instant.from_seconds(duration_s)

    provider = RequestProvider(server, stop_after=stop_after)
    arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=[server, sink],
        probes=[queue_probe],
    )
    summary = sim.run()

    return SimulationResult(
        sink=sink,
        server=server,
        queue_depth_data=queue_depth_data,
        source_generated=provider.generated_requests,
        profile=profile,
        summary=summary,
    )


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations of the simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use built-in bucketing instead of manual bucket_latencies()
    latency_buckets = result.sink.summary(window_s=1.0).to_dict()

    q_times = result.queue_depth_data.times()
    q_depths = result.queue_depth_data.raw_values()

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
    time_points = list(range(130))
    rates = [profile.get_rate(Instant.from_seconds(t)) for t in time_points]
    ax1.plot(time_points, rates, "b-", linewidth=2, label="Arrival Rate")
    ax1.axhline(y=10, color="r", linestyle="--", label="Service Capacity (10 req/s)")
    ax1.fill_between(
        [spike1_start, spike1_end], 0, 20, alpha=0.2, color="orange", label="Spike 1 (stable)"
    )
    ax1.fill_between(
        [spike2_start, spike2_end], 0, 20, alpha=0.2, color="red", label="Spike 2 (vulnerable)"
    )
    ax1.axvline(x=vulnerable_start, color="purple", linestyle=":", alpha=0.7)
    ax1.axvline(x=step_down_start, color="green", linestyle=":", alpha=0.7)
    ax1.set_ylabel("Rate (req/s)")
    ax1.set_title("Load Profile: Metastable Failure Demonstration")
    ax1.legend(loc="upper right")
    ax1.set_ylim(0, 18)
    ax1.grid(True, alpha=0.3)

    # Annotations
    ax1.annotate("Stable\n(50%)", xy=(12, 5), ha="center", fontsize=9)
    ax1.annotate("Vulnerable\n(90%)", xy=(45, 9), ha="center", fontsize=9)
    ax1.annotate("Step-down\nRecovery", xy=(95, 5), ha="center", fontsize=9)

    # Middle: Queue depth
    ax2 = axes[1]
    ax2.plot(q_times, q_depths, "g-", linewidth=1, alpha=0.8)
    ax2.fill_between(
        [spike1_start, spike1_end], 0, max(q_depths) if q_depths else 10, alpha=0.2, color="orange"
    )
    ax2.fill_between(
        [spike2_start, spike2_end], 0, max(q_depths) if q_depths else 10, alpha=0.2, color="red"
    )
    ax2.axvline(
        x=vulnerable_start, color="purple", linestyle=":", alpha=0.7, label="Enter vulnerable state"
    )
    ax2.axvline(x=step_down_start, color="green", linestyle=":", alpha=0.7, label="Begin step-down")
    ax2.set_ylabel("Queue Depth")
    ax2.set_title("Queue Depth Over Time")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Bottom: Latency (using built-in bucketed data)
    ax3 = axes[2]
    ax3.plot(latency_buckets["time_s"], latency_buckets["mean"], "b-", linewidth=2, label="Avg")
    ax3.plot(latency_buckets["time_s"], latency_buckets["p99"], "r-", linewidth=1.5, label="p99")
    ax3.fill_between(
        [spike1_start, spike1_end],
        0,
        max(latency_buckets["p99"]) if latency_buckets["p99"] else 1,
        alpha=0.2,
        color="orange",
    )
    ax3.fill_between(
        [spike2_start, spike2_end],
        0,
        max(latency_buckets["p99"]) if latency_buckets["p99"] else 1,
        alpha=0.2,
        color="red",
    )
    ax3.axvline(x=vulnerable_start, color="purple", linestyle=":", alpha=0.7)
    ax3.axvline(x=step_down_start, color="green", linestyle=":", alpha=0.7)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Latency (s)")
    ax3.set_title("End-to-End Latency Over Time")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "metastable_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'metastable_overview.png'}")

    # Figure 2: Detailed analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Spike 1 analysis - using Data.between() instead of manual filtering
    latency_data = result.sink.data
    pre_spike1 = latency_data.between(15, spike1_start)
    post_spike1 = latency_data.between(spike1_end, vulnerable_start)

    ax = axes[0, 0]
    if pre_spike1:
        ax.hist(pre_spike1.raw_values(), bins=30, alpha=0.5, label="Before spike", color="blue")
    if post_spike1:
        ax.hist(post_spike1.raw_values(), bins=30, alpha=0.5, label="After spike", color="green")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title("Spike 1 (Stable State): Latency Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spike 2 analysis
    pre_spike2 = latency_data.between(45, spike2_start)
    post_spike2 = latency_data.between(spike2_end, step_down_start)

    ax = axes[0, 1]
    if pre_spike2:
        ax.hist(pre_spike2.raw_values(), bins=30, alpha=0.5, label="Before spike", color="blue")
    if post_spike2:
        ax.hist(post_spike2.raw_values(), bins=30, alpha=0.5, label="After spike", color="red")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Count")
    ax.set_title("Spike 2 (Vulnerable State): Latency Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recovery analysis
    step_times = [
        step_down_start + i * profile.step_down_duration
        for i in range(len(profile.step_down_rates))
    ]

    ax = axes[1, 0]
    for i, (start_t, _rate) in enumerate(zip(step_times, profile.step_down_rates, strict=False)):
        end_t = start_t + profile.step_down_duration
        step_data = latency_data.between(start_t, end_t)
        if step_data:
            positions = [i + 1]
            ax.boxplot([step_data.raw_values()], positions=positions, widths=0.6)

    ax.set_xticks(range(1, len(profile.step_down_rates) + 1))
    ax.set_xticklabels([f"{r} req/s" for r in profile.step_down_rates])
    ax.set_xlabel("Load Level")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Recovery Search: Latency at Each Step-Down Level")
    ax.grid(True, alpha=0.3, axis="y")

    # Throughput over time - derive from sink data (one event = one completion)
    tp_buckets = result.sink.data.bucket(window_s=1.0)
    tp_times = tp_buckets.times()
    tp_values = tp_buckets.counts()  # completions per window

    ax = axes[1, 1]
    ax.bar(tp_times, tp_values, width=0.8, alpha=0.7)
    ax.axhline(y=10, color="r", linestyle="--", label="Capacity")
    ax.fill_between(
        [spike1_start, spike1_end],
        0,
        max(tp_values) if tp_values else 15,
        alpha=0.2,
        color="orange",
    )
    ax.fill_between(
        [spike2_start, spike2_end], 0, max(tp_values) if tp_values else 15, alpha=0.2, color="red"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Completions/second")
    ax.set_title("Throughput Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

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

    print("\nConfiguration:")
    print("  Service capacity: 10 req/s (mean service time = 100ms)")
    print(
        f"  Stable rate: {profile.stable_rate} req/s ({profile.stable_rate / 10 * 100:.0f}% utilization)"
    )
    print(
        f"  Vulnerable rate: {profile.vulnerable_rate} req/s ({profile.vulnerable_rate / 10 * 100:.0f}% utilization)"
    )
    print(
        f"  Spike rate: {profile.spike_rate} req/s ({profile.spike_rate / 10 * 100:.0f}% utilization)"
    )

    print("\nRequests:")
    print(f"  Generated: {result.source_generated}")
    print(f"  Completed: {result.sink.count}")
    print(f"  Processed by server: {result.server.stats_processed}")

    # Queue depth analysis using Data.between().mean() / .max()
    qd = result.queue_depth_data

    print("\nQueue Depth Analysis:")
    print(
        f"  Stable period (10-25s):     avg={qd.between(10, 25).mean():.1f}, max={qd.between(10, 25).max():.0f}"
    )
    print(
        f"  During spike 1 (25-30s):    avg={qd.between(25, 30).mean():.1f}, max={qd.between(25, 30).max():.0f}"
    )
    print(
        f"  Post-spike 1 (30-35s):      avg={qd.between(30, 35).mean():.1f}, max={qd.between(30, 35).max():.0f}"
    )
    print(
        f"  Vulnerable period (45-55s): avg={qd.between(45, 55).mean():.1f}, max={qd.between(45, 55).max():.0f}"
    )
    print(
        f"  During spike 2 (55-60s):    avg={qd.between(55, 60).mean():.1f}, max={qd.between(55, 60).max():.0f}"
    )
    print(
        f"  Post-spike 2 (60-65s):      avg={qd.between(60, 65).mean():.1f}, max={qd.between(60, 65).max():.0f}"
    )

    # Latency analysis using Data.between() methods
    lat = result.sink.data

    print("\nLatency Analysis (average / p99):")
    print(
        f"  Stable period (10-25s):     {lat.between(10, 25).mean() * 1000:.1f}ms / {lat.between(10, 25).percentile(0.99) * 1000:.1f}ms"
    )
    print(
        f"  Post-spike 1 (30-35s):      {lat.between(30, 35).mean() * 1000:.1f}ms / {lat.between(30, 35).percentile(0.99) * 1000:.1f}ms"
    )
    print(
        f"  Vulnerable period (45-55s): {lat.between(45, 55).mean() * 1000:.1f}ms / {lat.between(45, 55).percentile(0.99) * 1000:.1f}ms"
    )
    print(
        f"  Post-spike 2 (60-65s):      {lat.between(60, 65).mean() * 1000:.1f}ms / {lat.between(60, 65).percentile(0.99) * 1000:.1f}ms"
    )

    # Recovery search analysis
    print("\nRecovery Search (step-down phases):")
    step_start = profile.step_down_start
    for i, rate in enumerate(profile.step_down_rates):
        start_t = step_start + i * profile.step_down_duration
        end_t = start_t + profile.step_down_duration
        avg_lat = lat.between(start_t, end_t).mean()
        avg_q = qd.between(start_t, end_t).mean()
        print(
            f"  {rate} req/s ({rate / 10 * 100:.0f}%): avg_latency={avg_lat * 1000:.1f}ms, avg_queue={avg_q:.1f}"
        )

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("-" * 70)

    # Compare recovery behavior
    post_spike1_q = qd.between(30, 35).mean()
    post_spike2_q = qd.between(60, 65).mean()

    print("\n1. SPIKE IN STABLE STATE (50% utilization):")
    print("   Queue recovers quickly after spike ends.")
    print(f"   Post-spike queue depth: {post_spike1_q:.1f}")

    print("\n2. SPIKE IN VULNERABLE STATE (90% utilization):")
    print("   Queue buildup persists after spike ends - METASTABLE FAILURE.")
    print(f"   Post-spike queue depth: {post_spike2_q:.1f}")

    if post_spike1_q > 0 and post_spike2_q > post_spike1_q * 2:
        print(
            f"\n   The {post_spike2_q / post_spike1_q:.1f}x higher queue depth after spike 2 demonstrates"
        )
        print("   why systems near saturation are vulnerable to metastable failure.")

    # Print auto-generated simulation summary
    print(f"\n{result.summary}")

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
