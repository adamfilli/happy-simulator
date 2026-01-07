"""Test: QueuedEntity with FIFO and LIFO policies.

Scenario:
- Two queued entities (FIFO and LIFO) model a concurrency-limited server.
- Requests arrive according to a Poisson arrival distribution at 10 req/s.
- Each request takes 1 second to process, max concurrency=10.
- Completed requests are forwarded to a final sink entity.

The sink parses each event's trace context and reports latency statistics:
- p0 (min), average, p99

Simulation runs for 60 seconds total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from happysimulator.data.data import Data
from happysimulator.data.probe import Probe
from happysimulator.entities.entity import Entity
from happysimulator.entities.queue_policy import FIFOQueue, LIFOQueue
from happysimulator.entities.queued_entity import QueuedEntity
from happysimulator.events.event import Event
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.poisson_arrival_time_provider import PoissonArrivalTimeProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.simulation import Simulation
from happysimulator.utils.instant import Instant

def _percentile(values: List[float], p: float) -> float:
    """Return the pth percentile using a simple nearest-rank method."""
    if not values:
        raise ValueError("values must be non-empty")
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)

    ordered = sorted(values)
    idx = int(round(p * (len(ordered) - 1)))
    return ordered[idx]


@dataclass
class LatencyStats:
    p0_s: float
    p50_s: float
    avg_s: float
    p99_s: float
    count: int

class LatencySink(Entity):
    """Final destination that parses traces and computes latency stats."""

    def __init__(self, name: str):
        super().__init__(name)
        self.latencies_s: List[float] = []
        self.completion_times_s: List[float] = []  # When each request completed
        self.events: List[Event] = []

    def handle_event(self, event: Event):
        self.events.append(event)

        # Time in system = event completion time - original creation time.
        created_at: Instant = event.context.get("created_at", event.time)
        latency = (event.time - created_at).to_seconds()
        self.latencies_s.append(latency)
        self.completion_times_s.append(event.time.to_seconds())
        return []

    def stats(self) -> LatencyStats:
        if not self.latencies_s:
            return LatencyStats(p0_s=0.0, p50_s=0.0, avg_s=0.0, p99_s=0.0, count=0)
        avg = sum(self.latencies_s) / len(self.latencies_s)
        return LatencyStats(
            p0_s=_percentile(self.latencies_s, 0.0),
            p50_s=_percentile(self.latencies_s, 0.50),
            avg_s=avg,
            p99_s=_percentile(self.latencies_s, 0.99),
            count=len(self.latencies_s),
        )


class ConstantTwentyPerSecondProfile(Profile):
    """Returns a constant 20 events per second for 60 seconds."""

    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(60):
            return 20.0
        return 0.0


class SingleEventProfile(Profile):
    """Returns a single event at the start, then nothing."""

    def get_rate(self, time: Instant) -> float:
        # High rate for a very short window to produce exactly one event.
        if time <= Instant.from_seconds(0.001):
            return 1000.0  # 1 event in 0.001s window
        return 0.0


class RequestProvider(EventProvider):
    """Provides Request events targeting the queued server."""

    def __init__(self, server: Entity):
        super().__init__()
        self._server = server

    def get_events(self, time: Instant) -> List[Event]:
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._server,
            )
        ]

class ForwardingQueuedServer(QueuedEntity):
    """QueuedEntity that simulates 1s service and forwards to a sink."""

    def __init__(self, name: str, sink: LatencySink, queue_policy, service_time_s: float = 1.0):
        super().__init__(
            name=name,
            concurrency=10,
            queue=queue_policy,
        )
        self._sink = sink
        self._service_time_s = service_time_s

    def process_item(self, item: Event):
        # Simulate service time - yields control back to simulation loop.
        yield self._service_time_s

        # Completion time = original time + total yielded delay.
        completion_time = item.time + Instant.from_seconds(self._service_time_s)

        # Forward to sink with correct completion timestamp.
        forwarded = Event(
            time=completion_time,
            event_type="Completed",
            target=self._sink,
            context=item.context,
        )
        return [forwarded]


@dataclass
class CaseResult:
    stats: LatencyStats
    latencies_s: List[float]
    completion_times_s: List[float]  # Time series: when each request completed
    # Probe data: list of (time_s, value) tuples
    queue_depth_samples: List[tuple[float, float]]
    utilization_samples: List[tuple[float, float]]


def _run_case(queue_kind: str, single_event: bool = False) -> CaseResult:
    numpy = pytest.importorskip("numpy")
    pytest.importorskip("scipy")

    sink = LatencySink(name=f"sink_{queue_kind}")
    if queue_kind == "fifo":
        queue_policy = FIFOQueue(capacity=float("inf"))
    elif queue_kind == "lifo":
        queue_policy = LIFOQueue(capacity=float("inf"))
    else:
        raise ValueError(queue_kind)

    server = ForwardingQueuedServer(
        name=f"server_{queue_kind}",
        sink=sink,
        queue_policy=queue_policy,
        service_time_s=1.0,
    )

    # Create probes to track queue depth and utilization
    queue_depth_data = Data()
    utilization_data = Data()
    
    queue_depth_probe = Probe(
        target=server,
        metric="queue_depth",
        data=queue_depth_data,
        interval=0.1,  # Sample every 100ms
    )
    utilization_probe = Probe(
        target=server,
        metric="active_workers",
        data=utilization_data,
        interval=0.1,
    )

    if single_event:
        profile = SingleEventProfile()
        end_time = Instant.from_seconds(5.0)  # Enough time for 1s service + buffer
    else:
        profile = ConstantTwentyPerSecondProfile()
        end_time = Instant.from_seconds(60.0)

    provider = RequestProvider(server)
    arrival_time_provider = PoissonArrivalTimeProvider(profile, Instant.Epoch)

    source = Source(
        name=f"source_{queue_kind}",
        event_provider=provider,
        arrival_time_provider=arrival_time_provider,
    )

    sim = Simulation(
        sources=[source],
        entities=[server, sink],
        probes=[queue_depth_probe, utilization_probe],
        end_time=end_time,
    )
    sim.run()

    stats = sink.stats()
    print(
        f"\n{queue_kind.upper()} results: n={stats.count} p0={stats.p0_s:.3f}s p50={stats.p50_s:.3f}s avg={stats.avg_s:.3f}s p99={stats.p99_s:.3f}s"
    )
    return CaseResult(
        stats=stats,
        latencies_s=sink.latencies_s.copy(),
        completion_times_s=sink.completion_times_s.copy(),
        queue_depth_samples=queue_depth_data.values.copy(),
        utilization_samples=utilization_data.values.copy(),
    )


def test_queued_entity_fifo_vs_lifo_latency_stats(test_output_dir):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    fifo_result = _run_case("fifo")
    lifo_result = _run_case("lifo")

    fifo_stats = fifo_result.stats
    lifo_stats = lifo_result.stats

    # Sanity: we should complete a substantial number of requests in 60s.
    assert fifo_stats.count > 100
    assert lifo_stats.count > 100

    # Service time is 1s, so minimum observed latency should be ~1s.
    assert fifo_stats.p0_s >= 0.99
    assert lifo_stats.p0_s >= 0.99

    # Under stable load (10/s in, 10/s out), tail latency should remain bounded.
    # Keep assertions loose to avoid flakiness across platforms.
    assert fifo_stats.avg_s < 10.0
    assert lifo_stats.avg_s < 10.0
    assert fifo_stats.p99_s < 30.0
    assert lifo_stats.p99_s < 30.0

    # --- Visualization ---

    # Plot 1: Latency histogram comparison
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

    max_latency = max(max(fifo_result.latencies_s), max(lifo_result.latencies_s))
    bins = [i * 0.5 for i in range(int(max_latency / 0.5) + 3)]  # 0.5s bins

    axes[0].hist(fifo_result.latencies_s, bins=bins, alpha=0.7, label="FIFO", edgecolor="black")
    axes[0].axvline(fifo_stats.p50_s, color="blue", linestyle="--", label=f"p50={fifo_stats.p50_s:.2f}s")
    axes[0].axvline(fifo_stats.p99_s, color="red", linestyle="--", label=f"p99={fifo_stats.p99_s:.2f}s")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"FIFO Latency Distribution (n={fifo_stats.count}, avg={fifo_stats.avg_s:.2f}s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(lifo_result.latencies_s, bins=bins, alpha=0.7, label="LIFO", color="orange", edgecolor="black")
    axes[1].axvline(lifo_stats.p50_s, color="blue", linestyle="--", label=f"p50={lifo_stats.p50_s:.2f}s")
    axes[1].axvline(lifo_stats.p99_s, color="red", linestyle="--", label=f"p99={lifo_stats.p99_s:.2f}s")
    axes[1].set_xlabel("Latency (s)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"LIFO Latency Distribution (n={lifo_stats.count}, avg={lifo_stats.avg_s:.2f}s)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(test_output_dir / "latency_histograms.png", dpi=150)
    plt.close(fig)

    # Plot 2: CDF comparison (empirical cumulative distribution)
    fig2, ax = plt.subplots(figsize=(10, 6))

    for label, latencies, color in [
        ("FIFO", sorted(fifo_result.latencies_s), "blue"),
        ("LIFO", sorted(lifo_result.latencies_s), "orange"),
    ]:
        n = len(latencies)
        cdf_y = [(i + 1) / n for i in range(n)]
        ax.step(latencies, cdf_y, where="post", label=label, color=color, linewidth=2)

    # Mark key percentiles
    for p, style in [(0.50, "--"), (0.99, ":")]:
        ax.axhline(p, color="gray", linestyle=style, alpha=0.5)
        ax.text(0.5, p + 0.02, f"p{int(p*100)}", fontsize=9, color="gray")

    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("FIFO vs LIFO: Latency CDF Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)

    fig2.tight_layout()
    fig2.savefig(test_output_dir / "latency_cdf.png", dpi=150)
    plt.close(fig2)

    # Plot 3: Summary bar chart comparing percentiles
    fig3, ax = plt.subplots(figsize=(8, 5))

    metrics = ["p0", "p50", "avg", "p99"]
    fifo_vals = [fifo_stats.p0_s, fifo_stats.p50_s, fifo_stats.avg_s, fifo_stats.p99_s]
    lifo_vals = [lifo_stats.p0_s, lifo_stats.p50_s, lifo_stats.avg_s, lifo_stats.p99_s]

    x = range(len(metrics))
    width = 0.35
    ax.bar([i - width/2 for i in x], fifo_vals, width, label="FIFO", color="blue", alpha=0.7)
    ax.bar([i + width/2 for i in x], lifo_vals, width, label="LIFO", color="orange", alpha=0.7)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Latency (s)")
    ax.set_title("FIFO vs LIFO: Latency Metrics Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (fv, lv) in enumerate(zip(fifo_vals, lifo_vals)):
        ax.text(i - width/2, fv + 0.1, f"{fv:.2f}", ha="center", fontsize=8)
        ax.text(i + width/2, lv + 0.1, f"{lv:.2f}", ha="center", fontsize=8)

    fig3.tight_layout()
    fig3.savefig(test_output_dir / "latency_metrics_comparison.png", dpi=150)
    plt.close(fig3)

    # Plot 4: p50 and p99 latency over time (rolling window)
    def compute_rolling_percentiles(
        completion_times: List[float],
        latencies: List[float],
        window_s: float = 5.0,
        step_s: float = 1.0,
        duration_s: float = 60.0,
    ) -> tuple[List[float], List[float], List[float]]:
        """Compute rolling p50 and p99 latency over time windows.
        
        Returns:
            (time_centers, p50_values, p99_values)
        """
        import numpy as np

        times = np.array(completion_times)
        lats = np.array(latencies)
        
        time_centers = []
        p50_values = []
        p99_values = []
        
        t = window_s / 2  # Start at center of first full window
        while t <= duration_s - window_s / 2:
            # Find events in window [t - window_s/2, t + window_s/2]
            mask = (times >= t - window_s / 2) & (times < t + window_s / 2)
            window_lats = lats[mask]
            
            if len(window_lats) >= 5:  # Need enough samples for meaningful percentile
                time_centers.append(t)
                p50_values.append(float(np.percentile(window_lats, 50)))
                p99_values.append(float(np.percentile(window_lats, 99)))
            
            t += step_s
        
        return time_centers, p50_values, p99_values

    fig4, (ax_p50, ax_p99) = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)

    for label, result, color in [
        ("FIFO", fifo_result, "blue"),
        ("LIFO", lifo_result, "orange"),
    ]:
        times, p50s, p99s = compute_rolling_percentiles(
            result.completion_times_s,
            result.latencies_s,
            window_s=5.0,
            step_s=1.0,
        )
        ax_p50.plot(times, p50s, label=label, color=color, linewidth=2)
        ax_p99.plot(times, p99s, label=label, color=color, linewidth=2)

    ax_p50.set_ylabel("p50 Latency (s)")
    ax_p50.set_title("p50 Latency Over Time (5s rolling window)")
    ax_p50.legend()
    ax_p50.grid(True, alpha=0.3)
    ax_p50.axhline(1.0, color="gray", linestyle=":", alpha=0.5, label="Service time")

    ax_p99.set_xlabel("Simulation Time (s)")
    ax_p99.set_ylabel("p99 Latency (s)")
    ax_p99.set_title("p99 Latency Over Time (5s rolling window)")
    ax_p99.legend()
    ax_p99.grid(True, alpha=0.3)
    ax_p99.axhline(1.0, color="gray", linestyle=":", alpha=0.5)

    fig4.tight_layout()
    fig4.savefig(test_output_dir / "latency_over_time.png", dpi=150)
    plt.close(fig4)

    # Plot 5: Combined view - scatter plot of individual latencies over time
    fig5, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        fifo_result.completion_times_s,
        fifo_result.latencies_s,
        alpha=0.3,
        s=10,
        label="FIFO",
        color="blue",
    )
    ax.scatter(
        lifo_result.completion_times_s,
        lifo_result.latencies_s,
        alpha=0.3,
        s=10,
        label="LIFO",
        color="orange",
    )

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Service time (1s)")
    ax.set_xlabel("Completion Time (s)")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Individual Request Latencies Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)

    fig5.tight_layout()
    fig5.savefig(test_output_dir / "latency_scatter.png", dpi=150)
    plt.close(fig5)

    # Plot 6: Queue depth over time
    fig6, ax = plt.subplots(figsize=(12, 5))

    for label, result, color in [
        ("FIFO", fifo_result, "blue"),
        ("LIFO", lifo_result, "orange"),
    ]:
        times = [t for t, _ in result.queue_depth_samples]
        depths = [v for _, v in result.queue_depth_samples]
        ax.plot(times, depths, label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Queue Depth")
    ax.set_title("Queue Depth Over Time (sampled every 100ms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(bottom=0)

    fig6.tight_layout()
    fig6.savefig(test_output_dir / "queue_depth.png", dpi=150)
    plt.close(fig6)

    # Plot 7: Utilization (active workers) over time
    fig7, ax = plt.subplots(figsize=(12, 5))

    concurrency_limit = 10  # Matches ForwardingQueuedServer concurrency

    for label, result, color in [
        ("FIFO", fifo_result, "blue"),
        ("LIFO", lifo_result, "orange"),
    ]:
        times = [t for t, _ in result.utilization_samples]
        active = [v for _, v in result.utilization_samples]
        # Convert to utilization percentage
        utilization_pct = [100.0 * v / concurrency_limit for v in active]
        ax.plot(times, utilization_pct, label=label, color=color, linewidth=1.5, alpha=0.8)

    ax.axhline(100.0, color="red", linestyle="--", alpha=0.5, label="Max capacity")
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Utilization (%)")
    ax.set_title(f"Server Utilization Over Time (max concurrency={concurrency_limit})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 110)

    fig7.tight_layout()
    fig7.savefig(test_output_dir / "utilization.png", dpi=150)
    plt.close(fig7)

    # Plot 8: Combined queue depth and utilization (dual y-axis)
    fig8, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    for label, result, color, linestyle in [
        ("FIFO", fifo_result, "blue", "-"),
        ("LIFO", lifo_result, "orange", "-"),
    ]:
        times_qd = [t for t, _ in result.queue_depth_samples]
        depths = [v for _, v in result.queue_depth_samples]
        times_util = [t for t, _ in result.utilization_samples]
        active = [v for _, v in result.utilization_samples]

        ax1.plot(times_qd, depths, label=f"{label} Queue Depth", color=color, 
                 linestyle=linestyle, linewidth=1.5, alpha=0.8)
        ax2.plot(times_util, active, label=f"{label} Active Workers", color=color,
                 linestyle="--", linewidth=1.5, alpha=0.6)

    ax1.set_xlabel("Simulation Time (s)")
    ax1.set_ylabel("Queue Depth", color="black")
    ax2.set_ylabel("Active Workers", color="gray")
    ax1.set_title("Queue Depth and Active Workers Over Time")
    ax1.set_xlim(0, 60)
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(0, concurrency_limit + 1)
    ax2.axhline(concurrency_limit, color="red", linestyle=":", alpha=0.5)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.grid(True, alpha=0.3)

    fig8.tight_layout()
    fig8.savefig(test_output_dir / "queue_and_utilization.png", dpi=150)
    plt.close(fig8)

    print(f"\nSaved visualization to: {test_output_dir}")


def test_queued_entity_single_event_debug():
    """Debug test: sends a single event through the system for easier tracing."""
    result = _run_case("fifo", single_event=True)

    # Should have processed exactly 1 event.
    assert result.stats.count == 1

    # Service time is 1s, so latency should be ~1s.
    assert result.stats.p0_s >= 0.99

