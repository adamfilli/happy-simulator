"""GC-induced metastable collapse demonstration.

This example demonstrates how a GC (garbage collection) pause exceeding client
timeout triggers non-recoverable metastable collapse, even at moderate utilization.

Unlike load-spike metastability (which requires high utilization), GC-induced
collapse occurs at moderate load because:
1. GC pause > client timeout causes ALL in-flight requests to timeout
2. Retries create a "retry storm" that overwhelms recovery capacity
3. Queue compounds faster than it drains, leading to collapse

## Architecture Diagram

```
    REQUEST FLOW (70% utilization baseline)

    +-------------+      Request        +---------------------------------+
    |   Source    |-------------------->|         Retrying Client         |
    | (Constant)  |                     |                                 |
    |  7 req/s    |                     |  Timeout: 500ms                 |
    +-------------+                     |  Max Retries: 3                 |
                                        |  Retry Delay: 50ms              |
                                        +-----------------+---------------+
                                                          |
                                                          v
                                        +---------------------------------+
                                        |        GC-Aware Server          |
                                        |  +---------+   +-------------+  |
                                        |  |  Queue  |-->|   Worker    |  |
                                        |  | (FIFO)  |   | (100ms svc) |  |
                                        |  +---------+   +-------------+  |
                                        |         ^                       |
                                        |    GC Pause: 1.0s at t=30s      |
                                        +-----------------+---------------+
                                                          |
                                                          v
                                        +---------------------------------+
                                        |              Sink               |
                                        |  (Tracks latency, goodput)      |
                                        +---------------------------------+
```

## Collapse Mechanism

**Before GC** (t=0-30s, stable, zero timeouts):
- Arrival: 7 req/s, Service: 10 req/s (constant 100ms)
- 30% headroom, queue always empty
- Timeout 500ms = 5x service time, never triggers under normal load

**Single GC at t=30s** (1.0s pause):
- Server STOPS (processing = 0)
- ~7 new requests queue up
- ALL in-flight timeout (1.0s > 500ms timeout)
- Clients retry timed-out requests

**After GC** (t=31s+, unrecoverable):
- Queue = original arrivals + retry storm
- Retries amplify load ~4x → effective arrival 28 req/s vs 10 req/s capacity
- Queue grows indefinitely, system never recovers

## Parameters

| Parameter      | Value  | Rationale                                  |
|----------------|--------|--------------------------------------------|
| Arrival rate   | 7 req/s| 70% utilization - clearly below saturation |
| Service time   | 100ms  | Constant (deterministic) - no timeouts in steady state |
| Client timeout | 500ms  | 5x service time - generous, never triggers normally |
| GC at t=30s    | 1.0s   | Single pause, 2x timeout - guarantees all in-flight timeout |
| Max retries    | 3      | Amplifies load ~4x, overwhelming 30% headroom |
| Retry delay    | 50ms   | Fixed, contributes to retry storm          |
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    Probe,
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.analysis import analyze, detect_phases


# =============================================================================
# GC-Aware Server
# =============================================================================


class GCServer(QueuedResource):
    """Server that experiences periodic stop-the-world GC pauses.

    Simulates stop-the-world garbage collection where all processing halts
    for the duration of the GC pause. This can trigger metastable failure
    if GC duration exceeds client timeout.

    GC Schedule:
    - First GC at `gc_start_time_s`
    - Subsequent GCs at `gc_start_time_s + n * gc_interval_s`
    - Each GC pauses processing for `gc_duration_s`

    Args:
        name: Entity name
        service_time_s: Constant service time per request (default 100ms)
        gc_interval_s: Time between GC events (default 999 = single GC)
        gc_duration_s: Duration of each GC pause (default 1.0s)
        gc_start_time_s: When the GC occurs (default 30s)
        downstream: Entity to forward completed events to
        concurrency: Number of concurrent workers (default 1)
    """

    def __init__(
        self,
        name: str,
        *,
        service_time_s: float = 0.1,
        gc_interval_s: float = 999.0,
        gc_duration_s: float = 1.0,
        gc_start_time_s: float = 30.0,
        downstream: Entity | None = None,
        concurrency: int = 1,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.service_time_s = service_time_s
        self.gc_interval_s = gc_interval_s
        self.gc_duration_s = gc_duration_s
        self.gc_start_time_s = gc_start_time_s
        self.downstream = downstream
        self.concurrency = concurrency
        self._in_flight: int = 0

        # Stats
        self.stats_processed: int = 0
        self.stats_gc_pauses: int = 0

        # GC event times for visualization
        self.gc_events: list[tuple[float, float]] = []  # (start_time, end_time)

    def _get_gc_pause_remaining(self) -> float:
        """Calculate remaining GC pause time if currently in a GC window.

        Returns 0 if not in a GC pause, otherwise returns seconds remaining.
        """
        current_time = self.now.to_seconds()

        if current_time < self.gc_start_time_s:
            return 0.0

        time_since_first_gc = current_time - self.gc_start_time_s
        gc_cycle_position = time_since_first_gc % self.gc_interval_s

        if gc_cycle_position < self.gc_duration_s:
            return self.gc_duration_s - gc_cycle_position

        return 0.0

    def _record_gc_event(self) -> None:
        """Record a GC event for visualization."""
        current_time = self.now.to_seconds()
        time_since_first_gc = current_time - self.gc_start_time_s
        gc_cycle_position = time_since_first_gc % self.gc_interval_s

        gc_start = current_time - gc_cycle_position
        gc_end = gc_start + self.gc_duration_s

        # Only record if not already recorded
        if not self.gc_events or self.gc_events[-1][0] != gc_start:
            self.gc_events.append((gc_start, gc_end))

    def has_capacity(self) -> bool:
        return self._in_flight < self.concurrency

    def handle_queued_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]]:
        """Process request, pausing for GC if necessary."""
        self._in_flight += 1
        try:
            # Check if we're in a GC pause
            gc_remaining = self._get_gc_pause_remaining()
            if gc_remaining > 0:
                self.stats_gc_pauses += 1
                self._record_gc_event()
                yield gc_remaining, None  # Server completely stops

            # Constant service time (deterministic - no variance-induced timeouts)
            yield self.service_time_s, None

            self.stats_processed += 1

            if self.downstream is None:
                return []

            # Forward completion to downstream
            completed = self.forward(event, self.downstream, event_type="Completion")
            return [completed]
        finally:
            self._in_flight -= 1


# =============================================================================
# Retrying Client with Statistics
# =============================================================================


@dataclass
class InFlightRequest:
    """Tracks state for an in-flight request."""

    request_id: int
    created_at: Instant
    attempt: int
    timeout_event_id: int


class RetryingClientWithStats(Entity):
    """Client with timeout-based retries and detailed statistics.

    Sends requests with configurable retry behavior. When a timeout fires
    before the server responds, the client retries (up to max_retries).

    Args:
        name: Entity name
        server: Target server entity
        timeout_s: Client timeout before retry (default 500ms)
        max_retries: Maximum retry attempts per request (default 3)
        retry_delay_s: Delay before sending retry (default 50ms)
        retry_enabled: Whether retries are enabled (default True)
    """

    def __init__(
        self,
        name: str,
        *,
        server: Entity,
        timeout_s: float = 0.5,
        max_retries: int = 3,
        retry_delay_s: float = 0.05,
        retry_enabled: bool = True,
    ):
        super().__init__(name)
        self.server = server
        self.timeout_s = timeout_s
        self.max_retries = max_retries if retry_enabled else 0
        self.retry_delay_s = retry_delay_s
        self.retry_enabled = retry_enabled

        # In-flight tracking
        self._in_flight: dict[int, InFlightRequest] = {}
        self._next_timeout_id: int = 0

        # Core stats
        self.stats_requests_received: int = 0
        self.stats_attempts_sent: int = 0
        self.stats_completions: int = 0
        self.stats_timeouts: int = 0
        self.stats_retries: int = 0
        self.stats_gave_up: int = 0

        # Latency tracking
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []
        self.attempts_per_request: list[int] = []

        # Time series for retry amplification visualization
        self.attempts_by_time: list[tuple[float, int]] = []

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)

    def goodput_time_series(
        self, bucket_size_s: float = 1.0
    ) -> tuple[list[float], list[int]]:
        """Return (bucket_times, completion_counts) for plotting goodput."""
        if not self.completion_times:
            return [], []

        buckets: dict[int, int] = defaultdict(int)
        for t in self.completion_times:
            bucket = int(t.to_seconds() / bucket_size_s)
            buckets[bucket] += 1

        sorted_buckets = sorted(buckets.keys())
        bucket_times = [b * bucket_size_s for b in sorted_buckets]
        counts = [buckets[b] for b in sorted_buckets]

        return bucket_times, counts

    def handle_event(self, event: Event) -> list[Event]:
        """Handle incoming request, completion, timeout, or retry events."""
        event_type = event.event_type

        if event_type == "NewRequest":
            return self._handle_new_request(event)
        elif event_type == "Completion":
            return self._handle_completion(event)
        elif event_type == "Timeout":
            return self._handle_timeout(event)
        elif event_type == "DoRetry":
            return self._handle_do_retry(event)

        return []

    def _handle_new_request(self, event: Event) -> list[Event]:
        """Handle a new request from the source."""
        self.stats_requests_received += 1
        request_id = event.context.get("request_id", self.stats_requests_received)

        return self._send_request(request_id, event.time, attempt=1)

    def _send_request(
        self, request_id: int, created_at: Instant, attempt: int
    ) -> list[Event]:
        """Send a request to the server and schedule a timeout."""
        self.stats_attempts_sent += 1
        self._next_timeout_id += 1
        timeout_id = self._next_timeout_id

        # Record attempt for amplification tracking
        self.attempts_by_time.append((self.now.to_seconds(), attempt))

        # Track this request
        self._in_flight[request_id] = InFlightRequest(
            request_id=request_id,
            created_at=created_at,
            attempt=attempt,
            timeout_event_id=timeout_id,
        )

        # Create request event for server
        server_request = Event(
            time=self.now,
            event_type="Request",
            target=self.server,
            context={
                "request_id": request_id,
                "created_at": created_at,
                "attempt": attempt,
                "client": self,
            },
        )

        # Schedule timeout
        timeout_event = Event(
            time=self.now + self.timeout_s,
            event_type="Timeout",
            target=self,
            context={
                "request_id": request_id,
                "timeout_id": timeout_id,
            },
        )

        return [server_request, timeout_event]

    def _handle_completion(self, event: Event) -> list[Event]:
        """Handle completion from server."""
        request_id = event.context.get("request_id")

        if request_id not in self._in_flight:
            # Already timed out and retried, or duplicate completion
            return []

        in_flight = self._in_flight.pop(request_id)
        self.stats_completions += 1

        # Record latency from original creation time
        original_created_at = event.context.get("created_at", in_flight.created_at)
        latency_s = (event.time - original_created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)
        self.attempts_per_request.append(in_flight.attempt)

        return []

    def _handle_timeout(self, event: Event) -> list[Event]:
        """Handle timeout - schedule retry if enabled and within limits."""
        request_id = event.context.get("request_id")
        timeout_id = event.context.get("timeout_id")

        if request_id not in self._in_flight:
            # Already completed, ignore this timeout
            return []

        in_flight = self._in_flight[request_id]

        # Check if this timeout matches the current attempt
        if in_flight.timeout_event_id != timeout_id:
            # Stale timeout from a previous attempt
            return []

        self.stats_timeouts += 1

        # Check retry limit
        if in_flight.attempt >= self.max_retries + 1:
            # +1 because attempt starts at 1
            del self._in_flight[request_id]
            self.stats_gave_up += 1
            return []

        # Schedule retry after delay (don't remove from in_flight yet)
        self.stats_retries += 1

        retry_event = Event(
            time=self.now + self.retry_delay_s,
            event_type="DoRetry",
            target=self,
            context={
                "request_id": request_id,
                "created_at": in_flight.created_at,
                "next_attempt": in_flight.attempt + 1,
            },
        )

        # Remove from in-flight (will be re-added on retry)
        del self._in_flight[request_id]

        return [retry_event]

    def _handle_do_retry(self, event: Event) -> list[Event]:
        """Execute a scheduled retry."""
        request_id = event.context.get("request_id")
        created_at = event.context.get("created_at")
        next_attempt = event.context.get("next_attempt")

        return self._send_request(request_id, created_at, attempt=next_attempt)


# =============================================================================
# Event Provider
# =============================================================================


class ClientRequestProvider(EventProvider):
    """Generates request events targeting the retrying client."""

    def __init__(self, client: Entity, *, stop_after: Instant | None = None):
        self._client = client
        self._stop_after = stop_after
        self._request_id: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self._request_id += 1
        return [
            Event(
                time=time,
                event_type="NewRequest",
                target=self._client,
                context={"request_id": self._request_id},
            )
        ]


# =============================================================================
# Simulation Results
# =============================================================================


@dataclass
class ScenarioResult:
    """Results from a single simulation scenario."""

    client: RetryingClientWithStats
    server: GCServer
    queue_depth_data: Data
    requests_generated: int
    retry_enabled: bool
    summary: SimulationSummary | None = None


@dataclass
class ComparisonResult:
    """Results comparing with-retry vs without-retry scenarios."""

    with_retries: ScenarioResult
    without_retries: ScenarioResult


# =============================================================================
# Simulation Functions
# =============================================================================


def run_gc_collapse_simulation(
    *,
    duration_s: float = 60.0,
    drain_s: float = 10.0,
    arrival_rate: float = 7.0,
    service_time_s: float = 0.1,
    timeout_s: float = 0.5,
    max_retries: int = 3,
    retry_delay_s: float = 0.05,
    gc_interval_s: float = 999.0,
    gc_duration_s: float = 1.0,
    gc_start_time_s: float = 30.0,
    probe_interval_s: float = 0.1,
    retry_enabled: bool = True,
    seed: int | None = 42,
) -> ScenarioResult:
    """Run a GC collapse simulation scenario.

    Args:
        duration_s: How long to generate load
        drain_s: Extra time for in-flight requests to complete
        arrival_rate: Requests per second
        service_time_s: Constant server processing time per request
        timeout_s: Client timeout before retry
        max_retries: Maximum retry attempts per request
        retry_delay_s: Delay before sending retry
        gc_interval_s: Time between GC events
        gc_duration_s: Duration of each GC pause
        gc_start_time_s: When the first GC occurs
        probe_interval_s: Queue depth sampling interval
        retry_enabled: Whether retries are enabled
        seed: Random seed for reproducibility
    """
    if seed is not None:
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)

    # Create server with GC pauses
    server = GCServer(
        name="Server",
        service_time_s=service_time_s,
        gc_interval_s=gc_interval_s,
        gc_duration_s=gc_duration_s,
        gc_start_time_s=gc_start_time_s,
        downstream=None,  # Client handles completions via callback
    )

    # Create retrying client
    client = RetryingClientWithStats(
        name="Client",
        server=server,
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_delay_s=retry_delay_s,
        retry_enabled=retry_enabled,
    )

    # Server sends completions back to client
    server.downstream = client

    # Create queue depth probe

    queue_probe, queue_depth_data = Probe.on(server, "depth", interval=probe_interval_s)

    # Create source (constant arrivals for deterministic pre-GC stability)
    stop_after = Instant.from_seconds(duration_s)
    provider = ClientRequestProvider(client, stop_after=stop_after)
    profile = ConstantRateProfile(rate=arrival_rate)
    arrival = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)
    source = Source(name="Source", event_provider=provider, arrival_time_provider=arrival)

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + drain_s,
        sources=[source],
        entities=[client, server],
        probes=[queue_probe],
    )
    summary = sim.run()

    return ScenarioResult(
        client=client,
        server=server,
        queue_depth_data=queue_depth_data,
        requests_generated=provider._request_id,
        retry_enabled=retry_enabled,
        summary=summary,
    )


def run_comparison(
    *,
    duration_s: float = 100.0,
    seed: int = 42,
    **kwargs,
) -> ComparisonResult:
    """Run comparison between with-retry and without-retry scenarios.

    Uses the same random seed for both scenarios for fair comparison.
    """
    # Run with retries
    with_retries = run_gc_collapse_simulation(
        duration_s=duration_s,
        retry_enabled=True,
        seed=seed,
        **kwargs,
    )

    # Run without retries (same seed for comparable arrivals)
    without_retries = run_gc_collapse_simulation(
        duration_s=duration_s,
        retry_enabled=False,
        seed=seed,
        **kwargs,
    )

    return ComparisonResult(
        with_retries=with_retries,
        without_retries=without_retries,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _build_latency_data(client: RetryingClientWithStats) -> Data:
    """Build a Data object from client latency measurements for analysis."""
    d = Data()
    for t, lat in zip(client.completion_times, client.latencies_s):
        d.add_stat(lat, t)
    return d


def get_final_queue_depth(data: Data) -> int:
    """Get the final queue depth from probe data."""
    if not data.values:
        return 0
    return int(data.values[-1][1])


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: ComparisonResult, output_dir: Path) -> None:
    """Generate visualizations of the comparison results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    with_r = result.with_retries
    without_r = result.without_retries

    # Get GC events from server (both should have same GC timing)
    gc_events = with_r.server.gc_events

    # Get queue depth data
    q_times_with = [t for (t, _) in with_r.queue_depth_data.values]
    q_depths_with = [v for (_, v) in with_r.queue_depth_data.values]

    q_times_without = [t for (t, _) in without_r.queue_depth_data.values]
    q_depths_without = [v for (_, v) in without_r.queue_depth_data.values]

    # Get goodput data
    gp_times_with, gp_counts_with = with_r.client.goodput_time_series(bucket_size_s=1.0)
    gp_times_without, gp_counts_without = without_r.client.goodput_time_series(
        bucket_size_s=1.0
    )

    # =========================================================================
    # Figure 1: Overview (3 subplots, shared x-axis)
    # =========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(
        "GC-Induced Metastable Collapse: With Retries vs Without",
        fontsize=14,
        fontweight="bold",
    )

    # Plot 1: GC Events Timeline
    ax1 = axes[0]
    for gc_start, gc_end in gc_events:
        ax1.axvspan(gc_start, gc_end, alpha=0.3, color="red", label="_nolegend_")
        ax1.axvline(x=gc_start, color="red", linestyle="--", alpha=0.5)

    # Add a single legend entry for GC
    if gc_events:
        gc_dur_ms = with_r.server.gc_duration_s * 1000
        ax1.axvspan(0, 0, alpha=0.3, color="red", label=f"GC Pause ({gc_dur_ms:.0f}ms)")

    ax1.set_ylabel("GC Events")
    gc_dur_ms = with_r.server.gc_duration_s * 1000
    ax1.set_title(f"GC Pause ({gc_dur_ms:.0f}ms at t={with_r.server.gc_start_time_s:.0f}s)")
    if ax1.get_legend_handles_labels()[1]:
        ax1.legend(loc="upper right")
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Queue Depth
    ax2 = axes[1]
    ax2.plot(q_times_with, q_depths_with, "r-", linewidth=1.5, label="With Retries")
    ax2.plot(
        q_times_without, q_depths_without, "g-", linewidth=1.5, label="Without Retries"
    )

    for gc_start, gc_end in gc_events:
        ax2.axvspan(gc_start, gc_end, alpha=0.1, color="red")

    ax2.set_ylabel("Queue Depth")
    ax2.set_title("Server Queue Depth Over Time")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Goodput
    ax3 = axes[2]
    if gp_times_with and gp_counts_with:
        ax3.plot(
            gp_times_with, gp_counts_with, "r-", linewidth=1.5, label="With Retries"
        )
    if gp_times_without and gp_counts_without:
        ax3.plot(
            gp_times_without,
            gp_counts_without,
            "g-",
            linewidth=1.5,
            label="Without Retries",
        )

    for gc_start, gc_end in gc_events:
        ax3.axvspan(gc_start, gc_end, alpha=0.1, color="red")

    ax3.axhline(y=7, color="blue", linestyle="--", alpha=0.5, label="Arrival Rate (7/s)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Completions/second")
    ax3.set_title("Goodput Over Time (Successful Completions)")
    ax3.legend(loc="upper right")
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "gc_collapse_overview.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'gc_collapse_overview.png'}")

    # =========================================================================
    # Figure 2: Analysis (2x2 grid)
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GC Collapse Analysis", fontsize=14, fontweight="bold")

    # Plot 1: Retry Amplification Over Time
    ax1 = axes[0, 0]

    # Bucket attempts by time and calculate amplification
    bucket_size = 5.0  # 5-second buckets
    attempts_with = with_r.client.attempts_by_time
    requests_buckets: dict[int, int] = defaultdict(int)
    attempts_buckets: dict[int, int] = defaultdict(int)

    for t_s, attempt in attempts_with:
        bucket = int(t_s / bucket_size)
        if attempt == 1:
            requests_buckets[bucket] += 1
        attempts_buckets[bucket] += 1

    amp_times = []
    amp_values = []
    for bucket in sorted(attempts_buckets.keys()):
        if requests_buckets[bucket] > 0:
            amp = attempts_buckets[bucket] / requests_buckets[bucket]
            amp_times.append(bucket * bucket_size)
            amp_values.append(amp)

    ax1.plot(amp_times, amp_values, "b-", linewidth=2, marker="o", markersize=3)
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="No Amplification")
    for gc_start, gc_end in gc_events:
        ax1.axvspan(gc_start, gc_end, alpha=0.1, color="red")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Attempts / Original Request")
    ax1.set_title("Retry Amplification Over Time (With Retries)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Latency Distribution Comparison
    ax2 = axes[0, 1]

    latencies_with = with_r.client.latencies_s
    latencies_without = without_r.client.latencies_s

    if latencies_with:
        ax2.hist(
            [l * 1000 for l in latencies_with],
            bins=50,
            alpha=0.5,
            label="With Retries",
            color="red",
        )
    if latencies_without:
        ax2.hist(
            [l * 1000 for l in latencies_without],
            bins=50,
            alpha=0.5,
            label="Without Retries",
            color="green",
        )

    timeout_ms = with_r.client.timeout_s * 1000
    ax2.axvline(
        x=timeout_ms, color="orange", linestyle="--", alpha=0.7, label=f"Timeout ({timeout_ms:.0f}ms)"
    )
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Count")
    ax2.set_title("End-to-End Latency Distribution")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative Timeouts
    ax3 = axes[1, 0]

    # Build cumulative timeout curve from attempts_by_time
    timeout_times_with: list[float] = []
    cumulative_with: list[int] = []
    cumulative = 0
    for t_s, attempt in sorted(with_r.client.attempts_by_time):
        if attempt > 1:  # Retry means previous attempt timed out
            cumulative += 1
            timeout_times_with.append(t_s)
            cumulative_with.append(cumulative)

    if timeout_times_with:
        ax3.plot(
            timeout_times_with,
            cumulative_with,
            "r-",
            linewidth=2,
            label="With Retries",
        )

    for gc_start, gc_end in gc_events:
        ax3.axvspan(gc_start, gc_end, alpha=0.1, color="red")

    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Cumulative Timeouts")
    ax3.set_title("Cumulative Client Timeouts Over Time")
    if ax3.get_legend_handles_labels()[1]:
        ax3.legend(loc="upper left")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary Statistics Box
    ax4 = axes[1, 1]
    ax4.axis("off")

    # Calculate statistics
    success_rate_with = (
        with_r.client.stats_completions / max(1, with_r.client.stats_requests_received)
        * 100
    )
    success_rate_without = (
        without_r.client.stats_completions
        / max(1, without_r.client.stats_requests_received)
        * 100
    )

    amp_with = with_r.client.stats_attempts_sent / max(
        1, with_r.client.stats_requests_received
    )
    amp_without = without_r.client.stats_attempts_sent / max(
        1, without_r.client.stats_requests_received
    )

    final_q_with = get_final_queue_depth(with_r.queue_depth_data)
    final_q_without = get_final_queue_depth(without_r.queue_depth_data)

    lat_data_with = _build_latency_data(with_r.client)
    lat_data_without = _build_latency_data(without_r.client)

    avg_lat_with = lat_data_with.mean() * 1000 if lat_data_with.count() > 0 else 0
    avg_lat_without = lat_data_without.mean() * 1000 if lat_data_without.count() > 0 else 0

    p99_lat_with = lat_data_with.percentile(0.99) * 1000 if lat_data_with.count() > 0 else 0
    p99_lat_without = lat_data_without.percentile(0.99) * 1000 if lat_data_without.count() > 0 else 0

    summary = f"""
Summary Statistics
==================

                    WITH RETRIES    WITHOUT RETRIES
                    ------------    ---------------
Requests Generated:     {with_r.requests_generated:>6}            {without_r.requests_generated:>6}
Completions:            {with_r.client.stats_completions:>6}            {without_r.client.stats_completions:>6}
Success Rate:           {success_rate_with:>5.1f}%           {success_rate_without:>5.1f}%
Retry Amplification:    {amp_with:>5.2f}x           {amp_without:>5.2f}x
Timeouts:               {with_r.client.stats_timeouts:>6}            {without_r.client.stats_timeouts:>6}
Final Queue Depth:      {final_q_with:>6}            {final_q_without:>6}
Avg Latency:            {avg_lat_with:>5.0f}ms          {avg_lat_without:>5.0f}ms
p99 Latency:            {p99_lat_with:>5.0f}ms          {p99_lat_without:>5.0f}ms

GC: {with_r.server.gc_duration_s}s pause at t={with_r.server.gc_start_time_s}s ({with_r.server.gc_duration_s / with_r.client.timeout_s:.1f}x timeout)

Key Insight:
  A single GC pause causes {amp_with:.1f}x load amplification,
  overwhelming the 30% recovery headroom. System never
  recovers.
"""

    ax4.text(
        0.05,
        0.95,
        summary,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(output_dir / "gc_collapse_analysis.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'gc_collapse_analysis.png'}")


# =============================================================================
# Summary Output
# =============================================================================


def analyze_gc_impact(result: ComparisonResult) -> None:
    """Analyze the GC pause impact using phase detection and time-slicing."""
    with_r = result.with_retries
    gc_start = with_r.server.gc_start_time_s
    gc_end = gc_start + with_r.server.gc_duration_s
    qd = with_r.queue_depth_data
    latency_data = _build_latency_data(with_r.client)

    print("\n" + "=" * 75)
    print("GC IMPACT ANALYSIS (using observability APIs)")
    print("=" * 75)

    # --- Time-sliced queue depth comparison ---
    pre_gc = qd.between(0, gc_start)
    during_gc = qd.between(gc_start, gc_end)
    post_gc_5s = qd.between(gc_end, gc_end + 5)
    post_gc_30s = qd.between(gc_end + 20, gc_end + 30)

    print("\n  Queue Depth by Phase (Data.between):")
    print(f"    Pre-GC    [0, {gc_start:.0f}s):     "
          f"mean={pre_gc.mean():.1f}, max={pre_gc.max():.0f}")
    if during_gc.count() > 0:
        print(f"    During GC [{gc_start:.0f}, {gc_end:.0f}s):  "
              f"mean={during_gc.mean():.1f}, max={during_gc.max():.0f}")
    print(f"    Post-GC   [{gc_end:.0f}, {gc_end+5:.0f}s):  "
          f"mean={post_gc_5s.mean():.1f}, max={post_gc_5s.max():.0f}")
    if post_gc_30s.count() > 0:
        print(f"    Late       [{gc_end+20:.0f}, {gc_end+30:.0f}s): "
              f"mean={post_gc_30s.mean():.1f}, max={post_gc_30s.max():.0f}")

    # --- Time-sliced latency comparison ---
    lat_pre = latency_data.between(0, gc_start)
    lat_post_5s = latency_data.between(gc_end, gc_end + 5)
    lat_post_30s = latency_data.between(gc_end + 20, gc_end + 30)

    print("\n  Latency by Phase (Data.between):")
    if lat_pre.count() > 0:
        print(f"    Pre-GC:  p50={lat_pre.percentile(0.50)*1000:.0f}ms, "
              f"p99={lat_pre.percentile(0.99)*1000:.0f}ms, n={lat_pre.count()}")
    if lat_post_5s.count() > 0:
        print(f"    Post-GC (0-5s):  p50={lat_post_5s.percentile(0.50)*1000:.0f}ms, "
              f"p99={lat_post_5s.percentile(0.99)*1000:.0f}ms, n={lat_post_5s.count()}")
    if lat_post_30s.count() > 0:
        print(f"    Post-GC (20-30s): p50={lat_post_30s.percentile(0.50)*1000:.0f}ms, "
              f"p99={lat_post_30s.percentile(0.99)*1000:.0f}ms, n={lat_post_30s.count()}")

    # --- Phase detection on queue depth ---
    print("\n  Phase Detection (detect_phases on queue depth):")
    phases = detect_phases(qd, window_s=5.0)
    if phases:
        for p in phases:
            print(f"    [{p.label:>10}] {p.start_s:.0f}s - {p.end_s:.0f}s  "
                  f"mean={p.mean:.1f}  std={p.std:.1f}")
    else:
        print("    No phases detected.")

    # --- Full analysis pipeline ---
    if with_r.summary is not None:
        analysis = analyze(
            with_r.summary,
            latency=latency_data,
            queue_depth=qd,
        )

        if analysis.anomalies:
            print(f"\n  Anomalies Detected: {len(analysis.anomalies)}")
            for a in analysis.anomalies[:5]:
                print(f"    [{a.severity:>8}] t={a.time_s:.0f}s: {a.description}")
            if len(analysis.anomalies) > 5:
                print(f"    ... and {len(analysis.anomalies) - 5} more")

        if analysis.causal_chains:
            print(f"\n  Causal Chains: {len(analysis.causal_chains)}")
            for chain in analysis.causal_chains:
                print(f"    Trigger: {chain.trigger_description}")
                for effect in chain.effects:
                    print(f"      -> {effect}")

    # --- Queue depth rate of growth post-GC ---
    bucketed = qd.bucket(window_s=5.0)
    means = bucketed.means()
    times = bucketed.times()
    if len(means) >= 2:
        # Find the bucket right after GC and measure growth
        post_gc_buckets = [(t, m) for t, m in zip(times, means) if t >= gc_end]
        if len(post_gc_buckets) >= 2:
            t0, m0 = post_gc_buckets[0]
            t1, m1 = post_gc_buckets[min(3, len(post_gc_buckets) - 1)]
            if t1 > t0:
                growth_rate = (m1 - m0) / (t1 - t0)
                print(f"\n  Queue Growth Rate (post-GC): {growth_rate:+.1f} items/sec")
                if growth_rate > 0:
                    print("    -> Queue growing: system is NOT recovering")
                else:
                    print("    -> Queue draining: system is recovering")


def print_summary(result: ComparisonResult) -> None:
    """Print comparison summary statistics."""
    with_r = result.with_retries
    without_r = result.without_retries

    print("\n" + "=" * 75)
    print("GC-INDUCED METASTABLE COLLAPSE SIMULATION")
    print("=" * 75)

    svc_rate = 1.0 / with_r.server.service_time_s
    gc_timeout_ratio = with_r.server.gc_duration_s / with_r.client.timeout_s

    print("\nConfiguration:")
    print(f"  Service capacity: {svc_rate:.0f} req/s (service time = {with_r.server.service_time_s * 1000:.0f}ms)")
    print(f"  Client timeout: {with_r.client.timeout_s * 1000:.0f}ms")
    print(f"  GC pause: {with_r.server.gc_duration_s}s at t={with_r.server.gc_start_time_s}s ({gc_timeout_ratio:.1f}x timeout)")
    print(f"  Max retries: {with_r.client.max_retries}")
    print(f"  Retry delay: {with_r.client.retry_delay_s * 1000:.0f}ms")

    print("\n" + "-" * 75)
    print("SCENARIO 1: WITH RETRIES")
    print("-" * 75)
    _print_scenario_stats(with_r)

    print("\n" + "-" * 75)
    print("SCENARIO 2: WITHOUT RETRIES")
    print("-" * 75)
    _print_scenario_stats(without_r)

    # Analysis
    amp_with = with_r.client.stats_attempts_sent / max(
        1, with_r.client.stats_requests_received
    )

    print("\n" + "-" * 75)
    print("ANALYSIS")
    print("-" * 75)

    print("""
The GC-induced collapse demonstrates how retries amplify transient
failures into sustained metastable failure:

1. With retries: {:.2f}x request amplification caused queue to compound
   after each GC, never recovering to steady state.

2. Without retries: System experiences brief latency spikes during GC
   but returns to steady state between GC events.

3. Recovery would require reducing load to ~{:.0f}% utilization ({:.0f}%
   reduction from normal), matching the theoretical requirement
   to overcome {:.2f}x amplification.
""".format(
        amp_with,
        70 / amp_with,
        (1 - 1 / amp_with) * 100,
        amp_with,
    ))

    # Detailed GC impact analysis using new observability APIs
    analyze_gc_impact(result)

    print("=" * 75)


def _print_scenario_stats(scenario: ScenarioResult) -> None:
    """Print statistics for a single scenario."""
    client = scenario.client
    latency_data = _build_latency_data(client)

    success_rate = (
        client.stats_completions / max(1, client.stats_requests_received) * 100
    )
    amp = client.stats_attempts_sent / max(1, client.stats_requests_received)
    final_q = get_final_queue_depth(scenario.queue_depth_data)

    avg_lat = latency_data.mean() * 1000 if latency_data.count() > 0 else 0
    p99_lat = latency_data.percentile(0.99) * 1000 if latency_data.count() > 0 else 0

    print(f"  Requests generated:     {scenario.requests_generated}")
    print(f"  Successful completions: {client.stats_completions}")
    print(f"  Success rate:           {success_rate:.1f}%")
    print(f"  Timeouts:               {client.stats_timeouts}")
    print(f"  Retries:                {client.stats_retries}")
    print(f"  Gave up (max retries):  {client.stats_gave_up}")
    print(f"  Retry amplification:    {amp:.2f}x")
    print(f"  Final queue depth:      {final_q}")
    print(f"  Average latency:        {avg_lat:.1f}ms")
    print(f"  p99 latency:            {p99_lat:.1f}ms")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GC-induced metastable collapse simulation"
    )
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Simulation duration (s)"
    )
    parser.add_argument(
        "--drain", type=float, default=10.0, help="Drain time after load stops (s)"
    )
    parser.add_argument(
        "--arrival-rate", type=float, default=7.0, help="Arrival rate (req/s)"
    )
    parser.add_argument(
        "--service-time", type=float, default=0.1, help="Constant service time (s)"
    )
    parser.add_argument(
        "--timeout", type=float, default=0.5, help="Client timeout (s)"
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Max retries per request"
    )
    parser.add_argument(
        "--retry-delay", type=float, default=0.05, help="Retry delay (s)"
    )
    parser.add_argument(
        "--gc-interval", type=float, default=999.0, help="GC interval (s, 999=single GC)"
    )
    parser.add_argument(
        "--gc-duration", type=float, default=1.0, help="GC duration (s)"
    )
    parser.add_argument(
        "--gc-start", type=float, default=30.0, help="GC start time (s)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--output", type=str, default="output/gc_collapse", help="Output directory"
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization generation"
    )
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running GC-induced collapse simulation...")
    print(f"  Duration: {args.duration}s + {args.drain}s drain")
    print(f"  Arrival rate: {args.arrival_rate} req/s")
    print(f"  GC: {args.gc_duration}s pause at t={args.gc_start}s")
    print(f"  Random seed: {seed if seed is not None else 'random'}")

    result = run_comparison(
        duration_s=args.duration,
        drain_s=args.drain,
        arrival_rate=args.arrival_rate,
        service_time_s=args.service_time,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
        retry_delay_s=args.retry_delay,
        gc_interval_s=args.gc_interval,
        gc_duration_s=args.gc_duration,
        gc_start_time_s=args.gc_start,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
