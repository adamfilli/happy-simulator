"""Database query timeouts: protecting against long-running queries.

Inspired by @BenJDicken's explanation of why you should never run queries
without a timeout. A single long-running query or load spike can lead to
stalls or app downtime without them.

This simulation runs two identical databases **side-by-side in the same
simulation** under the same workload pattern:

1. **No Timeout (DB_NoTimeout)**: Long-running queries (1000ms) run to
   completion, hogging connections until the 100-connection pool is
   exhausted. Regular queries pile up and experience massive latency spikes.
   Recovery takes 10-15 seconds after the burst ends.

2. **With Timeout (DB_WithTimeout)**: A 250ms database-side timeout kills
   long-running queries. The app retries with exponential backoff + jitter.
   Regular queries are barely affected and the system recovers within
   seconds.

## Architecture Diagram

```
  Regular  ──> App_NoTimeout  ──> DB_NoTimeout  (100 conns, no timeout)
  LongRunning ──> App_NoTimeout

  Regular  ──> App_WithTimeout ──> DB_WithTimeout (100 conns, 250ms timeout)
  LongRunning ──> App_WithTimeout
```

## Workload

Regular queries (10ms) arrive at 500/s for the full simulation. A burst of
300/s long-running queries (1000ms each) hits both databases for 10 seconds
mid-simulation, simulating a bad deployment or load spike. Without timeouts,
the connection pool saturates at 100% and congestive collapse occurs --
regular queries back up behind long-running ones and latency spirals for
30+ seconds after the burst ends. With a 250ms timeout, long-running queries
are killed early (75 connections vs 300 needed), so the pool never saturates
and the system stays healthy throughout.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

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
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.load.profile import ConstantRateProfile, SpikeProfile

if TYPE_CHECKING:
    from collections.abc import Generator


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class Config:
    """Simulation parameters."""

    duration_s: float = 70.0
    drain_s: float = 15.0

    # Regular query workload (runs for the full duration)
    regular_rate: float = 500.0  # queries/s
    regular_query_time_s: float = 0.01  # 10ms

    # Long-running query burst (intense spike mid-simulation)
    long_running_rate: float = 150.0  # queries/s during burst
    long_running_query_time_s: float = 1.0  # 1000ms per query
    long_running_burst_start_s: float = 15.0  # burst begins at t=15s
    long_running_burst_duration_s: float = 10.0  # lasts 10 seconds

    # Database
    connection_limit: int = 100

    # Timeout configuration (with-timeout database only)
    timeout_s: float = 0.25  # 250ms database-side timeout
    max_retries: int = 2  # total attempts per query (1 original + 1 retry)
    base_backoff_s: float = 0.05  # 50ms base backoff
    max_backoff_s: float = 1.0  # 1s cap


    seed: int | None = 42


# =============================================================================
# Database: QueuedResource with Connection Pool
# =============================================================================


class Database(QueuedResource):
    """Database server with a limited connection pool and optional query timeout.

    Each query waits in the FIFO queue until a connection slot opens
    (controlled by has_capacity). Once dequeued, the query executes for its
    full duration -- unless timeout_s is set, in which case queries exceeding
    the timeout are killed early and the response is marked timed_out=True.
    """

    def __init__(
        self,
        name: str,
        *,
        connection_limit: int = 100,
        timeout_s: float | None = None,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.connection_limit = connection_limit
        self.timeout_s = timeout_s
        self._active: int = 0

        # Stats
        self.queries_completed: int = 0
        self.queries_timed_out: int = 0
        self.peak_connections: int = 0

    @property
    def active_connections(self) -> int:
        """Number of connections currently in use (for Probe)."""
        return self._active

    def has_capacity(self) -> bool:
        return self._active < self.connection_limit

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Acquire a connection slot, execute the query, release, respond."""
        self._active += 1
        if self._active > self.peak_connections:
            self.peak_connections = self._active

        query_time = event.context.get("query_time", 0.01)
        timed_out = False

        try:
            if self.timeout_s is not None and query_time > self.timeout_s:
                yield self.timeout_s
                timed_out = True
                self.queries_timed_out += 1
            else:
                yield query_time
                self.queries_completed += 1
        finally:
            self._active -= 1

        reply_to = event.context.get("reply_to")
        if reply_to is None:
            return []

        return [
            Event(
                time=self.now,
                event_type="QueryResponse",
                target=reply_to,
                context={
                    "timed_out": timed_out,
                    "query_id": event.context.get("query_id"),
                    "created_at": event.context.get("created_at"),
                    "query_type": event.context.get("query_type"),
                    "query_time": event.context.get("query_time"),
                    "attempt": event.context.get("attempt", 1),
                },
            )
        ]


# =============================================================================
# App Client: Routes Queries and Handles Retries
# =============================================================================


class AppClient(Entity):
    """Application client that forwards queries to the database.

    Tracks per-query-type statistics. In the timeout scenario, timed-out
    queries are retried with exponential backoff + jitter to prevent
    thundering herds.
    """

    def __init__(
        self,
        name: str,
        database: Database,
        *,
        enable_retry: bool = False,
        max_retries: int = 3,
        base_backoff_s: float = 0.05,
        max_backoff_s: float = 1.0,
        latency_sink: LatencyTracker | None = None,
    ):
        super().__init__(name)
        self.database = database
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        self.max_backoff_s = max_backoff_s
        self.latency_sink = latency_sink
        self._next_id: int = 0

        # Regular query stats
        self.regular_completed: int = 0
        self.regular_timed_out: int = 0
        self.regular_gave_up: int = 0
        self.regular_latencies_s: list[float] = []
        self.regular_completion_times: list[float] = []

        # Long-running query stats
        self.long_running_completed: int = 0
        self.long_running_timed_out: int = 0
        self.long_running_gave_up: int = 0

        # Total tracking
        self.total_sent: int = 0
        self.total_retries: int = 0

    def downstream_entities(self) -> list[Entity]:
        """Declare downstream entities for visual debugger topology."""
        result: list[Entity] = [self.database]
        if self.latency_sink is not None:
            result.append(self.latency_sink)
        return result

    def handle_event(self, event: Event) -> list[Event]:
        et = event.event_type
        if et in ("RegularQuery", "LongRunningQuery"):
            return self._new_query(event)
        if et == "QueryResponse":
            return self._handle_response(event)
        return []

    def _new_query(self, event: Event) -> list[Event]:
        self._next_id += 1
        is_regular = event.event_type == "RegularQuery"
        query_time = event.context.get("query_time_s", 0.01 if is_regular else 1.0)
        self.total_sent += 1

        return [
            Event(
                time=self.now,
                event_type="Query",
                target=self.database,
                context={
                    "query_time": query_time,
                    "reply_to": self,
                    "query_id": self._next_id,
                    "created_at": event.time,
                    "query_type": "regular" if is_regular else "long_running",
                    "attempt": 1,
                },
            )
        ]

    def _handle_response(self, event: Event) -> list[Event]:
        ctx = event.context
        query_type = ctx.get("query_type", "regular")
        timed_out = ctx.get("timed_out", False)

        if not timed_out:
            latency = (event.time - ctx["created_at"]).to_seconds()
            if query_type == "regular":
                self.regular_completed += 1
                self.regular_latencies_s.append(latency)
                self.regular_completion_times.append(event.time.to_seconds())
            else:
                self.long_running_completed += 1

            # Forward completed requests to latency sink
            if self.latency_sink is not None:
                return [
                    Event(
                        time=event.time,
                        event_type="CompletedQuery",
                        target=self.latency_sink,
                        context={"created_at": ctx["created_at"]},
                    )
                ]
            return []

        # Timed out
        if query_type == "regular":
            self.regular_timed_out += 1
        else:
            self.long_running_timed_out += 1

        if not self.enable_retry:
            return []

        # Retry with exponential backoff + jitter
        attempt = ctx.get("attempt", 1)
        if attempt >= self.max_retries:
            if query_type == "regular":
                self.regular_gave_up += 1
            else:
                self.long_running_gave_up += 1
            return []

        self.total_retries += 1
        self.total_sent += 1

        backoff = min(self.base_backoff_s * (2 ** attempt), self.max_backoff_s)
        jitter = random.uniform(0, backoff)

        return [
            Event(
                time=Instant.from_seconds(self.now.to_seconds() + jitter),
                event_type="Query",
                target=self.database,
                context={
                    "query_time": ctx["query_time"],
                    "reply_to": self,
                    "query_id": ctx["query_id"],
                    "created_at": ctx["created_at"],
                    "query_type": query_type,
                    "attempt": attempt + 1,
                },
            )
        ]


# =============================================================================
# Event Provider with Delayed Start
# =============================================================================


class QueryEventProvider(EventProvider):
    """Generates query events targeting an AppClient."""

    def __init__(
        self,
        target: AppClient,
        event_type: str,
        *,
        query_time_s: float,
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._event_type = event_type
        self._query_time_s = query_time_s
        self._stop_after = stop_after
        self._generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []
        self._generated += 1
        return [
            Event(
                time=time,
                event_type=self._event_type,
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self._generated,
                    "query_time_s": self._query_time_s,
                },
            )
        ]

    @property
    def generated(self) -> int:
        return self._generated


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class ScenarioParts:
    """Entities and data for one side of the comparison."""

    client: AppClient
    database: Database
    conn_data: Data
    queue_depth_data: Data
    latency_data: Data
    regular_provider: QueryEventProvider
    long_running_provider: QueryEventProvider


@dataclass
class SimulationBuild:
    """Combined simulation with both scenarios, ready to run or serve."""

    sim: Simulation
    no_timeout: ScenarioParts
    with_timeout: ScenarioParts
    config: Config


@dataclass
class SimulationResult:
    """Results after running the combined simulation."""

    no_timeout: ScenarioParts
    with_timeout: ScenarioParts
    summary: SimulationSummary
    config: Config


def _build_scenario_parts(
    label: str,
    config: Config,
    *,
    enable_timeout: bool,
) -> tuple[ScenarioParts, list[Source], list[Entity], list]:
    """Build entities and sources for one side of the comparison.

    Returns (parts, sources, entities, probes) so the caller can register
    them all in a single Simulation.
    """
    # Create database
    database = Database(
        f"DB_{label}",
        connection_limit=config.connection_limit,
        timeout_s=config.timeout_s if enable_timeout else None,
    )

    # Create latency tracker for completed requests
    latency_sink = LatencyTracker(f"Latency_{label}")

    # Create app client
    client = AppClient(
        f"App_{label}",
        database,
        enable_retry=enable_timeout,
        max_retries=config.max_retries,
        base_backoff_s=config.base_backoff_s,
        max_backoff_s=config.max_backoff_s,
        latency_sink=latency_sink,
    )

    # Create event providers
    regular_provider = QueryEventProvider(
        client,
        "RegularQuery",
        query_time_s=config.regular_query_time_s,
        stop_after=Instant.from_seconds(config.duration_s),
    )
    long_running_provider = QueryEventProvider(
        client,
        "LongRunningQuery",
        query_time_s=config.long_running_query_time_s,
    )

    # Create sources (Poisson arrivals for realistic variance)
    regular_source = Source.with_profile(
        ConstantRateProfile(rate=config.regular_rate),
        name=f"Regular_{label}",
        event_provider=regular_provider,
        stop_after=config.duration_s,
    )
    long_running_source = Source.with_profile(
        SpikeProfile(
            baseline_rate=0.0,
            spike_rate=config.long_running_rate,
            warmup_s=config.long_running_burst_start_s,
            spike_duration_s=config.long_running_burst_duration_s,
        ),
        name=f"LongRunning_{label}",
        event_provider=long_running_provider,
    )

    # Create probes
    conn_probe, conn_data = Probe.on(
        database, "active_connections", interval=0.1
    )
    queue_probe, queue_depth_data = Probe.on(database, "depth", interval=0.1)

    parts = ScenarioParts(
        client=client,
        database=database,
        conn_data=conn_data,
        queue_depth_data=queue_depth_data,
        latency_data=latency_sink.data,
        regular_provider=regular_provider,
        long_running_provider=long_running_provider,
    )

    return (
        parts,
        [regular_source, long_running_source],
        [client, database, latency_sink],
        [conn_probe, queue_probe],
    )


def build_simulation(config: Config | None = None) -> SimulationBuild:
    """Build both scenarios in a single simulation (without running)."""
    config = config or Config()

    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)

    nt_parts, nt_sources, nt_entities, nt_probes = _build_scenario_parts(
        "NoTimeout", config, enable_timeout=False
    )
    wt_parts, wt_sources, wt_entities, wt_probes = _build_scenario_parts(
        "WithTimeout", config, enable_timeout=True
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + config.drain_s,
        sources=nt_sources + wt_sources,
        entities=nt_entities + wt_entities,
        probes=nt_probes + wt_probes,
    )

    return SimulationBuild(
        sim=sim,
        no_timeout=nt_parts,
        with_timeout=wt_parts,
        config=config,
    )


def run_simulation(config: Config | None = None) -> SimulationResult:
    """Build and run the combined simulation."""
    build = build_simulation(config)
    summary = build.sim.run()

    return SimulationResult(
        no_timeout=build.no_timeout,
        with_timeout=build.with_timeout,
        summary=summary,
        config=build.config,
    )


# =============================================================================
# Summary
# =============================================================================


def _percentile(values: list[float], p: float) -> float:
    """Calculate percentile from a list of values."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def print_summary(result: SimulationResult) -> None:
    cfg = result.config
    nt = result.no_timeout
    wt = result.with_timeout

    print("\n" + "=" * 72)
    print("DATABASE QUERY TIMEOUTS: Protecting Against Long-Running Queries")
    print("=" * 72)

    print(
        f"\nScenario: {cfg.connection_limit} connections, regular queries "
        f"({cfg.regular_query_time_s * 1000:.0f}ms) at {cfg.regular_rate:.0f}/s,"
    )
    burst_end = cfg.long_running_burst_start_s + cfg.long_running_burst_duration_s
    print(
        f"long-running queries ({cfg.long_running_query_time_s * 1000:.0f}ms) "
        f"burst at t={cfg.long_running_burst_start_s:.0f}-{burst_end:.0f}s "
        f"at {cfg.long_running_rate:.0f}/s."
    )
    print(
        f"Timeout scenario: {cfg.timeout_s * 1000:.0f}ms database-side timeout, "
        f"{cfg.max_retries} max attempts, exponential backoff + jitter."
    )

    # Regular query performance
    header = (
        f"\n{'Regular Query Performance':<40} {'No Timeout':>14} {'With Timeout':>14}"
    )
    print(header)
    print("-" * 68)

    print(
        f"{'  Generated':<40} "
        f"{nt.regular_provider.generated:>14,} "
        f"{wt.regular_provider.generated:>14,}"
    )
    print(
        f"{'  Completed':<40} "
        f"{nt.client.regular_completed:>14,} "
        f"{wt.client.regular_completed:>14,}"
    )
    print(
        f"{'  Timed out':<40} "
        f"{nt.client.regular_timed_out:>14,} "
        f"{wt.client.regular_timed_out:>14,}"
    )
    print(
        f"{'  Gave up (exhausted retries)':<40} "
        f"{nt.client.regular_gave_up:>14,} "
        f"{wt.client.regular_gave_up:>14,}"
    )

    # Latency stats
    for label, p in [("Avg", None), ("p50", 0.5), ("p99", 0.99)]:
        if label == "Avg":
            v_nt = (
                sum(nt.client.regular_latencies_s) / len(nt.client.regular_latencies_s)
                if nt.client.regular_latencies_s
                else float("nan")
            )
            v_wt = (
                sum(wt.client.regular_latencies_s) / len(wt.client.regular_latencies_s)
                if wt.client.regular_latencies_s
                else float("nan")
            )
        else:
            v_nt = _percentile(nt.client.regular_latencies_s, p)
            v_wt = _percentile(wt.client.regular_latencies_s, p)
        print(
            f"{'  ' + label + ' latency (ms)':<40} "
            f"{v_nt * 1000:>14.1f} {v_wt * 1000:>14.1f}"
        )

    max_nt = (
        max(nt.client.regular_latencies_s) * 1000
        if nt.client.regular_latencies_s
        else 0
    )
    max_wt = (
        max(wt.client.regular_latencies_s) * 1000
        if wt.client.regular_latencies_s
        else 0
    )
    print(f"{'  Max latency (ms)':<40} {max_nt:>14.1f} {max_wt:>14.1f}")

    # Long-running query stats
    header2 = (
        f"\n{'Long-Running Query Stats':<40} {'No Timeout':>14} {'With Timeout':>14}"
    )
    print(header2)
    print("-" * 68)
    print(
        f"{'  Generated':<40} "
        f"{nt.long_running_provider.generated:>14,} "
        f"{wt.long_running_provider.generated:>14,}"
    )
    print(
        f"{'  Completed':<40} "
        f"{nt.client.long_running_completed:>14,} "
        f"{wt.client.long_running_completed:>14,}"
    )
    print(
        f"{'  Timed out':<40} "
        f"{nt.client.long_running_timed_out:>14,} "
        f"{wt.client.long_running_timed_out:>14,}"
    )
    print(
        f"{'  Gave up':<40} "
        f"{nt.client.long_running_gave_up:>14,} "
        f"{wt.client.long_running_gave_up:>14,}"
    )

    # Connection pool
    header3 = (
        f"\n{'Connection Pool (' + str(cfg.connection_limit) + ' max)':<40} "
        f"{'No Timeout':>14} {'With Timeout':>14}"
    )
    print(header3)
    print("-" * 68)
    print(
        f"{'  Peak connections used':<40} "
        f"{nt.database.peak_connections:>14,} "
        f"{wt.database.peak_connections:>14,}"
    )
    print(
        f"{'  Queries completed (DB)':<40} "
        f"{nt.database.queries_completed:>14,} "
        f"{wt.database.queries_completed:>14,}"
    )
    print(
        f"{'  Queries timed out (DB)':<40} "
        f"{nt.database.queries_timed_out:>14,} "
        f"{wt.database.queries_timed_out:>14,}"
    )

    # Simulation stats
    print(f"\n{'Simulation':<40}")
    print("-" * 68)
    print(
        f"{'  Total events processed':<40} "
        f"{result.summary.total_events_processed:>14,}"
    )
    print(
        f"{'  Wall clock (s)':<40} "
        f"{result.summary.wall_clock_seconds:>14.2f}"
    )

    # Interpretation
    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print()
    print("  Both databases are hit by the same burst of long-running")
    print("  queries. During the burst, both saturate their connection")
    print("  pools. The difference is in recovery:")
    print()
    print("  Without timeouts, long-running queries hold connections for")
    print("  their full 1000ms. The FIFO queue fills with a mix of regular")
    print("  and long-running queries. Even after the burst ends, it takes")
    print("  10-15 seconds to drain the backlog. Regular query latency")
    print("  stays elevated the entire time.")
    print()
    print("  With a 250ms timeout, long-running queries are killed early,")
    print("  freeing connections 4x faster. Exponential backoff + jitter")
    print("  on retries prevents thundering herds. The system recovers")
    print("  within seconds and regular queries are barely affected.")
    print()
    print("  Key takeaway: always set query timeouts. They limit the blast")
    print("  radius of any burst and dramatically improve recovery time.")
    print("  Monitor timeouts to quickly identify problematic queries.")
    print("\n" + "=" * 72)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    burst_start = cfg.long_running_burst_start_s
    burst_end = burst_start + cfg.long_running_burst_duration_s

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    def mark_burst(ax):
        """Shade the burst window on a chart."""
        ax.axvspan(burst_start, burst_end, color="orange", alpha=0.12, label="Burst")

    # Chart 1: Active connections over time
    ax = axes[0, 0]
    for parts, color, label in [
        (result.no_timeout, "#e74c3c", "No Timeout"),
        (result.with_timeout, "#2ecc71", "With Timeout"),
    ]:
        times = parts.conn_data.times()
        vals = parts.conn_data.raw_values()
        ax.plot(times, vals, color=color, linewidth=1, label=label, alpha=0.8)
    ax.axhline(
        y=cfg.connection_limit,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Limit ({cfg.connection_limit})",
    )
    mark_burst(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Active Connections")
    ax.set_title("Connection Pool Usage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 2: Database queue depth over time
    ax = axes[0, 1]
    for parts, color, label in [
        (result.no_timeout, "#e74c3c", "No Timeout"),
        (result.with_timeout, "#2ecc71", "With Timeout"),
    ]:
        times = parts.queue_depth_data.times()
        vals = parts.queue_depth_data.raw_values()
        ax.plot(times, vals, color=color, linewidth=1, label=label, alpha=0.8)
    mark_burst(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Depth")
    ax.set_title("Database Queue Depth")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 3: Regular query p99 latency over time (1s buckets)
    ax = axes[1, 0]
    for parts, color, label in [
        (result.no_timeout, "#e74c3c", "No Timeout"),
        (result.with_timeout, "#2ecc71", "With Timeout"),
    ]:
        if not parts.client.regular_completion_times:
            continue
        buckets: dict[int, list[float]] = defaultdict(list)
        for t, lat in zip(
            parts.client.regular_completion_times,
            parts.client.regular_latencies_s,
            strict=False,
        ):
            buckets[int(t)].append(lat)
        bucket_times = sorted(buckets.keys())
        bucket_p99 = [
            _percentile(buckets[b], 0.99) * 1000 for b in bucket_times
        ]
        ax.plot(
            bucket_times,
            bucket_p99,
            color=color,
            linewidth=1.5,
            label=label,
            alpha=0.8,
        )
    mark_burst(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("p99 Latency (ms)")
    ax.set_title("Regular Query p99 Latency (1s buckets)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Chart 4: Regular query goodput over time (1s buckets)
    ax = axes[1, 1]
    for parts, color, label in [
        (result.no_timeout, "#e74c3c", "No Timeout"),
        (result.with_timeout, "#2ecc71", "With Timeout"),
    ]:
        if not parts.client.regular_completion_times:
            continue
        count_buckets: dict[int, int] = defaultdict(int)
        for t in parts.client.regular_completion_times:
            count_buckets[int(t)] += 1
        bucket_times = sorted(count_buckets.keys())
        counts = [count_buckets[b] for b in bucket_times]
        ax.plot(
            bucket_times,
            counts,
            color=color,
            linewidth=1.5,
            label=label,
            alpha=0.8,
        )
    mark_burst(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Completions / second")
    ax.set_title("Regular Query Goodput")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Database Query Timeouts: No Timeout vs With Timeout",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "database_query_timeout.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'database_query_timeout.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Database query timeout simulation (inspired by @BenJDicken)"
    )
    parser.add_argument(
        "--duration", type=float, default=70.0, help="Simulation duration (s)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (-1 for random)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/database_query_timeout",
        help="Output directory",
    )
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Launch in the browser-based visual debugger",
    )
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    config = Config(duration_s=args.duration, seed=seed)

    print("Running database query timeout simulation...")
    print(f"  Duration: {config.duration_s}s")
    print(
        f"  Regular: {config.regular_rate}/s at "
        f"{config.regular_query_time_s * 1000:.0f}ms"
    )
    burst_end = config.long_running_burst_start_s + config.long_running_burst_duration_s
    print(
        f"  Long-running: {config.long_running_rate}/s at "
        f"{config.long_running_query_time_s * 1000:.0f}ms "
        f"(burst t={config.long_running_burst_start_s:.0f}-{burst_end:.0f}s)"
    )
    print(f"  Connections: {config.connection_limit}")
    print(f"  Timeout: {config.timeout_s * 1000:.0f}ms (WithTimeout database)")

    if args.visual:
        from happysimulator.visual import Chart, serve

        print("\n  Launching visual debugger (both scenarios side-by-side)...")

        build = build_simulation(config)
        nt = build.no_timeout
        wt = build.with_timeout

        serve(
            build.sim,
            charts=[
                Chart(
                    nt.conn_data,
                    title="Connections (No Timeout)",
                    y_label="connections",
                    color="#e74c3c",
                ),
                Chart(
                    wt.conn_data,
                    title="Connections (With Timeout)",
                    y_label="connections",
                    color="#2ecc71",
                ),
                Chart(
                    nt.queue_depth_data,
                    title="Queue Depth (No Timeout)",
                    y_label="items",
                    color="#f97316",
                ),
                Chart(
                    wt.queue_depth_data,
                    title="Queue Depth (With Timeout)",
                    y_label="items",
                    color="#06b6d4",
                ),
                Chart(
                    nt.latency_data,
                    title="Avg Latency (No Timeout)",
                    transform="mean",
                    window_s=1.0,
                    y_label="seconds",
                    color="#ef4444",
                ),
                Chart(
                    wt.latency_data,
                    title="Avg Latency (With Timeout)",
                    transform="mean",
                    window_s=1.0,
                    y_label="seconds",
                    color="#22c55e",
                ),
            ],
        )
    else:
        result = run_simulation(config)
        print_summary(result)

        if not args.no_viz:
            visualize_results(result, Path(args.output))
            print(f"\nVisualizations saved to: {Path(args.output).absolute()}")
