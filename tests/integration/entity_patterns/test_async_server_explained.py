"""Integration tests explaining AsyncServer behavior with visualizations.

This test suite demonstrates how AsyncServer models event-loop style servers
(like Node.js or Python asyncio) and how they differ from thread-pool servers.

KEY CONCEPTS:

1. SINGLE-THREADED EVENT LOOP
   - AsyncServer has ONE "thread" that handles all requests
   - CPU-bound work (computation) BLOCKS the entire server
   - I/O-bound work (waiting for database, network) is NON-BLOCKING

2. CPU WORK IS SERIALIZED
   - Only ONE request can do CPU work at a time
   - Other requests queue up waiting for CPU
   - This is why Node.js says "don't block the event loop"

3. HIGH CONNECTION CONCURRENCY
   - Despite single-threaded CPU, can handle MANY connections
   - Connections waiting for I/O don't block others
   - This is why async servers excel at I/O-bound workloads

VISUALIZATION OUTPUT:
    test_output/test_async_server_explained/<test_name>/

Run:
    pytest tests/integration/test_async_server_explained.py -v
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pytest

from happysimulator.components.server.async_server import AsyncServer
from happysimulator.components.server.server import Server
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Duration, Instant
from happysimulator.distributions.constant import ConstantLatency
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


# --- Metrics Collector ---


@dataclass
class TimelineCollector(Entity):
    """Collects timeline data for visualization."""
    name: str
    sample_interval_s: float = 0.01  # 10ms resolution

    timestamps: list[float] = field(default_factory=list, init=False)

    # For AsyncServer
    async_active_connections: list[int] = field(default_factory=list, init=False)
    async_cpu_queue_depth: list[int] = field(default_factory=list, init=False)
    async_cpu_busy: list[int] = field(default_factory=list, init=False)
    async_completed: list[int] = field(default_factory=list, init=False)

    # For comparison Server
    server_active_requests: list[int] = field(default_factory=list, init=False)
    server_queue_depth: list[int] = field(default_factory=list, init=False)
    server_completed: list[int] = field(default_factory=list, init=False)

    _async_server: AsyncServer | None = field(default=None, init=False)
    _server: Server | None = field(default=None, init=False)

    def set_targets(self, async_server: AsyncServer | None = None, server: Server | None = None):
        self._async_server = async_server
        self._server = server

    def handle_event(self, event: Event) -> list[Event] | None:
        """Record metrics and schedule next sample."""
        current_time = self.now.to_seconds()
        self.timestamps.append(current_time)

        # Record AsyncServer state
        if self._async_server is not None:
            self.async_active_connections.append(self._async_server.active_connections)
            self.async_cpu_queue_depth.append(self._async_server.cpu_queue_depth)
            self.async_cpu_busy.append(1 if self._async_server.is_cpu_busy else 0)
            self.async_completed.append(self._async_server.stats.requests_completed)

        # Record Server state
        if self._server is not None:
            self.server_active_requests.append(self._server.active_requests)
            self.server_queue_depth.append(self._server.depth)
            self.server_completed.append(self._server.stats.requests_completed)

        # Schedule next sample
        next_time = self.now + Duration.from_seconds(self.sample_interval_s)
        return [
            Event(
                time=next_time,
                event_type="collect_metrics",
                target=self,
            )
        ]


# --- Request Tracking ---


@dataclass
class RequestTracker(Entity):
    """Tracks individual request lifecycle for timeline visualization."""
    name: str

    # List of (request_id, arrive_time, complete_time)
    request_timeline: list[tuple[int, float, float]] = field(default_factory=list, init=False)
    _pending: dict[int, float] = field(default_factory=dict, init=False)

    def record_arrival(self, request_id: int, time: float):
        self._pending[request_id] = time

    def record_completion(self, request_id: int, time: float):
        if request_id in self._pending:
            arrive_time = self._pending.pop(request_id)
            self.request_timeline.append((request_id, arrive_time, time))

    def handle_event(self, event: Event) -> None:
        # Handle completion notifications
        request_id = event.context.get("metadata", {}).get("request_id")
        if request_id is not None:
            self.record_completion(request_id, self.now.to_seconds())


class TrackedRequestProvider(EventProvider):
    """Generates requests with tracking hooks."""

    def __init__(
        self,
        target: Entity,
        tracker: RequestTracker,
        stop_after: Instant | None = None,
    ):
        self.target = target
        self.tracker = tracker
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        request_id = self.generated

        # Record arrival
        self.tracker.record_arrival(request_id, time.to_seconds())

        event = Event(
            time=time,
            event_type=f"Request-{request_id}",
            target=self.target,
            context={"metadata": {"request_id": request_id}},
        )

        # Add completion hook to notify tracker
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="request_complete",
                target=self.tracker,
                context={"metadata": {"request_id": request_id}},
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


# --- Test Cases ---


class TestAsyncServerExplained:
    """Integration tests that explain AsyncServer behavior visually."""

    def test_cpu_serialization_explained(self, test_output_dir: Path):
        """
        DEMONSTRATES: CPU work is serialized in AsyncServer

        Scenario:
        - 5 requests arrive at EXACTLY the same time (t=0)
        - Each request needs 100ms of CPU work
        - AsyncServer processes them ONE AT A TIME

        Expected behavior:
        - Request 1: CPU work t=0 to t=0.1s, completes at t=0.1s
        - Request 2: CPU work t=0.1s to t=0.2s, completes at t=0.2s
        - Request 3: CPU work t=0.2s to t=0.3s, completes at t=0.3s
        - etc.

        Total time: 5 requests × 100ms = 500ms
        (NOT 100ms as it would be with 5 parallel workers!)
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Create AsyncServer with 100ms CPU work per request
        server = AsyncServer(
            name="async_server",
            max_connections=100,
            cpu_work_distribution=ConstantLatency(0.100),  # 100ms CPU each
        )

        # Create tracker
        tracker = RequestTracker(name="tracker")

        # Create metrics collector
        collector = TimelineCollector(name="collector", sample_interval_s=0.01)
        collector.set_targets(async_server=server)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=1.0,
            sources=[],
            entities=[server, tracker, collector],
        )

        # Schedule 5 requests at exactly t=0
        for i in range(5):
            request_id = i + 1
            tracker.record_arrival(request_id, 0.0)

            event = Event(
                time=Instant.Epoch,
                event_type=f"Request-{request_id}",
                target=server,
                context={"metadata": {"request_id": request_id}},
            )

            def make_hook(rid):
                def on_complete(finish_time: Instant):
                    return Event(
                        time=finish_time,
                        event_type="complete",
                        target=tracker,
                        context={"metadata": {"request_id": rid}},
                    )
                return on_complete

            event.add_completion_hook(make_hook(request_id))
            sim.schedule(event)

        # Start metrics collection
        sim.schedule(Event(time=Instant.Epoch, event_type="collect_metrics", target=collector))

        sim.run()

        # --- Generate Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Request Timeline (Gantt-style)
        ax = axes[0, 0]
        colors = plt.cm.Set3(range(len(tracker.request_timeline)))
        for i, (req_id, arrive, complete) in enumerate(sorted(tracker.request_timeline)):
            ax.barh(req_id, complete - arrive, left=arrive, height=0.6,
                   color=colors[i % len(colors)], edgecolor='black', linewidth=1)
            ax.text(complete + 0.01, req_id, f'{(complete-arrive)*1000:.0f}ms',
                   va='center', fontsize=9)

        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Request ID')
        ax.set_title('Request Timeline: CPU Work is SERIALIZED\n(Each request waits for previous to finish CPU work)')
        ax.set_xlim(-0.02, 0.6)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='All requests arrive')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 2: CPU Queue Depth Over Time
        ax = axes[0, 1]
        ax.fill_between(collector.timestamps, collector.async_cpu_queue_depth,
                       alpha=0.5, color='coral', label='Waiting in CPU queue')
        ax.plot(collector.timestamps, collector.async_cpu_queue_depth,
               color='coral', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Queue Depth')
        ax.set_title('CPU Queue: Requests waiting for their turn\n(Queue drains as CPU processes each request)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Active Connections Over Time
        ax = axes[1, 0]
        ax.fill_between(collector.timestamps, collector.async_active_connections,
                       alpha=0.5, color='steelblue')
        ax.plot(collector.timestamps, collector.async_active_connections,
               color='steelblue', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Active Connections')
        ax.set_title('Active Connections: All 5 connected simultaneously\n(But only 1 uses CPU at a time)')
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5 requests arrived')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Explanation Text
        ax = axes[1, 1]
        ax.axis('off')
        explanation = """
        HOW ASYNC SERVER WORKS:
        ════════════════════════

        1. SINGLE EVENT LOOP (like Node.js)
           • One "thread" handles everything
           • CPU work is done ONE request at a time

        2. THIS EXAMPLE:
           • 5 requests arrive at t=0
           • Each needs 100ms CPU work
           • Total time: 5 × 100ms = 500ms

        3. THE CPU QUEUE:
           • Request 1 starts immediately
           • Requests 2-5 wait in queue
           • Each starts when previous finishes

        4. WHY USE ASYNC SERVERS?
           • Great for I/O-bound work (DB, network)
           • I/O waits DON'T block other requests
           • Bad for CPU-heavy work (blocks everyone)

        KEY INSIGHT:
        ────────────
        "Don't block the event loop!"
        Heavy CPU work should be offloaded to workers.
        """
        ax.text(0.05, 0.95, explanation, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle('AsyncServer: CPU Work Serialization Explained', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / 'cpu_serialization_explained.png', dpi=150)
        plt.close(fig)

        # Write data
        _write_csv(
            test_output_dir / 'request_timeline.csv',
            header=['request_id', 'arrive_time_s', 'complete_time_s', 'total_time_ms'],
            rows=[[r[0], r[1], r[2], (r[2]-r[1])*1000] for r in sorted(tracker.request_timeline)]
        )

        # Verify behavior
        assert server.stats.requests_completed == 5
        # Total CPU time should be 500ms (5 × 100ms)
        assert server.stats.total_cpu_time == pytest.approx(0.5, rel=0.01)

        # The cumulative completed count should show staggered completions
        # At ~0.1s: 1 complete, at ~0.2s: 2 complete, etc.
        # Verify this through the collector data
        completed_at_times = list(zip(collector.timestamps, collector.async_completed))
        # Find when each completion happened
        completion_count = 0
        for t, c in completed_at_times:
            if c > completion_count:
                expected_time = c * 0.1  # Should complete at n * 0.1s
                assert t == pytest.approx(expected_time, abs=0.02), \
                    f"Request {c} should complete around {expected_time}s, got {t}s"
                completion_count = c

        print(f"\nVisualization saved to: {test_output_dir / 'cpu_serialization_explained.png'}")

    def test_async_vs_threaded_comparison(self, test_output_dir: Path):
        """
        DEMONSTRATES: AsyncServer vs ThreadPool Server comparison

        Scenario: Same workload on both server types
        - 20 requests over 2 seconds (10 req/s)
        - Each request needs 100ms of work

        AsyncServer (1 CPU thread):
        - Can only do 10 req/s (1 / 0.1s)
        - Queue will build up
        - High latency for later requests

        ThreadPool Server (4 workers):
        - Can do 40 req/s (4 / 0.1s)
        - No queue buildup at 10 req/s
        - Low, consistent latency
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        duration_s = 2.0
        request_rate = 10.0  # 10 req/s
        work_time = 0.100    # 100ms per request

        # --- Run AsyncServer scenario ---
        async_server = AsyncServer(
            name="async_server",
            max_connections=1000,
            cpu_work_distribution=ConstantLatency(work_time),
        )
        async_tracker = RequestTracker(name="async_tracker")
        async_collector = TimelineCollector(name="async_collector", sample_interval_s=0.02)
        async_collector.set_targets(async_server=async_server)

        async_provider = TrackedRequestProvider(
            target=async_server,
            tracker=async_tracker,
            stop_after=Instant.from_seconds(duration_s),
        )
        async_arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=request_rate),
            start_time=Instant.Epoch,
        )
        async_source = Source("async_source", async_provider, async_arrival)

        async_sim = Simulation(
            start_time=Instant.Epoch,
            duration=duration_s + 2.0,
            sources=[async_source],
            entities=[async_server, async_tracker, async_collector],
        )
        async_sim.schedule(Event(time=Instant.Epoch, event_type="collect", target=async_collector))
        async_sim.run()

        # --- Run ThreadPool Server scenario ---
        threaded_server = Server(
            name="threaded_server",
            concurrency=4,  # 4 workers
            service_time=ConstantLatency(work_time),
        )
        threaded_tracker = RequestTracker(name="threaded_tracker")
        threaded_collector = TimelineCollector(name="threaded_collector", sample_interval_s=0.02)
        threaded_collector.set_targets(server=threaded_server)

        threaded_provider = TrackedRequestProvider(
            target=threaded_server,
            tracker=threaded_tracker,
            stop_after=Instant.from_seconds(duration_s),
        )
        threaded_arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=request_rate),
            start_time=Instant.Epoch,
        )
        threaded_source = Source("threaded_source", threaded_provider, threaded_arrival)

        threaded_sim = Simulation(
            start_time=Instant.Epoch,
            duration=duration_s + 2.0,
            sources=[threaded_source],
            entities=[threaded_server, threaded_tracker, threaded_collector],
        )
        threaded_sim.schedule(Event(time=Instant.Epoch, event_type="collect", target=threaded_collector))
        threaded_sim.run()

        # --- Generate Comparison Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Queue Depth Comparison
        ax = axes[0, 0]
        ax.plot(async_collector.timestamps, async_collector.async_cpu_queue_depth,
               color='coral', linewidth=2, label='AsyncServer (1 CPU)')
        ax.plot(threaded_collector.timestamps, threaded_collector.server_queue_depth,
               color='steelblue', linewidth=2, label='ThreadPool (4 workers)')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Queue Depth')
        ax.set_title('Queue Depth Over Time\n(AsyncServer queues up, ThreadPool handles load easily)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Response Time Distribution
        ax = axes[0, 1]
        async_latencies = [(r[2] - r[1]) * 1000 for r in async_tracker.request_timeline]
        threaded_latencies = [(r[2] - r[1]) * 1000 for r in threaded_tracker.request_timeline]

        if async_latencies:
            ax.hist(async_latencies, bins=20, alpha=0.5, color='coral',
                   label=f'AsyncServer (avg: {sum(async_latencies)/len(async_latencies):.0f}ms)', edgecolor='black')
        if threaded_latencies:
            ax.hist(threaded_latencies, bins=20, alpha=0.5, color='steelblue',
                   label=f'ThreadPool (avg: {sum(threaded_latencies)/len(threaded_latencies):.0f}ms)', edgecolor='black')
        ax.axvline(x=100, color='green', linestyle='--', alpha=0.7, label='Work time (100ms)')
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Response Time Distribution\n(AsyncServer has high variance due to queueing)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Cumulative Completions
        ax = axes[1, 0]
        ax.plot(async_collector.timestamps, async_collector.async_completed,
               color='coral', linewidth=2, label='AsyncServer')
        ax.plot(threaded_collector.timestamps, threaded_collector.server_completed,
               color='steelblue', linewidth=2, label='ThreadPool')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Requests Completed')
        ax.set_title('Cumulative Completions\n(Both eventually complete all requests)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Comparison Summary
        ax = axes[1, 1]
        ax.axis('off')

        async_avg = sum(async_latencies)/len(async_latencies) if async_latencies else 0
        threaded_avg = sum(threaded_latencies)/len(threaded_latencies) if threaded_latencies else 0
        async_max = max(async_latencies) if async_latencies else 0
        threaded_max = max(threaded_latencies) if threaded_latencies else 0

        comparison = f"""
        COMPARISON: AsyncServer vs ThreadPool
        ══════════════════════════════════════

        WORKLOAD:
        • {request_rate:.0f} requests/second for {duration_s}s
        • Each request needs {work_time*1000:.0f}ms of work

        ASYNC SERVER (Single Event Loop):
        ┌─────────────────────────────────┐
        │ Capacity: {1/work_time:.0f} req/s (1 CPU)        │
        │ Completed: {async_server.stats.requests_completed}                    │
        │ Avg Latency: {async_avg:.0f}ms               │
        │ Max Latency: {async_max:.0f}ms               │
        │ [!] Queue buildup under load!    │
        └─────────────────────────────────┘

        THREAD POOL (4 Workers):
        ┌─────────────────────────────────┐
        │ Capacity: {4/work_time:.0f} req/s (4 workers)    │
        │ Completed: {threaded_server.stats.requests_completed}                    │
        │ Avg Latency: {threaded_avg:.0f}ms               │
        │ Max Latency: {threaded_max:.0f}ms               │
        │ [OK] Handles load easily        │
        └─────────────────────────────────┘

        KEY TAKEAWAY:
        AsyncServer excels at I/O-bound work (waiting),
        but struggles with CPU-bound work (computing).
        """
        ax.text(0.05, 0.95, comparison, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle('AsyncServer vs ThreadPool: CPU-Bound Workload Comparison',
                    fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / 'async_vs_threaded_comparison.png', dpi=150)
        plt.close(fig)

        print(f"\nVisualization saved to: {test_output_dir / 'async_vs_threaded_comparison.png'}")

    def test_connection_vs_cpu_capacity(self, test_output_dir: Path):
        """
        DEMONSTRATES: Connection limit vs CPU capacity

        AsyncServer can have HIGH connection limits but LOW CPU throughput.
        This is the key insight for async servers:
        - 10,000 connections possible
        - But only ~10 req/s if each needs 100ms CPU

        This test shows what happens when:
        1. Connections fill up (connection-limited)
        2. CPU can't keep up (CPU-limited)
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Scenario: Low connection limit, moderate CPU
        low_conn_server = AsyncServer(
            name="low_connections",
            max_connections=5,  # Only 5 connections allowed
            cpu_work_distribution=ConstantLatency(0.050),  # 50ms CPU
        )

        # Scenario: High connection limit, slow CPU
        slow_cpu_server = AsyncServer(
            name="slow_cpu",
            max_connections=1000,  # Many connections allowed
            cpu_work_distribution=ConstantLatency(0.200),  # 200ms CPU (slow!)
        )

        # Run both with same high load
        request_rate = 20.0  # 20 req/s
        duration_s = 3.0

        # --- Low connection scenario ---
        low_conn_collector = TimelineCollector(name="low_conn_collector", sample_interval_s=0.02)
        low_conn_collector.set_targets(async_server=low_conn_server)

        class SimpleProvider(EventProvider):
            def __init__(self, target, stop_after):
                self.target = target
                self.stop_after = stop_after
                self.generated = 0
            def get_events(self, time):
                if self.stop_after and time > self.stop_after:
                    return []
                self.generated += 1
                return [Event(time=time, event_type=f"Req-{self.generated}", target=self.target)]

        low_conn_provider = SimpleProvider(low_conn_server, Instant.from_seconds(duration_s))
        low_conn_source = Source("source", low_conn_provider,
            ConstantArrivalTimeProvider(ConstantRateProfile(request_rate), Instant.Epoch))

        low_conn_sim = Simulation(
            start_time=Instant.Epoch,
            duration=duration_s + 2.0,
            sources=[low_conn_source],
            entities=[low_conn_server, low_conn_collector],
        )
        low_conn_sim.schedule(Event(time=Instant.Epoch, event_type="collect", target=low_conn_collector))
        low_conn_sim.run()

        # --- Slow CPU scenario ---
        slow_cpu_collector = TimelineCollector(name="slow_cpu_collector", sample_interval_s=0.02)
        slow_cpu_collector.set_targets(async_server=slow_cpu_server)

        slow_cpu_provider = SimpleProvider(slow_cpu_server, Instant.from_seconds(duration_s))
        slow_cpu_source = Source("source", slow_cpu_provider,
            ConstantArrivalTimeProvider(ConstantRateProfile(request_rate), Instant.Epoch))

        slow_cpu_sim = Simulation(
            start_time=Instant.Epoch,
            duration=duration_s + 15.0,  # More time to drain
            sources=[slow_cpu_source],
            entities=[slow_cpu_server, slow_cpu_collector],
        )
        slow_cpu_sim.schedule(Event(time=Instant.Epoch, event_type="collect", target=slow_cpu_collector))
        slow_cpu_sim.run()

        # --- Generate Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Connection-Limited Server
        ax = axes[0, 0]
        ax.plot(low_conn_collector.timestamps, low_conn_collector.async_active_connections,
               color='steelblue', linewidth=2, label='Active Connections')
        ax.axhline(y=5, color='red', linestyle='--', label='Max Connections (5)')
        ax.fill_between(low_conn_collector.timestamps, low_conn_collector.async_active_connections,
                       alpha=0.3, color='steelblue')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Connections')
        ax.set_title(f'Connection-Limited (max=5)\nRejected: {low_conn_server.stats.requests_rejected} requests')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: CPU-Limited Server
        ax = axes[0, 1]
        ax.plot(slow_cpu_collector.timestamps, slow_cpu_collector.async_cpu_queue_depth,
               color='coral', linewidth=2, label='CPU Queue Depth')
        ax.fill_between(slow_cpu_collector.timestamps, slow_cpu_collector.async_cpu_queue_depth,
                       alpha=0.3, color='coral')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Queue Depth')
        ax.set_title(f'CPU-Limited (200ms/req = 5 req/s capacity)\nQueue builds up at {request_rate} req/s load')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Completions comparison
        ax = axes[1, 0]
        ax.plot(low_conn_collector.timestamps, low_conn_collector.async_completed,
               color='steelblue', linewidth=2, label='Connection-Limited')
        ax.plot(slow_cpu_collector.timestamps, slow_cpu_collector.async_completed,
               color='coral', linewidth=2, label='CPU-Limited')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Completed Requests')
        ax.set_title('Cumulative Completions\n(Both are bottlenecked, different reasons)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Explanation
        ax = axes[1, 1]
        ax.axis('off')
        explanation = f"""
        TWO WAYS TO BOTTLENECK AN ASYNC SERVER
        ═══════════════════════════════════════

        1. CONNECTION-LIMITED (Left plots):
           ┌────────────────────────────────┐
           │ Max Connections: 5             │
           │ CPU per request: 50ms          │
           │ CPU Capacity: 20 req/s         │
           │ Load: {request_rate:.0f} req/s                 │
           │                                │
           │ Result: Requests REJECTED      │
           │ Rejected: {low_conn_server.stats.requests_rejected} requests          │
           │ [!] Connection limit hit!       │
           └────────────────────────────────┘

        2. CPU-LIMITED (Right plots):
           ┌────────────────────────────────┐
           │ Max Connections: 1000          │
           │ CPU per request: 200ms         │
           │ CPU Capacity: 5 req/s          │
           │ Load: {request_rate:.0f} req/s                 │
           │                                │
           │ Result: QUEUE builds up        │
           │ Peak Queue: {max(slow_cpu_collector.async_cpu_queue_depth)}                  │
           │ [!] CPU is bottleneck!          │
           └────────────────────────────────┘

        KEY INSIGHT:
        ────────────
        Async servers can accept MANY connections,
        but CPU throughput is LIMITED by:

            Throughput = 1 / CPU_time_per_request
        """
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle('AsyncServer: Connection Limit vs CPU Capacity', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / 'connection_vs_cpu_capacity.png', dpi=150)
        plt.close(fig)

        print(f"\nVisualization saved to: {test_output_dir / 'connection_vs_cpu_capacity.png'}")

    def test_event_loop_blocking_demo(self, test_output_dir: Path):
        """
        DEMONSTRATES: "Don't block the event loop!"

        Shows what happens when ONE slow request blocks everyone else.
        This is the classic Node.js anti-pattern.

        Scenario:
        - Normal requests: 10ms CPU each
        - One SLOW request at t=0.5s: 500ms CPU (blocks everything!)
        - Watch how all subsequent requests get delayed
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Custom server that has one slow request
        class BlockingDemoServer(AsyncServer):
            def __init__(self, *args, slow_request_time: float = 0.5, **kwargs):
                super().__init__(*args, **kwargs)
                self.slow_request_time = slow_request_time
                self._slow_triggered = False
                self.request_completions: list[tuple[float, float, bool]] = []  # (arrive, complete, was_slow)

            def handle_event(self, event):
                # Track if this is the slow request
                if event.event_type.startswith("Request-"):
                    arrive_time = self.now.to_seconds()
                    is_slow = (0.45 <= arrive_time <= 0.55) and not self._slow_triggered

                    if is_slow:
                        self._slow_triggered = True
                        # Override CPU work for this request
                        event.context["_is_slow"] = True

                return super().handle_event(event)

        # Create server with variable CPU time
        from happysimulator.distributions.latency_distribution import LatencyDistribution
        from happysimulator.core.temporal import Duration

        class VariableCPU(LatencyDistribution):
            def __init__(self, normal_time: float, slow_time: float):
                self.normal_time = normal_time
                self.slow_time = slow_time
                self._next_is_slow = False

            def set_next_slow(self):
                self._next_is_slow = True

            def get_latency(self, time) -> Duration:
                if self._next_is_slow:
                    self._next_is_slow = False
                    return Duration.from_seconds(self.slow_time)
                return Duration.from_seconds(self.normal_time)

        variable_cpu = VariableCPU(normal_time=0.010, slow_time=0.500)

        server = AsyncServer(
            name="blocking_demo",
            max_connections=1000,
            cpu_work_distribution=variable_cpu,
        )

        # Track request times
        request_times: list[tuple[int, float, float, bool]] = []  # (id, arrive, complete, is_slow)

        collector = TimelineCollector(name="collector", sample_interval_s=0.01)
        collector.set_targets(async_server=server)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=2.0,
            sources=[],
            entities=[server, collector],
        )

        # Schedule requests every 50ms for 1.5 seconds
        request_id = 0
        for t in range(0, 1500, 50):  # Every 50ms
            t_sec = t / 1000.0
            request_id += 1
            rid = request_id
            arrive_time = t_sec

            # The request at t=0.5s will be slow
            is_slow = (0.45 <= t_sec <= 0.55)
            if is_slow:
                variable_cpu.set_next_slow()

            event = Event(
                time=Instant.from_seconds(t_sec),
                event_type=f"Request-{rid}",
                target=server,
            )

            def make_hook(r_id, arr_time, slow):
                def on_complete(finish_time: Instant):
                    request_times.append((r_id, arr_time, finish_time.to_seconds(), slow))
                    return None
                return on_complete

            event.add_completion_hook(make_hook(rid, arrive_time, is_slow))
            sim.schedule(event)

        sim.schedule(Event(time=Instant.Epoch, event_type="collect", target=collector))
        sim.run()

        # --- Generate Visualization ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sort by arrival time
        request_times.sort(key=lambda x: x[1])

        # Plot 1: Request Timeline showing the blocking
        ax = axes[0, 0]
        for rid, arrive, complete, is_slow in request_times:
            color = 'red' if is_slow else 'steelblue'
            alpha = 1.0 if is_slow else 0.6
            ax.barh(rid, complete - arrive, left=arrive, height=0.8,
                   color=color, alpha=alpha, edgecolor='black' if is_slow else 'none',
                   linewidth=2 if is_slow else 0)

        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        ax.annotate('SLOW REQUEST\n(500ms CPU)', xy=(0.5, 12), fontsize=9,
                   color='red', ha='center')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Request ID')
        ax.set_title('Request Timeline: One Slow Request Blocks Everyone!\n(Red bar = 500ms blocking request)')
        ax.grid(True, alpha=0.3, axis='x')

        # Plot 2: Response time over time
        ax = axes[0, 1]
        latencies = [(arrive, (complete - arrive) * 1000, is_slow)
                    for rid, arrive, complete, is_slow in request_times]

        normal_lat = [(a, l) for a, l, s in latencies if not s]
        slow_lat = [(a, l) for a, l, s in latencies if s]

        if normal_lat:
            ax.scatter([x[0] for x in normal_lat], [x[1] for x in normal_lat],
                      c='steelblue', alpha=0.6, label='Normal requests', s=50)
        if slow_lat:
            ax.scatter([x[0] for x in slow_lat], [x[1] for x in slow_lat],
                      c='red', s=100, marker='*', label='Slow request (500ms CPU)')

        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
        ax.set_xlabel('Arrival Time (seconds)')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title('Response Time vs Arrival Time\n(Spike after the blocking request)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: CPU Queue
        ax = axes[1, 0]
        ax.fill_between(collector.timestamps, collector.async_cpu_queue_depth,
                       alpha=0.5, color='coral')
        ax.plot(collector.timestamps, collector.async_cpu_queue_depth,
               color='coral', linewidth=2)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Slow request arrives')
        ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Slow request completes')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('CPU Queue Depth')
        ax.set_title('CPU Queue During Blocking\n(Queue builds up while slow request runs)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Explanation
        ax = axes[1, 1]
        ax.axis('off')
        explanation = """
        "DON'T BLOCK THE EVENT LOOP!"
        ══════════════════════════════

        WHAT HAPPENED:
        ──────────────
        1. Requests arriving every 50ms
        2. Normal requests: 10ms CPU each ✓
        3. At t=0.5s: ONE request needs 500ms CPU
        4. That request BLOCKS everything for 500ms!
        5. All requests arriving 0.5s-1.0s must wait

        THE PROBLEM:
        ────────────
        • Single-threaded event loop
        • CPU work cannot be interrupted
        • ONE slow operation affects EVERYONE

        REAL-WORLD EXAMPLES:
        ────────────────────
        [X] Synchronous file I/O
        [X] Complex regex on user input
        [X] JSON parsing large payloads
        [X] Image processing in request handler
        [X] Synchronous crypto operations

        SOLUTIONS:
        ──────────
        [OK] Use async/await for I/O
        [OK] Offload CPU work to worker threads
        [OK] Use streaming for large data
        [OK] Set timeouts on operations
        [OK] Use worker pools for CPU tasks
        """
        ax.text(0.02, 0.98, explanation, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        fig.suptitle('AsyncServer: Event Loop Blocking Demonstration', fontsize=14, fontweight='bold')
        fig.tight_layout()
        fig.savefig(test_output_dir / 'event_loop_blocking_demo.png', dpi=150)
        plt.close(fig)

        print(f"\nVisualization saved to: {test_output_dir / 'event_loop_blocking_demo.png'}")
