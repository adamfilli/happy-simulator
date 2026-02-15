"""Integration tests with visualizations for ConnectionPool and PooledClient.

These tests demonstrate connection pooling behavior through visual output,
showing connection reuse, wait queues, and the impact of pool sizing.

Run:
    pytest tests/integration/test_connection_pool_visualization.py -v

Output:
    tests/integration/output/connection_pool/<test_name>.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import pytest

from happysimulator.components.client.connection_pool import ConnectionPool
from happysimulator.components.client.pooled_client import PooledClient
from happysimulator.components.client.client import Client
from happysimulator.components.client.retry import FixedRetry
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


@dataclass
class DatabaseServer(Entity):
    """Simulates a database server with connection handling."""
    name: str
    query_time: float = 0.020  # 20ms query time
    connection_overhead: float = 0.005  # 5ms per-connection overhead

    queries_processed: int = field(default=0, init=False)
    _query_times: list = field(default_factory=list, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        start = self.now
        self.queries_processed += 1
        # Simulate query execution
        yield self.query_time + self.connection_overhead
        end = self.now
        self._query_times.append((start.to_seconds(), end.to_seconds()))


@dataclass
class VariableLatencyServer(Entity):
    """Server with variable response times based on load."""
    name: str
    base_time: float = 0.010
    load_factor: float = 0.001  # Additional time per concurrent request

    _active_requests: int = field(default=0, init=False)
    requests_processed: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self._active_requests += 1
        self.requests_processed += 1
        # Response time increases with load
        response_time = self.base_time + (self._active_requests * self.load_factor)
        yield response_time
        self._active_requests -= 1


class PooledClientRequestProvider(EventProvider):
    """Generates requests through a pooled client."""

    def __init__(self, client: PooledClient, stop_after: Instant | None = None):
        self.client = client
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        request = self.client.send_request(payload=f"query-{self.generated}")
        request.time = time
        return [request]


class DirectClientRequestProvider(EventProvider):
    """Generates requests through a direct client (no pooling)."""

    def __init__(self, client: Client, stop_after: Instant | None = None):
        self.client = client
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        request = self.client.send_request(payload=f"request-{self.generated}")
        request.time = time
        return [request]


class TestConnectionPoolVisualization:
    """Visual tests for connection pool behavior."""

    def test_connection_reuse_pattern(self, test_output_dir):
        """
        Visualize how connections are reused over time.

        Shows:
        - Connection creation events
        - Connection reuse patterns
        - Idle pool size over time
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        server = DatabaseServer(name="db", query_time=0.020)
        pool = ConnectionPool(
            name="pool",
            target=server,
            min_connections=0,
            max_connections=5,
            connection_latency=ConstantLatency(0.010),  # 10ms to establish
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        # Track events over time
        connection_events = []  # (time, event_type, connection_id)
        pool_state = []  # (time, active, idle, total)

        original_acquire = pool._activate_connection
        original_release = pool.release

        def track_acquire(conn):
            connection_events.append((pool.now.to_seconds(), "acquire", conn.id))
            pool_state.append((
                pool.now.to_seconds(),
                pool.active_connections + 1,  # Will be active after this
                pool.idle_connections,
                pool.total_connections,
            ))
            return original_acquire(conn)

        def track_release(conn):
            connection_events.append((pool.now.to_seconds(), "release", conn.id))
            result = original_release(conn)
            pool_state.append((
                pool.now.to_seconds(),
                pool.active_connections,
                pool.idle_connections,
                pool.total_connections,
            ))
            return result

        pool._activate_connection = track_acquire
        pool.release = track_release

        # Run simulation with moderate load
        provider = PooledClientRequestProvider(client, stop_after=Instant.from_seconds(2.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=50.0),  # 50 requests/sec
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=3.0,
            sources=[source],
            entities=[server, pool, client],
        )
        sim.run()

        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle("Connection Pool Reuse Pattern", fontsize=14, fontweight='bold')

        # Plot 1: Connection events timeline
        ax1 = axes[0]
        acquires = [(t, cid) for t, evt, cid in connection_events if evt == "acquire"]
        releases = [(t, cid) for t, evt, cid in connection_events if evt == "release"]

        if acquires:
            ax1.scatter([a[0] for a in acquires], [a[1] for a in acquires],
                       c='green', marker='o', s=20, alpha=0.6, label='Acquire')
        if releases:
            ax1.scatter([r[0] for r in releases], [r[1] for r in releases],
                       c='red', marker='x', s=20, alpha=0.6, label='Release')

        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Connection ID")
        ax1.set_title("Connection Acquire/Release Events")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Pool state over time
        ax2 = axes[1]
        if pool_state:
            times = [s[0] for s in pool_state]
            active = [s[1] for s in pool_state]
            idle = [s[2] for s in pool_state]
            total = [s[3] for s in pool_state]

            ax2.step(times, active, where='post', label='Active', color='blue', linewidth=1.5)
            ax2.step(times, idle, where='post', label='Idle', color='orange', linewidth=1.5)
            ax2.step(times, total, where='post', label='Total', color='green',
                    linestyle='--', linewidth=1.5)
            ax2.axhline(y=pool.max_connections, color='red', linestyle=':',
                       label=f'Max ({pool.max_connections})')

        ax2.set_xlabel("Time (seconds)")
        ax2.set_ylabel("Connection Count")
        ax2.set_title("Pool State Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Statistics summary
        ax3 = axes[2]
        ax3.axis('off')

        stats_text = f"""
Connection Pool Statistics:
--------------------------
Connections Created:    {pool.stats.connections_created}
Connections Closed:     {pool.stats.connections_closed}
Total Acquisitions:     {pool.stats.acquisitions}
Total Releases:         {pool.stats.releases}
Timeouts:               {pool.stats.timeouts}

Client Statistics:
-----------------
Requests Sent:          {client.stats.requests_sent}
Responses Received:     {client.stats.responses_received}
Average Response Time:  {client.average_response_time*1000:.2f} ms

Efficiency:
----------
Reuse Ratio:            {pool.stats.acquisitions / max(1, pool.stats.connections_created):.1f}x
"""
        ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        fig.savefig(test_output_dir / 'connection_reuse_pattern.png', dpi=150)
        plt.close(fig)

        # Verify reuse is happening
        assert pool.stats.connections_created < pool.stats.acquisitions
        assert client.stats.responses_received > 0

    def test_pool_sizing_impact(self, test_output_dir):
        """
        Compare different pool sizes under the same load.

        Shows how pool size affects:
        - Response latency
        - Wait time for connections
        - Connection utilization
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        results = []
        pool_sizes = [1, 2, 5, 10, 20]
        request_rate = 100  # requests/sec
        sim_duration = 2.0

        for pool_size in pool_sizes:
            server = VariableLatencyServer(name="server", base_time=0.015)
            pool = ConnectionPool(
                name="pool",
                target=server,
                max_connections=pool_size,
                connection_latency=ConstantLatency(0.005),
                connection_timeout=5.0,
                idle_timeout=300.0,
            )
            client = PooledClient(name="client", connection_pool=pool)

            provider = PooledClientRequestProvider(
                client, stop_after=Instant.from_seconds(sim_duration - 0.5)
            )
            arrival = ConstantArrivalTimeProvider(
                ConstantRateProfile(rate_per_s=request_rate),
                start_time=Instant.Epoch,
            )
            source = Source("source", provider, arrival)

            sim = Simulation(
                start_time=Instant.Epoch,
                duration=sim_duration + 1.0,
                sources=[source],
                entities=[server, pool, client],
            )
            sim.run()

            results.append({
                'pool_size': pool_size,
                'requests_sent': client.stats.requests_sent,
                'responses_received': client.stats.responses_received,
                'avg_response_time': client.average_response_time * 1000,  # ms
                'p50_response_time': client.get_response_time_percentile(0.50) * 1000,
                'p99_response_time': client.get_response_time_percentile(0.99) * 1000,
                'connections_created': pool.stats.connections_created,
                'avg_wait_time': pool.average_wait_time * 1000,
                'timeouts': client.stats.timeouts + client.stats.connection_wait_timeouts,
            })

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Connection Pool Sizing Impact (Load: {request_rate} req/s)",
                    fontsize=14, fontweight='bold')

        pool_sizes_arr = [r['pool_size'] for r in results]

        # Plot 1: Response time vs pool size
        ax1 = axes[0, 0]
        ax1.plot(pool_sizes_arr, [r['avg_response_time'] for r in results],
                'bo-', linewidth=2, markersize=8, label='Average')
        ax1.plot(pool_sizes_arr, [r['p50_response_time'] for r in results],
                'g^--', linewidth=1.5, markersize=6, label='P50')
        ax1.plot(pool_sizes_arr, [r['p99_response_time'] for r in results],
                'rs--', linewidth=1.5, markersize=6, label='P99')
        ax1.set_xlabel("Pool Size (max connections)")
        ax1.set_ylabel("Response Time (ms)")
        ax1.set_title("Response Latency vs Pool Size")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(pool_sizes_arr)

        # Plot 2: Wait time vs pool size
        ax2 = axes[0, 1]
        ax2.bar(pool_sizes_arr, [r['avg_wait_time'] for r in results],
               color='orange', alpha=0.7, width=0.8)
        ax2.set_xlabel("Pool Size (max connections)")
        ax2.set_ylabel("Average Wait Time (ms)")
        ax2.set_title("Connection Wait Time vs Pool Size")
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_xticks(pool_sizes_arr)

        # Plot 3: Throughput and success rate
        ax3 = axes[1, 0]
        success_rate = [r['responses_received'] / max(1, r['requests_sent']) * 100
                       for r in results]
        bars = ax3.bar(pool_sizes_arr, success_rate, color='green', alpha=0.7, width=0.8)
        ax3.set_xlabel("Pool Size (max connections)")
        ax3.set_ylabel("Success Rate (%)")
        ax3.set_title("Request Success Rate vs Pool Size")
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(pool_sizes_arr)

        # Add value labels on bars
        for bar, rate in zip(bars, success_rate):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

        # Plot 4: Connection efficiency
        ax4 = axes[1, 1]
        reuse_ratio = [r['requests_sent'] / max(1, r['connections_created'])
                      for r in results]
        ax4.bar(pool_sizes_arr, reuse_ratio, color='purple', alpha=0.7, width=0.8)
        ax4.set_xlabel("Pool Size (max connections)")
        ax4.set_ylabel("Requests per Connection")
        ax4.set_title("Connection Reuse Efficiency")
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_xticks(pool_sizes_arr)

        plt.tight_layout()
        fig.savefig(test_output_dir / 'pool_sizing_impact.png', dpi=150)
        plt.close(fig)

        # Save data to CSV
        with open(test_output_dir / 'pool_sizing_data.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        # Verify larger pools have lower wait times
        assert results[-1]['avg_wait_time'] <= results[0]['avg_wait_time']

    def test_pooled_vs_direct_client(self, test_output_dir):
        """
        Compare pooled client vs direct client performance.

        Shows the benefit of connection pooling when connection
        establishment has significant overhead.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        connection_overhead = 0.050  # 50ms to establish connection
        query_time = 0.010  # 10ms query
        request_rate = 50
        sim_duration = 2.0

        results = {}

        # Test 1: Pooled client
        server1 = DatabaseServer(name="db1", query_time=query_time,
                                connection_overhead=0.0)  # No overhead per query
        pool = ConnectionPool(
            name="pool",
            target=server1,
            max_connections=10,
            connection_latency=ConstantLatency(connection_overhead),
            idle_timeout=300.0,
        )
        pooled_client = PooledClient(name="pooled_client", connection_pool=pool)

        provider1 = PooledClientRequestProvider(
            pooled_client, stop_after=Instant.from_seconds(sim_duration - 0.2)
        )
        arrival1 = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=request_rate),
            start_time=Instant.Epoch,
        )
        source1 = Source("source1", provider1, arrival1)

        sim1 = Simulation(
            start_time=Instant.Epoch,
            duration=sim_duration + 0.5,
            sources=[source1],
            entities=[server1, pool, pooled_client],
        )
        sim1.run()

        results['pooled'] = {
            'requests': pooled_client.stats.requests_sent,
            'responses': pooled_client.stats.responses_received,
            'avg_response_time': pooled_client.average_response_time * 1000,
            'p50': pooled_client.get_response_time_percentile(0.50) * 1000,
            'p99': pooled_client.get_response_time_percentile(0.99) * 1000,
            'connections': pool.stats.connections_created,
        }

        # Test 2: Direct client (simulates connection per request)
        # We simulate this with a server that includes connection overhead
        server2 = DatabaseServer(name="db2", query_time=query_time,
                                connection_overhead=connection_overhead)
        direct_client = Client(name="direct_client", target=server2)

        provider2 = DirectClientRequestProvider(
            direct_client, stop_after=Instant.from_seconds(sim_duration - 0.2)
        )
        arrival2 = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=request_rate),
            start_time=Instant.Epoch,
        )
        source2 = Source("source2", provider2, arrival2)

        sim2 = Simulation(
            start_time=Instant.Epoch,
            duration=sim_duration + 0.5,
            sources=[source2],
            entities=[server2, direct_client],
        )
        sim2.run()

        results['direct'] = {
            'requests': direct_client.stats.requests_sent,
            'responses': direct_client.stats.responses_received,
            'avg_response_time': direct_client.average_response_time * 1000,
            'p50': direct_client.get_response_time_percentile(0.50) * 1000,
            'p99': direct_client.get_response_time_percentile(0.99) * 1000,
            'connections': direct_client.stats.requests_sent,  # One per request
        }

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Pooled vs Direct Client (Connection Overhead: {connection_overhead*1000:.0f}ms)",
                    fontsize=14, fontweight='bold')

        labels = ['Pooled', 'Direct']
        x = np.arange(len(labels))
        width = 0.35

        # Plot 1: Response time comparison
        ax1 = axes[0]
        avg_times = [results['pooled']['avg_response_time'],
                    results['direct']['avg_response_time']]
        p99_times = [results['pooled']['p99'], results['direct']['p99']]

        bars1 = ax1.bar(x - width/2, avg_times, width, label='Average', color='steelblue')
        bars2 = ax1.bar(x + width/2, p99_times, width, label='P99', color='coral')

        ax1.set_ylabel('Response Time (ms)')
        ax1.set_title('Response Time Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar in bars1:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

        # Plot 2: Connections used
        ax2 = axes[1]
        connections = [results['pooled']['connections'], results['direct']['connections']]
        bars = ax2.bar(labels, connections, color=['green', 'orange'], alpha=0.7)
        ax2.set_ylabel('Connections Created')
        ax2.set_title('Connection Overhead')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, conn in zip(bars, connections):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{conn}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 3: Summary comparison
        ax3 = axes[2]
        ax3.axis('off')

        speedup = results['direct']['avg_response_time'] / max(0.01, results['pooled']['avg_response_time'])
        conn_savings = (1 - results['pooled']['connections'] / max(1, results['direct']['connections'])) * 100

        summary_text = f"""
Performance Comparison
======================

Pooled Client:
  - Avg Response: {results['pooled']['avg_response_time']:.2f} ms
  - P99 Response: {results['pooled']['p99']:.2f} ms
  - Connections:  {results['pooled']['connections']}

Direct Client:
  - Avg Response: {results['direct']['avg_response_time']:.2f} ms
  - P99 Response: {results['direct']['p99']:.2f} ms
  - Connections:  {results['direct']['connections']}

Benefits of Pooling:
  - Latency Improvement: {speedup:.1f}x faster
  - Connection Savings:  {conn_savings:.0f}% fewer
"""
        ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        fig.savefig(test_output_dir / 'pooled_vs_direct.png', dpi=150)
        plt.close(fig)

        # Pooled should be faster due to connection reuse
        assert results['pooled']['avg_response_time'] < results['direct']['avg_response_time']

    def test_burst_handling(self, test_output_dir):
        """
        Test how the pool handles traffic bursts.

        Shows:
        - Pool expansion during bursts
        - Wait queue formation
        - Recovery after burst
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        server = DatabaseServer(name="db", query_time=0.020)
        pool = ConnectionPool(
            name="pool",
            target=server,
            min_connections=2,
            max_connections=10,
            connection_latency=ConstantLatency(0.010),
            connection_timeout=2.0,
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        # Track pool state over time
        pool_history = []  # (time, active, idle, pending)

        original_acquire = pool.acquire

        def track_state_acquire():
            gen = original_acquire()
            try:
                while True:
                    val = next(gen)
                    pool_history.append((
                        pool.now.to_seconds(),
                        pool.active_connections,
                        pool.idle_connections,
                        pool.pending_requests,
                    ))
                    val = yield val
            except StopIteration as e:
                pool_history.append((
                    pool.now.to_seconds(),
                    pool.active_connections,
                    pool.idle_connections,
                    pool.pending_requests,
                ))
                return e.value

        pool.acquire = track_state_acquire

        # Create burst pattern: low -> high -> low
        burst_events = []

        # Phase 1: Low rate (0-1s)
        for i in range(10):
            t = i * 0.1
            request = client.send_request(payload=f"low-{i}")
            request.time = Instant.from_seconds(t)
            burst_events.append(request)

        # Phase 2: High burst (1-1.5s) - 100 requests in 0.5s
        for i in range(50):
            t = 1.0 + (i * 0.01)
            request = client.send_request(payload=f"burst-{i}")
            request.time = Instant.from_seconds(t)
            burst_events.append(request)

        # Phase 3: Low rate again (2-3s)
        for i in range(10):
            t = 2.0 + (i * 0.1)
            request = client.send_request(payload=f"low2-{i}")
            request.time = Instant.from_seconds(t)
            burst_events.append(request)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[],
            entities=[server, pool, client],
        )

        for event in burst_events:
            sim.schedule(event)

        # Warmup pool
        warmup = pool.warmup()
        warmup.time = Instant.Epoch
        sim.schedule(warmup)

        sim.run()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Connection Pool Burst Handling", fontsize=14, fontweight='bold')

        # Plot 1: Pool state timeline
        ax1 = axes[0, 0]
        if pool_history:
            times = [h[0] for h in pool_history]
            active = [h[1] for h in pool_history]
            pending = [h[3] for h in pool_history]

            ax1.fill_between(times, active, alpha=0.3, color='blue', label='Active Connections')
            ax1.plot(times, active, 'b-', linewidth=1)
            ax1.fill_between(times, pending, alpha=0.3, color='red', label='Pending Requests')
            ax1.plot(times, pending, 'r-', linewidth=1)
            ax1.axhline(y=pool.max_connections, color='green', linestyle='--',
                       label=f'Max Connections ({pool.max_connections})')

        # Mark burst period
        ax1.axvspan(1.0, 1.5, alpha=0.1, color='yellow', label='Burst Period')
        ax1.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Count")
        ax1.set_title("Pool State During Burst")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Plot 2: Request timeline
        ax2 = axes[0, 1]
        phases = [
            (0, 1.0, 'Normal (10 req)', 'green'),
            (1.0, 1.5, 'Burst (50 req)', 'red'),
            (2.0, 3.0, 'Normal (10 req)', 'green'),
        ]
        for start, end, label, color in phases:
            ax2.axvspan(start, end, alpha=0.3, color=color, label=label)

        ax2.set_xlim(0, 3.5)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time (seconds)")
        ax2.set_title("Traffic Pattern")
        ax2.legend()
        ax2.set_yticks([])

        # Plot 3: Statistics by phase
        ax3 = axes[1, 0]
        stats_labels = ['Sent', 'Received', 'Timeouts', 'Failures']
        stats_values = [
            client.stats.requests_sent,
            client.stats.responses_received,
            client.stats.timeouts,
            client.stats.failures,
        ]
        colors = ['steelblue', 'green', 'orange', 'red']
        bars = ax3.bar(stats_labels, stats_values, color=colors, alpha=0.7)
        ax3.set_ylabel("Count")
        ax3.set_title("Request Statistics")
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, stats_values):
            if val > 0:
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val}', ha='center', va='bottom', fontsize=10)

        # Plot 4: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        success_rate = client.stats.responses_received / max(1, client.stats.requests_sent) * 100

        summary = f"""
Burst Handling Summary
======================

Traffic Pattern:
  - Phase 1: 10 requests over 1s (10 req/s)
  - Phase 2: 50 requests over 0.5s (100 req/s) - BURST
  - Phase 3: 10 requests over 1s (10 req/s)

Pool Configuration:
  - Min Connections: {pool.min_connections}
  - Max Connections: {pool.max_connections}
  - Connection Timeout: {pool.connection_timeout}s

Results:
  - Total Requests:      {client.stats.requests_sent}
  - Successful:          {client.stats.responses_received}
  - Success Rate:        {success_rate:.1f}%
  - Connections Created: {pool.stats.connections_created}
  - Max Active (peak):   {max(h[1] for h in pool_history) if pool_history else 0}
"""
        ax4.text(0.05, 0.5, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        fig.savefig(test_output_dir / 'burst_handling.png', dpi=150)
        plt.close(fig)

        # Verify pool handled the burst
        assert client.stats.responses_received > 0
        assert pool.stats.connections_created <= pool.max_connections

    def test_connection_pool_warmup(self, test_output_dir):
        """
        Compare cold start vs warmed up pool performance.

        Shows the benefit of pre-warming connections.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        results = {'cold': {}, 'warm': {}}
        connection_time = 0.050  # 50ms to establish
        query_time = 0.010

        for scenario in ['cold', 'warm']:
            server = DatabaseServer(name=f"db_{scenario}", query_time=query_time)
            pool = ConnectionPool(
                name=f"pool_{scenario}",
                target=server,
                min_connections=5 if scenario == 'warm' else 0,
                max_connections=10,
                connection_latency=ConstantLatency(connection_time),
                idle_timeout=300.0,
            )
            client = PooledClient(name=f"client_{scenario}", connection_pool=pool)

            # Track first N request latencies
            request_latencies = []
            original_handle = client.handle_event

            def track_latency(event, start_times={}):
                req_id = event.context.get("metadata", {}).get("request_id")
                if event.event_type not in ["_pooled_client_timeout", "_pooled_client_response"]:
                    start_times[req_id] = client.now.to_seconds()
                result = original_handle(event)
                if event.event_type == "_pooled_client_response" and req_id in start_times:
                    latency = client.now.to_seconds() - start_times[req_id]
                    request_latencies.append((req_id, latency * 1000))
                return result

            client.handle_event = track_latency

            sim = Simulation(
                start_time=Instant.Epoch,
                duration=2.0,
                sources=[],
                entities=[server, pool, client],
            )

            # Warmup if needed
            if scenario == 'warm':
                warmup = pool.warmup()
                warmup.time = Instant.Epoch
                sim.schedule(warmup)

            # Send burst of requests at start
            for i in range(20):
                request = client.send_request(payload=f"req-{i}")
                request.time = Instant.from_seconds(0.1 + i * 0.01)
                sim.schedule(request)

            sim.run()

            results[scenario] = {
                'latencies': request_latencies[:10],  # First 10
                'avg_first_10': np.mean([l[1] for l in request_latencies[:10]]) if request_latencies else 0,
                'connections_created': pool.stats.connections_created,
            }

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Connection Pool Warmup Impact (Connection Time: {connection_time*1000:.0f}ms)",
                    fontsize=14, fontweight='bold')

        # Plot 1: First 10 request latencies
        ax1 = axes[0]
        x = range(1, 11)
        if results['cold']['latencies']:
            cold_latencies = [l[1] for l in results['cold']['latencies']]
            ax1.plot(x, cold_latencies, 'ro-', label='Cold Start', linewidth=2, markersize=8)
        if results['warm']['latencies']:
            warm_latencies = [l[1] for l in results['warm']['latencies']]
            ax1.plot(x, warm_latencies, 'go-', label='Warm Start', linewidth=2, markersize=8)

        ax1.set_xlabel("Request Number")
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("First 10 Request Latencies")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Average latency comparison
        ax2 = axes[1]
        scenarios = ['Cold Start', 'Warm Start']
        avg_latencies = [results['cold']['avg_first_10'], results['warm']['avg_first_10']]
        colors = ['coral', 'lightgreen']
        bars = ax2.bar(scenarios, avg_latencies, color=colors, alpha=0.8)
        ax2.set_ylabel("Average Latency (ms)")
        ax2.set_title("Average Latency (First 10 Requests)")
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, avg_latencies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}ms', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Plot 3: Summary
        ax3 = axes[2]
        ax3.axis('off')

        improvement = ((results['cold']['avg_first_10'] - results['warm']['avg_first_10']) /
                      max(0.01, results['cold']['avg_first_10']) * 100)

        summary = f"""
Warmup Impact Summary
=====================

Cold Start:
  - Avg First 10: {results['cold']['avg_first_10']:.2f} ms
  - Connections Created: {results['cold']['connections_created']}
  - First requests pay connection cost

Warm Start:
  - Avg First 10: {results['warm']['avg_first_10']:.2f} ms
  - Connections Created: {results['warm']['connections_created']}
  - Pre-warmed 5 connections

Improvement:
  - Latency Reduction: {improvement:.0f}%
  - Benefit: Eliminates cold start penalty

Recommendation:
  For latency-sensitive workloads,
  pre-warm connections at startup.
"""
        ax3.text(0.05, 0.5, summary, transform=ax3.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        plt.tight_layout()
        fig.savefig(test_output_dir / 'warmup_impact.png', dpi=150)
        plt.close(fig)

        # Warm start should be faster
        if results['cold']['avg_first_10'] > 0 and results['warm']['avg_first_10'] > 0:
            assert results['warm']['avg_first_10'] <= results['cold']['avg_first_10']
