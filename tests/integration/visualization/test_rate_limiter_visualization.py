"""Integration tests with visualizations for rate limiter components.

These tests demonstrate rate limiting behavior through visual output,
showing request acceptance, rejection, and rate adaptation.

Run:
    pytest tests/integration/test_rate_limiter_visualization.py -v

Output:
    test_output/test_rate_limiter_visualization/<test_name>/
"""

from __future__ import annotations

import random
from typing import List, Generator, Any

import pytest

from happysimulator.components.rate_limiter import (
    RateLimitedEntity,
    FixedWindowPolicy,
    AdaptivePolicy,
    DistributedRateLimiter,
    RateAdjustmentReason,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class DummyServer(Entity):
    """Simple server for testing."""

    def __init__(self, name: str = "server"):
        super().__init__(name)
        self.requests_handled = 0

    def handle_event(self, event: Event) -> list[Event]:
        self.requests_handled += 1
        return []


class MockKVStore(Entity):
    """Mock KVStore for distributed rate limiter testing."""

    def __init__(self):
        super().__init__("mock_store")
        self._data: dict[str, Any] = {}

    def handle_event(self, event: Event) -> list[Event]:
        return []

    def get(self, key: str) -> Generator[float, None, Any]:
        yield 0.001  # 1ms latency
        return self._data.get(key)

    def put(self, key: str, value: Any) -> Generator[float, None, None]:
        yield 0.001  # 1ms latency
        self._data[key] = value


class TestFixedWindowVisualization:
    """Visual tests for FixedWindowPolicy behavior."""

    def test_fixed_window_boundary_burst(self, test_output_dir):
        """
        Visualize the boundary burst problem in fixed window limiting.

        Shows how requests clustered at window boundaries can exceed
        the intended rate limit.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        window_size = 1.0
        requests_per_window = 10
        server = DummyServer()
        policy = FixedWindowPolicy(
            requests_per_window=requests_per_window,
            window_size=window_size,
        )
        limiter = RateLimitedEntity(
            name="fixed_window",
            downstream=server,
            policy=policy,
            queue_capacity=10000,
        )

        # Need a simulation for clock
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            sources=[],
            entities=[limiter, server],
        )

        # Generate requests clustered at window boundary
        # 10 requests at t=0.9 (end of window 0) + 10 requests at t=1.1 (start of window 1)
        request_times = []
        for i in range(10):
            request_times.append(0.9 + i * 0.01)  # 0.9, 0.91, ..., 0.99
        for i in range(10):
            request_times.append(1.0 + i * 0.01)  # 1.0, 1.01, ..., 1.09

        forwarded = []
        queued = []

        for t in request_times:
            event = Event(
                time=Instant.from_seconds(t),
                event_type="request",
                target=limiter,
            )
            result = limiter.handle_event(event)
            # Check if a forward event was produced (vs just poll events)
            has_forward = any(
                e.event_type.startswith("forward::") for e in result
            )
            if has_forward:
                forwarded.append(t)
            else:
                queued.append(t)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Request timeline
        ax1.scatter(forwarded, [1] * len(forwarded), c='green', marker='o', s=100, label='Forwarded', alpha=0.7)
        ax1.scatter(queued, [1] * len(queued), c='orange', marker='s', s=100, label='Queued', alpha=0.7)

        # Window boundaries
        ax1.axvline(x=1.0, color='blue', linestyle='--', linewidth=2, label='Window boundary')
        ax1.fill_betweenx([0.5, 1.5], 0, 1.0, alpha=0.1, color='blue', label='Window 0')
        ax1.fill_betweenx([0.5, 1.5], 1.0, 2.0, alpha=0.1, color='orange', label='Window 1')

        ax1.set_xlim(0.8, 1.2)
        ax1.set_ylim(0.5, 1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_yticks([])
        ax1.set_title(f'Fixed Window Boundary Burst (limit={requests_per_window}/window)')
        ax1.legend(loc='upper right')

        # Cumulative request count
        all_times = sorted(forwarded)
        counts = list(range(1, len(all_times) + 1))

        ax2.step(all_times, counts, where='post', linewidth=2, color='green')
        ax2.axvline(x=1.0, color='blue', linestyle='--', linewidth=2, label='Window boundary')
        ax2.axhline(y=requests_per_window, color='red', linestyle='--', label='Per-window limit')

        # Highlight the burst
        burst_count = len([t for t in forwarded if 0.9 <= t <= 1.1])
        ax2.annotate(f'{burst_count} requests in 0.2s window\n(2x limit allows burst)',
                    xy=(1.0, burst_count / 2),
                    xytext=(1.15, burst_count / 2),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cumulative Forwarded Requests')
        ax2.set_title('Boundary Burst: Requests Passed Near Window Edge')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(test_output_dir / 'fixed_window_boundary_burst.png', dpi=150)
        plt.close()

        # Verify the boundary burst effect — all 20 forwarded immediately
        assert len(forwarded) == 20  # 10 per window = 20 total in 0.2s span


class TestAdaptiveVisualization:
    """Visual tests for AdaptivePolicy behavior."""

    def test_adaptive_rate_convergence(self, test_output_dir):
        """
        Visualize adaptive rate limiter converging to optimal rate.

        Shows how AIMD algorithm adjusts rate based on success/failure feedback.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Simulate a server with capacity of ~50 req/s
        server_capacity = 50.0
        server = DummyServer()

        policy = AdaptivePolicy(
            initial_rate=10.0,
            min_rate=5.0,
            max_rate=200.0,
            increase_step=5.0,
            decrease_factor=0.7,
        )
        limiter = RateLimitedEntity(
            name="adaptive",
            downstream=server,
            policy=policy,
            queue_capacity=10000,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(20.0),
            sources=[],
            entities=[limiter, server],
        )

        # Simulate requests over time
        times = []
        rates = []
        outcomes = []  # 'success', 'failure', 'dropped'

        current_time = 0.0
        random.seed(42)

        for i in range(200):
            # Send a request
            event = Event(
                time=Instant.from_seconds(current_time),
                event_type="request",
                target=limiter,
            )
            result = limiter.handle_event(event)

            has_forward = any(
                e.event_type.startswith("forward::") for e in result
            )

            if has_forward:
                # Request was forwarded - simulate success/failure based on load
                failure_probability = max(0, (policy.current_rate - server_capacity) / server_capacity)
                if random.random() < failure_probability:
                    policy.record_failure(Instant.from_seconds(current_time))
                    outcomes.append('failure')
                else:
                    policy.record_success(Instant.from_seconds(current_time))
                    outcomes.append('success')
            else:
                outcomes.append('queued')

            times.append(current_time)
            rates.append(policy.current_rate)
            current_time += 0.05  # 20 req/s incoming

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Rate over time
        ax1 = axes[0, 0]
        ax1.plot(times, rates, 'b-', linewidth=2)
        ax1.axhline(y=server_capacity, color='red', linestyle='--', linewidth=2, label=f'Server capacity ({server_capacity}/s)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Rate Limit (req/s)')
        ax1.set_title('Adaptive Rate Limiter: Rate Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rate history with annotations
        ax2 = axes[0, 1]
        for snapshot in policy.rate_history[:50]:  # First 50 changes
            color = 'green' if snapshot.reason == RateAdjustmentReason.SUCCESS else 'red'
            marker = '^' if snapshot.reason == RateAdjustmentReason.SUCCESS else 'v'
            ax2.scatter(snapshot.time.to_seconds(), snapshot.rate, c=color, marker=marker, s=50, alpha=0.6)

        ax2.axhline(y=server_capacity, color='red', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Rate')
        ax2.set_title('Rate Adjustments (^ increase, v decrease)')
        ax2.grid(True, alpha=0.3)

        # Outcome distribution
        ax3 = axes[1, 0]
        outcome_counts = {
            'success': outcomes.count('success'),
            'failure': outcomes.count('failure'),
            'queued': outcomes.count('queued'),
        }
        colors = {'success': 'green', 'failure': 'red', 'queued': 'orange'}
        bars = ax3.bar(outcome_counts.keys(), outcome_counts.values(),
                      color=[colors[k] for k in outcome_counts.keys()], alpha=0.7)
        ax3.set_ylabel('Count')
        ax3.set_title('Request Outcomes')
        ax3.grid(True, alpha=0.3, axis='y')

        for bar, count in zip(bars, outcome_counts.values()):
            ax3.annotate(str(count), xy=(bar.get_x() + bar.get_width() / 2, count),
                        ha='center', va='bottom')

        # AIMD explanation
        ax4 = axes[1, 1]
        ax4.axis('off')
        explanation = f"""
AIMD (Additive Increase, Multiplicative Decrease)

Configuration:
  - Initial rate: 10 req/s
  - Min rate: 5 req/s
  - Max rate: 200 req/s
  - Increase step: +5 req/s on success
  - Decrease factor: 0.7x on failure

Behavior:
  - Rate slowly increases on success
  - Rate quickly drops on failure
  - Converges to sustainable throughput

Results:
  - Final rate: {rates[-1]:.1f} req/s
  - Server capacity: {server_capacity} req/s
  - Successes: {outcome_counts['success']}
  - Failures: {outcome_counts['failure']}
  - Queued: {outcome_counts['queued']}
"""
        ax4.text(0.1, 0.9, explanation, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Adaptive Rate Limiter with AIMD', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'adaptive_rate_convergence.png', dpi=150)
        plt.close()

        # Verify rate adapted reasonably
        assert rates[-1] > rates[0]  # Should have increased from initial low rate


class TestDistributedVisualization:
    """Visual tests for DistributedRateLimiter behavior."""

    def test_distributed_coordination(self, test_output_dir):
        """
        Visualize distributed rate limiting across multiple instances.

        Shows how multiple limiters share a global limit via backing store.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        store = MockKVStore()
        global_limit = 20
        window_size = 1.0

        # Create multiple limiter instances
        servers = [DummyServer(f"server{i}") for i in range(3)]
        limiters = [
            DistributedRateLimiter(
                name=f"limiter{i}",
                downstream=servers[i],
                backing_store=store,
                global_limit=global_limit,
                window_size=window_size,
            )
            for i in range(3)
        ]

        # Simulate requests across instances
        request_log = {i: {'forwarded': [], 'dropped': []} for i in range(3)}
        random.seed(42)

        for t_ms in range(500):  # 500ms of simulation
            t = t_ms / 1000.0
            instance = random.randint(0, 2)

            event = Event(
                time=Instant.from_seconds(t),
                event_type="request",
                target=limiters[instance],
            )

            gen = limiters[instance].handle_event(event)
            result = None
            try:
                while True:
                    next(gen)
            except StopIteration as e:
                result = e.value

            if result:
                request_log[instance]['forwarded'].append(t)
            else:
                request_log[instance]['dropped'].append(t)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ['#e74c3c', '#3498db', '#2ecc71']

        # Request timeline by instance
        ax1 = axes[0, 0]
        for i in range(3):
            y_base = i
            forwarded = request_log[i]['forwarded']
            dropped = request_log[i]['dropped']

            ax1.scatter(forwarded, [y_base + 0.2] * len(forwarded),
                       c=colors[i], marker='o', s=30, alpha=0.6, label=f'Instance {i} forwarded')
            ax1.scatter(dropped, [y_base - 0.2] * len(dropped),
                       c='gray', marker='x', s=20, alpha=0.4)

        ax1.set_yticks(range(3))
        ax1.set_yticklabels([f'Instance {i}' for i in range(3)])
        ax1.set_xlabel('Time (s)')
        ax1.set_title('Request Distribution Across Instances')
        ax1.grid(True, alpha=0.3, axis='x')

        # Cumulative forwarded requests
        ax2 = axes[0, 1]
        all_forwarded = []
        for i in range(3):
            all_forwarded.extend([(t, i) for t in request_log[i]['forwarded']])
        all_forwarded.sort()

        times = [t for t, _ in all_forwarded]
        cumulative = list(range(1, len(times) + 1))

        ax2.step(times, cumulative, where='post', linewidth=2, color='blue')
        ax2.axhline(y=global_limit, color='red', linestyle='--', label=f'Global limit ({global_limit})')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Cumulative Forwarded')
        ax2.set_title('Global Request Count (Shared Limit)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Per-instance statistics
        ax3 = axes[1, 0]
        instance_stats = []
        for i in range(3):
            forwarded = len(request_log[i]['forwarded'])
            dropped = len(request_log[i]['dropped'])
            instance_stats.append({'forwarded': forwarded, 'dropped': dropped})

        x = np.arange(3)
        width = 0.35
        ax3.bar(x - width/2, [s['forwarded'] for s in instance_stats], width,
               label='Forwarded', color='green', alpha=0.7)
        ax3.bar(x + width/2, [s['dropped'] for s in instance_stats], width,
               label='Dropped', color='red', alpha=0.7)
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'Instance {i}' for i in range(3)])
        ax3.set_ylabel('Request Count')
        ax3.set_title('Per-Instance Statistics')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        total_forwarded = sum(len(request_log[i]['forwarded']) for i in range(3))
        total_dropped = sum(len(request_log[i]['dropped']) for i in range(3))
        total_requests = total_forwarded + total_dropped

        summary = f"""
Distributed Rate Limiter Summary

Configuration:
  - Global limit: {global_limit} requests/window
  - Window size: {window_size}s
  - Instances: 3

Results:
  - Total requests: {total_requests}
  - Total forwarded: {total_forwarded}
  - Total dropped: {total_dropped}
  - Drop rate: {total_dropped / total_requests * 100:.1f}%

Per-Instance Breakdown:
"""
        for i in range(3):
            summary += f"  Instance {i}: {instance_stats[i]['forwarded']} forwarded, {instance_stats[i]['dropped']} dropped\n"

        ax4.text(0.1, 0.9, summary, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.suptitle('Distributed Rate Limiting Across Multiple Instances', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'distributed_coordination.png', dpi=150)
        plt.close()

        # Verify global limit was approximately respected per window
        first_window_forwarded = sum(
            1 for t in all_forwarded if t[0] < 1.0
        )
        assert first_window_forwarded <= global_limit


class TestRateLimiterComparison:
    """Compare different rate limiting strategies."""

    def test_algorithm_comparison(self, test_output_dir):
        """
        Compare fixed window and adaptive limiters.

        Shows different characteristics of each algorithm.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Simulate bursty traffic
        np.random.seed(42)
        request_times = []
        t = 0.0
        for _ in range(200):
            if np.random.random() < 0.2:
                burst_size = np.random.randint(5, 11)
                for _ in range(burst_size):
                    request_times.append(t)
                    t += 0.001
            else:
                t += np.random.exponential(0.05)
                request_times.append(t)

        # Test each limiter
        results = {}

        # Fixed Window
        server1 = DummyServer()
        fw_policy = FixedWindowPolicy(requests_per_window=50, window_size=1.0)
        fw_limiter = RateLimitedEntity(
            name="fixed", downstream=server1, policy=fw_policy, queue_capacity=10000,
        )

        sim1 = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(20.0),
            sources=[],
            entities=[fw_limiter, server1],
        )

        fw_forwarded = []
        fw_queued = []
        for t in request_times:
            event = Event(time=Instant.from_seconds(t), event_type="req", target=fw_limiter)
            result = fw_limiter.handle_event(event)
            has_forward = any(e.event_type.startswith("forward::") for e in result)
            if has_forward:
                fw_forwarded.append(t)
            else:
                fw_queued.append(t)
        results['Fixed Window'] = {'forwarded': fw_forwarded, 'queued': fw_queued}

        # Adaptive (simulating good conditions)
        server2 = DummyServer()
        adp_policy = AdaptivePolicy(
            initial_rate=50.0, min_rate=10.0, max_rate=100.0,
        )
        adaptive_limiter = RateLimitedEntity(
            name="adaptive", downstream=server2, policy=adp_policy, queue_capacity=10000,
        )

        sim2 = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(20.0),
            sources=[],
            entities=[adaptive_limiter, server2],
        )

        adp_forwarded = []
        adp_queued = []
        for t in request_times:
            event = Event(time=Instant.from_seconds(t), event_type="req", target=adaptive_limiter)
            result = adaptive_limiter.handle_event(event)
            has_forward = any(e.event_type.startswith("forward::") for e in result)
            if has_forward:
                adp_forwarded.append(t)
                adp_policy.record_success(Instant.from_seconds(t))
            else:
                adp_queued.append(t)
        results['Adaptive'] = {'forwarded': adp_forwarded, 'queued': adp_queued}

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Forwarded over time comparison
        ax1 = axes[0, 0]
        for name, data in results.items():
            times = sorted(data['forwarded'])
            cumulative = list(range(1, len(times) + 1))
            ax1.step(times, cumulative, where='post', linewidth=2, label=name)

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Cumulative Forwarded')
        ax1.set_title('Forwarded Requests Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Queue rate comparison
        ax2 = axes[0, 1]
        names = list(results.keys())
        total_per_algo = [
            len(results[name]['forwarded']) + len(results[name]['queued'])
            for name in names
        ]
        queue_rates = [
            len(results[name]['queued']) / total * 100 if total > 0 else 0
            for name, total in zip(names, total_per_algo)
        ]
        colors = ['#3498db', '#e74c3c']
        bars = ax2.bar(names, queue_rates, color=colors, alpha=0.7)
        ax2.set_ylabel('Queue Rate (%)')
        ax2.set_title('Request Queue Rate by Algorithm')
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, rate in zip(bars, queue_rates):
            ax2.annotate(f'{rate:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, rate + 1),
                        ha='center')

        # Traffic pattern
        ax3 = axes[1, 0]
        ax3.hist(request_times, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Request Count')
        ax3.set_title('Input Traffic Pattern (Bursty)')
        ax3.grid(True, alpha=0.3)

        # Algorithm characteristics
        ax4 = axes[1, 1]
        ax4.axis('off')
        characteristics = """
Rate Limiter Algorithm Comparison

Fixed Window:
  - Simple implementation
  - O(1) memory
  - Susceptible to boundary bursts
  - Best for: Simple rate limiting

Adaptive (AIMD):
  - Self-tuning rate
  - Responds to downstream health
  - More complex implementation
  - Best for: Unknown/variable capacity

Sliding Window (not shown):
  - Smoother limiting
  - O(n) memory (n = requests in window)
  - No boundary bursts
  - Best for: Precise rate control

Token Bucket (not shown):
  - Allows controlled bursting
  - O(1) memory
  - Good for bursty traffic
  - Best for: APIs with burst allowance
"""
        ax4.text(0.05, 0.95, characteristics, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Rate Limiter Algorithm Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(test_output_dir / 'algorithm_comparison.png', dpi=150)
        plt.close()

        assert len(results) == 2
