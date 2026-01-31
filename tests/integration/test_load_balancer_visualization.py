"""Integration tests with visualizations for LoadBalancer.

These tests demonstrate load balancer behavior through visual output,
showing traffic distribution, strategy comparison, and health checking.

Run:
    pytest tests/integration/test_load_balancer_visualization.py -v

Output:
    test_output/test_load_balancer_visualization/<test_name>/
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List

import pytest

from happysimulator.components.load_balancer import (
    LoadBalancer,
    HealthChecker,
    RoundRobin,
    Random,
    LeastConnections,
    PowerOfTwoChoices,
    WeightedRoundRobin,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
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
class BackendServer(Entity):
    """Backend server with configurable response time."""
    name: str
    base_response_time: float = 0.010
    load_factor: float = 0.001  # Additional time per concurrent request

    requests_received: int = field(default=0, init=False)
    requests_completed: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)
    _response_times: list = field(default_factory=list, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        # Handle health checks
        if event.event_type == "health_check":
            yield self.base_response_time
            return

        self.requests_received += 1
        self.active_connections += 1

        # Response time increases with load
        response_time = self.base_response_time + (self.active_connections * self.load_factor)
        self._response_times.append(response_time)

        yield response_time

        self.active_connections -= 1
        self.requests_completed += 1


@dataclass
class SlowBackend(Entity):
    """Slow backend server."""
    name: str
    response_time: float = 0.100  # 100ms - much slower

    requests_received: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        if event.event_type == "health_check":
            yield self.response_time
            return

        self.requests_received += 1
        self.active_connections += 1
        yield self.response_time
        self.active_connections -= 1


@dataclass
class FailingBackend(Entity):
    """Backend that becomes slow/unresponsive after some time."""
    name: str
    normal_response: float = 0.010
    slow_response: float = 10.0
    fail_after_seconds: float = 1.0

    requests_received: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        self.active_connections += 1

        # Become slow after fail_after_seconds
        if self.now.to_seconds() > self.fail_after_seconds:
            yield self.slow_response
        else:
            yield self.normal_response

        self.active_connections -= 1


class LoadBalancerRequestProvider(EventProvider):
    """Generates requests to a load balancer."""

    def __init__(self, lb: LoadBalancer, stop_after: Instant | None = None):
        self.lb = lb
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        return [Event(
            time=time,
            event_type="request",
            target=self.lb,
            context={"metadata": {"request_id": self.generated}},
        )]


class TestLoadBalancerVisualization:
    """Visual tests for load balancer behavior."""

    def test_strategy_comparison(self, test_output_dir):
        """
        Compare different load balancing strategies.

        Shows how different strategies distribute load across backends.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        strategies = [
            ("RoundRobin", RoundRobin()),
            ("Random", Random()),
            ("LeastConnections", LeastConnections()),
            ("PowerOfTwo", PowerOfTwoChoices()),
        ]

        request_rate = 100
        sim_duration = 2.0
        results = {}

        for name, strategy in strategies:
            # Create 5 backends with varying speed
            backends = [
                BackendServer(
                    name=f"s{i}",
                    base_response_time=0.005 + (i * 0.005),  # 5ms to 25ms
                    load_factor=0.001,
                )
                for i in range(5)
            ]

            lb = LoadBalancer(name="lb", backends=backends, strategy=strategy)

            provider = LoadBalancerRequestProvider(
                lb, stop_after=Instant.from_seconds(sim_duration - 0.2)
            )
            arrival = ConstantArrivalTimeProvider(
                ConstantRateProfile(rate_per_s=request_rate),
                start_time=Instant.Epoch,
            )
            source = Source("source", provider, arrival)

            sim = Simulation(
                start_time=Instant.Epoch,
                end_time=Instant.from_seconds(sim_duration + 0.5),
                sources=[source],
                entities=backends + [lb],
            )
            sim.run()

            results[name] = {
                'distribution': [b.requests_received for b in backends],
                'backend_names': [b.name for b in backends],
                'total': sum(b.requests_received for b in backends),
                'variance': np.var([b.requests_received for b in backends]),
            }

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Load Balancing Strategy Comparison ({request_rate} req/s)",
                    fontsize=14, fontweight='bold')

        for idx, (name, data) in enumerate(results.items()):
            ax = axes[idx // 2, idx % 2]
            x = np.arange(len(data['backend_names']))
            bars = ax.bar(x, data['distribution'], color='steelblue', alpha=0.7)
            ax.set_xlabel("Backend")
            ax.set_ylabel("Requests Received")
            ax.set_title(f"{name} (variance: {data['variance']:.1f})")
            ax.set_xticks(x)
            ax.set_xticklabels(data['backend_names'])
            ax.axhline(y=data['total']/5, color='red', linestyle='--',
                      label='Ideal (uniform)')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels
            for bar, val in zip(bars, data['distribution']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig.savefig(test_output_dir / 'strategy_comparison.png', dpi=150)
        plt.close(fig)

        # Save data to CSV
        with open(test_output_dir / 'strategy_data.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Strategy', 'Backend', 'Requests', 'Variance'])
            for name, data in results.items():
                for backend, count in zip(data['backend_names'], data['distribution']):
                    writer.writerow([name, backend, count, data['variance']])

    def test_load_aware_strategies(self, test_output_dir):
        """
        Compare load-aware vs load-agnostic strategies.

        Shows how LeastConnections and PowerOfTwo adapt to slow backends.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        strategies = [
            ("RoundRobin", RoundRobin()),
            ("LeastConnections", LeastConnections()),
            ("PowerOfTwo", PowerOfTwoChoices()),
        ]

        request_rate = 50
        sim_duration = 2.0
        results = {}

        for name, strategy in strategies:
            # Create 3 fast backends and 1 slow backend
            backends = [
                BackendServer(name="fast1", base_response_time=0.010),
                BackendServer(name="fast2", base_response_time=0.010),
                BackendServer(name="fast3", base_response_time=0.010),
                SlowBackend(name="slow", response_time=0.100),
            ]

            lb = LoadBalancer(name="lb", backends=backends, strategy=strategy)

            provider = LoadBalancerRequestProvider(
                lb, stop_after=Instant.from_seconds(sim_duration - 0.2)
            )
            arrival = ConstantArrivalTimeProvider(
                ConstantRateProfile(rate_per_s=request_rate),
                start_time=Instant.Epoch,
            )
            source = Source("source", provider, arrival)

            sim = Simulation(
                start_time=Instant.Epoch,
                end_time=Instant.from_seconds(sim_duration + 0.5),
                sources=[source],
                entities=backends + [lb],
            )
            sim.run()

            results[name] = {
                'distribution': [b.requests_received for b in backends],
                'backend_names': [b.name for b in backends],
                'slow_traffic': backends[3].requests_received,
                'fast_traffic': sum(b.requests_received for b in backends[:3]),
            }

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Load-Aware vs Load-Agnostic Strategies (1 slow backend)",
                    fontsize=14, fontweight='bold')

        colors = ['green', 'green', 'green', 'red']
        labels = ['Fast', 'Fast', 'Fast', 'Slow']

        for idx, (name, data) in enumerate(results.items()):
            ax = axes[idx]
            x = np.arange(len(data['backend_names']))
            bars = ax.bar(x, data['distribution'], color=colors, alpha=0.7)
            ax.set_xlabel("Backend")
            ax.set_ylabel("Requests")
            ax.set_title(f"{name}")
            ax.set_xticks(x)
            ax.set_xticklabels(data['backend_names'])
            ax.grid(True, alpha=0.3, axis='y')

            # Annotate slow backend traffic percentage
            total = sum(data['distribution'])
            slow_pct = data['slow_traffic'] / total * 100 if total > 0 else 0
            ax.text(0.95, 0.95, f"Slow: {slow_pct:.0f}%",
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

        plt.tight_layout()
        fig.savefig(test_output_dir / 'load_aware_comparison.png', dpi=150)
        plt.close(fig)

        # LeastConnections and PowerOfTwo should send less to slow backend
        assert results['LeastConnections']['slow_traffic'] < results['RoundRobin']['slow_traffic']

    def test_health_check_failover(self, test_output_dir):
        """
        Visualize health check detecting and failing over from unhealthy backend.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Create backends - one will fail after 1 second
        backends = [
            BackendServer(name="healthy1", base_response_time=0.010),
            BackendServer(name="healthy2", base_response_time=0.010),
            FailingBackend(name="failing", fail_after_seconds=1.0),
        ]

        lb = LoadBalancer(name="lb", backends=backends, strategy=RoundRobin())

        hc = HealthChecker(
            name="hc",
            load_balancer=lb,
            interval=0.3,
            timeout=0.1,
            unhealthy_threshold=2,
            healthy_threshold=2,
        )

        # Track health states over time
        health_history = []  # (time, backend_name, is_healthy)

        provider = LoadBalancerRequestProvider(
            lb, stop_after=Instant.from_seconds(3.0)
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=30),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(4.0),
            sources=[source],
            entities=backends + [lb, hc],
        )

        # Start health checking
        start_event = hc.start()
        start_event.time = Instant.Epoch
        sim.schedule(start_event)

        sim.run()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Health Check Failover Behavior", fontsize=14, fontweight='bold')

        # Plot 1: Request distribution
        ax1 = axes[0, 0]
        backend_names = [b.name for b in backends]
        requests = [b.requests_received for b in backends]
        colors = ['green', 'green', 'red']
        ax1.bar(backend_names, requests, color=colors, alpha=0.7)
        ax1.set_xlabel("Backend")
        ax1.set_ylabel("Requests Received")
        ax1.set_title("Request Distribution")
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Health check statistics
        ax2 = axes[0, 1]
        hc_stats = [
            ('Checks Performed', hc.stats.checks_performed),
            ('Checks Passed', hc.stats.checks_passed),
            ('Checks Failed', hc.stats.checks_failed),
            ('Timed Out', hc.stats.checks_timed_out),
        ]
        stat_names = [s[0] for s in hc_stats]
        stat_values = [s[1] for s in hc_stats]
        ax2.bar(stat_names, stat_values, color='steelblue', alpha=0.7)
        ax2.set_ylabel("Count")
        ax2.set_title("Health Check Statistics")
        ax2.tick_params(axis='x', rotation=15)
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Load balancer statistics
        ax3 = axes[1, 0]
        lb_stats = [
            ('Requests Received', lb.stats.requests_received),
            ('Forwarded', lb.stats.requests_forwarded),
            ('No Backend', lb.stats.no_backend_available),
            ('Marked Unhealthy', lb.stats.backends_marked_unhealthy),
        ]
        stat_names = [s[0] for s in lb_stats]
        stat_values = [s[1] for s in lb_stats]
        ax3.bar(stat_names, stat_values, color='coral', alpha=0.7)
        ax3.set_ylabel("Count")
        ax3.set_title("Load Balancer Statistics")
        ax3.tick_params(axis='x', rotation=15)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Summary
        ax4 = axes[1, 1]
        ax4.axis('off')

        # Check final health status
        healthy_names = [b.name for b in lb.healthy_backends]
        unhealthy_names = [b.name for b in lb.unhealthy_backends]

        summary = f"""
Health Check Failover Summary
=============================

Configuration:
  - Health Check Interval: {hc.interval}s
  - Timeout: {hc.timeout}s
  - Unhealthy Threshold: {hc.unhealthy_threshold}
  - Healthy Threshold: {hc.healthy_threshold}

Backend Health Status:
  - Healthy: {', '.join(healthy_names) or 'None'}
  - Unhealthy: {', '.join(unhealthy_names) or 'None'}

Traffic Distribution:
  - healthy1: {backends[0].requests_received} requests
  - healthy2: {backends[1].requests_received} requests
  - failing:  {backends[2].requests_received} requests

Result:
  Failing backend {"was" if backends[2] in lb.unhealthy_backends else "was NOT"}
  detected and removed from pool.
"""
        ax4.text(0.05, 0.5, summary, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        fig.savefig(test_output_dir / 'health_check_failover.png', dpi=150)
        plt.close(fig)

        # Failing backend should be marked unhealthy
        assert backends[2] in lb.unhealthy_backends

    def test_weighted_distribution(self, test_output_dir):
        """
        Visualize weighted round-robin distribution.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # Create backends with different weights
        backends = [
            BackendServer(name="small"),
            BackendServer(name="medium"),
            BackendServer(name="large"),
        ]

        strategy = WeightedRoundRobin()
        lb = LoadBalancer(name="lb", strategy=strategy)

        # Add backends with weights - must be done after LoadBalancer creation
        # because add_backend() propagates weights to the strategy
        lb.add_backend(backends[0], weight=1)
        lb.add_backend(backends[1], weight=2)
        lb.add_backend(backends[2], weight=4)

        provider = LoadBalancerRequestProvider(
            lb, stop_after=Instant.from_seconds(2.0)
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=100),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.5),
            sources=[source],
            entities=backends + [lb],
        )
        sim.run()

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Weighted Round-Robin Distribution", fontsize=14, fontweight='bold')

        # Plot 1: Actual distribution
        ax1 = axes[0]
        names = [b.name for b in backends]
        requests = [b.requests_received for b in backends]
        weights = [1, 2, 4]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, requests, width, label='Actual', color='steelblue')

        # Expected based on weights
        total = sum(requests)
        total_weight = sum(weights)
        expected = [total * w / total_weight for w in weights]
        bars2 = ax1.bar(x + width/2, expected, width, label='Expected', color='lightgreen')

        ax1.set_xlabel("Backend")
        ax1.set_ylabel("Requests")
        ax1.set_title("Actual vs Expected Distribution")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{n}\n(weight={w})" for n, w in zip(names, weights)])
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Percentage distribution
        ax2 = axes[1]
        percentages = [r / total * 100 for r in requests]
        expected_pct = [w / total_weight * 100 for w in weights]

        bars = ax2.bar(names, percentages, color='coral', alpha=0.7)
        ax2.set_xlabel("Backend")
        ax2.set_ylabel("Traffic %")
        ax2.set_title("Traffic Share")
        ax2.grid(True, alpha=0.3, axis='y')

        # Add expected line markers
        for i, exp in enumerate(expected_pct):
            ax2.hlines(exp, i - 0.4, i + 0.4, colors='green', linestyles='--', linewidth=2)

        # Add labels
        for bar, pct, exp in zip(bars, percentages, expected_pct):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%\n(exp: {exp:.1f}%)', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig.savefig(test_output_dir / 'weighted_distribution.png', dpi=150)
        plt.close(fig)

        # Verify weighted distribution is roughly correct
        # Large should get ~4x the traffic of small
        ratio = backends[2].requests_received / max(1, backends[0].requests_received)
        assert 2.5 < ratio < 5.5  # Allow some variance
