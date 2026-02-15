"""Integration tests with visualizations for resilience components.

These tests demonstrate resilience patterns through visual output,
showing circuit breaker behavior, bulkhead isolation, and hedging.

Run:
    pytest tests/integration/test_resilience_visualization.py -v

Output:
    test_output/test_resilience_visualization/<test_name>/
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.components.resilience import (
    Bulkhead,
    CircuitBreaker,
    CircuitState,
    Fallback,
    Hedge,
    TimeoutWrapper,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""

    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


@dataclass
class ReliableServer(Entity):
    """Server that always responds successfully."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class FailingServer(Entity):
    """Server that fails after a specified time."""

    name: str
    normal_response: float = 0.010
    slow_response: float = 10.0
    fail_after_seconds: float = 1.0

    requests_received: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        if self.now.to_seconds() > self.fail_after_seconds:
            self.failures += 1
            yield self.slow_response
        else:
            yield self.normal_response


@dataclass
class IntermittentServer(Entity):
    """Server with intermittent failures."""

    name: str
    normal_response: float = 0.010
    slow_response: float = 10.0
    failure_rate: float = 0.3
    _request_count: int = field(default=0, init=False)

    requests_received: int = field(default=0, init=False)
    successes: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        self._request_count += 1

        # Use deterministic pattern for testing
        if self._request_count % 10 < int(self.failure_rate * 10):
            self.failures += 1
            yield self.slow_response
        else:
            self.successes += 1
            yield self.normal_response


@dataclass
class VariableLatencyServer(Entity):
    """Server with variable response times."""

    name: str
    base_latency: float = 0.010
    p99_latency: float = 0.100
    p99_threshold: int = 99

    requests_received: int = field(default=0, init=False)
    _counter: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        self._counter += 1

        # Every 100th request is slow (simulating p99)
        if self._counter % 100 >= self.p99_threshold:
            yield self.p99_latency
        else:
            yield self.base_latency


class ResilienceRequestProvider(EventProvider):
    """Generates requests to a resilience component."""

    def __init__(self, target: Entity, stop_after: Instant | None = None):
        self.target = target
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        return [
            Event(
                time=time,
                event_type="request",
                target=self.target,
                context={"metadata": {"request_id": self.generated}},
            )
        ]


class TestResilienceVisualization:
    """Visual tests for resilience components."""

    def test_circuit_breaker_behavior(self, test_output_dir):
        """
        Visualize circuit breaker state transitions and request handling.

        Shows how the circuit breaker opens during failure and recovers.
        This test demonstrates the circuit breaker with manual failure injection.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Server that always responds
        server = ReliableServer(name="server", response_time=0.010)

        cb = CircuitBreaker(
            name="cb",
            target=server,
            failure_threshold=5,
            success_threshold=3,
            timeout=1.0,
        )

        # Manually simulate failures to demonstrate circuit breaker behavior
        # In a real scenario, these would come from timeouts, errors, etc.

        # Phase 1: Normal operation - some successful requests
        for _ in range(10):
            cb.record_success()

        # Phase 2: Failures cause circuit to open
        for _ in range(5):
            cb.record_failure()

        # Now circuit should be open
        assert cb.state == CircuitState.OPEN

        # Phase 3: Run simulation with circuit open - requests rejected
        provider = ResilienceRequestProvider(cb, stop_after=Instant.from_seconds(0.5))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=20),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=1.0,
            sources=[source],
            entities=[server, cb],
        )
        sim.run()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Circuit Breaker Behavior", fontsize=14, fontweight="bold")

        # Plot 1: Request statistics
        ax1 = axes[0, 0]
        stats = [
            ("Total", cb.stats.total_requests),
            ("Success", cb.stats.successful_requests),
            ("Failed", cb.stats.failed_requests),
            ("Rejected", cb.stats.rejected_requests),
        ]
        names = [s[0] for s in stats]
        values = [s[1] for s in stats]
        colors = ["steelblue", "green", "red", "orange"]
        ax1.bar(names, values, color=colors, alpha=0.7)
        ax1.set_ylabel("Count")
        ax1.set_title("Request Statistics")
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: State changes
        ax2 = axes[0, 1]
        state_stats = [
            ("Times Opened", cb.stats.times_opened),
            ("Times Closed", cb.stats.times_closed),
            ("Total Changes", cb.stats.state_changes),
        ]
        names = [s[0] for s in state_stats]
        values = [s[1] for s in state_stats]
        ax2.bar(names, values, color="coral", alpha=0.7)
        ax2.set_ylabel("Count")
        ax2.set_title("State Transitions")
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Server stats
        ax3 = axes[1, 0]
        server_stats = [
            ("Received by Server", server.requests_received),
            ("Blocked by CB", cb.stats.rejected_requests),
        ]
        names = [s[0] for s in server_stats]
        values = [s[1] for s in server_stats]
        ax3.bar(names, values, color=["green", "red"], alpha=0.7)
        ax3.set_ylabel("Count")
        ax3.set_title("Traffic Routing")
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: Summary
        ax4 = axes[1, 1]
        ax4.axis("off")

        rejection_rate = cb.stats.rejected_requests / max(1, cb.stats.total_requests) * 100

        summary = f"""
Circuit Breaker Summary
=======================

Configuration:
  - Failure Threshold: {cb.failure_threshold}
  - Success Threshold: {cb.success_threshold}
  - Timeout: {cb.timeout}s

Results:
  - Total Requests: {cb.stats.total_requests}
  - Rejected (fast-fail): {cb.stats.rejected_requests} ({rejection_rate:.1f}%)
  - Circuit opened {cb.stats.times_opened} time(s)

Final State: {cb.state.value}

Note: This demo uses manual failure injection
to illustrate circuit breaker state transitions.
"""
        ax4.text(
            0.05,
            0.5,
            summary,
            transform=ax4.transAxes,
            fontsize=11,
            verticalalignment="center",
            fontfamily="monospace",
            bbox={"boxstyle": "round", "facecolor": "lightblue", "alpha": 0.8},
        )

        plt.tight_layout()
        fig.savefig(test_output_dir / "circuit_breaker_behavior.png", dpi=150)
        plt.close(fig)

        # Verify circuit breaker did its job
        assert cb.stats.rejected_requests > 0, "Circuit breaker should have rejected some requests"
        assert cb.stats.times_opened >= 1, "Circuit should have opened at least once"

    def test_bulkhead_isolation(self, test_output_dir):
        """
        Visualize bulkhead concurrency limiting.

        Shows how bulkhead prevents resource exhaustion.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Compare with and without bulkhead
        results = {}

        for use_bulkhead in [False, True]:
            server = ReliableServer(name="server", response_time=0.050)

            if use_bulkhead:
                bulkhead = Bulkhead(
                    name="bulkhead",
                    target=server,
                    max_concurrent=5,
                    max_wait_queue=10,
                    max_wait_time=0.5,
                )
                target = bulkhead
            else:
                bulkhead = None
                target = server

            provider = ResilienceRequestProvider(target, stop_after=Instant.from_seconds(1.5))
            arrival = ConstantArrivalTimeProvider(
                ConstantRateProfile(rate_per_s=100),
                start_time=Instant.Epoch,
            )
            source = Source("source", provider, arrival)

            entities = [server]
            if bulkhead:
                entities.append(bulkhead)

            sim = Simulation(
                start_time=Instant.Epoch,
                duration=2.0,
                sources=[source],
                entities=entities,
            )
            sim.run()

            label = "With Bulkhead" if use_bulkhead else "Without Bulkhead"
            results[label] = {
                "requests_to_server": server.requests_received,
                "rejected": bulkhead.stats.rejected_requests if bulkhead else 0,
                "queued": bulkhead.stats.queued_requests if bulkhead else 0,
                "timed_out": bulkhead.stats.timed_out_requests if bulkhead else 0,
                "peak_concurrent": bulkhead.stats.peak_concurrent if bulkhead else 0,
            }

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Bulkhead Isolation Effect", fontsize=14, fontweight="bold")

        # Plot 1: Requests reaching server
        ax1 = axes[0]
        labels = list(results.keys())
        server_requests = [results[k]["requests_to_server"] for k in labels]
        ax1.bar(labels, server_requests, color=["red", "green"], alpha=0.7)
        ax1.set_ylabel("Requests to Server")
        ax1.set_title("Server Load Comparison")
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Bulkhead statistics
        ax2 = axes[1]
        bh_data = results["With Bulkhead"]
        stats = [
            ("Accepted", bh_data["requests_to_server"]),
            ("Rejected", bh_data["rejected"]),
            ("Queued", bh_data["queued"]),
            ("Timed Out", bh_data["timed_out"]),
        ]
        names = [s[0] for s in stats]
        values = [s[1] for s in stats]
        colors = ["green", "red", "orange", "gray"]
        ax2.bar(names, values, color=colors, alpha=0.7)
        ax2.set_ylabel("Count")
        ax2.set_title("Bulkhead Statistics")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(test_output_dir / "bulkhead_isolation.png", dpi=150)
        plt.close(fig)

        # Verify bulkhead limited concurrency
        assert results["With Bulkhead"]["peak_concurrent"] <= 5

    def test_hedge_latency_reduction(self, test_output_dir):
        """
        Visualize how hedging reduces tail latency.

        Shows the effect of hedged requests on response time distribution.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        results = {}

        for use_hedge in [False, True]:
            server = VariableLatencyServer(
                name="server",
                base_latency=0.010,
                p99_latency=0.100,
                p99_threshold=95,
            )

            if use_hedge:
                hedge = Hedge(
                    name="hedge",
                    target=server,
                    hedge_delay=0.020,  # Send hedge after 20ms
                    max_hedges=1,
                )
                target = hedge
            else:
                hedge = None
                target = server

            provider = ResilienceRequestProvider(target, stop_after=Instant.from_seconds(2.0))
            arrival = ConstantArrivalTimeProvider(
                ConstantRateProfile(rate_per_s=50),
                start_time=Instant.Epoch,
            )
            source = Source("source", provider, arrival)

            entities = [server]
            if hedge:
                entities.append(hedge)

            sim = Simulation(
                start_time=Instant.Epoch,
                duration=2.5,
                sources=[source],
                entities=entities,
            )
            sim.run()

            label = "With Hedge" if use_hedge else "Without Hedge"
            results[label] = {
                "total_requests": provider.generated,
                "server_requests": server.requests_received,
                "primary_wins": hedge.stats.primary_wins if hedge else provider.generated,
                "hedge_wins": hedge.stats.hedge_wins if hedge else 0,
                "hedges_sent": hedge.stats.hedges_sent if hedge else 0,
            }

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Hedge Request Behavior", fontsize=14, fontweight="bold")

        # Plot 1: Request comparison
        ax1 = axes[0]
        labels = ["Without Hedge", "With Hedge"]
        total = [results[l]["total_requests"] for l in labels]
        server = [results[l]["server_requests"] for l in labels]

        x = np.arange(len(labels))
        width = 0.35

        ax1.bar(x - width / 2, total, width, label="Client Requests", color="steelblue")
        ax1.bar(x + width / 2, server, width, label="Server Requests", color="coral")
        ax1.set_ylabel("Count")
        ax1.set_title("Request Volume")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Hedge statistics
        ax2 = axes[1]
        hedge_data = results["With Hedge"]
        stats = [
            ("Primary Wins", hedge_data["primary_wins"]),
            ("Hedge Wins", hedge_data["hedge_wins"]),
            ("Hedges Sent", hedge_data["hedges_sent"]),
        ]
        names = [s[0] for s in stats]
        values = [s[1] for s in stats]
        colors = ["green", "blue", "orange"]
        ax2.bar(names, values, color=colors, alpha=0.7)
        ax2.set_ylabel("Count")
        ax2.set_title("Hedge Statistics")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(test_output_dir / "hedge_behavior.png", dpi=150)
        plt.close(fig)

    def test_fallback_graceful_degradation(self, test_output_dir):
        """
        Visualize fallback behavior during primary failure.

        Shows how fallback provides graceful degradation.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Primary server fails after 1 second
        primary = FailingServer(
            name="primary",
            fail_after_seconds=1.0,
            normal_response=0.010,
            slow_response=10.0,
        )

        # Cache server as fallback
        fallback_server = ReliableServer(name="cache", response_time=0.001)

        fallback = Fallback(
            name="fallback",
            primary=primary,
            fallback=fallback_server,
            timeout=0.1,
        )

        provider = ResilienceRequestProvider(fallback, stop_after=Instant.from_seconds(3.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=30),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=4.0,
            sources=[source],
            entities=[primary, fallback_server, fallback],
        )
        sim.run()

        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Fallback Graceful Degradation", fontsize=14, fontweight="bold")

        # Plot 1: Request routing
        ax1 = axes[0]
        stats = [
            ("Total Requests", fallback.stats.total_requests),
            ("Primary Success", fallback.stats.primary_successes),
            ("Primary Fail", fallback.stats.primary_failures),
            ("Fallback Success", fallback.stats.fallback_successes),
        ]
        names = [s[0] for s in stats]
        values = [s[1] for s in stats]
        colors = ["steelblue", "green", "red", "orange"]
        ax1.bar(names, values, color=colors, alpha=0.7)
        ax1.set_ylabel("Count")
        ax1.set_title("Request Routing")
        ax1.tick_params(axis="x", rotation=15)
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Server load
        ax2 = axes[1]
        server_stats = [
            ("Primary", primary.requests_received),
            ("Fallback", fallback_server.requests_received),
        ]
        names = [s[0] for s in server_stats]
        values = [s[1] for s in server_stats]
        colors = ["coral", "lightgreen"]
        ax2.bar(names, values, color=colors, alpha=0.7)
        ax2.set_ylabel("Requests Received")
        ax2.set_title("Server Load Distribution")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(test_output_dir / "fallback_degradation.png", dpi=150)
        plt.close(fig)

        # Verify fallback was used
        assert fallback.stats.fallback_invocations > 0

    def test_combined_resilience_stack(self, test_output_dir):
        """
        Visualize a composed resilience stack.

        Shows Circuit Breaker + Bulkhead + Timeout working together.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Intermittent server
        server = IntermittentServer(
            name="server",
            normal_response=0.020,
            slow_response=1.0,
            failure_rate=0.2,
        )

        # Build resilience stack: Bulkhead -> Circuit Breaker -> Timeout -> Server
        timeout = TimeoutWrapper(name="timeout", target=server, timeout=0.1)
        cb = CircuitBreaker(
            name="cb",
            target=timeout,
            failure_threshold=5,
            success_threshold=2,
            timeout=2.0,
        )
        bulkhead = Bulkhead(
            name="bulkhead",
            target=cb,
            max_concurrent=10,
            max_wait_queue=20,
        )

        provider = ResilienceRequestProvider(bulkhead, stop_after=Instant.from_seconds(5.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=50),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=6.0,
            sources=[source],
            entities=[server, timeout, cb, bulkhead],
        )
        sim.run()

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Combined Resilience Stack", fontsize=14, fontweight="bold")

        # Plot 1: Bulkhead
        ax1 = axes[0, 0]
        bh_stats = [
            ("Accepted", bulkhead.stats.accepted_requests),
            ("Rejected", bulkhead.stats.rejected_requests),
            ("Queued", bulkhead.stats.queued_requests),
        ]
        ax1.bar(
            [s[0] for s in bh_stats],
            [s[1] for s in bh_stats],
            color=["green", "red", "orange"],
            alpha=0.7,
        )
        ax1.set_title("Bulkhead")
        ax1.set_ylabel("Count")
        ax1.grid(True, alpha=0.3, axis="y")

        # Plot 2: Circuit Breaker
        ax2 = axes[0, 1]
        cb_stats = [
            ("Total", cb.stats.total_requests),
            ("Success", cb.stats.successful_requests),
            ("Failed", cb.stats.failed_requests),
            ("Rejected", cb.stats.rejected_requests),
        ]
        ax2.bar(
            [s[0] for s in cb_stats],
            [s[1] for s in cb_stats],
            color=["steelblue", "green", "red", "orange"],
            alpha=0.7,
        )
        ax2.set_title("Circuit Breaker")
        ax2.set_ylabel("Count")
        ax2.grid(True, alpha=0.3, axis="y")

        # Plot 3: Timeout
        ax3 = axes[1, 0]
        tw_stats = [
            ("Total", timeout.stats.total_requests),
            ("Success", timeout.stats.successful_requests),
            ("Timed Out", timeout.stats.timed_out_requests),
        ]
        ax3.bar(
            [s[0] for s in tw_stats],
            [s[1] for s in tw_stats],
            color=["steelblue", "green", "gray"],
            alpha=0.7,
        )
        ax3.set_title("Timeout Wrapper")
        ax3.set_ylabel("Count")
        ax3.grid(True, alpha=0.3, axis="y")

        # Plot 4: Server
        ax4 = axes[1, 1]
        srv_stats = [
            ("Received", server.requests_received),
            ("Successes", server.successes),
            ("Failures", server.failures),
        ]
        ax4.bar(
            [s[0] for s in srv_stats],
            [s[1] for s in srv_stats],
            color=["steelblue", "green", "red"],
            alpha=0.7,
        )
        ax4.set_title("Server")
        ax4.set_ylabel("Count")
        ax4.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(test_output_dir / "combined_resilience.png", dpi=150)
        plt.close(fig)

        # Save data to CSV
        with (test_output_dir / "resilience_stats.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Component", "Metric", "Value"])
            writer.writerow(["Bulkhead", "Accepted", bulkhead.stats.accepted_requests])
            writer.writerow(["Bulkhead", "Rejected", bulkhead.stats.rejected_requests])
            writer.writerow(["CircuitBreaker", "Total", cb.stats.total_requests])
            writer.writerow(["CircuitBreaker", "Rejected", cb.stats.rejected_requests])
            writer.writerow(["Timeout", "TimedOut", timeout.stats.timed_out_requests])
            writer.writerow(["Server", "Received", server.requests_received])
