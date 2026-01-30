"""Integration tests for SimpleServer with PercentileFittedLatency distribution."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from happysimulator.modules.client_server.simple_client import SimpleClient
from happysimulator.modules.client_server.simple_server import SimpleServer
from happysimulator.core.event import Event
from happysimulator.modules.client_server.request import Request
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.instant import Instant

from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.percentile_fitted import PercentileFittedLatency


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


class RequestProvider(EventProvider):
    """Generates Request events targeting a client/server pair."""

    def __init__(
        self,
        client: SimpleClient,
        server: SimpleServer,
        network_latency=None,
        stop_after: Instant | None = None,
    ):
        self.client = client
        self.server = server
        self.network_latency = network_latency or ConstantLatency(
            Instant.from_seconds(0.001)
        )
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        request = Request(
            time=time,
            event_type=f"Request-{self.generated}",
            client=self.client,
            server=self.server,
            network_latency=self.network_latency,
            callback=self.client.send_request,
        )
        return [request]


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Write data to a CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _percentile_sorted(sorted_values: list[float], p: float) -> float:
    """Calculate percentile from sorted values using linear interpolation."""
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


@dataclass
class PercentileFittedScenarioResult:
    """Results from running a percentile-fitted latency scenario."""
    processing_times: list[float]
    client_latencies: list[float]
    distribution: PercentileFittedLatency
    requests_generated: int
    requests_completed: int


def run_percentile_fitted_scenario(
    *,
    distribution: PercentileFittedLatency,
    network_latency_s: float = 0.001,
    duration_s: float = 100.0,
    request_rate: float = 2.0,
    test_output_dir: Path | None = None,
) -> PercentileFittedScenarioResult:
    """Run a simulation with PercentileFittedLatency and collect statistics.

    Args:
        distribution: The PercentileFittedLatency to use for server processing.
        network_latency_s: Fixed network latency in seconds.
        duration_s: How long to generate requests.
        request_rate: Requests per second.
        test_output_dir: Optional directory for output files and plots.

    Returns:
        PercentileFittedScenarioResult with collected data.
    """
    client = SimpleClient("client")
    server = SimpleServer("server", processing_latency=distribution)

    provider = RequestProvider(
        client, server,
        network_latency=ConstantLatency(Instant.from_seconds(network_latency_s)),
        stop_after=Instant.from_seconds(duration_s),
    )
    arrival = ConstantArrivalTimeProvider(
        ConstantRateProfile(rate_per_s=request_rate),
        start_time=Instant.Epoch,
    )
    source = Source("source", provider, arrival)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 10.0),  # Extra time to complete
        sources=[source],
        entities=[client, server],
    )
    sim.run()

    processing_times = [v for _, v in server.stats_processing_time.values]
    client_latencies = [v for _, v in client.stats_latency.values]

    result = PercentileFittedScenarioResult(
        processing_times=processing_times,
        client_latencies=client_latencies,
        distribution=distribution,
        requests_generated=provider.generated,
        requests_completed=len(processing_times),
    )

    if test_output_dir is not None:
        _generate_outputs(result, test_output_dir, network_latency_s)

    return result


def _generate_outputs(
    result: PercentileFittedScenarioResult,
    output_dir: Path,
    network_latency_s: float,
) -> None:
    """Generate CSV files and plots for the scenario results."""
    proc_times_sorted = sorted(result.processing_times)
    n = len(proc_times_sorted)

    # Calculate observed percentiles
    percentiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
    observed = {p: _percentile_sorted(proc_times_sorted, p) for p in percentiles}
    theoretical = {p: result.distribution.get_percentile(p).to_seconds() for p in percentiles}

    # Write raw processing times
    _write_csv(
        output_dir / "processing_times.csv",
        header=["index", "processing_time_s"],
        rows=[[i, t] for i, t in enumerate(result.processing_times)],
    )

    # Write percentile comparison
    _write_csv(
        output_dir / "percentile_comparison.csv",
        header=["percentile", "theoretical_s", "observed_s", "error_pct"],
        rows=[
            [
                f"p{int(p*100)}" if p < 0.999 else f"p{p*1000:.0f}",
                theoretical[p],
                observed[p],
                100 * (observed[p] - theoretical[p]) / theoretical[p] if theoretical[p] > 0 else 0,
            ]
            for p in percentiles
        ],
    )

    # Write summary statistics
    mean_observed = sum(proc_times_sorted) / n if n > 0 else 0
    mean_theoretical = result.distribution._mean_latency
    variance = sum((t - mean_observed) ** 2 for t in proc_times_sorted) / n if n > 0 else 0
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_observed if mean_observed > 0 else 0

    _write_csv(
        output_dir / "summary_statistics.csv",
        header=["metric", "value"],
        rows=[
            ["samples", n],
            ["mean_theoretical_s", mean_theoretical],
            ["mean_observed_s", mean_observed],
            ["std_dev_s", std_dev],
            ["coefficient_of_variation", cv],
            ["min_s", min(proc_times_sorted) if proc_times_sorted else 0],
            ["max_s", max(proc_times_sorted) if proc_times_sorted else 0],
            ["lambda", result.distribution._lambda],
        ],
    )

    # Generate plots
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot 1: Histogram of processing times with theoretical PDF overlay
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(proc_times_sorted, bins=50, density=True, alpha=0.7, label="Observed", color="steelblue")

    # Theoretical exponential PDF: f(x) = λ * e^(-λx)
    lam = result.distribution._lambda
    x_max = max(proc_times_sorted) * 1.1
    x_vals = [i * x_max / 200 for i in range(201)]
    pdf_vals = [lam * math.exp(-lam * x) for x in x_vals]
    ax.plot(x_vals, pdf_vals, "r-", linewidth=2, label=f"Theoretical (λ={lam:.3f})")

    # Add vertical lines for key percentiles
    for p, label in [(0.50, "p50"), (0.90, "p90"), (0.99, "p99")]:
        ax.axvline(observed[p], color="green", linestyle="--", alpha=0.7)
        ax.axvline(theoretical[p], color="red", linestyle=":", alpha=0.7)

    ax.set_title("Processing Time Distribution: Observed vs Theoretical")
    ax.set_xlabel("Processing Time (s)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "distribution_histogram.png", dpi=150)
    plt.close(fig)

    # Plot 2: CDF comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Empirical CDF
    ecdf_x = proc_times_sorted
    ecdf_y = [(i + 1) / n for i in range(n)]
    ax.step(ecdf_x, ecdf_y, where="post", label="Observed (ECDF)", color="steelblue", linewidth=1.5)

    # Theoretical CDF: F(x) = 1 - e^(-λx)
    cdf_vals = [1 - math.exp(-lam * x) for x in x_vals]
    ax.plot(x_vals, cdf_vals, "r-", linewidth=2, label="Theoretical CDF")

    ax.set_title("Cumulative Distribution: Observed vs Theoretical")
    ax.set_xlabel("Processing Time (s)")
    ax.set_ylabel("Cumulative Probability")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "distribution_cdf.png", dpi=150)
    plt.close(fig)

    # Plot 3: Percentile comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    percentile_labels = [f"p{int(p*100)}" if p < 0.999 else f"p{p*1000:.0f}" for p in percentiles]
    x_pos = range(len(percentiles))
    width = 0.35

    theoretical_vals = [theoretical[p] for p in percentiles]
    observed_vals = [observed[p] for p in percentiles]

    bars1 = ax.bar([x - width/2 for x in x_pos], theoretical_vals, width, label="Theoretical", color="coral")
    bars2 = ax.bar([x + width/2 for x in x_pos], observed_vals, width, label="Observed", color="steelblue")

    ax.set_title("Percentile Comparison: Theoretical vs Observed")
    ax.set_xlabel("Percentile")
    ax.set_ylabel("Latency (s)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(percentile_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "percentile_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 4: Q-Q plot (quantile-quantile)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Generate theoretical quantiles for each observed value
    qq_percentiles = [(i + 0.5) / n for i in range(n)]
    theoretical_quantiles = [-math.log(1 - p) / lam for p in qq_percentiles]

    ax.scatter(theoretical_quantiles, proc_times_sorted, alpha=0.3, s=10, color="steelblue")

    # Add diagonal reference line
    max_val = max(max(theoretical_quantiles), max(proc_times_sorted))
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="Perfect fit")

    ax.set_title("Q-Q Plot: Theoretical vs Observed Quantiles")
    ax.set_xlabel("Theoretical Quantiles (s)")
    ax.set_ylabel("Observed Quantiles (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_dir / "qq_plot.png", dpi=150)
    plt.close(fig)


class TestPercentileFittedServerLatency:
    """Test server with PercentileFittedLatency processing distribution."""

    @pytest.fixture(autouse=True)
    def set_seed(self):
        """Set random seed for reproducible tests."""
        random.seed(42)

    def test_server_uses_percentile_fitted_latency(self):
        """Server should use PercentileFittedLatency for processing time."""
        processing_latency = PercentileFittedLatency(
            p50=Instant.from_seconds(0.050),
            p99=Instant.from_seconds(0.200),
        )

        client = SimpleClient("client")
        server = SimpleServer("server", processing_latency=processing_latency)

        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(Instant.from_seconds(0.001)),
            stop_after=Instant.from_seconds(50.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=2.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(60.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        proc_times = [v for _, v in server.stats_processing_time.values]
        assert len(proc_times) >= 50, f"Expected >= 50 samples, got {len(proc_times)}"
        assert len(set(proc_times)) > 1

    def test_observed_percentiles_match_distribution(self, test_output_dir: Path):
        """Observed processing times should match fitted percentile targets."""
        # Use percentiles consistent with exponential for accurate fit
        target_p50 = 0.0693
        target_p90 = 0.2303
        target_p99 = 0.4605

        distribution = PercentileFittedLatency(
            p50=Instant.from_seconds(target_p50),
            p90=Instant.from_seconds(target_p90),
            p99=Instant.from_seconds(target_p99),
        )

        result = run_percentile_fitted_scenario(
            distribution=distribution,
            network_latency_s=0.0001,
            duration_s=500.0,
            request_rate=1.0,
            test_output_dir=test_output_dir,
        )

        proc_times_sorted = sorted(result.processing_times)
        n = len(proc_times_sorted)
        assert n >= 400, f"Expected >= 400 samples, got {n}"

        observed_p50 = _percentile_sorted(proc_times_sorted, 0.50)
        observed_p90 = _percentile_sorted(proc_times_sorted, 0.90)
        observed_p99 = _percentile_sorted(proc_times_sorted, 0.99)

        assert abs(observed_p50 - target_p50) / target_p50 < 0.15
        assert abs(observed_p90 - target_p90) / target_p90 < 0.15
        assert abs(observed_p99 - target_p99) / target_p99 < 0.25

    def test_end_to_end_latency_includes_processing(self):
        """Client-observed latency should include server processing time."""
        target_processing_p50 = 0.100
        processing_latency = PercentileFittedLatency(
            p50=Instant.from_seconds(target_processing_p50),
        )

        network_latency_value = 0.010

        client = SimpleClient("client")
        server = SimpleServer("server", processing_latency=processing_latency)

        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(Instant.from_seconds(network_latency_value)),
            stop_after=Instant.from_seconds(100.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=1.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(120.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        latencies = sorted([v for _, v in client.stats_latency.values])
        n = len(latencies)
        assert n >= 50, f"Expected >= 50 samples, got {n}"

        expected_total_p50 = 2 * network_latency_value + target_processing_p50
        observed_total_p50 = _percentile_sorted(latencies, 0.50)

        assert abs(observed_total_p50 - expected_total_p50) / expected_total_p50 < 0.20

    def test_high_percentile_tail_latency(self, test_output_dir: Path):
        """High percentile (p99) should show tail latency behavior."""
        target_p99 = 0.500

        distribution = PercentileFittedLatency(
            p99=Instant.from_seconds(target_p99),
        )

        result = run_percentile_fitted_scenario(
            distribution=distribution,
            network_latency_s=0.001,
            duration_s=200.0,
            request_rate=1.0,
            test_output_dir=test_output_dir,
        )

        proc_times_sorted = sorted(result.processing_times)
        n = len(proc_times_sorted)
        assert n >= 150, f"Expected >= 150 samples, got {n}"

        observed_p99 = _percentile_sorted(proc_times_sorted, 0.99)

        assert abs(observed_p99 - target_p99) / target_p99 < 0.50
        assert max(proc_times_sorted) > observed_p99

    def test_multiple_servers_independent_distributions(self, test_output_dir: Path):
        """Multiple servers can have independent latency distributions."""
        fast_latency = PercentileFittedLatency(p50=Instant.from_seconds(0.020))
        slow_latency = PercentileFittedLatency(p50=Instant.from_seconds(0.100))

        client1 = SimpleClient("client1")
        client2 = SimpleClient("client2")
        fast_server = SimpleServer("fast-server", processing_latency=fast_latency)
        slow_server = SimpleServer("slow-server", processing_latency=slow_latency)

        provider1 = RequestProvider(
            client1, fast_server,
            network_latency=ConstantLatency(Instant.from_seconds(0.001)),
            stop_after=Instant.from_seconds(50.0),
        )
        provider2 = RequestProvider(
            client2, slow_server,
            network_latency=ConstantLatency(Instant.from_seconds(0.001)),
            stop_after=Instant.from_seconds(50.0),
        )

        arrival1 = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=2.0),
            start_time=Instant.Epoch,
        )
        arrival2 = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=2.0),
            start_time=Instant.Epoch,
        )

        source1 = Source("source1", provider1, arrival1)
        source2 = Source("source2", provider2, arrival2)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(60.0),
            sources=[source1, source2],
            entities=[client1, client2, fast_server, slow_server],
        )
        sim.run()

        fast_times = sorted([v for _, v in fast_server.stats_processing_time.values])
        slow_times = sorted([v for _, v in slow_server.stats_processing_time.values])

        assert len(fast_times) >= 50
        assert len(slow_times) >= 50

        fast_median = _percentile_sorted(fast_times, 0.50)
        slow_median = _percentile_sorted(slow_times, 0.50)

        assert fast_median < slow_median
        assert slow_median / fast_median > 3.0

        # Write comparison data
        _write_csv(
            test_output_dir / "fast_server_times.csv",
            header=["index", "processing_time_s"],
            rows=[[i, t] for i, t in enumerate(fast_times)],
        )
        _write_csv(
            test_output_dir / "slow_server_times.csv",
            header=["index", "processing_time_s"],
            rows=[[i, t] for i, t in enumerate(slow_times)],
        )

        # Generate comparison plot
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram comparison
        ax = axes[0]
        ax.hist(fast_times, bins=30, density=True, alpha=0.7, label=f"Fast (p50={fast_median:.3f}s)", color="green")
        ax.hist(slow_times, bins=30, density=True, alpha=0.7, label=f"Slow (p50={slow_median:.3f}s)", color="orange")
        ax.set_title("Processing Time Distribution: Fast vs Slow Server")
        ax.set_xlabel("Processing Time (s)")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # CDF comparison
        ax = axes[1]
        fast_ecdf_y = [(i + 1) / len(fast_times) for i in range(len(fast_times))]
        slow_ecdf_y = [(i + 1) / len(slow_times) for i in range(len(slow_times))]
        ax.step(fast_times, fast_ecdf_y, where="post", label="Fast Server", color="green", linewidth=1.5)
        ax.step(slow_times, slow_ecdf_y, where="post", label="Slow Server", color="orange", linewidth=1.5)
        ax.set_title("CDF Comparison: Fast vs Slow Server")
        ax.set_xlabel("Processing Time (s)")
        ax.set_ylabel("Cumulative Probability")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(test_output_dir / "server_comparison.png", dpi=150)
        plt.close(fig)
