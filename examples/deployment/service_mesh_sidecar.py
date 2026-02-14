"""Service mesh sidecar absorbing backend failures.

Demonstrates a sidecar proxy that protects a flaky backend with rate
limiting, circuit breaking, timeout, and retry. Shows how the circuit
breaker transitions and the effective error rate seen by upstream callers.

## Architecture

```
   ┌──────────┐     ┌──────────────────────────────────┐     ┌──────────┐
   │  Source   │────►│          Sidecar Proxy           │────►│  Backend  │
   │ (Poisson) │     │                                  │     │  Service  │
   └──────────┘     │  Rate Limit ─► Circuit Breaker   │     │ (flaky)   │
                     │  ─► Timeout ─► Retry             │     └──────────┘
                     └──────────────────────────────────┘
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    EventProvider,
    Instant,
    Simulation,
    Source,
    TokenBucketPolicy,
)
from happysimulator.components.microservice import Sidecar


# =============================================================================
# Components
# =============================================================================


class FlakyBackend(Entity):
    """Backend that becomes slow/unresponsive periodically."""

    def __init__(self, name: str, normal_latency: float = 0.01,
                 slow_latency: float = 20.0, failure_interval: float = 5.0,
                 failure_duration: float = 2.0):
        super().__init__(name)
        self.normal_latency = normal_latency
        self.slow_latency = slow_latency
        self.failure_interval = failure_interval
        self.failure_duration = failure_duration
        self.requests_received = 0

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        t = self.now.to_seconds()
        cycle = t % self.failure_interval
        if cycle < self.failure_duration:
            yield self.slow_latency  # Simulate being slow/down
        else:
            yield self.normal_latency


class RequestProvider(EventProvider):
    """Generates requests targeting the sidecar."""

    def __init__(self, target: Entity, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after and time > self._stop_after:
            return []
        return [Event(time=time, event_type="request", target=self._target)]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the sidecar simulation."""

    sidecar: Sidecar
    backend: FlakyBackend
    circuit_state_log: list[tuple[float, str]]


def run_sidecar_simulation(
    *,
    duration_s: float = 30.0,
    arrival_rate: float = 50.0,
    rate_limit_capacity: float = 100.0,
    rate_limit_refill: float = 50.0,
    request_timeout: float = 0.5,
    max_retries: int = 2,
    circuit_failure_threshold: int = 5,
    circuit_timeout: float = 3.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the sidecar simulation."""
    if seed is not None:
        random.seed(seed)

    backend = FlakyBackend("Backend")

    # Track circuit state over time
    circuit_log: list[tuple[float, str]] = []

    sidecar = Sidecar(
        name="Sidecar",
        target=backend,
        rate_limit_policy=TokenBucketPolicy(
            capacity=rate_limit_capacity, refill_rate=rate_limit_refill,
        ),
        circuit_failure_threshold=circuit_failure_threshold,
        circuit_success_threshold=2,
        circuit_timeout=circuit_timeout,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )

    stop = Instant.from_seconds(duration_s)
    provider = RequestProvider(sidecar, stop_after=stop)
    profile = ConstantRateProfile(rate=arrival_rate)
    source = Source(
        name="Traffic",
        event_provider=provider,
        arrival_time_provider=ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 5.0),
        sources=[source],
        entities=[sidecar, backend],
    )
    sim.run()

    return SimulationResult(
        sidecar=sidecar,
        backend=backend,
        circuit_state_log=circuit_log,
    )


def print_summary(result: SimulationResult) -> None:
    """Print sidecar statistics."""
    s = result.sidecar.stats

    print("\n" + "=" * 60)
    print("SERVICE MESH SIDECAR — RESULTS")
    print("=" * 60)

    print(f"\nTraffic:")
    print(f"  Total requests:     {s.total_requests}")
    print(f"  Successful:         {s.successful_requests}")
    print(f"  Failed:             {s.failed_requests}")

    print(f"\nResilience:")
    print(f"  Rate limited:       {s.rate_limited}")
    print(f"  Circuit broken:     {s.circuit_broken}")
    print(f"  Timed out:          {s.timed_out}")
    print(f"  Retries:            {s.retries}")

    print(f"\nCircuit state:        {result.sidecar.circuit_state}")
    print(f"Backend received:     {result.backend.requests_received}")

    if s.total_requests > 0:
        effective_success = s.successful_requests / s.total_requests * 100
        effective_error = (s.failed_requests + s.circuit_broken + s.rate_limited) / s.total_requests * 100
        print(f"\nEffective success rate: {effective_success:.1f}%")
        print(f"Effective error rate:   {effective_error:.1f}%")

    print("=" * 60)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate sidecar statistics visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    s = result.sidecar.stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Outcome breakdown
    ax = axes[0]
    categories = ["Successful", "Failed", "Rate Limited", "Circuit Broken", "Timed Out"]
    values = [s.successful_requests, s.failed_requests, s.rate_limited,
              s.circuit_broken, s.timed_out]
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#e67e22", "#9b59b6"]
    ax.bar(categories, values, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Request Outcomes")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")

    # Pie chart
    ax = axes[1]
    nonzero = [(c, v) for c, v in zip(categories, values) if v > 0]
    if nonzero:
        labels, sizes = zip(*nonzero)
        color_map = dict(zip(categories, colors))
        pie_colors = [color_map[l] for l in labels]
        ax.pie(sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Outcome Distribution")

    fig.tight_layout()
    fig.savefig(output_dir / "sidecar_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'sidecar_results.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Service mesh sidecar simulation")
    parser.add_argument("--duration", type=float, default=30.0, help="Load duration (s)")
    parser.add_argument("--rate", type=float, default=50.0, help="Arrival rate (req/s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/sidecar", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running service mesh sidecar simulation...")
    result = run_sidecar_simulation(duration_s=args.duration, arrival_rate=args.rate, seed=seed)
    print_summary(result)

    if not args.no_viz:
        visualize_results(result, Path(args.output))
