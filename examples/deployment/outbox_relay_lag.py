"""Outbox relay lag under varying write rates and poll intervals.

Demonstrates the tradeoff between relay latency (how quickly outbox
entries are delivered downstream) and polling overhead (how often the
relay polls). Shows that faster polling reduces lag but increases
CPU/network overhead.

## Architecture

```
   ┌──────────┐     ┌──────────────────────────────────┐     ┌──────────┐
   │  Order   │────►│        Outbox Relay               │────►│ Message  │
   │ Service  │     │                                    │     │  Queue   │
   └──────────┘     │  write() ──► [outbox] ──► poll()  │     └──────────┘
                     │              batch relay           │
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
)
from happysimulator.components.microservice import OutboxRelay


# =============================================================================
# Components
# =============================================================================


class MessageCollector(Entity):
    """Downstream message sink."""

    def __init__(self, name: str):
        super().__init__(name)
        self.messages_received = 0

    def handle_event(self, event: Event) -> None:
        self.messages_received += 1


class OrderService(Entity):
    """Service that writes to outbox during event handling."""

    def __init__(self, name: str, outbox: OutboxRelay, processing_time: float = 0.005):
        super().__init__(name)
        self._outbox = outbox
        self._processing_time = processing_time
        self.orders_processed = 0
        self._poll_primed = False

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self.orders_processed += 1
        yield self._processing_time

        self._outbox.write({
            "order_id": self.orders_processed,
            "event_type": "order_created",
        })

        # Prime the poll loop on first write
        if not self._poll_primed:
            self._poll_primed = True
            return [self._outbox.prime_poll()]
        return []


class OrderProvider(EventProvider):
    """Generates order events."""

    def __init__(self, target: Entity, stop_after: Instant | None = None):
        self._target = target
        self._stop_after = stop_after

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after and time > self._stop_after:
            return []
        return [Event(time=time, event_type="new_order", target=self._target)]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class ScenarioResult:
    """Results from a single outbox scenario."""

    poll_interval: float
    entries_written: int
    entries_relayed: int
    poll_cycles: int
    avg_relay_lag: float
    max_relay_lag: float
    messages_received: int


@dataclass
class SimulationResult:
    """Results comparing multiple poll intervals."""

    scenarios: list[ScenarioResult]


def run_single_scenario(
    *,
    duration_s: float,
    arrival_rate: float,
    poll_interval: float,
    batch_size: int,
    seed: int | None,
) -> ScenarioResult:
    """Run a single outbox scenario with given poll_interval."""
    if seed is not None:
        random.seed(seed)

    collector = MessageCollector("MessageQueue")
    outbox = OutboxRelay(
        name="Outbox",
        downstream=collector,
        poll_interval=poll_interval,
        batch_size=batch_size,
        relay_latency=0.0001,
    )
    order_svc = OrderService("OrderService", outbox)

    stop = Instant.from_seconds(duration_s)
    provider = OrderProvider(order_svc, stop_after=stop)
    profile = ConstantRateProfile(rate=arrival_rate)
    source = Source(
        name="Orders",
        event_provider=provider,
        arrival_time_provider=ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 5.0,
        sources=[source],
        entities=[order_svc, outbox, collector],
    )
    sim.run()

    return ScenarioResult(
        poll_interval=poll_interval,
        entries_written=outbox.stats.entries_written,
        entries_relayed=outbox.stats.entries_relayed,
        poll_cycles=outbox.stats.poll_cycles,
        avg_relay_lag=outbox.stats.avg_relay_lag,
        max_relay_lag=outbox.stats.relay_lag_max,
        messages_received=collector.messages_received,
    )


def run_outbox_simulation(
    *,
    duration_s: float = 10.0,
    arrival_rate: float = 50.0,
    batch_size: int = 50,
    poll_intervals: list[float] | None = None,
    seed: int | None = 42,
) -> SimulationResult:
    """Run outbox simulations across multiple poll intervals."""
    if poll_intervals is None:
        poll_intervals = [0.01, 0.05, 0.1, 0.5, 1.0]

    scenarios = []
    for interval in poll_intervals:
        result = run_single_scenario(
            duration_s=duration_s,
            arrival_rate=arrival_rate,
            poll_interval=interval,
            batch_size=batch_size,
            seed=seed,
        )
        scenarios.append(result)

    return SimulationResult(scenarios=scenarios)


def print_summary(result: SimulationResult) -> None:
    """Print comparison summary across poll intervals."""
    print("\n" + "=" * 70)
    print("OUTBOX RELAY LAG — RESULTS")
    print("=" * 70)

    print(f"\n{'Poll Interval':>14s}  {'Written':>8s}  {'Relayed':>8s}  {'Polls':>6s}  "
          f"{'Avg Lag':>10s}  {'Max Lag':>10s}")
    print("-" * 70)

    for s in result.scenarios:
        print(f"{s.poll_interval:>13.3f}s  {s.entries_written:>8d}  {s.entries_relayed:>8d}  "
              f"{s.poll_cycles:>6d}  {s.avg_relay_lag * 1000:>9.2f}ms  "
              f"{s.max_relay_lag * 1000:>9.2f}ms")

    print("=" * 70)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate lag vs poll interval visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    intervals = [s.poll_interval for s in result.scenarios]
    avg_lags = [s.avg_relay_lag * 1000 for s in result.scenarios]
    max_lags = [s.max_relay_lag * 1000 for s in result.scenarios]
    poll_counts = [s.poll_cycles for s in result.scenarios]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Lag vs poll interval
    ax = axes[0]
    ax.plot(intervals, avg_lags, "bo-", label="Avg Lag", markersize=6)
    ax.plot(intervals, max_lags, "rs-", label="Max Lag", markersize=6)
    ax.set_xlabel("Poll Interval (s)")
    ax.set_ylabel("Relay Lag (ms)")
    ax.set_title("Relay Lag vs Poll Interval")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    # Overhead (poll cycles) vs poll interval
    ax = axes[1]
    ax.bar(range(len(intervals)), poll_counts, tick_label=[f"{i:.3f}s" for i in intervals],
           color="#3498db", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Poll Interval")
    ax.set_ylabel("Poll Cycles")
    ax.set_title("Polling Overhead vs Interval")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "outbox_lag_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'outbox_lag_results.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Outbox relay lag simulation")
    parser.add_argument("--duration", type=float, default=10.0, help="Load duration (s)")
    parser.add_argument("--rate", type=float, default=50.0, help="Arrival rate (req/s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/outbox", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running outbox relay lag simulation...")
    result = run_outbox_simulation(duration_s=args.duration, arrival_rate=args.rate, seed=seed)
    print_summary(result)

    if not args.no_viz:
        visualize_results(result, Path(args.output))
