"""Saga failure cascade with compensation waterfall.

Demonstrates a multi-step distributed transaction (saga) where a
mid-saga failure triggers compensation of all previously completed
steps in reverse order.

## Architecture

```
   ┌─────────┐     ┌──────────────────────────────────────────────┐
   │ Source   │────►│              Saga Orchestrator               │
   └─────────┘     │                                              │
                    │  Step 1: Reserve ──► Step 2: Charge ──►     │
                    │  Step 3: Ship (TIMEOUT!) ──► COMPENSATE     │
                    │                                              │
                    │  Comp 2: Refund ──► Comp 1: Release         │
                    └──────────────────────────────────────────────┘
                         │           │           │
                         ▼           ▼           ▼
                    ┌─────────┐ ┌─────────┐ ┌─────────┐
                    │Inventory│ │ Payment │ │Shipping │
                    │ Service │ │ Service │ │ Service │
                    └─────────┘ └─────────┘ └─────────┘
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
from happysimulator.components.microservice import Saga, SagaState, SagaStep

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Components
# =============================================================================


class ReliableService(Entity):
    """Service that always succeeds with a given latency."""

    def __init__(self, name: str, latency: float = 0.01):
        super().__init__(name)
        self.latency = latency
        self.actions = 0
        self.compensations = 0

    def handle_event(self, event: Event) -> Generator[float]:
        metadata = event.context.get("metadata", {})
        if metadata.get("_saga_compensation"):
            self.compensations += 1
        else:
            self.actions += 1
        yield self.latency


class UnreliableService(Entity):
    """Service that is sometimes very slow (simulating failures)."""

    def __init__(
        self,
        name: str,
        normal_latency: float = 0.01,
        slow_latency: float = 100.0,
        failure_rate: float = 0.5,
    ):
        super().__init__(name)
        self.normal_latency = normal_latency
        self.slow_latency = slow_latency
        self.failure_rate = failure_rate
        self.actions = 0
        self.compensations = 0

    def handle_event(self, event: Event) -> Generator[float]:
        metadata = event.context.get("metadata", {})
        if metadata.get("_saga_compensation"):
            self.compensations += 1
            yield self.normal_latency
        else:
            self.actions += 1
            if random.random() < self.failure_rate:
                yield self.slow_latency  # Will timeout
            else:
                yield self.normal_latency


class OrderProvider(EventProvider):
    """Generates saga trigger events."""

    def __init__(self, saga: Saga, stop_after: Instant | None = None):
        self._saga = saga
        self._stop_after = stop_after
        self._order_id = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after and time > self._stop_after:
            return []
        self._order_id += 1
        return [
            Event(
                time=time,
                event_type="start_order",
                target=self._saga,
                context={"payload": {"order_id": self._order_id}},
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the saga simulation."""

    saga: Saga
    inventory: ReliableService
    payment: ReliableService
    shipping: UnreliableService
    saga_outcomes: list[tuple[int, SagaState]]


def run_saga_simulation(
    *,
    duration_s: float = 10.0,
    arrival_rate: float = 5.0,
    shipping_failure_rate: float = 0.5,
    step_timeout: float = 0.5,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the saga failure cascade simulation."""
    if seed is not None:
        random.seed(seed)

    inventory = ReliableService("Inventory", latency=0.01)
    payment = ReliableService("Payment", latency=0.02)
    shipping = UnreliableService(
        "Shipping",
        normal_latency=0.03,
        slow_latency=100.0,
        failure_rate=shipping_failure_rate,
    )

    outcomes: list[tuple[int, SagaState]] = []

    def on_complete(saga_id, state, step_results):
        outcomes.append((saga_id, state))

    saga = Saga(
        name="OrderSaga",
        steps=[
            SagaStep(
                "reserve_inventory",
                inventory,
                "reserve",
                inventory,
                "release",
                timeout=step_timeout,
            ),
            SagaStep("charge_payment", payment, "charge", payment, "refund", timeout=step_timeout),
            SagaStep(
                "ship_order", shipping, "ship", shipping, "cancel_shipment", timeout=step_timeout
            ),
        ],
        on_complete=on_complete,
    )

    stop = Instant.from_seconds(duration_s)
    provider = OrderProvider(saga, stop_after=stop)
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
        entities=[inventory, payment, shipping, saga],
    )
    sim.run()

    return SimulationResult(
        saga=saga,
        inventory=inventory,
        payment=payment,
        shipping=shipping,
        saga_outcomes=outcomes,
    )


def print_summary(result: SimulationResult) -> None:
    """Print saga execution summary."""
    s = result.saga.stats

    print("\n" + "=" * 60)
    print("SAGA FAILURE CASCADE — RESULTS")
    print("=" * 60)

    sum(1 for _, st in result.saga_outcomes if st == SagaState.COMPLETED)
    sum(1 for _, st in result.saga_outcomes if st == SagaState.COMPENSATED)

    print("\nSagas:")
    print(f"  Started:       {s.sagas_started}")
    print(f"  Completed:     {s.sagas_completed}")
    print(f"  Compensated:   {s.sagas_compensated}")
    print(f"  Failed:        {s.sagas_failed}")

    print("\nSteps:")
    print(f"  Executed:      {s.steps_executed}")
    print(f"  Failed:        {s.steps_failed}")
    print(f"  Compensated:   {s.compensations_executed}")

    print("\nServices:")
    print(
        f"  Inventory: {result.inventory.actions} actions, {result.inventory.compensations} compensations"
    )
    print(
        f"  Payment:   {result.payment.actions} actions, {result.payment.compensations} compensations"
    )
    print(
        f"  Shipping:  {result.shipping.actions} actions, {result.shipping.compensations} compensations"
    )

    if s.sagas_started > 0:
        success_rate = s.sagas_completed / s.sagas_started * 100
        print(f"\nSaga success rate: {success_rate:.1f}%")

    print("=" * 60)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate saga outcome visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    s = result.saga.stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Saga outcomes
    ax = axes[0]
    outcomes = ["Completed", "Compensated", "Failed"]
    counts = [s.sagas_completed, s.sagas_compensated, s.sagas_failed]
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    ax.bar(outcomes, counts, color=colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Saga Outcomes")
    ax.grid(True, alpha=0.3, axis="y")

    # Service action/compensation breakdown
    ax = axes[1]
    services = ["Inventory", "Payment", "Shipping"]
    actions = [result.inventory.actions, result.payment.actions, result.shipping.actions]
    comps = [
        result.inventory.compensations,
        result.payment.compensations,
        result.shipping.compensations,
    ]
    x = range(len(services))
    w = 0.35
    ax.bar([i - w / 2 for i in x], actions, w, label="Actions", color="#3498db", alpha=0.8)
    ax.bar([i + w / 2 for i in x], comps, w, label="Compensations", color="#e74c3c", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(services)
    ax.set_ylabel("Count")
    ax.set_title("Service Action vs Compensation")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "saga_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'saga_results.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Saga failure cascade simulation")
    parser.add_argument("--duration", type=float, default=10.0, help="Load duration (s)")
    parser.add_argument("--rate", type=float, default=5.0, help="Arrival rate (req/s)")
    parser.add_argument("--failure-rate", type=float, default=0.5, help="Shipping failure rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/saga", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running saga failure cascade simulation...")
    result = run_saga_simulation(
        duration_s=args.duration,
        arrival_rate=args.rate,
        shipping_failure_rate=args.failure_rate,
        seed=seed,
    )
    print_summary(result)

    if not args.no_viz:
        visualize_results(result, Path(args.output))
