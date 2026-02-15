"""Idempotency store under client retries.

Demonstrates duplicate suppression when a retrying client sends the same
request multiple times. Compares processed counts with and without the
IdempotencyStore to show how duplicates are suppressed.

## Architecture

```
                    WITH IDEMPOTENCY STORE
   ┌──────────┐     ┌─────────────────┐     ┌──────────┐
   │ Retrying │────►│  Idempotency    │────►│  Payment  │
   │  Client  │     │     Store       │     │  Service  │
   └──────────┘     └─────────────────┘     └──────────┘
                     Dedup by key             Processes
                     Cache hit = no-op        unique only

                    WITHOUT IDEMPOTENCY STORE
   ┌──────────┐     ┌──────────┐
   │ Retrying │────►│  Payment  │
   │  Client  │     │  Service  │
   └──────────┘     └──────────┘
                     Processes ALL
                     (including dupes)
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
from happysimulator.components.microservice import IdempotencyStore


# =============================================================================
# Components
# =============================================================================


class PaymentService(Entity):
    """Service that processes payments with random latency."""

    def __init__(self, name: str, mean_latency: float = 0.05):
        super().__init__(name)
        self.mean_latency = mean_latency
        self.requests_processed = 0
        self.unique_keys: set[str] = set()

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_processed += 1
        key = event.context.get("metadata", {}).get("idempotency_key")
        if key:
            self.unique_keys.add(key)
        yield random.expovariate(1.0 / self.mean_latency)


class RetryingClient(Entity):
    """Client that retries on timeout."""

    def __init__(self, name: str, target: Entity, timeout: float = 0.03,
                 max_retries: int = 3):
        super().__init__(name)
        self._target = target
        self._timeout = timeout
        self._max_retries = max_retries
        self._in_flight: dict[str, int] = {}  # key -> attempt
        self.requests_sent = 0
        self.retries = 0

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type == "_rc_timeout":
            return self._handle_timeout(event)
        if event.event_type == "_rc_done":
            return self._handle_done(event)
        return self._start_request(event)

    def _start_request(self, event: Event) -> list[Event]:
        key = event.context.get("metadata", {}).get("idempotency_key", str(self.requests_sent))
        return self._send(key, attempt=0)

    def _send(self, key: str, attempt: int) -> list[Event]:
        self.requests_sent += 1
        self._in_flight[key] = attempt

        from happysimulator.core.temporal import Duration

        forward = Event(
            time=self.now,
            event_type="payment",
            target=self._target,
            context={"metadata": {"idempotency_key": key}},
        )

        def on_complete(t: Instant):
            return Event(time=t, event_type="_rc_done", target=self,
                         context={"metadata": {"key": key}})
        forward.add_completion_hook(on_complete)

        timeout = Event(
            time=self.now + Duration.from_seconds(self._timeout),
            event_type="_rc_timeout", target=self,
            context={"metadata": {"key": key, "attempt": attempt}},
        )
        return [forward, timeout]

    def _handle_done(self, event: Event) -> None:
        key = event.context.get("metadata", {}).get("key")
        self._in_flight.pop(key, None)

    def _handle_timeout(self, event: Event) -> list[Event]:
        md = event.context.get("metadata", {})
        key = md.get("key")
        attempt = md.get("attempt", 0)
        if key not in self._in_flight:
            return []
        if attempt >= self._max_retries:
            del self._in_flight[key]
            return []
        self.retries += 1
        return self._send(key, attempt + 1)


class RequestProvider(EventProvider):
    """Generates payment requests with idempotency keys."""

    def __init__(self, client: RetryingClient, stop_after: Instant | None = None):
        self._client = client
        self._stop_after = stop_after
        self._next_id = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after and time > self._stop_after:
            return []
        self._next_id += 1
        return [Event(
            time=time, event_type="new_payment", target=self._client,
            context={"metadata": {"idempotency_key": f"pay_{self._next_id}"}},
        )]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the idempotency simulation."""

    with_store_processed: int
    without_store_processed: int
    with_store_unique: int
    without_store_unique: int
    total_requests_sent_with: int
    total_requests_sent_without: int
    store_cache_hits: int
    store_cache_misses: int


def run_idempotency_simulation(
    *,
    duration_s: float = 10.0,
    arrival_rate: float = 20.0,
    mean_service_latency: float = 0.05,
    client_timeout: float = 0.03,
    max_retries: int = 3,
    seed: int | None = 42,
) -> SimulationResult:
    """Run paired simulations with and without idempotency store."""
    # --- With IdempotencyStore ---
    if seed is not None:
        random.seed(seed)

    payment_with = PaymentService("PaymentWithStore", mean_latency=mean_service_latency)
    store = IdempotencyStore(
        name="IdempotencyStore",
        target=payment_with,
        key_extractor=lambda e: e.context.get("metadata", {}).get("idempotency_key"),
        ttl=60.0,
    )
    client_with = RetryingClient("ClientWith", target=store,
                                  timeout=client_timeout, max_retries=max_retries)

    stop = Instant.from_seconds(duration_s)
    provider_with = RequestProvider(client_with, stop_after=stop)
    profile = ConstantRateProfile(rate=arrival_rate)
    source_with = Source(
        name="Source",
        event_provider=provider_with,
        arrival_time_provider=ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch),
    )

    sim_with = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 5.0,
        sources=[source_with],
        entities=[client_with, store, payment_with],
    )
    sim_with.run()

    # --- Without IdempotencyStore ---
    if seed is not None:
        random.seed(seed)

    payment_without = PaymentService("PaymentWithout", mean_latency=mean_service_latency)
    client_without = RetryingClient("ClientWithout", target=payment_without,
                                     timeout=client_timeout, max_retries=max_retries)

    provider_without = RequestProvider(client_without, stop_after=stop)
    source_without = Source(
        name="Source",
        event_provider=provider_without,
        arrival_time_provider=ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch),
    )

    sim_without = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 5.0,
        sources=[source_without],
        entities=[client_without, payment_without],
    )
    sim_without.run()

    return SimulationResult(
        with_store_processed=payment_with.requests_processed,
        without_store_processed=payment_without.requests_processed,
        with_store_unique=len(payment_with.unique_keys),
        without_store_unique=len(payment_without.unique_keys),
        total_requests_sent_with=client_with.requests_sent,
        total_requests_sent_without=client_without.requests_sent,
        store_cache_hits=store.stats.cache_hits,
        store_cache_misses=store.stats.cache_misses,
    )


def print_summary(result: SimulationResult) -> None:
    """Print comparison summary."""
    print("\n" + "=" * 60)
    print("IDEMPOTENCY UNDER RETRIES — RESULTS")
    print("=" * 60)

    print(f"\nWith IdempotencyStore:")
    print(f"  Requests sent by client:  {result.total_requests_sent_with}")
    print(f"  Processed by service:     {result.with_store_processed}")
    print(f"  Unique keys processed:    {result.with_store_unique}")
    print(f"  Cache hits (suppressed):  {result.store_cache_hits}")
    print(f"  Cache misses (forwarded): {result.store_cache_misses}")

    print(f"\nWithout IdempotencyStore:")
    print(f"  Requests sent by client:  {result.total_requests_sent_without}")
    print(f"  Processed by service:     {result.without_store_processed}")
    print(f"  Unique keys processed:    {result.without_store_unique}")

    if result.without_store_processed > 0:
        dup_rate = 1.0 - result.without_store_unique / result.without_store_processed
        print(f"  Duplicate rate:           {dup_rate * 100:.1f}%")

    reduction = result.without_store_processed - result.with_store_processed
    if result.without_store_processed > 0:
        pct = reduction / result.without_store_processed * 100
        print(f"\nDuplicate processing reduction: {reduction} ({pct:.1f}%)")

    print("=" * 60)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate comparison bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["With Store", "Without Store"]
    processed = [result.with_store_processed, result.without_store_processed]
    unique = [result.with_store_unique, result.without_store_unique]

    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], processed, width, label="Total Processed", alpha=0.8)
    ax.bar([i + width / 2 for i in x], unique, width, label="Unique Keys", alpha=0.8)

    ax.set_ylabel("Count")
    ax.set_title("Idempotency Store: Duplicate Suppression")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "idempotency_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'idempotency_results.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Idempotency under retries simulation")
    parser.add_argument("--duration", type=float, default=10.0, help="Load duration (s)")
    parser.add_argument("--rate", type=float, default=20.0, help="Arrival rate (req/s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/idempotency", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running idempotency simulation...")
    result = run_idempotency_simulation(
        duration_s=args.duration,
        arrival_rate=args.rate,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
