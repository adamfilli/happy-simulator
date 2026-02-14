"""Bank branch discrete-event simulation with balking and reneging.

Simulates a bank branch where customers arrive via Poisson process,
may balk (leave immediately if the queue is too long), wait with
limited patience (renege if not served in time), and are served by
tellers whose staffing varies by shift (peak hours add extra tellers).

Customer types:
  - deposit  (40%): ~2 min mean service time
  - inquiry  (40%): ~5 min mean service time
  - loan     (20%): ~15 min mean service time

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                       BANK BRANCH SIMULATION                          |
+-----------------------------------------------------------------------+

    +---------+      +------------------+      +---------+      +------+
    | Source  |----->| BalkingQueue     |----->| Tellers |----->| Sink |
    |(Poisson)|      | (threshold=8,    |      | (2-3,   |      |      |
    |         |      |  p=1.0)          |      |  shift) |      |      |
    +---------+      +------------------+      +---------+      +------+
                           |                       |
                           | balked                 | reneged
                           v                       v
                       (dropped)              +----------+
                                              | Reneged  |
                                              | Counter  |
                                              +----------+

    Shift schedule:
      [0, peak_start)     : num_tellers      (normal staffing)
      [peak_start, peak_end) : num_tellers + peak_extra (peak staffing)
      [peak_end, end)      : num_tellers      (normal staffing)
```
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    LatencyTracker,
    Probe,
    QueuedResource,
    Simulation,
    SimulationSummary,
    Source,
)
from happysimulator.components.common import Counter
from happysimulator.components.industrial import (
    BalkingQueue,
    RenegingQueuedResource,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class BankConfig:
    """Configuration for the bank branch simulation."""

    duration_s: float = 3600.0          # 1 hour of operation
    arrival_rate_per_min: float = 1.5   # customers per minute
    num_tellers: int = 2                # baseline teller count
    peak_teller_extra: int = 1          # additional tellers during peak
    peak_start_min: float = 20.0        # peak period start (minutes)
    peak_end_min: float = 40.0          # peak period end (minutes)
    balk_threshold: int = 8             # queue depth that triggers balking
    default_patience_min: float = 10.0  # mean patience (minutes)
    seed: int = 42


# Customer type definitions: (name, probability, mean_service_time_minutes)
CUSTOMER_TYPES: list[tuple[str, float, float]] = [
    ("deposit", 0.40, 2.0),
    ("inquiry", 0.40, 5.0),
    ("loan", 0.20, 15.0),
]


# =============================================================================
# Customer Event Provider
# =============================================================================


class CustomerProvider(EventProvider):
    """Generates customer arrival events with type-specific context.

    Each customer is assigned a random type (deposit, inquiry, loan)
    and given a patience drawn from an exponential distribution.
    The event context carries customer_type, patience_s, and created_at
    for downstream processing and latency tracking.
    """

    def __init__(
        self,
        target: Entity,
        default_patience_min: float,
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._default_patience_min = default_patience_min
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1

        # Assign customer type based on weighted probabilities
        r = random.random()
        cumulative = 0.0
        customer_type = CUSTOMER_TYPES[-1][0]
        for name, prob, _ in CUSTOMER_TYPES:
            cumulative += prob
            if r < cumulative:
                customer_type = name
                break

        # Patience drawn from exponential distribution (in seconds)
        patience_s = random.expovariate(1.0 / (self._default_patience_min * 60))

        return [
            Event(
                time=time,
                event_type="Customer",
                target=self._target,
                context={
                    "created_at": time,
                    "request_id": self.generated,
                    "customer_type": customer_type,
                    "patience_s": patience_s,
                },
            )
        ]


# =============================================================================
# Bank Teller (RenegingQueuedResource with concurrency control)
# =============================================================================


class BankTeller(RenegingQueuedResource):
    """Bank teller station with reneging, concurrency control, and shift-based capacity.

    Customers who have waited longer than their patience are routed to
    reneged_target. Service time is exponentially distributed with a
    mean that depends on customer type.

    The teller capacity changes at shift boundaries: normal hours use
    num_tellers, peak hours add peak_extra.
    """

    def __init__(
        self,
        name: str,
        *,
        downstream: Entity,
        reneged_target: Entity | None = None,
        default_patience_s: float,
        concurrency: int,
        policy: BalkingQueue | None = None,
    ):
        super().__init__(
            name,
            reneged_target=reneged_target,
            default_patience_s=default_patience_s,
            policy=policy or FIFOQueue(),
        )
        self.downstream = downstream
        self._concurrency = concurrency
        self._active = 0
        self._processed = 0
        self._service_time_total = 0.0

    @property
    def processed(self) -> int:
        return self._processed

    def has_capacity(self) -> bool:
        return self._active < self._concurrency

    def _handle_served_event(
        self, event: Event
    ) -> Generator[float, None, list[Event]]:
        """Process a customer with type-dependent exponential service time."""
        customer_type = event.context.get("customer_type", "inquiry")

        # Look up mean service time for this customer type
        mean_service_min = 5.0  # default: inquiry
        for name, _, mean_min in CUSTOMER_TYPES:
            if name == customer_type:
                mean_service_min = mean_min
                break

        mean_service_s = mean_service_min * 60.0
        service_time = random.expovariate(1.0 / mean_service_s)

        self._active += 1
        try:
            yield service_time
        finally:
            self._active -= 1

        self._processed += 1
        self._service_time_total += service_time

        return [
            Event(
                time=self.now,
                event_type="Served",
                target=self.downstream,
                context=event.context,
            )
        ]


# =============================================================================
# Result
# =============================================================================


@dataclass
class BankResult:
    """Results from the bank branch simulation."""

    sink: LatencyTracker
    teller: BankTeller
    reneged_counter: Counter
    balking_queue: BalkingQueue
    queue_depth_data: Data
    customer_provider: CustomerProvider
    config: BankConfig
    summary: SimulationSummary


# =============================================================================
# Simulation Runner
# =============================================================================


def run_bank_simulation(config: BankConfig | None = None) -> BankResult:
    """Run the bank branch simulation.

    Args:
        config: Simulation configuration. Uses defaults if None.

    Returns:
        BankResult with all metrics and entity references.
    """
    if config is None:
        config = BankConfig()

    random.seed(config.seed)

    # Build pipeline from end to start
    sink = LatencyTracker("Sink")
    reneged_counter = Counter("RenegedCounter")

    # Create balking queue policy
    balking_queue = BalkingQueue(
        inner=FIFOQueue(),
        balk_threshold=config.balk_threshold,
        balk_probability=1.0,
    )

    # Create bank teller with reneging + balking
    teller = BankTeller(
        "Tellers",
        downstream=sink,
        reneged_target=reneged_counter,
        default_patience_s=config.default_patience_min * 60,
        concurrency=config.num_tellers,
        policy=balking_queue,
    )

    # Queue depth probe
    queue_depth_data = Data()
    queue_probe = Probe(
        target=teller,
        metric="depth",
        data=queue_depth_data,
        interval=10.0,
        start_time=Instant.Epoch,
    )

    # Customer source (Poisson arrivals)
    arrival_rate_per_s = config.arrival_rate_per_min / 60.0
    stop_after = Instant.from_seconds(config.duration_s)

    customer_provider = CustomerProvider(
        target=teller,
        default_patience_min=config.default_patience_min,
        stop_after=stop_after,
    )

    from happysimulator.load.providers.poisson_arrival import PoissonArrivalTimeProvider
    from happysimulator.load.profile import ConstantRateProfile

    source = Source(
        name="Arrivals",
        event_provider=customer_provider,
        arrival_time_provider=PoissonArrivalTimeProvider(
            ConstantRateProfile(rate=arrival_rate_per_s),
            start_time=Instant.Epoch,
        ),
    )

    # Schedule a capacity change for peak hours
    # We handle this by scheduling events that adjust concurrency
    peak_start_s = config.peak_start_min * 60
    peak_end_s = config.peak_end_min * 60
    peak_concurrency = config.num_tellers + config.peak_teller_extra

    capacity_events = [
        Event.once(
            time=Instant.from_seconds(peak_start_s),
            event_type="PeakStart",
            fn=lambda e, t=teller, c=peak_concurrency: _set_concurrency(t, c),
        ),
        Event.once(
            time=Instant.from_seconds(peak_end_s),
            event_type="PeakEnd",
            fn=lambda e, t=teller, c=config.num_tellers: _set_concurrency(t, c),
        ),
    ]

    # Run simulation with drain buffer
    end_time = Instant.from_seconds(config.duration_s + 1200)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=end_time,
        sources=[source],
        entities=[teller, sink, reneged_counter],
        probes=[queue_probe],
    )

    # Inject shift-change events into the heap before running
    for ev in capacity_events:
        sim.schedule(ev)

    summary = sim.run()

    return BankResult(
        sink=sink,
        teller=teller,
        reneged_counter=reneged_counter,
        balking_queue=balking_queue,
        queue_depth_data=queue_depth_data,
        customer_provider=customer_provider,
        config=config,
        summary=summary,
    )


def _set_concurrency(teller: BankTeller, concurrency: int) -> None:
    """Helper to adjust teller concurrency at shift boundaries."""
    teller._concurrency = concurrency


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: BankResult) -> None:
    """Print a formatted summary of the bank simulation results."""
    config = result.config

    print("\n" + "=" * 65)
    print("BANK BRANCH SIMULATION RESULTS")
    print("=" * 65)

    print(f"\nConfiguration:")
    print(f"  Duration:            {config.duration_s / 60:.0f} minutes")
    print(f"  Arrival rate:        {config.arrival_rate_per_min:.1f} customers/min")
    print(f"  Tellers (normal):    {config.num_tellers}")
    print(f"  Tellers (peak):      {config.num_tellers + config.peak_teller_extra}")
    print(f"  Peak window:         {config.peak_start_min:.0f}-{config.peak_end_min:.0f} min")
    print(f"  Balk threshold:      {config.balk_threshold} customers in queue")
    print(f"  Mean patience:       {config.default_patience_min:.0f} min")

    total_arrived = result.customer_provider.generated
    served = result.teller.served
    reneged = result.teller.reneged
    balked = result.balking_queue.balked
    completed = result.sink.count

    print(f"\nCustomer Flow:")
    print(f"  Arrived:             {total_arrived}")
    print(f"  Balked (left queue): {balked} ({100 * balked / max(total_arrived, 1):.1f}%)")
    print(f"  Reneged (gave up):   {reneged} ({100 * reneged / max(total_arrived, 1):.1f}%)")
    print(f"  Served:              {served} ({100 * served / max(total_arrived, 1):.1f}%)")
    print(f"  Completed (at sink): {completed}")

    if completed > 0:
        print(f"\nService Latency (end-to-end, including wait):")
        print(f"  Mean:    {result.sink.mean_latency() / 60:.2f} min")
        print(f"  p50:     {result.sink.p50() / 60:.2f} min")
        print(f"  p99:     {result.sink.p99() / 60:.2f} min")

    if result.teller.processed > 0:
        avg_service = result.teller._service_time_total / result.teller.processed
        print(f"\nService Time (processing only):")
        print(f"  Mean:    {avg_service / 60:.2f} min")
        print(f"  Total:   {result.teller._service_time_total / 60:.1f} min")

    print(f"\nQueue Depth:")
    qd = result.queue_depth_data
    if len(qd.raw_values()) > 0:
        print(f"  Mean:    {qd.mean():.1f}")
        print(f"  Max:     {qd.max():.0f}")
        print(f"  p99:     {qd.percentile(0.99):.0f}")

    print(f"\n{result.summary}")
    print("=" * 65)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bank branch simulation")
    parser.add_argument(
        "--duration", type=float, default=3600.0,
        help="Simulation duration in seconds (default: 3600)",
    )
    parser.add_argument(
        "--arrival-rate", type=float, default=1.5,
        help="Customer arrival rate per minute (default: 1.5)",
    )
    parser.add_argument(
        "--tellers", type=int, default=2,
        help="Number of tellers during normal hours (default: 2)",
    )
    parser.add_argument(
        "--peak-extra", type=int, default=1,
        help="Extra tellers during peak hours (default: 1)",
    )
    parser.add_argument(
        "--balk-threshold", type=int, default=8,
        help="Queue depth that triggers balking (default: 8)",
    )
    parser.add_argument(
        "--patience", type=float, default=10.0,
        help="Mean customer patience in minutes (default: 10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42, use -1 for random)",
    )
    args = parser.parse_args()

    config = BankConfig(
        duration_s=args.duration,
        arrival_rate_per_min=args.arrival_rate,
        num_tellers=args.tellers,
        peak_teller_extra=args.peak_extra,
        balk_threshold=args.balk_threshold,
        default_patience_min=args.patience,
        seed=args.seed if args.seed != -1 else random.randint(0, 2**31),
    )

    print("Running bank branch simulation...")
    print(f"  Duration: {config.duration_s / 60:.0f} min")
    print(f"  Arrival rate: {config.arrival_rate_per_min:.1f} customers/min")
    print(f"  Tellers: {config.num_tellers} (+ {config.peak_teller_extra} during peak)")
    print(f"  Seed: {config.seed}")

    result = run_bank_simulation(config)
    print_summary(result)
