"""Shuffle Fair Queuing (SFQ) with Shuffle Sharding and Best-of-2.

Based on: https://brooker.co.za/blog/2026/02/25/sfq.html

Demonstrates noisy-neighbor isolation using the combined SFQ approach:
1. **Shuffle Sharding**: Each customer hashes to 2 candidate queues (out of 6)
2. **Best-of-2**: Requests go to the shorter of the 2 candidate queues
3. **Hash Perturbation**: Every 5 seconds the hash seed changes, reshuffling
   customer-to-queue mappings so collisions are short-lived

## Architecture

```
  Customer0 (2 req/s) ──┐
  Customer1 (2 req/s) ──┤
  Customer2 (2 req/s) ──┤                    ┌─── Queue0 ───┐
  Customer3 (2 req/s) ──┼──> SFQ Router ───> ├─── Queue1 ───┤
  Customer4 (2 req/s) ──┤   (best-of-n)      ├─── Queue2 ───┤──> RoundRobin ──> LatencySink
  Customer5 (2 req/s) ──┤        ↑            ├─── Queue3 ───┤     Poller
  Customer6 (2 req/s) ──┤    PerturbHash      ├─── Queue4 ───┤
  Customer7 (NOISY)   ──┘   (every s seconds) └─── Queue5 ───┘
```

Customer 7 is "noisy": low rate normally, but bursts every 30 seconds
for 10 seconds. The periodic hash perturbation limits how long the noisy
customer can saturate any particular queue pair.

## Key Metrics

- Per-queue depth over time (shows burst spreading)
- Per-customer latency (noisy vs normal isolation)
- Hash perturbation events visible as queue-depth shifts
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    Entity,
    Event,
    EventProvider,
    Instant,
    PoissonArrivalTimeProvider,
    Probe,
    Simulation,
    Source,
)
from happysimulator.load.profile import Profile

if TYPE_CHECKING:
    pass

# =============================================================================
# Tunable knobs — edit these to experiment
# =============================================================================

CUSTOMER_REQ_PER_SECOND = 2.0       # req/s for each normal customer
BEST_OF_N = 2                        # candidate queues per request (best-of-N)
PERTURB_INTERVAL_S = 60.0                # seconds between hash perturbations
NOISY_BURST_RATE = 40.0              # req/s during noisy customer burst
NOISY_BURST_DURATION_S = 10.0        # how long each burst lasts (seconds)
NOISY_BURST_PERIOD_S = 30.0          # seconds between burst starts
MAX_QUEUE_DEPTH = 20                 # max items per queue (0 = unlimited)
POLL_RATE = 30.0                     # polls/s for the round-robin poller

# =============================================================================
# Profiles
# =============================================================================


@dataclass(frozen=True)
class PeriodicBurstProfile(Profile):
    """Repeating burst pattern: normal rate with periodic high-rate bursts.

    Timeline repeats every ``period_s`` seconds:
    - [0, burst_duration_s): burst_rate
    - [burst_duration_s, period_s): normal_rate
    """

    normal_rate: float
    burst_rate: float
    period_s: float
    burst_duration_s: float

    def get_rate(self, time: Instant) -> float:
        phase = time.to_seconds() % self.period_s
        if phase < self.burst_duration_s:
            return self.burst_rate
        return self.normal_rate


# =============================================================================
# Buffer Queue — simple FIFO buffer entity
# =============================================================================


class BufferQueue(Entity):
    """A simple FIFO buffer that accepts events and allows external dequeue.

    Events arrive via ``handle_event`` and are stored internally.
    The ``RoundRobinPoller`` calls ``dequeue()`` directly to pull items.
    """

    def __init__(self, name: str, *, max_depth: int = 0) -> None:
        super().__init__(name)
        self._buffer: deque[Event] = deque()
        self._max_depth = max_depth  # 0 = unlimited
        self.depth: int = 0
        self.enqueued: int = 0
        self.dequeued: int = 0

    @property
    def is_full(self) -> bool:
        return self._max_depth > 0 and self.depth >= self._max_depth

    def handle_event(self, event: Event) -> list[Event]:
        self._buffer.append(event)
        self.depth += 1
        self.enqueued += 1
        return []

    def dequeue(self) -> Event | None:
        if self._buffer:
            self.depth -= 1
            self.dequeued += 1
            return self._buffer.popleft()
        return None

    def downstream_entities(self) -> list[Entity]:
        return []  # leaf — poller pulls from us, no outgoing edges


# =============================================================================
# SFQ Router — shuffle sharding + best-of-N
# =============================================================================


class SFQRouter(Entity):
    """Routes events to queues using Shuffle Fair Queuing.

    For each event, hashes (customer_id, seed) to pick N candidate queues,
    then sends the event to the shortest non-full candidate (best-of-N).

    If all candidate queues are full, the event is dropped and forwarded
    to the ``drop_sink`` (if provided).

    Handles ``PerturbHash`` events by changing the hash seed.
    """

    def __init__(
        self,
        name: str,
        queues: list[BufferQueue],
        *,
        best_of_n: int = 2,
        seed: int = 0,
        drop_sink: Entity | None = None,
    ) -> None:
        super().__init__(name)
        self._queues = queues
        self._num_queues = len(queues)
        self._best_of_n = min(best_of_n, self._num_queues)
        self._seed = seed
        self._drop_sink = drop_sink
        self.perturbations: int = 0
        self.routed: int = 0
        self.dropped: int = 0

    def downstream_entities(self) -> list[Entity]:
        result: list[Entity] = list(self._queues)
        if self._drop_sink is not None:
            result.append(self._drop_sink)
        return result

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type == "PerturbHash":
            self._seed = event.context.get("new_seed", self._seed + 1)
            self.perturbations += 1
            return []

        customer_id = event.context.get("customer_id", 0)
        candidates = self._candidate_queues(customer_id)

        # Pick shortest non-full candidate, or drop
        best: BufferQueue | None = None
        for idx in candidates:
            q = self._queues[idx]
            if not q.is_full and (best is None or q.depth < best.depth):
                best = q

        if best is None:
            self.dropped += 1
            if self._drop_sink is not None:
                return [
                    Event(
                        time=self.now,
                        event_type="Dropped",
                        target=self._drop_sink,
                        context=event.context,
                    )
                ]
            return []

        self.routed += 1
        return [
            Event(
                time=self.now,
                event_type="Enqueue",
                target=best,
                context=event.context,
            )
        ]

    def _candidate_queues(self, customer_id: int) -> list[int]:
        """Hash customer to N distinct queue indices."""
        h = abs(hash((customer_id, self._seed)))
        # Build N distinct indices using Fisher-Yates-style selection
        available = list(range(self._num_queues))
        chosen: list[int] = []
        for i in range(self._best_of_n):
            pick = h % len(available)
            chosen.append(available[pick])
            # Remove picked slot so subsequent picks are distinct
            available[pick] = available[-1]
            available.pop()
            h = h // (len(available) + 1)
        return chosen


# =============================================================================
# Round-Robin Poller
# =============================================================================


class RoundRobinPoller(Entity):
    """Polls queues in round-robin order at a configurable rate.

    Each poll dequeues one event from the current queue (if non-empty)
    and forwards it to the downstream sink with zero additional latency.
    """

    def __init__(
        self,
        name: str,
        queues: list[BufferQueue],
        downstream: Entity,
        *,
        poll_rate: float = 20.0,
    ) -> None:
        super().__init__(name)
        self._queues = queues
        self._downstream = downstream
        self._poll_rate = poll_rate
        self._current_idx: int = 0
        self.polls: int = 0
        self.forwarded: int = 0
        self.idle_polls: int = 0

    def downstream_entities(self) -> list[Entity]:
        return [*self._queues, self._downstream]

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type != "Poll":
            return []

        self.polls += 1
        result_events: list[Event] = []

        # Try current queue
        queue = self._queues[self._current_idx]
        item = queue.dequeue()
        if item is not None:
            self.forwarded += 1
            result_events.append(
                Event(
                    time=self.now,
                    event_type="Completed",
                    target=self._downstream,
                    context=item.context,
                )
            )
        else:
            self.idle_polls += 1

        # Advance round-robin
        self._current_idx = (self._current_idx + 1) % len(self._queues)

        # Schedule next poll
        poll_interval = 1.0 / self._poll_rate
        result_events.append(
            Event(
                time=self.now + poll_interval,
                event_type="Poll",
                target=self,
            )
        )
        return result_events


# =============================================================================
# Latency Sink
# =============================================================================


class LatencySink(Entity):
    """Collects queue-wait latency from completed events.

    Latency = current time - context["created_at"].
    Tracks per-customer latency for fairness analysis.

    Maintains a ``Data`` time-series per customer so the visual debugger
    can chart per-customer latency over time.

    Exposes ``customer_N_latency_ms`` attributes for ``Probe.on()``
    (returns the last observed latency in ms for customer N).
    """

    def __init__(self, name: str, *, num_customers: int = 0) -> None:
        super().__init__(name)
        self._num_customers = num_customers
        self.events_received: int = 0
        self.latencies_s: list[float] = []
        self.customer_latencies: dict[int, list[float]] = {}
        self.completion_times: list[Instant] = []
        self.customer_ids: list[int] = []

        # Per-customer Data time-series (for Charts)
        self.customer_latency_data: dict[int, Data] = {
            cid: Data() for cid in range(num_customers)
        }

        # Last observed latency per customer (for Probe sampling)
        self._last_latency_ms: dict[int, float] = {
            cid: 0.0 for cid in range(num_customers)
        }

    def __getattr__(self, name: str) -> float:
        # Allow Probe.on(sink, "customer_3_latency_ms") etc.
        if name.startswith("customer_") and name.endswith("_latency_ms"):
            try:
                cid = int(name.removeprefix("customer_").removesuffix("_latency_ms"))
                return self._last_latency_ms.get(cid, 0.0)
            except ValueError:
                pass
        raise AttributeError(name)

    def handle_event(self, event: Event) -> list[Event]:
        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()
        latency_ms = latency_s * 1000.0

        customer_id = event.context.get("customer_id", -1)

        self.events_received += 1
        self.latencies_s.append(latency_s)
        self.completion_times.append(event.time)
        self.customer_ids.append(customer_id)

        if customer_id not in self.customer_latencies:
            self.customer_latencies[customer_id] = []
        self.customer_latencies[customer_id].append(latency_s)

        # Update last latency and Data for charting
        if customer_id in self._last_latency_ms:
            self._last_latency_ms[customer_id] = latency_ms
        if customer_id in self.customer_latency_data:
            self.customer_latency_data[customer_id].add_stat(latency_ms, event.time)

        return []


# =============================================================================
# Drop Sink — records events dropped due to full queues
# =============================================================================


class DropSink(Entity):
    """Records events that were dropped because both candidate queues were full.

    Tracks drop count per customer for fairness analysis.

    Exposes ``customer_N_drops`` attributes for ``Probe.on()``
    (returns cumulative drop count for customer N).
    """

    def __init__(self, name: str, *, num_customers: int = 0) -> None:
        super().__init__(name)
        self._num_customers = num_customers
        self.events_dropped: int = 0
        self.drop_times: list[Instant] = []
        self.customer_drops: dict[int, int] = {
            cid: 0 for cid in range(num_customers)
        }

    def __getattr__(self, name: str) -> int:
        # Allow Probe.on(drop_sink, "customer_3_drops") etc.
        if name.startswith("customer_") and name.endswith("_drops"):
            try:
                cid = int(name.removeprefix("customer_").removesuffix("_drops"))
                return self.customer_drops.get(cid, 0)
            except ValueError:
                pass
        raise AttributeError(name)

    def handle_event(self, event: Event) -> list[Event]:
        self.events_dropped += 1
        self.drop_times.append(event.time)
        customer_id = event.context.get("customer_id", -1)
        self.customer_drops[customer_id] = self.customer_drops.get(customer_id, 0) + 1
        return []


# =============================================================================
# Event Providers
# =============================================================================


class CustomerEventProvider(EventProvider):
    """Generates request events for a specific customer."""

    def __init__(
        self,
        target: Entity,
        customer_id: int,
        *,
        stop_after: Instant | None = None,
    ) -> None:
        self._target = target
        self._customer_id = customer_id
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []
        self.generated += 1
        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={
                    "customer_id": self._customer_id,
                    "created_at": time,
                },
            )
        ]


class PerturbHashProvider(EventProvider):
    """Generates hash-perturbation events targeting the SFQ router."""

    def __init__(
        self,
        target: Entity,
        *,
        rng: random.Random,
        stop_after: Instant | None = None,
    ) -> None:
        self._target = target
        self._rng = rng
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []
        self.generated += 1
        return [
            Event(
                time=time,
                event_type="PerturbHash",
                target=self._target,
                context={"new_seed": self._rng.randint(0, 2**31)},
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SFQConfig:
    """Configuration for the SFQ simulation.

    Defaults are sourced from the top-level knobs so you can tweak
    either the knobs or pass explicit values here.
    """

    duration_s: float = 600.0
    drain_s: float = 5.0
    num_queues: int = 6
    num_customers: int = 8
    noisy_customer_id: int = 7
    normal_rate: float = CUSTOMER_REQ_PER_SECOND
    noisy_normal_rate: float = CUSTOMER_REQ_PER_SECOND
    noisy_burst_rate: float = NOISY_BURST_RATE
    burst_period_s: float = NOISY_BURST_PERIOD_S
    burst_duration_s: float = NOISY_BURST_DURATION_S
    perturb_interval_s: float = PERTURB_INTERVAL_S
    best_of_n: int = BEST_OF_N
    poll_rate: float = POLL_RATE
    max_queue_depth: int = MAX_QUEUE_DEPTH
    probe_interval_s: float = 0.5
    seed: int = 42


@dataclass
class SFQSetup:
    """Pre-run simulation objects (for visual debugger or manual control)."""

    sim: Simulation
    config: SFQConfig
    sink: LatencySink
    drop_sink: DropSink
    queues: list[BufferQueue]
    router: SFQRouter
    poller: RoundRobinPoller
    customer_providers: list[CustomerEventProvider]
    perturb_provider: PerturbHashProvider


@dataclass
class SFQResult:
    """Results from the SFQ simulation (after run)."""

    config: SFQConfig
    sink: LatencySink
    drop_sink: DropSink
    queues: list[BufferQueue]
    router: SFQRouter
    poller: RoundRobinPoller
    customer_providers: list[CustomerEventProvider]
    perturb_provider: PerturbHashProvider


def build_sfq_simulation(config: SFQConfig | None = None) -> SFQSetup:
    """Build the SFQ simulation without running it.

    Use this when you want to pass the simulation to ``serve()``
    for the visual debugger, or when you need manual control.
    """
    if config is None:
        config = SFQConfig()

    rng = random.Random(config.seed)

    # --- Entities ---
    queues = [
        BufferQueue(f"Queue{i}", max_depth=config.max_queue_depth)
        for i in range(config.num_queues)
    ]
    drop_sink = DropSink("DropSink", num_customers=config.num_customers)
    router = SFQRouter(
        "SFQRouter", queues,
        best_of_n=config.best_of_n,
        seed=rng.randint(0, 2**31),
        drop_sink=drop_sink,
    )
    sink = LatencySink("LatencySink", num_customers=config.num_customers)
    poller = RoundRobinPoller(
        "Poller", queues, sink, poll_rate=config.poll_rate
    )

    # --- Sources: one per customer ---
    stop_after = Instant.from_seconds(config.duration_s)
    sources: list[Source] = []
    customer_providers: list[CustomerEventProvider] = []

    for cid in range(config.num_customers):
        provider = CustomerEventProvider(router, cid, stop_after=stop_after)
        customer_providers.append(provider)

        if cid == config.noisy_customer_id:
            profile = PeriodicBurstProfile(
                normal_rate=config.noisy_normal_rate,
                burst_rate=config.noisy_burst_rate,
                period_s=config.burst_period_s,
                burst_duration_s=config.burst_duration_s,
            )
            arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)
        else:
            profile = ConstantRateProfile(rate=config.normal_rate)
            arrival = PoissonArrivalTimeProvider(profile, start_time=Instant.Epoch)

        sources.append(
            Source(
                name=f"Customer{cid}",
                event_provider=provider,
                arrival_time_provider=arrival,
            )
        )

    # --- Perturb hash source ---
    perturb_provider = PerturbHashProvider(router, rng=rng, stop_after=stop_after)
    perturb_profile = ConstantRateProfile(rate=1.0 / config.perturb_interval_s)
    perturb_arrival = ConstantArrivalTimeProvider(
        perturb_profile, start_time=Instant.Epoch
    )
    perturb_source = Source(
        name="PerturbHash",
        event_provider=perturb_provider,
        arrival_time_provider=perturb_arrival,
    )
    sources.append(perturb_source)

    # --- Bootstrap: first poll event ---
    first_poll = Event(
        time=Instant.Epoch,
        event_type="Poll",
        target=poller,
    )

    # --- Entities list ---
    entities: list[Entity] = [router, *queues, poller, sink, drop_sink]

    # --- Probes: per-customer latency + drops ---
    probes = []
    for cid in range(config.num_customers):
        latency_probe, _ = Probe.on(
            sink, f"customer_{cid}_latency_ms", interval=config.probe_interval_s
        )
        drop_probe, _ = Probe.on(
            drop_sink, f"customer_{cid}_drops", interval=config.probe_interval_s
        )
        probes.extend([latency_probe, drop_probe])

    # --- Build (don't run) ---
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=config.duration_s + config.drain_s,
        sources=sources,
        entities=entities,
        probes=probes,
    )
    sim.schedule(first_poll)

    return SFQSetup(
        sim=sim,
        config=config,
        sink=sink,
        drop_sink=drop_sink,
        queues=queues,
        router=router,
        poller=poller,
        customer_providers=customer_providers,
        perturb_provider=perturb_provider,
    )


def run_sfq_simulation(config: SFQConfig | None = None) -> SFQResult:
    """Build and run the SFQ simulation."""
    setup = build_sfq_simulation(config)
    summary = setup.sim.run()

    print(f"Simulation complete: {summary.total_events_processed} events "
          f"in {summary.wall_clock_seconds:.1f}s")

    return SFQResult(
        config=setup.config,
        sink=setup.sink,
        drop_sink=setup.drop_sink,
        queues=setup.queues,
        router=setup.router,
        poller=setup.poller,

        customer_providers=setup.customer_providers,
        perturb_provider=setup.perturb_provider,
    )


# =============================================================================
# Output
# =============================================================================


def print_summary(result: SFQResult) -> None:
    """Print summary statistics."""
    cfg = result.config
    sink = result.sink

    print("\n" + "=" * 70)
    print("SHUFFLE FAIR QUEUING (SFQ) SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"  Queues: {cfg.num_queues}, best-of-{cfg.best_of_n} routing")
    print(f"  Customers: {cfg.num_customers} "
          f"(noisy = Customer{cfg.noisy_customer_id})")
    print(f"  Normal customer rate: {cfg.normal_rate} req/s each")
    print(f"  Noisy customer: {cfg.noisy_normal_rate} req/s normal, "
          f"{cfg.noisy_burst_rate} req/s burst")
    print(f"  Burst pattern: {cfg.burst_duration_s}s on / "
          f"{cfg.burst_period_s - cfg.burst_duration_s}s off "
          f"(period {cfg.burst_period_s}s)")
    print(f"  Max queue depth: {cfg.max_queue_depth}")
    print(f"  Hash perturbation: every {cfg.perturb_interval_s}s")
    print(f"  Poller rate: {cfg.poll_rate} polls/s (round-robin)")
    print(f"  Duration: {cfg.duration_s}s + {cfg.drain_s}s drain")

    print(f"\nLoad Generation:")
    for i, p in enumerate(result.customer_providers):
        tag = " (NOISY)" if i == cfg.noisy_customer_id else ""
        print(f"  Customer{i}{tag}: {p.generated} events")
    print(f"  Hash perturbations: {result.perturb_provider.generated}")

    print(f"\nRouter: {result.router.routed} routed, "
          f"{result.router.dropped} dropped, "
          f"{result.router.perturbations} perturbations")

    print(f"\nQueue Stats:")
    for i, q in enumerate(result.queues):
        print(f"  {q.name}: enqueued={q.enqueued}, dequeued={q.dequeued}, "
              f"remaining={q.depth}")

    print(f"\nPoller: {result.poller.polls} polls, "
          f"{result.poller.forwarded} forwarded, "
          f"{result.poller.idle_polls} idle")

    drop_sink = result.drop_sink
    print(f"\nDropped Events: {drop_sink.events_dropped} total")
    if drop_sink.customer_drops:
        for cid in sorted(drop_sink.customer_drops.keys()):
            tag = " ** NOISY **" if cid == cfg.noisy_customer_id else ""
            print(f"  Customer{cid}{tag}: {drop_sink.customer_drops[cid]} dropped")

    print(f"\nLatency Summary ({sink.events_received} events served):")
    if sink.latencies_s:
        sorted_lat = sorted(sink.latencies_s)
        n = len(sorted_lat)
        print(f"  Overall:  avg={sum(sorted_lat)/n*1000:.1f}ms  "
              f"p50={sorted_lat[n//2]*1000:.1f}ms  "
              f"p99={sorted_lat[int(n*0.99)]*1000:.1f}ms  "
              f"max={sorted_lat[-1]*1000:.1f}ms")

    print(f"\nPer-Customer Latency:")
    noisy_id = cfg.noisy_customer_id
    for cid in sorted(sink.customer_latencies.keys()):
        lats = sorted(sink.customer_latencies[cid])
        n = len(lats)
        if n == 0:
            continue
        tag = " ** NOISY **" if cid == noisy_id else ""
        print(f"  Customer{cid}{tag}: "
              f"count={n}  "
              f"avg={sum(lats)/n*1000:.1f}ms  "
              f"p50={lats[n//2]*1000:.1f}ms  "
              f"p99={lats[int(n*0.99)]*1000:.1f}ms  "
              f"max={lats[-1]*1000:.1f}ms")

    print("\n" + "=" * 70)


def visualize_results(result: SFQResult, output_dir: Path) -> None:
    """Generate visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = result.config
    noisy_id = cfg.noisy_customer_id

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Helper: shade burst windows
    def add_burst_shading(ax: plt.Axes) -> None:
        t = 0.0
        while t < cfg.duration_s:
            ax.axvspan(t, t + cfg.burst_duration_s,
                       alpha=0.08, color="red")
            t += cfg.burst_period_s

    # Helper: mark perturbation times
    def add_perturb_markers(ax: plt.Axes) -> None:
        t = cfg.perturb_interval_s
        while t < cfg.duration_s:
            ax.axvline(t, color="green", alpha=0.15, linewidth=0.5)
            t += cfg.perturb_interval_s

    # ---- 1. Per-customer average latency over time ----
    ax = axes[0, 0]
    sink = result.sink
    for cid in range(cfg.num_customers):
        data = sink.customer_latency_data.get(cid)
        if data and data.values:
            bucketed = data.bucket(window_s=0.5)
            label = f"Customer{cid} (NOISY)" if cid == noisy_id else f"Customer{cid}"
            color = "red" if cid == noisy_id else None
            ax.plot(bucketed.times(), bucketed.means(), label=label,
                    alpha=0.9 if cid == noisy_id else 0.6, color=color)
    add_burst_shading(ax)
    add_perturb_markers(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Per-Customer Average Latency (500ms windows)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ---- 2. Per-customer average latency (bucketed by 5s) ----
    ax = axes[0, 1]
    bucket_width = 5.0

    # Separate noisy vs normal
    normal_buckets: dict[int, list[float]] = {}
    noisy_buckets: dict[int, list[float]] = {}

    for t, lat, cid in zip(
        sink.completion_times, sink.latencies_s, sink.customer_ids, strict=False
    ):
        bucket = int(t.to_seconds() / bucket_width)
        target = noisy_buckets if cid == noisy_id else normal_buckets
        if bucket not in target:
            target[bucket] = []
        target[bucket].append(lat)

    if normal_buckets:
        bk = sorted(normal_buckets.keys())
        times = [b * bucket_width for b in bk]
        avgs = [sum(normal_buckets[b]) / len(normal_buckets[b]) * 1000 for b in bk]
        ax.plot(times, avgs, "b-", label="Normal customers", alpha=0.8)

    if noisy_buckets:
        bk = sorted(noisy_buckets.keys())
        times = [b * bucket_width for b in bk]
        avgs = [sum(noisy_buckets[b]) / len(noisy_buckets[b]) * 1000 for b in bk]
        ax.plot(times, avgs, "r-", label=f"Customer{noisy_id} (noisy)", alpha=0.8)

    add_burst_shading(ax)
    add_perturb_markers(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Latency Over Time (noisy vs normal)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ---- 3. Latency distribution comparison ----
    ax = axes[1, 0]

    normal_lats = [
        lat * 1000
        for cid, lats in sink.customer_latencies.items()
        if cid != noisy_id
        for lat in lats
    ]
    noisy_lats = [
        lat * 1000
        for lat in sink.customer_latencies.get(noisy_id, [])
    ]

    if normal_lats:
        ax.hist(normal_lats, bins=50, alpha=0.5, label="Normal", color="blue")
    if noisy_lats:
        ax.hist(noisy_lats, bins=50, alpha=0.5,
                label=f"Customer{noisy_id} (noisy)", color="red")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # ---- 4. Total enqueue rate over time (1s buckets) ----
    ax = axes[1, 1]

    normal_rate_buckets: dict[int, int] = {}
    noisy_rate_buckets: dict[int, int] = {}

    # We don't have arrival times on the sink side, but we can use
    # completion_times as a proxy with the bucket approach
    # Better: count from customer providers isn't timestamped, so use
    # the sink data which shows when events were served
    for t, cid in zip(sink.completion_times, sink.customer_ids, strict=False):
        bucket = int(t.to_seconds())
        if cid == noisy_id:
            noisy_rate_buckets[bucket] = noisy_rate_buckets.get(bucket, 0) + 1
        else:
            normal_rate_buckets[bucket] = normal_rate_buckets.get(bucket, 0) + 1

    if normal_rate_buckets:
        bk = sorted(normal_rate_buckets.keys())
        ax.plot(bk, [normal_rate_buckets[b] for b in bk],
                "b-", label="Normal throughput", alpha=0.7)
    if noisy_rate_buckets:
        bk = sorted(noisy_rate_buckets.keys())
        ax.plot(bk, [noisy_rate_buckets[b] for b in bk],
                "r-", label=f"Customer{noisy_id} throughput", alpha=0.7)

    add_burst_shading(ax)
    add_perturb_markers(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Events/s")
    ax.set_title("Served Throughput Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "SFQ + Shuffle Sharding + Best-of-2  "
        f"({cfg.num_queues} queues, {cfg.num_customers} customers, "
        f"perturb every {cfg.perturb_interval_s}s)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "shuffle_fair_queuing.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'shuffle_fair_queuing.png'}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SFQ + Shuffle Sharding + Best-of-N simulation"
    )
    parser.add_argument("--duration", type=float, default=600.0,
                        help="Load duration (s)")
    parser.add_argument("--drain", type=float, default=5.0,
                        help="Drain time (s)")
    parser.add_argument("--queues", type=int, default=6,
                        help="Number of queues")
    parser.add_argument("--customers", type=int, default=8,
                        help="Number of customers")
    parser.add_argument("--noisy-id", type=int, default=7,
                        help="Noisy customer ID")
    parser.add_argument("--normal-rate", type=float, default=CUSTOMER_REQ_PER_SECOND,
                        help="Normal customer rate (req/s)")
    parser.add_argument("--noisy-normal-rate", type=float, default=CUSTOMER_REQ_PER_SECOND,
                        help="Noisy customer normal rate (req/s)")
    parser.add_argument("--noisy-burst-rate", type=float, default=NOISY_BURST_RATE,
                        help="Noisy customer burst rate (req/s)")
    parser.add_argument("--burst-period", type=float, default=NOISY_BURST_PERIOD_S,
                        help="Burst period (s)")
    parser.add_argument("--burst-duration", type=float, default=NOISY_BURST_DURATION_S,
                        help="Burst duration (s)")
    parser.add_argument("--perturb-interval", type=float, default=PERTURB_INTERVAL_S,
                        help="Hash perturbation interval (s)")
    parser.add_argument("--best-of-n", type=int, default=BEST_OF_N,
                        help="Candidate queues per request (best-of-N)")
    parser.add_argument("--poll-rate", type=float, default=POLL_RATE,
                        help="Poller rate (polls/s)")
    parser.add_argument("--max-queue-depth", type=int, default=MAX_QUEUE_DEPTH,
                        help="Max queue depth (0=unlimited)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str,
                        default="output/shuffle_fair_queuing",
                        help="Output directory")
    parser.add_argument("--visual", action="store_true",
                        help="Launch browser-based visual debugger")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip matplotlib visualization")
    args = parser.parse_args()

    cfg = SFQConfig(
        duration_s=args.duration,
        drain_s=args.drain,
        num_queues=args.queues,
        num_customers=args.customers,
        noisy_customer_id=args.noisy_id,
        normal_rate=args.normal_rate,
        noisy_normal_rate=args.noisy_normal_rate,
        noisy_burst_rate=args.noisy_burst_rate,
        burst_period_s=args.burst_period,
        burst_duration_s=args.burst_duration,
        perturb_interval_s=args.perturb_interval,
        best_of_n=args.best_of_n,
        poll_rate=args.poll_rate,
        max_queue_depth=args.max_queue_depth,
        seed=None if args.seed == -1 else args.seed,
    )

    print("SFQ simulation")
    print(f"  {cfg.num_queues} queues, best-of-{cfg.best_of_n} routing")
    print(f"  {cfg.num_customers} customers @ {cfg.normal_rate} req/s each")
    print(f"  Customer{cfg.noisy_customer_id} (noisy): "
          f"{cfg.noisy_normal_rate} / {cfg.noisy_burst_rate} req/s  "
          f"({cfg.burst_duration_s}s burst every {cfg.burst_period_s}s)")
    print(f"  Max queue depth: {cfg.max_queue_depth} (0=unlimited)")
    print(f"  Hash perturbation every {cfg.perturb_interval_s}s")
    print(f"  Poller: {cfg.poll_rate} polls/s round-robin")

    if args.visual:
        from happysimulator.visual import Chart, serve

        setup = build_sfq_simulation(cfg)
        print(f"\nLaunching visual debugger at http://127.0.0.1:8765 ...")
        serve(setup.sim)
    else:
        result = run_sfq_simulation(cfg)
        print_summary(result)

        if not args.no_viz:
            output_dir = Path(args.output)
            visualize_results(result, output_dir)
            print(f"\nVisualizations saved to: {output_dir.absolute()}")
