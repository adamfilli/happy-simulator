"""Load-aware routing simulation demonstrating convergence time.

This example demonstrates how thick clients learn about server load and adaptively
reroute traffic to avoid saturated servers.

## Architecture

```
                                                         ┌─────────┐
  Source (0-1000)      ┌─────────┐   ┌─────────┐   hash  │ Server1 │
  Random Customers ───>│         │──>│ Client1 │────────>├─────────┤
                       │         │   └─────────┘         │ Server2 │
                       │ Router  │                       ├─────────┤
                       │(random) │   ┌─────────┐   hash  │ Server3 │
  Source (1001)        │         │──>│ Client2 │────────>└─────────┘
  High-rate        ───>│         │   └─────────┘
                       │         │
                       │         │   ┌─────────┐   hash
                       │         │──>│ Client3 │────────>(also to servers)
                       └─────────┘   └─────────┘
```

## Phases

1. **Basic routing**: Router distributes to clients randomly. Clients hash customerID
   to select server. Customer 1001 always goes to same server -> saturation.

2. **Smart routing**: Clients track server load from responses. If target server
   load > 0.9, client rehashes to different server. Load redistributes.

## Key Metrics

- Server queue depths over time
- Latency by customer type (random vs customer 1001)
- Convergence time: how long until smart routing kicks in
- Load distribution across servers

"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Data,
    Entity,
    Event,
    EventProvider,
    FIFOQueue,
    Instant,
    PoissonArrivalTimeProvider,
    Probe,
    QueuedResource,
    RandomRouter,
    Simulation,
    Source,
    SpikeProfile,
)


# =============================================================================
# Load Reporting Server
# =============================================================================


class LoadReportingServer(QueuedResource):
    """A queued server that reports its load in responses.

    Load is calculated as in_flight / concurrency. Responses include
    server_load and server_id for client-side load tracking.
    """

    def __init__(
        self,
        name: str,
        *,
        server_id: int,
        concurrency: int = 10,
        mean_service_time_s: float = 0.1,
    ):
        super().__init__(name, policy=FIFOQueue())
        self.server_id = server_id
        self.concurrency = concurrency
        self.mean_service_time_s = mean_service_time_s
        self._in_flight: int = 0

        # Stats
        self.stats_processed: int = 0
        self.completion_times: list[Instant] = []
        self.service_times_s: list[float] = []

    @property
    def load(self) -> float:
        """Current load as fraction of concurrency (0.0 to 1.0+)."""
        return self._in_flight / self.concurrency

    def has_capacity(self) -> bool:
        """Allow concurrent processing up to concurrency limit."""
        return self._in_flight < self.concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process request with exponential service time, then send response."""
        self._in_flight += 1

        # Exponential service time
        service_time = random.expovariate(1.0 / self.mean_service_time_s)
        yield service_time, None

        self._in_flight -= 1
        self.stats_processed += 1

        # Record stats
        self.completion_times.append(self.now)
        self.service_times_s.append(service_time)

        # Send response back to client
        client: Entity | None = event.context.get("client")
        if client is None:
            return []

        response = Event(
            time=self.now,
            event_type="Response",
            target=client,
            context={
                "server_id": self.server_id,
                "server_load": self.load,
                "customer_id": event.context.get("customer_id"),
                "created_at": event.context.get("created_at"),
                "service_time_s": service_time,
            },
        )
        return [response]


# =============================================================================
# Load-Aware Client
# =============================================================================


@dataclass
class LoadSample:
    """A timestamped load sample from a server."""
    time: Instant
    load: float


class LoadAwareClient(Entity):
    """Thick client that routes requests to servers using hash-based routing.

    Tracks server load from responses and can rehash to avoid overloaded servers
    when smart routing is enabled.
    """

    def __init__(
        self,
        name: str,
        *,
        client_id: int,
        servers: list[Entity],
        load_window_s: float = 5.0,
        load_threshold: float = 0.9,
        enable_smart_routing: bool = True,
    ):
        super().__init__(name)
        self.client_id = client_id
        self.servers = servers
        self.load_window_s = load_window_s
        self.load_threshold = load_threshold
        self.enable_smart_routing = enable_smart_routing

        # Recent load samples per server: server_id -> deque of LoadSample
        self._server_loads: dict[int, deque[LoadSample]] = defaultdict(deque)

        # Stats
        self.stats_requests: int = 0
        self.stats_responses: int = 0
        self.stats_rehashes: int = 0  # Times we picked non-default server

        # Latency tracking
        self.completion_times: list[Instant] = []
        self.latencies_s: list[float] = []
        self.customer_ids: list[int] = []
        self.server_choices: list[int] = []

        # Routing decisions over time for visualization
        self.routing_times: list[Instant] = []
        self.routing_decisions: list[tuple[int, int]]  = []  # (customer_id, server_id)

    def handle_event(self, event: Event) -> list[Event]:
        """Handle incoming request or server response."""
        event_type = event.event_type

        if event_type == "Request":
            return self._handle_request(event)
        elif event_type == "Response":
            return self._handle_response(event)

        return []

    def _handle_request(self, event: Event) -> list[Event]:
        """Route request to appropriate server."""
        self.stats_requests += 1

        customer_id = event.context.get("customer_id", 0)
        server_idx = self._select_server(customer_id)

        # Track routing decision
        self.routing_times.append(self.now)
        self.routing_decisions.append((customer_id, server_idx))

        # Forward to server
        server_request = Event(
            time=self.now,
            event_type="Request",
            target=self.servers[server_idx],
            context={
                "customer_id": customer_id,
                "created_at": event.context.get("created_at", self.now),
                "client": self,
            },
        )
        return [server_request]

    def _handle_response(self, event: Event) -> list[Event]:
        """Handle response from server and update load tracking."""
        self.stats_responses += 1

        # Update load tracking
        server_id = event.context.get("server_id", 0)
        server_load = event.context.get("server_load", 0.0)
        self._server_loads[server_id].append(LoadSample(self.now, server_load))

        # Record latency
        created_at: Instant = event.context.get("created_at", event.time)
        latency_s = (event.time - created_at).to_seconds()

        self.completion_times.append(event.time)
        self.latencies_s.append(latency_s)
        self.customer_ids.append(event.context.get("customer_id", 0))
        self.server_choices.append(server_id)

        return []

    def _select_server(self, customer_id: int) -> int:
        """Select server for request, optionally avoiding overloaded servers."""
        num_servers = len(self.servers)
        base_idx = hash(customer_id) % num_servers

        if not self.enable_smart_routing:
            return base_idx

        # Prune old load samples first
        self._prune_old_samples()

        # Try servers starting from base_idx
        for attempt in range(num_servers):
            idx = (base_idx + attempt) % num_servers
            recent_load = self._get_recent_load(idx)

            if recent_load < self.load_threshold:
                if attempt > 0:
                    self.stats_rehashes += 1
                return idx

        # All servers overloaded, use original
        return base_idx

    def _prune_old_samples(self) -> None:
        """Remove load samples older than the window."""
        cutoff = self.now - Instant.from_seconds(self.load_window_s)

        for server_id, samples in self._server_loads.items():
            while samples and samples[0].time < cutoff:
                samples.popleft()

    def _get_recent_load(self, server_id: int) -> float:
        """Get average recent load for a server."""
        samples = self._server_loads.get(server_id)
        if not samples:
            return 0.0  # No data = assume not loaded

        total = sum(s.load for s in samples)
        return total / len(samples)

    def latency_time_series_seconds(self) -> tuple[list[float], list[float]]:
        """Return (completion_times_s, latencies_s) for plotting."""
        return [t.to_seconds() for t in self.completion_times], list(self.latencies_s)


# =============================================================================
# Event Providers
# =============================================================================


class RandomCustomerProvider(EventProvider):
    """Generates requests with random customer IDs in [0, 1000)."""

    def __init__(
        self,
        target: Entity,
        *,
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1
        customer_id = random.randint(0, 999)

        return [
            Event(
                time=time,
                event_type="Request",
                target=self._target,
                context={
                    "customer_id": customer_id,
                    "created_at": time,
                    "source": "random",
                },
            )
        ]


class HighRateCustomerProvider(EventProvider):
    """Generates requests for a single high-volume customer (1001)."""

    def __init__(
        self,
        target: Entity,
        *,
        customer_id: int = 1001,
        stop_after: Instant | None = None,
    ):
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
                    "source": "high_rate",
                },
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the load-aware routing simulation."""
    servers: list[LoadReportingServer]
    clients: list[LoadAwareClient]
    router: RandomRouter
    server_depth_data: list[Data]
    server_load_data: list[Data]
    random_generated: int
    high_rate_generated: int
    enable_smart_routing: bool
    spike_profile: SpikeProfile


def run_load_aware_routing_simulation(
    *,
    duration_s: float = 60.0,
    drain_s: float = 5.0,
    num_servers: int = 3,
    num_clients: int = 3,
    server_concurrency: int = 10,
    mean_service_time_s: float = 0.1,
    random_customer_rate: float = 20.0,
    # Spike profile parameters for high-rate customer
    spike_baseline_rate: float = 10.0,
    spike_rate: float = 150.0,
    spike_warmup_s: float = 10.0,
    spike_duration_s: float = 15.0,
    load_window_s: float = 5.0,
    load_threshold: float = 0.9,
    enable_smart_routing: bool = True,
    probe_interval_s: float = 0.1,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the load-aware routing simulation.

    Args:
        duration_s: How long to generate load
        drain_s: Extra time for in-flight requests to complete
        num_servers: Number of backend servers
        num_clients: Number of thick clients
        server_concurrency: Max concurrent requests per server
        mean_service_time_s: Mean server processing time
        random_customer_rate: Requests/s for random customers (0-999)
        spike_baseline_rate: Customer 1001 baseline rate before/after spike
        spike_rate: Customer 1001 rate during spike (should saturate server)
        spike_warmup_s: Seconds before spike starts
        spike_duration_s: How long the spike lasts
        load_window_s: Sliding window for load averaging
        load_threshold: Load threshold to trigger rehash (0.9 = 90%)
        enable_smart_routing: Enable load-aware routing
        probe_interval_s: Probe sampling interval
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)

    # Create servers
    servers = [
        LoadReportingServer(
            name=f"Server{i}",
            server_id=i,
            concurrency=server_concurrency,
            mean_service_time_s=mean_service_time_s,
        )
        for i in range(num_servers)
    ]

    # Create clients (each knows all servers)
    clients = [
        LoadAwareClient(
            name=f"Client{i}",
            client_id=i,
            servers=servers,
            load_window_s=load_window_s,
            load_threshold=load_threshold,
            enable_smart_routing=enable_smart_routing,
        )
        for i in range(num_clients)
    ]

    # Create router
    router = RandomRouter(name="Router", targets=clients)

    # Create probes for server metrics
    server_depth_data = [Data() for _ in range(num_servers)]
    server_load_data = [Data() for _ in range(num_servers)]

    probes = []
    for i, server in enumerate(servers):
        probes.append(
            Probe(
                target=server,
                metric="depth",
                data=server_depth_data[i],
                interval=probe_interval_s,
                start_time=Instant.Epoch,
            )
        )
        probes.append(
            Probe(
                target=server,
                metric="load",
                data=server_load_data[i],
                interval=probe_interval_s,
                start_time=Instant.Epoch,
            )
        )

    # Create sources
    stop_after = Instant.from_seconds(duration_s)

    # Random customers: constant rate
    random_provider = RandomCustomerProvider(router, stop_after=stop_after)
    random_profile = ConstantRateProfile(rate=random_customer_rate)
    random_arrival = ConstantArrivalTimeProvider(random_profile, start_time=Instant.Epoch)
    random_source = Source(
        name="RandomSource",
        event_provider=random_provider,
        arrival_time_provider=random_arrival,
    )

    # Customer 1001: spike profile (baseline -> spike -> recovery)
    high_rate_provider = HighRateCustomerProvider(router, stop_after=stop_after)
    spike_profile = SpikeProfile(
        baseline_rate=spike_baseline_rate,
        spike_rate=spike_rate,
        warmup_s=spike_warmup_s,
        spike_duration_s=spike_duration_s,
    )
    high_rate_arrival = PoissonArrivalTimeProvider(spike_profile, start_time=Instant.Epoch)
    high_rate_source = Source(
        name="HighRateSource",
        event_provider=high_rate_provider,
        arrival_time_provider=high_rate_arrival,
    )

    # Collect all entities
    entities: list[Entity] = [router] + clients + servers

    # Run simulation
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + drain_s),
        sources=[random_source, high_rate_source],
        entities=entities,
        probes=probes,
    )
    sim.run()

    return SimulationResult(
        servers=servers,
        clients=clients,
        router=router,
        server_depth_data=server_depth_data,
        server_load_data=server_load_data,
        random_generated=random_provider.generated,
        high_rate_generated=high_rate_provider.generated,
        enable_smart_routing=enable_smart_routing,
        spike_profile=spike_profile,
    )


# =============================================================================
# Output and Visualization
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print summary statistics."""
    print("\n" + "=" * 70)
    print("LOAD-AWARE ROUTING SIMULATION RESULTS")
    print("=" * 70)

    profile = result.spike_profile

    print(f"\nConfiguration:")
    print(f"  Smart routing: {'ENABLED' if result.enable_smart_routing else 'DISABLED'}")
    print(f"  Servers: {len(result.servers)}")
    print(f"  Clients: {len(result.clients)}")

    print(f"\nSpike Profile (Customer 1001):")
    print(f"  [0-{profile.warmup_s}s]: {profile.baseline_rate} req/s (warmup)")
    print(f"  [{profile.warmup_s}-{profile.warmup_s + profile.spike_duration_s}s]: {profile.spike_rate} req/s (SPIKE)")
    print(f"  [{profile.warmup_s + profile.spike_duration_s}s+]: {profile.baseline_rate} req/s (recovery)")

    print(f"\nLoad Generation:")
    print(f"  Random customers (0-999): {result.random_generated} requests")
    print(f"  High-rate customer (1001): {result.high_rate_generated} requests")

    print(f"\nRouter Distribution:")
    for i, count in result.router.target_counts.items():
        pct = count / result.router.stats_routed * 100 if result.router.stats_routed else 0
        print(f"  Client{i}: {count} ({pct:.1f}%)")

    print(f"\nServer Processing:")
    for server in result.servers:
        print(f"  {server.name}: {server.stats_processed} requests")

    print(f"\nClient Stats:")
    total_rehashes = 0
    for client in result.clients:
        rehash_pct = client.stats_rehashes / client.stats_requests * 100 if client.stats_requests else 0
        total_rehashes += client.stats_rehashes
        print(f"  {client.name}: {client.stats_requests} requests, {client.stats_rehashes} rehashes ({rehash_pct:.1f}%)")

    # Latency analysis
    print(f"\nLatency Analysis:")

    # Aggregate latencies from all clients
    all_latencies: list[tuple[float, int]] = []  # (latency_s, customer_id)
    for client in result.clients:
        for lat, cid in zip(client.latencies_s, client.customer_ids):
            all_latencies.append((lat, cid))

    random_latencies = [lat for lat, cid in all_latencies if cid != 1001]
    high_rate_latencies = [lat for lat, cid in all_latencies if cid == 1001]

    if random_latencies:
        random_sorted = sorted(random_latencies)
        avg = sum(random_sorted) / len(random_sorted)
        p50 = random_sorted[len(random_sorted) // 2]
        p99 = random_sorted[int(len(random_sorted) * 0.99)]
        print(f"  Random customers (0-999):")
        print(f"    Count: {len(random_sorted)}")
        print(f"    Avg: {avg * 1000:.1f}ms, p50: {p50 * 1000:.1f}ms, p99: {p99 * 1000:.1f}ms")

    if high_rate_latencies:
        hr_sorted = sorted(high_rate_latencies)
        avg = sum(hr_sorted) / len(hr_sorted)
        p50 = hr_sorted[len(hr_sorted) // 2]
        p99 = hr_sorted[int(len(hr_sorted) * 0.99)]
        print(f"  High-rate customer (1001):")
        print(f"    Count: {len(hr_sorted)}")
        print(f"    Avg: {avg * 1000:.1f}ms, p50: {p50 * 1000:.1f}ms, p99: {p99 * 1000:.1f}ms")

    # Server load analysis
    print(f"\nServer Load (average):")
    for i, server in enumerate(result.servers):
        load_samples = result.server_load_data[i].values
        if load_samples:
            avg_load = sum(v for _, v in load_samples) / len(load_samples)
            max_load = max(v for _, v in load_samples)
            print(f"  {server.name}: avg={avg_load:.2f}, max={max_load:.2f}")

    print("\n" + "=" * 70)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate visualizations."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    num_servers = len(result.servers)
    profile = result.spike_profile
    spike_start = profile.warmup_s
    spike_end = profile.warmup_s + profile.spike_duration_s

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Helper to add spike phase shading
    def add_spike_shading(ax, ymax=None):
        if ymax is None:
            ymax = ax.get_ylim()[1]
        ax.axvspan(spike_start, spike_end, alpha=0.15, color="red", label="Spike phase")
        ax.axvline(spike_start, color="red", linestyle=":", alpha=0.5)
        ax.axvline(spike_end, color="red", linestyle=":", alpha=0.5)

    # 1. Server queue depths over time
    ax = axes[0, 0]
    for i in range(num_servers):
        times = [t for t, _ in result.server_depth_data[i].values]
        depths = [v for _, v in result.server_depth_data[i].values]
        ax.plot(times, depths, label=f"Server{i}", alpha=0.8)
    add_spike_shading(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Queue Depth")
    ax.set_title("Server Queue Depths Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Server load over time
    ax = axes[0, 1]
    for i in range(num_servers):
        times = [t for t, _ in result.server_load_data[i].values]
        loads = [v for _, v in result.server_load_data[i].values]
        ax.plot(times, loads, label=f"Server{i}", alpha=0.8)
    ax.axhline(y=0.9, color="orange", linestyle="--", alpha=0.7, label="Threshold (0.9)")
    add_spike_shading(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Load (in_flight / concurrency)")
    ax.set_title("Server Load Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Latency over time (binned by second)
    ax = axes[1, 0]

    # Aggregate from all clients
    all_times_s: list[float] = []
    all_latencies_s: list[float] = []
    all_customer_ids: list[int] = []

    for client in result.clients:
        for t, lat, cid in zip(client.completion_times, client.latencies_s, client.customer_ids):
            all_times_s.append(t.to_seconds())
            all_latencies_s.append(lat)
            all_customer_ids.append(cid)

    # Bucket by second
    random_buckets: dict[int, list[float]] = defaultdict(list)
    hr_buckets: dict[int, list[float]] = defaultdict(list)

    for t, lat, cid in zip(all_times_s, all_latencies_s, all_customer_ids):
        bucket = int(t)
        if cid == 1001:
            hr_buckets[bucket].append(lat)
        else:
            random_buckets[bucket].append(lat)

    # Plot average latency per bucket
    random_bucket_times = sorted(random_buckets.keys())
    random_bucket_avg = [sum(random_buckets[b]) / len(random_buckets[b]) * 1000
                         for b in random_bucket_times]

    hr_bucket_times = sorted(hr_buckets.keys())
    hr_bucket_avg = [sum(hr_buckets[b]) / len(hr_buckets[b]) * 1000
                     for b in hr_bucket_times]

    ax.plot(random_bucket_times, random_bucket_avg, "b-", label="Random (0-999)", alpha=0.8)
    ax.plot(hr_bucket_times, hr_bucket_avg, "r-", label="Customer 1001", alpha=0.8)
    add_spike_shading(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Avg Latency (ms)")
    ax.set_title("Latency Over Time by Customer Type")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Latency distribution comparison
    ax = axes[1, 1]

    random_latencies = [lat * 1000 for lat, cid in zip(all_latencies_s, all_customer_ids) if cid != 1001]
    hr_latencies = [lat * 1000 for lat, cid in zip(all_latencies_s, all_customer_ids) if cid == 1001]

    if random_latencies:
        ax.hist(random_latencies, bins=50, alpha=0.5, label="Random (0-999)", color="blue")
    if hr_latencies:
        ax.hist(hr_latencies, bins=50, alpha=0.5, label="Customer 1001", color="red")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Latency Distribution by Customer Type")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Server selection over time for customer 1001
    ax = axes[2, 0]

    # Track which server customer 1001 was routed to over time
    hr_routing_times: list[float] = []
    hr_server_choices: list[int] = []

    for client in result.clients:
        for t, (cid, server_id) in zip(client.routing_times, client.routing_decisions):
            if cid == 1001:
                hr_routing_times.append(t.to_seconds())
                hr_server_choices.append(server_id)

    if hr_routing_times:
        # Bucket by second and count per server
        server_buckets: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for t, sid in zip(hr_routing_times, hr_server_choices):
            bucket = int(t)
            server_buckets[bucket][sid] += 1

        bucket_times = sorted(server_buckets.keys())
        for sid in range(num_servers):
            counts = [server_buckets[b][sid] for b in bucket_times]
            ax.plot(bucket_times, counts, label=f"Server{sid}", alpha=0.8)

    add_spike_shading(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests to Server (per second)")
    ax.set_title("Customer 1001 Server Selection Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Rehash rate over time
    ax = axes[2, 1]

    rehash_by_time: dict[int, int] = defaultdict(int)
    request_by_time: dict[int, int] = defaultdict(int)

    for client in result.clients:
        # Approximate: count routing decisions over time
        for t in client.routing_times:
            bucket = int(t.to_seconds())
            request_by_time[bucket] += 1

    # Get rehash events (when server chosen != hash-based default)
    for client in result.clients:
        for t, (cid, chosen_server) in zip(client.routing_times, client.routing_decisions):
            bucket = int(t.to_seconds())
            expected_server = hash(cid) % num_servers
            if chosen_server != expected_server:
                rehash_by_time[bucket] += 1

    bucket_times = sorted(request_by_time.keys())
    rehash_rates = [
        rehash_by_time[b] / request_by_time[b] * 100 if request_by_time[b] else 0
        for b in bucket_times
    ]

    ax.plot(bucket_times, rehash_rates, "g-", alpha=0.8)
    add_spike_shading(ax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rehash Rate (%)")
    ax.set_title("Rehash Rate Over Time (Smart Routing Activity)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "load_aware_routing.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'load_aware_routing.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load-aware routing simulation")
    parser.add_argument("--duration", type=float, default=60.0, help="Load duration (s)")
    parser.add_argument("--drain", type=float, default=5.0, help="Drain time (s)")
    parser.add_argument("--servers", type=int, default=3, help="Number of servers")
    parser.add_argument("--clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--concurrency", type=int, default=10, help="Server concurrency")
    parser.add_argument("--service-time", type=float, default=0.1, help="Mean service time (s)")
    parser.add_argument("--random-rate", type=float, default=20.0, help="Random customer rate (req/s)")
    parser.add_argument("--spike-baseline", type=float, default=10.0, help="Customer 1001 baseline rate (req/s)")
    parser.add_argument("--spike-rate", type=float, default=150.0, help="Customer 1001 spike rate (req/s)")
    parser.add_argument("--spike-warmup", type=float, default=10.0, help="Seconds before spike starts")
    parser.add_argument("--spike-duration", type=float, default=15.0, help="Spike duration (s)")
    parser.add_argument("--load-window", type=float, default=5.0, help="Load tracking window (s)")
    parser.add_argument("--load-threshold", type=float, default=0.9, help="Load threshold for rehash")
    parser.add_argument("--no-smart-routing", action="store_true", help="Disable smart routing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/load_aware_routing", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    enable_smart_routing = not args.no_smart_routing

    print("Running load-aware routing simulation...")
    print(f"  Duration: {args.duration}s + {args.drain}s drain")
    print(f"  Servers: {args.servers}, Clients: {args.clients}")
    print(f"  Random customers: {args.random_rate} req/s (constant)")
    print(f"  Customer 1001: {args.spike_baseline} -> {args.spike_rate} -> {args.spike_baseline} req/s")
    print(f"    Spike at t={args.spike_warmup}s for {args.spike_duration}s")
    print(f"  Smart routing: {'ENABLED' if enable_smart_routing else 'DISABLED'}")

    result = run_load_aware_routing_simulation(
        duration_s=args.duration,
        drain_s=args.drain,
        num_servers=args.servers,
        num_clients=args.clients,
        server_concurrency=args.concurrency,
        mean_service_time_s=args.service_time,
        random_customer_rate=args.random_rate,
        spike_baseline_rate=args.spike_baseline,
        spike_rate=args.spike_rate,
        spike_warmup_s=args.spike_warmup,
        spike_duration_s=args.spike_duration,
        load_window_s=args.load_window,
        load_threshold=args.load_threshold,
        enable_smart_routing=enable_smart_routing,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
