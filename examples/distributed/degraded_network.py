"""Degraded network with fault injection.

Demonstrates how scheduled network faults (latency spikes, packet loss)
affect end-to-end client-server communication. Uses the fault injection
framework to declaratively schedule faults across five phases:

1. **Normal** (0-20s): Baseline datacenter network (~1ms RTT)
2. **Latency Spike** (20-40s): +50ms injected on both directions
3. **Packet Loss** (40-60s): 15% loss on both directions
4. **Combined** (60-80s): +100ms latency AND 10% loss
5. **Recovery** (80-100s): All faults removed, return to baseline

## Architecture

```
   Source ──► Client ──► Network ──► Server
  (50 req/s)               │  (faulted)   (Exp ~5ms)
                            │
              Client ◄──── Network ◄──── Server
             (records       (faulted)    (response)
              latency)
```

## Key Observations

- Latency injection is visible as a step change in round-trip time
- Packet loss doesn't affect latency of successful requests but drops goodput
- Combined faults show both effects simultaneously
- Recovery to baseline is nearly instant once faults are removed
- Bidirectional loss compounds: 15% per-hop → ~28% end-to-end loss
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    FaultSchedule,
    Instant,
    Network,
    Probe,
    Simulation,
    Source,
    datacenter_network,
)
from happysimulator.faults import InjectLatency, InjectPacketLoss


# =============================================================================
# Configuration
# =============================================================================

PHASES = [
    (0, 20, "Normal"),
    (20, 40, "+50ms Latency"),
    (40, 60, "15% Pkt Loss"),
    (60, 80, "+100ms & 10% Loss"),
    (80, 100, "Recovery"),
]

PHASE_COLORS = ["#c8e6c9", "#ffe0b2", "#f8bbd0", "#e1bee7", "#c8e6c9"]


# =============================================================================
# Entities
# =============================================================================


class NetworkClient(Entity):
    """Client that sends requests through a network and tracks latency.

    Receives "NewRequest" events from a Source, routes them through the
    Network to the server, and records round-trip latency when "Response"
    events arrive back through the network.
    """

    def __init__(self, name: str, *, network: Network):
        super().__init__(name)
        self.network = network
        self.server: Entity | None = None  # Set after construction
        self._pending: dict[int, Instant] = {}
        self._next_id = 0

        # Time series
        self.sent_times: list[float] = []
        self.received_times: list[float] = []
        self.latencies: list[tuple[float, float]] = []  # (time_s, latency_s)

    @property
    def pending_count(self) -> int:
        """Number of requests awaiting a response."""
        return len(self._pending)

    def handle_event(self, event: Event) -> list[Event]:
        if event.event_type == "NewRequest":
            return self._send_request(event)
        if event.event_type == "Response":
            return self._record_response(event)
        return []

    def _send_request(self, event: Event) -> list[Event]:
        self._next_id += 1
        rid = self._next_id
        self._pending[rid] = self.now
        self.sent_times.append(self.now.to_seconds())
        return [self.network.send(
            self, self.server, "Request",
            payload={"request_id": rid},
        )]

    def _record_response(self, event: Event) -> list[Event]:
        rid = event.context.get("metadata", {}).get("request_id")
        if rid is not None and rid in self._pending:
            send_time = self._pending.pop(rid)
            latency_s = (self.now - send_time).to_seconds()
            self.received_times.append(self.now.to_seconds())
            self.latencies.append((self.now.to_seconds(), latency_s))
        return []

    # -- Derived series for plotting --

    def throughput_series(
        self, bucket_s: float = 1.0,
    ) -> tuple[list[float], list[int], list[int]]:
        """(times, sent_per_s, received_per_s) bucketed time series."""
        sent: dict[int, int] = defaultdict(int)
        recv: dict[int, int] = defaultdict(int)
        for t in self.sent_times:
            sent[int(t / bucket_s)] += 1
        for t in self.received_times:
            recv[int(t / bucket_s)] += 1
        keys = sorted(set(sent) | set(recv))
        return (
            [k * bucket_s for k in keys],
            [sent.get(k, 0) for k in keys],
            [recv.get(k, 0) for k in keys],
        )

    def latency_series(self) -> tuple[list[float], list[float]]:
        """(time_s, latency_ms) for every successful response."""
        return (
            [t for t, _ in self.latencies],
            [lat * 1000 for _, lat in self.latencies],
        )

    def success_rate_series(
        self, bucket_s: float = 1.0,
    ) -> tuple[list[float], list[float]]:
        """(times, pct_success) — responses received / requests sent per bucket."""
        sent: dict[int, int] = defaultdict(int)
        recv: dict[int, int] = defaultdict(int)
        for t in self.sent_times:
            sent[int(t / bucket_s)] += 1
        for t in self.received_times:
            recv[int(t / bucket_s)] += 1
        keys = sorted(sent)
        return (
            [k * bucket_s for k in keys],
            [recv.get(k, 0) / sent[k] * 100 if sent[k] else 0 for k in keys],
        )


class NetworkServer(Entity):
    """Server that processes requests and sends responses through the network."""

    def __init__(
        self, name: str, *, network: Network, mean_service_time_s: float = 0.005,
    ):
        super().__init__(name)
        self.network = network
        self.client: Entity | None = None  # Set after construction
        self.mean_service_time_s = mean_service_time_s
        self.processed = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self.processed += 1
        yield random.expovariate(1.0 / self.mean_service_time_s)
        return [self.network.send(
            self, self.client, "Response",
            payload={"request_id": event.context.get("metadata", {}).get("request_id")},
        )]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Collected results from the degraded network simulation."""

    client: NetworkClient
    server: NetworkServer
    pending_data: Data
    fault_stats: object


def run_simulation(
    *,
    duration_s: float = 100.0,
    arrival_rate: float = 50.0,
    mean_service_time_s: float = 0.005,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the degraded network simulation.

    Args:
        duration_s: Total simulation time (seconds).
        arrival_rate: Client request rate (req/s).
        mean_service_time_s: Mean server processing time (seconds).
        seed: Random seed for reproducibility (-1 for random).
    """
    if seed is not None:
        random.seed(seed)

    # -- Topology --
    network = Network(name="net")
    client = NetworkClient("Client", network=network)
    server = NetworkServer(
        "Server", network=network, mean_service_time_s=mean_service_time_s,
    )
    client.server = server
    server.client = client
    network.add_bidirectional_link(client, server, datacenter_network("dc_link"))

    # -- Fault schedule --
    schedule = FaultSchedule()

    # Phase 2 (20-40s): latency spike, both directions
    schedule.add(InjectLatency("Client", "Server", extra_ms=50, start=20.0, end=40.0))
    schedule.add(InjectLatency("Server", "Client", extra_ms=50, start=20.0, end=40.0))

    # Phase 3 (40-60s): packet loss, both directions
    schedule.add(InjectPacketLoss("Client", "Server", loss_rate=0.15, start=40.0, end=60.0))
    schedule.add(InjectPacketLoss("Server", "Client", loss_rate=0.15, start=40.0, end=60.0))

    # Phase 4 (60-80s): latency + loss, both directions
    schedule.add(InjectLatency("Client", "Server", extra_ms=100, start=60.0, end=80.0))
    schedule.add(InjectLatency("Server", "Client", extra_ms=100, start=60.0, end=80.0))
    schedule.add(InjectPacketLoss("Client", "Server", loss_rate=0.10, start=60.0, end=80.0))
    schedule.add(InjectPacketLoss("Server", "Client", loss_rate=0.10, start=60.0, end=80.0))

    # -- Probe: track in-flight requests --

    probe, pending_data = Probe.on(client, "pending_count", interval=0.5)

    # -- Source --
    source = Source.constant(
        rate=arrival_rate,
        target=client,
        event_type="NewRequest",
        stop_after=duration_s,
    )

    # -- Run --
    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 5.0,
        sources=[source],
        entities=[client, server, network],
        probes=[probe],
        fault_schedule=schedule,
    )
    sim.run()

    return SimulationResult(
        client=client,
        server=server,
        pending_data=pending_data,
        fault_stats=schedule.stats,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    """Print per-phase statistics."""
    client = result.client

    print("\n" + "=" * 70)
    print("DEGRADED NETWORK SIMULATION RESULTS")
    print("=" * 70)

    total_sent = len(client.sent_times)
    total_recv = len(client.received_times)
    total_lost = total_sent - total_recv
    overall_loss = total_lost / total_sent * 100 if total_sent else 0

    print(f"\nOverall:")
    print(f"  Requests sent:      {total_sent}")
    print(f"  Responses received: {total_recv}")
    print(f"  Requests lost:      {total_lost} ({overall_loss:.1f}%)")
    print(f"  Server processed:   {result.server.processed}")

    if client.latencies:
        all_lats = sorted(lat for _, lat in client.latencies)
        avg = sum(all_lats) / len(all_lats)
        p50 = all_lats[len(all_lats) // 2]
        p99 = all_lats[int(len(all_lats) * 0.99)]
        print(f"\nOverall Latency:")
        print(f"  Average: {avg * 1000:.1f}ms")
        print(f"  p50:     {p50 * 1000:.1f}ms")
        print(f"  p99:     {p99 * 1000:.1f}ms")

    print(f"\nPer-Phase Breakdown:")
    header = f"  {'Phase':<22s} {'Sent':>6s} {'Recv':>6s} {'Loss%':>7s} {'Avg(ms)':>9s} {'p99(ms)':>9s}"
    print(header)
    print(f"  {'-' * 60}")

    for start, end, label in PHASES:
        sent = sum(1 for t in client.sent_times if start <= t < end)
        recv = sum(1 for t in client.received_times if start <= t < end)
        loss = (1 - recv / sent) * 100 if sent else 0
        phase_lats = sorted(
            lat * 1000 for t, lat in client.latencies if start <= t < end
        )
        avg_ms = sum(phase_lats) / len(phase_lats) if phase_lats else 0
        p99_ms = phase_lats[int(len(phase_lats) * 0.99)] if len(phase_lats) > 1 else avg_ms
        print(f"  {label:<22s} {sent:>6d} {recv:>6d} {loss:>6.1f}% {avg_ms:>8.1f} {p99_ms:>8.1f}")

    print(f"\nFault Stats: {result.fault_stats}")
    print("=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def _shade_phases(ax) -> None:
    """Add semi-transparent phase shading to an axes."""
    for (start, end, _label), color in zip(PHASES, PHASE_COLORS):
        ax.axvspan(start, end, alpha=0.3, color=color)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate a 3x2 dashboard of simulation results."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    client = result.client
    lat_times, lat_ms = client.latency_series()

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # -- 1. Latency scatter + 1s rolling average --
    ax = axes[0, 0]
    _shade_phases(ax)
    if lat_times:
        ax.scatter(lat_times, lat_ms, s=2, alpha=0.2, color="steelblue", rasterized=True)
        buckets: dict[int, list[float]] = defaultdict(list)
        for t, l in zip(lat_times, lat_ms):
            buckets[int(t)].append(l)
        avg_t = sorted(buckets)
        avg_l = [sum(buckets[k]) / len(buckets[k]) for k in avg_t]
        ax.plot(avg_t, avg_l, color="darkblue", linewidth=1.5, label="1s avg")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("End-to-End Latency")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.2)

    # -- 2. Throughput: sent vs received --
    ax = axes[0, 1]
    _shade_phases(ax)
    times, sent, recv = client.throughput_series()
    ax.step(times, sent, where="mid", color="coral", linewidth=1, label="Sent")
    ax.step(times, recv, where="mid", color="seagreen", linewidth=1, label="Received")
    ax.fill_between(times, recv, sent, step="mid", alpha=0.12, color="red", label="Lost")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Requests / second")
    ax.set_title("Throughput (Sent vs Received)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # -- 3. Success rate --
    ax = axes[1, 0]
    _shade_phases(ax)
    sr_times, sr_rates = client.success_rate_series()
    ax.step(sr_times, sr_rates, where="mid", color="purple", linewidth=1)
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="100%")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Request Success Rate")
    ax.set_ylim(-5, 115)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # -- 4. In-flight (pending) requests --
    ax = axes[1, 1]
    _shade_phases(ax)
    p_times = [t for t, _ in result.pending_data.values]
    p_vals = [v for _, v in result.pending_data.values]
    ax.plot(p_times, p_vals, color="darkorange", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pending Requests")
    ax.set_title("In-Flight Requests (Client Perspective)")
    ax.grid(True, alpha=0.2)

    # -- 5. Latency box plots by phase --
    ax = axes[2, 0]
    phase_lats = [[] for _ in PHASES]
    for t, l in zip(lat_times, lat_ms):
        for i, (start, end, _) in enumerate(PHASES):
            if start <= t < end:
                phase_lats[i].append(l)
                break
    valid = [(i, pl) for i, pl in enumerate(phase_lats) if pl]
    if valid:
        bp = ax.boxplot(
            [pl for _, pl in valid],
            tick_labels=[PHASES[i][2] for i, _ in valid],
            patch_artist=True,
            medianprops=dict(color="black"),
            showfliers=False,
        )
        for patch, (i, _) in zip(bp["boxes"], valid):
            patch.set_facecolor(PHASE_COLORS[i])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution by Phase")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(True, alpha=0.2, axis="y")

    # -- 6. Phase summary table --
    ax = axes[2, 1]
    ax.axis("off")
    headers = ["Phase", "Sent", "Recv", "Loss %", "Avg (ms)", "p99 (ms)"]
    rows = []
    for i, (start, end, label) in enumerate(PHASES):
        s = sum(1 for t in client.sent_times if start <= t < end)
        r = sum(1 for t in client.received_times if start <= t < end)
        loss = (1 - r / s) * 100 if s else 0
        pl = phase_lats[i]
        avg = sum(pl) / len(pl) if pl else 0
        p99 = sorted(pl)[int(len(pl) * 0.99)] if len(pl) > 1 else avg
        rows.append([label, str(s), str(r), f"{loss:.1f}", f"{avg:.1f}", f"{p99:.1f}"])

    table = ax.table(
        cellText=rows, colLabels=headers, cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for i in range(len(PHASES)):
        table[i + 1, 0].set_facecolor(PHASE_COLORS[i])
        table[i + 1, 0].set_alpha(0.5)
    ax.set_title("Phase Summary", pad=20)

    fig.suptitle(
        "Network Degradation Simulation \u2014 Fault Injection Impact",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "degraded_network.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Degraded network simulation with fault injection",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Sim duration (s)")
    parser.add_argument("--rate", type=float, default=50.0, help="Request rate (req/s)")
    parser.add_argument("--service-time", type=float, default=0.005, help="Mean service (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 = random)")
    parser.add_argument("--output", type=str, default="output/degraded_network")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running degraded network simulation...")
    print(f"  Duration: {args.duration}s | Rate: {args.rate} req/s")
    print(f"  Phases:")
    for start, end, label in PHASES:
        print(f"    [{start:>3.0f}s - {end:>3.0f}s] {label}")

    result = run_simulation(
        duration_s=args.duration,
        arrival_rate=args.rate,
        mean_service_time_s=args.service_time,
        seed=seed,
    )

    print_summary(result)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(result, output_dir)
        print(f"\nVisualizations saved to: {output_dir.absolute()}")
