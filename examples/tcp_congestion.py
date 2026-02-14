"""TCP congestion control: AIMD vs Cubic vs BBR.

This example demonstrates how different TCP congestion control algorithms
behave under packet loss. The key insight: AIMD (Reno) is conservative
and recovers slowly; Cubic grows more aggressively; BBR maintains higher
throughput by estimating bottleneck bandwidth rather than reacting to loss.

## Architecture Diagram

```
    Source (constant rate)
        |
        v
    DataSender ──> TCPConnection (AIMD / Cubic / BBR)
        |
        v
      Sink
```

## Key Metrics

- Final congestion window size
- Retransmission count
- Total bytes sent
- Throughput in segments/second
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Data,
    Entity,
    Event,
    Instant,
    Probe,
    Simulation,
    SimulationSummary,
    Sink,
    Source,
)
from happysimulator.components.infrastructure import (
    TCPConnection,
    TCPStats,
    AIMD,
    Cubic,
    BBR,
)


# =============================================================================
# Custom Entity
# =============================================================================


class DataSender(Entity):
    """Sends data over a TCP connection on each incoming event."""

    def __init__(
        self,
        name: str,
        *,
        tcp: TCPConnection,
        downstream: Entity | None = None,
        send_size_bytes: int = 65536,
    ) -> None:
        super().__init__(name)
        self._tcp = tcp
        self._downstream = downstream
        self._send_size = send_size_bytes
        self._sends: int = 0

    @property
    def sends(self) -> int:
        return self._sends

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield from self._tcp.send(self._send_size)
        self._sends += 1

        if self._downstream:
            return [Event(
                time=self.now,
                event_type="Sent",
                target=self._downstream,
                context=event.context,
            )]
        return []


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class TCPResult:
    algorithm: str
    stats: TCPStats
    summary: SimulationSummary
    cwnd_data: Data


@dataclass
class SimulationResult:
    aimd: TCPResult
    cubic: TCPResult
    bbr: TCPResult
    duration_s: float


def _run_algorithm(
    algo_name: str,
    tcp: TCPConnection,
    *,
    duration_s: float,
    rate: float,
    seed: int | None,
) -> TCPResult:
    if seed is not None:
        random.seed(seed)

    sink = Sink()
    sender = DataSender(f"Sender_{algo_name}", tcp=tcp, downstream=sink)

    source = Source.constant(
        rate=rate,
        target=sender,
        event_type="Send",
        stop_after=Instant.from_seconds(duration_s),
    )

    cwnd_data = Data()
    cwnd_probe = Probe(
        target=tcp,
        metric="cwnd",
        data=cwnd_data,
        interval=0.1,
        start_time=Instant.Epoch,
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 1.0),
        sources=[source],
        entities=[tcp, sender, sink],
        probes=[cwnd_probe],
    )
    summary = sim.run()

    return TCPResult(
        algorithm=algo_name,
        stats=tcp.stats,
        summary=summary,
        cwnd_data=cwnd_data,
    )


def run_simulation(
    *,
    duration_s: float = 10.0,
    rate: float = 50.0,
    loss_rate: float = 0.01,
    seed: int | None = 42,
) -> SimulationResult:
    """Compare AIMD, Cubic, and BBR under packet loss."""
    aimd = _run_algorithm(
        "AIMD",
        TCPConnection("TCP_AIMD", congestion_control=AIMD(), loss_rate=loss_rate),
        duration_s=duration_s, rate=rate, seed=seed,
    )
    cubic = _run_algorithm(
        "Cubic",
        TCPConnection("TCP_Cubic", congestion_control=Cubic(), loss_rate=loss_rate),
        duration_s=duration_s, rate=rate, seed=seed,
    )
    bbr = _run_algorithm(
        "BBR",
        TCPConnection("TCP_BBR", congestion_control=BBR(), loss_rate=loss_rate),
        duration_s=duration_s, rate=rate, seed=seed,
    )

    return SimulationResult(
        aimd=aimd, cubic=cubic, bbr=bbr,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    print("\n" + "=" * 72)
    print("TCP CONGESTION CONTROL: AIMD vs Cubic vs BBR")
    print("=" * 72)

    results = [result.aimd, result.cubic, result.bbr]
    header = f"{'Metric':<30} " + " ".join(f"{r.algorithm:>15}" for r in results)
    print(f"\n{header}")
    print("-" * len(header))

    print(f"{'Segments sent':<30} " + " ".join(f"{r.stats.segments_sent:>15,}" for r in results))
    print(f"{'Segments ACKed':<30} " + " ".join(f"{r.stats.segments_acked:>15,}" for r in results))
    print(f"{'Retransmissions':<30} " + " ".join(f"{r.stats.retransmissions:>15,}" for r in results))
    print(f"{'Final cwnd':<30} " + " ".join(f"{r.stats.cwnd:>15.1f}" for r in results))
    print(f"{'Final ssthresh':<30} " + " ".join(f"{r.stats.ssthresh:>15.1f}" for r in results))
    print(f"{'RTT (ms)':<30} " + " ".join(f"{r.stats.rtt_s*1000:>15.2f}" for r in results))
    print(f"{'Throughput (seg/s)':<30} " + " ".join(f"{r.stats.throughput_segments_per_s:>15.1f}" for r in results))
    print(f"{'Total bytes sent':<30} " + " ".join(f"{r.stats.total_bytes_sent:>15,}" for r in results))

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  AIMD (Reno): Conservative — halves cwnd on loss, linear increase.")
    print("  Good fairness but slow recovery on high-bandwidth links.")
    print("\n  Cubic: Aggressive — cubic function growth after loss. Better")
    print("  utilization on high-bandwidth, high-latency networks.")
    print("\n  BBR: Model-based — estimates bottleneck bandwidth and RTT instead")
    print("  of reacting to loss. Maintains higher throughput but can be")
    print("  unfair to loss-based algorithms sharing the same link.")
    print("\n" + "=" * 72)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Chart 1: cwnd over time
    ax = axes[0]
    for r in [result.aimd, result.cubic, result.bbr]:
        times = r.cwnd_data.times()
        vals = r.cwnd_data.raw_values()
        ax.plot(times, vals, linewidth=1.5, label=r.algorithm, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Congestion Window (segments)")
    ax.set_title("Congestion Window Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Retransmissions and throughput
    ax = axes[1]
    algos = ["AIMD", "Cubic", "BBR"]
    retrans = [r.stats.retransmissions for r in [result.aimd, result.cubic, result.bbr]]
    throughputs = [r.stats.throughput_segments_per_s for r in [result.aimd, result.cubic, result.bbr]]

    x = range(len(algos))
    ax2 = ax.twinx()
    bars = ax.bar(list(x), retrans, 0.4, label="Retransmissions", color="#DD8452", alpha=0.8)
    line = ax2.plot(list(x), throughputs, "bo-", linewidth=2, label="Throughput")
    ax.set_xticks(list(x))
    ax.set_xticklabels(algos)
    ax.set_ylabel("Retransmissions")
    ax2.set_ylabel("Throughput (seg/s)")
    ax.set_title("Retransmissions vs Throughput")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("TCP Congestion Control Comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "tcp_congestion.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'tcp_congestion.png'}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TCP congestion control comparison")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--loss-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/tcp_congestion")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    print("Running TCP congestion control comparison...")
    result = run_simulation(
        duration_s=args.duration, loss_rate=args.loss_rate, seed=seed,
    )
    print_summary(result)

    if not args.no_viz:
        visualize_results(result, Path(args.output))
