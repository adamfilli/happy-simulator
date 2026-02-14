"""Chain replication: head-to-tail writes, tail reads, CRAQ variant.

Demonstrates write propagation through a chain and compares standard
chain replication (tail reads only) with CRAQ (any-node reads for
committed keys).

## Architecture

```
  Writer ──► HEAD ──► MID-1 ──► MID-2 ──► TAIL
             (write)  (propagate)          (ack → HEAD)
                                            ▲
  Reader ──────────────────────────────────┘ (reads)
```

## Key Observations

- Write latency scales linearly with chain length (more hops to TAIL ack).
- Tail reads are strongly consistent — always see committed data.
- CRAQ allows reading from any node for committed keys, improving read
  throughput without sacrificing consistency for clean keys.
- Dirty keys (in-flight writes) are forwarded to TAIL in CRAQ mode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from happysimulator import (
    Entity,
    Event,
    Instant,
    Network,
    Simulation,
    SimFuture,
    Source,
    datacenter_network,
)
from happysimulator.components.datastore import KVStore
from happysimulator.components.replication.chain_replication import (
    ChainNode,
    ChainNodeRole,
    build_chain,
)


# =============================================================================
# Client entities
# =============================================================================


class ChainWriter(Entity):
    """Sends keyed writes to the chain HEAD and records latency."""

    def __init__(self, name: str, head: ChainNode):
        super().__init__(name)
        self.head = head
        self._count = 0
        self.latencies: list[tuple[float, float]] = []

    def handle_event(self, event: Event) -> Generator[float | SimFuture | tuple[float, list[Event]], None, None]:
        self._count += 1
        key = f"key-{self._count % 50}"

        reply = SimFuture()
        write = Event(
            time=self.now, event_type="Write", target=self.head,
            context={"metadata": {"key": key, "value": self._count, "reply_future": reply}},
        )
        start = self.now
        yield 0.0, [write]
        yield reply
        self.latencies.append((self.now.to_seconds(), (self.now - start).to_seconds()))


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class ChainResult:
    """Results from a chain replication run."""

    chain_length: int
    craq: bool
    nodes: list[ChainNode]
    writer: ChainWriter


def run_chain(
    chain_length: int = 3,
    *,
    craq: bool = False,
    duration_s: float = 20.0,
    write_rate: float = 50.0,
    seed: int = 42,
) -> ChainResult:
    """Run a chain replication simulation."""
    random.seed(seed)

    network = Network(name="net")
    names = [f"node-{i}" for i in range(chain_length)]

    nodes = build_chain(
        names, network,
        store_factory=lambda n: KVStore(n, write_latency=0.001, read_latency=0.001),
        craq_enabled=craq,
    )

    # Wire network links between adjacent nodes + head↔tail for acks
    for i in range(len(nodes) - 1):
        network.add_bidirectional_link(
            nodes[i], nodes[i + 1],
            datacenter_network(f"link-{i}-{i+1}"),
        )
    # Head ↔ tail direct link for ack path
    if len(nodes) > 2:
        network.add_bidirectional_link(
            nodes[0], nodes[-1],
            datacenter_network("link-head-tail"),
        )

    writer = ChainWriter("writer", head=nodes[0])

    source = Source.constant(
        rate=write_rate, target=writer,
        event_type="NewWrite", stop_after=duration_s,
    )

    all_entities: list = [writer, network, *nodes]
    for n in nodes:
        all_entities.append(n.store)

    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration_s + 5.0),
        sources=[source],
        entities=all_entities,
    )
    sim.run()

    return ChainResult(
        chain_length=chain_length, craq=craq,
        nodes=nodes, writer=writer,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(results: list[ChainResult]) -> None:
    """Print comparison of chain lengths and CRAQ."""
    print("\n" + "=" * 70)
    print("CHAIN REPLICATION — LATENCY VS CHAIN LENGTH")
    print("=" * 70)

    header = f"  {'Length':>6s} {'CRAQ':>5s} {'Writes':>7s} {'AvgLat(ms)':>11s} {'p99Lat(ms)':>11s} {'TailKeys':>9s}"
    print(header)
    print(f"  {'-' * 55}")

    for r in results:
        lats = sorted(l for _, l in r.writer.latencies) if r.writer.latencies else [0]
        avg_ms = sum(lats) / len(lats) * 1000
        p99_ms = lats[int(len(lats) * 0.99)] * 1000 if len(lats) > 1 else avg_ms
        tail_keys = r.nodes[-1].store.size

        print(
            f"  {r.chain_length:>6d} "
            f"{'Yes' if r.craq else 'No':>5s} "
            f"{len(r.writer.latencies):>7d} "
            f"{avg_ms:>10.2f} "
            f"{p99_ms:>10.2f} "
            f"{tail_keys:>9d}"
        )

    print("=" * 70)


# =============================================================================
# Visualization
# =============================================================================


def visualize_results(results: list[ChainResult], output_dir: Path) -> None:
    """Generate latency comparison chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["steelblue", "coral", "seagreen", "purple"]

    for i, r in enumerate(results):
        if r.writer.latencies:
            times = [t for t, _ in r.writer.latencies]
            lats = [l * 1000 for _, l in r.writer.latencies]
            label = f"len={r.chain_length}"
            if r.craq:
                label += " (CRAQ)"
            ax.scatter(times, lats, s=2, alpha=0.3, color=colors[i % len(colors)], label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Write Latency (ms)")
    ax.set_title("Chain Replication — Write Latency by Chain Length")
    ax.legend(fontsize=8, markerscale=5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    path = output_dir / "chain_replication.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chain replication demo")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="output/chain_replication")
    parser.add_argument("--no-viz", action="store_true")
    args = parser.parse_args()

    print("Running chain replication simulation...")

    results = []
    for length in [2, 3, 5]:
        print(f"  Chain length={length}...")
        r = run_chain(length, duration_s=args.duration, write_rate=args.rate, seed=args.seed)
        results.append(r)

    print_summary(results)

    if not args.no_viz:
        output_dir = Path(args.output)
        visualize_results(results, output_dir)
