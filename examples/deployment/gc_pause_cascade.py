"""GC pause cascade: how GC strategy affects tail latency.

This example demonstrates how different garbage collection strategies
(StopTheWorld, ConcurrentGC, GenerationalGC) inject pauses into a
request-processing pipeline. The key insight: StopTheWorld creates
large but infrequent pauses that cause timeout cascades, while
ConcurrentGC and GenerationalGC trade shorter pauses for more
frequent interruptions.

## Architecture Diagram

```
    Source (constant rate)
        |
        v
    GCAwareServer ──gc.pause()──> GarbageCollector (STW / Concurrent / Gen)
        |
        v
      Sink
```

## Key Metrics

- Total GC pause time
- Max single pause (affects tail latency)
- Minor vs major collections (GenerationalGC)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
    SimulationSummary,
    Sink,
    Source,
)
from happysimulator.components.infrastructure import (
    ConcurrentGC,
    GarbageCollector,
    GCStats,
    GenerationalGC,
    StopTheWorld,
)

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Custom Entity
# =============================================================================


class GCAwareServer(Entity):
    """Server that experiences GC pauses during request processing."""

    def __init__(
        self,
        name: str,
        *,
        gc: GarbageCollector,
        downstream: Entity,
        service_time_s: float = 0.01,
        gc_every_n: int = 50,
    ) -> None:
        super().__init__(name)
        self._gc = gc
        self._downstream = downstream
        self._service_time_s = service_time_s
        self._gc_every_n = gc_every_n
        self._request_count: int = 0

    @property
    def request_count(self) -> int:
        return self._request_count

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._request_count += 1

        # GC pause every N requests
        if self._request_count % self._gc_every_n == 0:
            yield from self._gc.pause()

        # Normal service time
        yield self._service_time_s

        return [self.forward(event, self._downstream, event_type="Response")]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class GCResult:
    strategy_name: str
    gc_stats: GCStats
    summary: SimulationSummary
    requests: int


@dataclass
class SimulationResult:
    stw: GCResult
    concurrent: GCResult
    generational: GCResult
    duration_s: float


def _run_strategy(
    strategy_name: str,
    gc: GarbageCollector,
    *,
    duration_s: float,
    rate: float,
    seed: int | None,
) -> GCResult:
    if seed is not None:
        random.seed(seed)

    sink = Sink()
    server = GCAwareServer(
        f"Server_{strategy_name}",
        gc=gc,
        downstream=sink,
    )

    source = Source.constant(
        rate=rate,
        target=server,
        event_type="Request",
        stop_after=Instant.from_seconds(duration_s),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[gc, server, sink],
    )
    summary = sim.run()

    return GCResult(
        strategy_name=strategy_name,
        gc_stats=gc.stats,
        summary=summary,
        requests=server.request_count,
    )


def run_simulation(
    *,
    duration_s: float = 10.0,
    rate: float = 200.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Compare GC strategies and their impact on request processing."""
    stw = _run_strategy(
        "StopTheWorld",
        GarbageCollector(
            "GC_STW",
            strategy=StopTheWorld(
                base_pause_s=0.05,
                interval_s=10.0,
            ),
            heap_pressure=0.6,
        ),
        duration_s=duration_s,
        rate=rate,
        seed=seed,
    )
    concurrent = _run_strategy(
        "ConcurrentGC",
        GarbageCollector(
            "GC_Concurrent",
            strategy=ConcurrentGC(
                pause_s=0.005,
                interval_s=2.0,
            ),
            heap_pressure=0.6,
        ),
        duration_s=duration_s,
        rate=rate,
        seed=seed,
    )
    generational = _run_strategy(
        "GenerationalGC",
        GarbageCollector(
            "GC_Gen",
            strategy=GenerationalGC(
                minor_pause_s=0.002,
                major_pause_s=0.03,
                minor_interval_s=1.0,
            ),
            heap_pressure=0.6,
        ),
        duration_s=duration_s,
        rate=rate,
        seed=seed,
    )

    return SimulationResult(
        stw=stw,
        concurrent=concurrent,
        generational=generational,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    print("\n" + "=" * 72)
    print("GC PAUSE CASCADE: Strategy Comparison")
    print("=" * 72)

    results = [result.stw, result.concurrent, result.generational]
    header = f"{'Metric':<30} " + " ".join(f"{r.strategy_name:>15}" for r in results)
    print(f"\n{header}")
    print("-" * len(header))

    print(f"{'Requests processed':<30} " + " ".join(f"{r.requests:>15,}" for r in results))
    print(f"{'GC collections':<30} " + " ".join(f"{r.gc_stats.collections:>15,}" for r in results))
    print(
        f"{'Total pause (ms)':<30} "
        + " ".join(f"{r.gc_stats.total_pause_s * 1000:>15.1f}" for r in results)
    )
    print(
        f"{'Max pause (ms)':<30} "
        + " ".join(f"{r.gc_stats.max_pause_s * 1000:>15.2f}" for r in results)
    )
    print(
        f"{'Avg pause (ms)':<30} "
        + " ".join(f"{r.gc_stats.avg_pause_s * 1000:>15.2f}" for r in results)
    )
    print(
        f"{'Minor collections':<30} "
        + " ".join(f"{r.gc_stats.minor_collections:>15,}" for r in results)
    )
    print(
        f"{'Major collections':<30} "
        + " ".join(f"{r.gc_stats.major_collections:>15,}" for r in results)
    )

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  StopTheWorld: Fewest but longest pauses. A single 50ms+ pause can")
    print("  cause cascading timeouts in upstream services.")
    print("\n  ConcurrentGC: More frequent but much shorter pauses (~5ms).")
    print("  Better tail latency at the cost of slightly higher base overhead.")
    print("\n  GenerationalGC: Frequent minor collections (~2ms) with rare")
    print("  major collections (~30ms). Best for workloads with many")
    print("  short-lived objects.")
    print("\n" + "=" * 72)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GC pause cascade comparison")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    print("Running GC pause cascade comparison...")
    result = run_simulation(duration_s=args.duration, seed=seed)
    print_summary(result)
