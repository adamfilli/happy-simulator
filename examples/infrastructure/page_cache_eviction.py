"""Page cache hit rate under different workload patterns.

This example demonstrates how OS page cache size and read-ahead
settings affect hit rates under sequential vs random access patterns.
The key insight: sequential access benefits enormously from read-ahead,
while random access is dominated by cache capacity.

## Architecture Diagram

```
    Source (constant rate)
        |
        v
    CacheWorkloadDriver ──> PageCache (small / large / with readahead)
        |
        v
      Sink
```

## Key Metrics

- Cache hit rate
- Eviction count
- Read-ahead effectiveness
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
    Simulation,
    SimulationSummary,
    Sink,
    Source,
)
from happysimulator.components.infrastructure import (
    PageCache,
    PageCacheStats,
)


# =============================================================================
# Custom Entity
# =============================================================================


class CacheWorkloadDriver(Entity):
    """Drives a page access workload against a PageCache.

    Alternates between sequential and random access patterns.
    """

    def __init__(
        self,
        name: str,
        *,
        cache: PageCache,
        downstream: Entity | None = None,
        total_pages: int = 10000,
        sequential_fraction: float = 0.7,
    ) -> None:
        super().__init__(name)
        self._cache = cache
        self._downstream = downstream
        self._total_pages = total_pages
        self._sequential_fraction = sequential_fraction
        self._next_seq_page: int = 0
        self._ops: int = 0

    @property
    def ops(self) -> int:
        return self._ops

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        if random.random() < self._sequential_fraction:
            page_id = self._next_seq_page % self._total_pages
            self._next_seq_page += 1
        else:
            page_id = random.randint(0, self._total_pages - 1)

        # 80% reads, 20% writes
        if random.random() < 0.8:
            yield from self._cache.read_page(page_id)
        else:
            yield from self._cache.write_page(page_id)

        self._ops += 1

        if self._downstream:
            return [self.forward(event, self._downstream, event_type="Done")]
        return []


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class CacheResult:
    config_name: str
    stats: PageCacheStats
    summary: SimulationSummary


@dataclass
class SimulationResult:
    small_cache: CacheResult
    large_cache: CacheResult
    readahead_cache: CacheResult
    duration_s: float


def _run_config(
    config_name: str,
    cache: PageCache,
    *,
    duration_s: float,
    rate: float,
    seed: int | None,
) -> CacheResult:
    if seed is not None:
        random.seed(seed)

    sink = Sink()
    driver = CacheWorkloadDriver(
        f"Driver_{config_name}",
        cache=cache,
        downstream=sink,
    )

    source = Source.constant(
        rate=rate,
        target=driver,
        event_type="Access",
        stop_after=Instant.from_seconds(duration_s),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[cache, driver, sink],
    )
    summary = sim.run()

    return CacheResult(
        config_name=config_name,
        stats=cache.stats,
        summary=summary,
    )


def run_simulation(
    *,
    duration_s: float = 10.0,
    rate: float = 1000.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Compare page cache configurations."""
    small = _run_config(
        "Small (100 pages)",
        PageCache("small_cache", capacity_pages=100),
        duration_s=duration_s, rate=rate, seed=seed,
    )
    large = _run_config(
        "Large (1000 pages)",
        PageCache("large_cache", capacity_pages=1000),
        duration_s=duration_s, rate=rate, seed=seed,
    )
    readahead = _run_config(
        "Readahead (100 + ra=4)",
        PageCache("readahead_cache", capacity_pages=100, readahead_pages=4),
        duration_s=duration_s, rate=rate, seed=seed,
    )

    return SimulationResult(
        small_cache=small,
        large_cache=large,
        readahead_cache=readahead,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    print("\n" + "=" * 72)
    print("PAGE CACHE HIT RATE COMPARISON")
    print("=" * 72)

    configs = [result.small_cache, result.large_cache, result.readahead_cache]
    header = f"{'Metric':<25} " + " ".join(f"{c.config_name:>20}" for c in configs)
    print(f"\n{header}")
    print("-" * len(header))

    for label, fn in [
        ("Hits", lambda s: f"{s.hits:,}"),
        ("Misses", lambda s: f"{s.misses:,}"),
        ("Hit rate", lambda s: f"{s.hit_rate:.1%}"),
        ("Evictions", lambda s: f"{s.evictions:,}"),
        ("Dirty writebacks", lambda s: f"{s.dirty_writebacks:,}"),
        ("Readaheads", lambda s: f"{s.readaheads:,}"),
        ("Pages cached", lambda s: f"{s.pages_cached:,}"),
    ]:
        vals = " ".join(f"{fn(c.stats):>20}" for c in configs)
        print(f"{label:<25} {vals}")

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  A larger cache captures more of the working set, reducing misses.")
    print("  Read-ahead helps sequential access by prefetching adjacent pages,")
    print("  but fills the cache faster with random access patterns.")
    print("\n" + "=" * 72)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Page cache hit rate comparison")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    print("Running page cache comparison...")
    result = run_simulation(duration_s=args.duration, seed=seed)
    print_summary(result)
