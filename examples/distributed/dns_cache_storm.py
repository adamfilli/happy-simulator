"""DNS cache miss storm: TTL expiration under high request rate.

This example demonstrates how DNS cache configuration affects
resolution latency. The key insight: when DNS cache entries expire
simultaneously under high load, a "cache miss storm" occurs where
many requests must perform full hierarchical lookups, adding
significant latency.

## Architecture Diagram

```
    Source (constant rate)
        |
        v
    ServiceCaller ──> DNSResolver (short TTL / long TTL / large cache)
        |
        v
      Sink
```

## Key Metrics

- Cache hit rate
- Total resolution latency
- Cache miss storms (visible in latency spikes)
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
    DNSRecord,
    DNSResolver,
    DNSStats,
)

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Custom Entity
# =============================================================================


class ServiceCaller(Entity):
    """Resolves DNS before each request to a backend service."""

    def __init__(
        self,
        name: str,
        *,
        dns: DNSResolver,
        downstream: Entity,
        hostnames: list[str],
    ) -> None:
        super().__init__(name)
        self._dns = dns
        self._downstream = downstream
        self._hostnames = hostnames
        self._calls: int = 0
        self._dns_miss_calls: int = 0

    @property
    def calls(self) -> int:
        return self._calls

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._calls += 1
        hostname = random.choice(self._hostnames)

        misses_before = self._dns.stats.cache_misses
        ip = yield from self._dns.resolve(hostname)
        if self._dns.stats.cache_misses > misses_before:
            self._dns_miss_calls += 1

        # Simulate backend call
        yield 0.005

        return [
            Event(
                time=self.now,
                event_type="Response",
                target=self._downstream,
                context={**event.context, "ip": ip},
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


HOSTNAMES = [
    "api.example.com",
    "auth.example.com",
    "db.example.com",
    "cache.example.com",
    "worker.example.com",
]

RECORDS = {h: DNSRecord(h, f"10.0.0.{i + 1}", ttl_s=60.0) for i, h in enumerate(HOSTNAMES)}


@dataclass
class DNSResult:
    config_name: str
    stats: DNSStats
    summary: SimulationSummary
    calls: int


@dataclass
class SimulationResult:
    short_ttl: DNSResult
    long_ttl: DNSResult
    large_cache: DNSResult
    duration_s: float


def _make_records(ttl_s: float) -> dict[str, DNSRecord]:
    return {h: DNSRecord(h, f"10.0.0.{i + 1}", ttl_s=ttl_s) for i, h in enumerate(HOSTNAMES)}


def _run_config(
    config_name: str,
    dns: DNSResolver,
    *,
    duration_s: float,
    rate: float,
    seed: int | None,
) -> DNSResult:
    if seed is not None:
        random.seed(seed)

    sink = Sink()
    caller = ServiceCaller(
        f"Caller_{config_name}",
        dns=dns,
        downstream=sink,
        hostnames=HOSTNAMES,
    )

    source = Source.constant(
        rate=rate,
        target=caller,
        event_type="Request",
        stop_after=Instant.from_seconds(duration_s),
    )

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 1.0,
        sources=[source],
        entities=[dns, caller, sink],
    )
    summary = sim.run()

    return DNSResult(
        config_name=config_name,
        stats=dns.stats,
        summary=summary,
        calls=caller.calls,
    )


def run_simulation(
    *,
    duration_s: float = 30.0,
    rate: float = 100.0,
    seed: int | None = 42,
) -> SimulationResult:
    """Compare DNS configurations under load."""
    short_ttl = _run_config(
        "Short TTL (5s)",
        DNSResolver("DNS_Short", records=_make_records(5.0)),
        duration_s=duration_s,
        rate=rate,
        seed=seed,
    )
    long_ttl = _run_config(
        "Long TTL (300s)",
        DNSResolver("DNS_Long", records=_make_records(300.0)),
        duration_s=duration_s,
        rate=rate,
        seed=seed,
    )
    large_cache = _run_config(
        "Large Cache (5000)",
        DNSResolver("DNS_Large", records=_make_records(5.0), cache_capacity=5000),
        duration_s=duration_s,
        rate=rate,
        seed=seed,
    )

    return SimulationResult(
        short_ttl=short_ttl,
        long_ttl=long_ttl,
        large_cache=large_cache,
        duration_s=duration_s,
    )


# =============================================================================
# Summary
# =============================================================================


def print_summary(result: SimulationResult) -> None:
    print("\n" + "=" * 72)
    print("DNS CACHE MISS STORM: TTL & Cache Configuration")
    print("=" * 72)

    configs = [result.short_ttl, result.long_ttl, result.large_cache]
    header = f"{'Metric':<30} " + " ".join(f"{c.config_name:>18}" for c in configs)
    print(f"\n{header}")
    print("-" * len(header))

    print(f"{'Total lookups':<30} " + " ".join(f"{c.stats.lookups:>18,}" for c in configs))
    print(f"{'Cache hits':<30} " + " ".join(f"{c.stats.cache_hits:>18,}" for c in configs))
    print(f"{'Cache misses':<30} " + " ".join(f"{c.stats.cache_misses:>18,}" for c in configs))
    print(f"{'Hit rate':<30} " + " ".join(f"{c.stats.hit_rate:>18.1%}" for c in configs))
    print(
        f"{'TTL expirations':<30} " + " ".join(f"{c.stats.cache_expirations:>18,}" for c in configs)
    )
    print(
        f"{'Avg resolution (ms)':<30} "
        + " ".join(f"{c.stats.avg_resolution_latency_s * 1000:>18.2f}" for c in configs)
    )
    print(
        f"{'Total resolution (ms)':<30} "
        + " ".join(f"{c.stats.total_resolution_latency_s * 1000:>18.1f}" for c in configs)
    )
    print(f"{'Requests processed':<30} " + " ".join(f"{c.calls:>18,}" for c in configs))

    print("\n" + "=" * 72)
    print("INTERPRETATION:")
    print("-" * 72)
    print("\n  Short TTL (5s): Entries expire frequently, forcing repeated full")
    print("  hierarchical lookups (root + TLD + auth = ~45ms per miss).")
    print("  Under high load, many requests hit expired entries simultaneously.")
    print("\n  Long TTL (300s): Cache entries persist across the simulation,")
    print("  achieving near-perfect hit rate after initial population.")
    print("\n  Large Cache: Same short TTL but more capacity — doesn't help with")
    print("  TTL-based expiration since the bottleneck is TTL, not capacity.")
    print("\n" + "=" * 72)


# =============================================================================
# Entry Point
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DNS cache miss storm simulation")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed
    print("Running DNS cache miss storm simulation...")
    result = run_simulation(duration_s=args.duration, seed=seed)
    print_summary(result)
