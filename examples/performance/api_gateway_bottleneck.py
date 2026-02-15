"""API gateway bottleneck with per-route rate limiting.

Demonstrates how per-route rate limiting in an API gateway prevents
a hot route from starving other routes. Shows request distribution
and rejection rates across routes.

## Architecture

```
                     ┌────────────────────────────────────────┐
   ┌──────────┐     │           API Gateway                   │
   │  Mixed   │────►│                                         │
   │  Traffic │     │  /api/search  ──► [Search Backends]     │
   │  Source  │     │    (rate limited: 10 req/s)             │
   └──────────┘     │  /api/users   ──► [User Backends]      │
                     │    (rate limited: 50 req/s)             │
                     │  /api/health  ──► [Health Backend]     │
                     │    (no rate limit)                      │
                     └────────────────────────────────────────┘
```
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from happysimulator import (
    ConstantArrivalTimeProvider,
    ConstantRateProfile,
    Entity,
    Event,
    EventProvider,
    Instant,
    Simulation,
    Source,
    TokenBucketPolicy,
)
from happysimulator.components.microservice import APIGateway, RouteConfig

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Components
# =============================================================================


class BackendService(Entity):
    """Simple backend with configurable latency."""

    def __init__(self, name: str, latency: float = 0.01):
        super().__init__(name)
        self.latency = latency
        self.requests_received = 0

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.latency


class MixedTrafficProvider(EventProvider):
    """Generates requests across routes with configurable distribution."""

    def __init__(
        self,
        target: Entity,
        route_weights: dict[str, float],
        stop_after: Instant | None = None,
    ):
        self._target = target
        self._routes = list(route_weights.keys())
        self._weights = list(route_weights.values())
        self._stop_after = stop_after

    def get_events(self, time: Instant) -> list[Event]:
        if self._stop_after and time > self._stop_after:
            return []
        route = random.choices(self._routes, weights=self._weights, k=1)[0]
        return [
            Event(
                time=time,
                event_type="request",
                target=self._target,
                context={"metadata": {"route": route}},
            )
        ]


# =============================================================================
# Simulation
# =============================================================================


@dataclass
class SimulationResult:
    """Results from the API gateway simulation."""

    gateway: APIGateway
    backends: dict[str, list[BackendService]]
    routes: list[str]


def run_gateway_simulation(
    *,
    duration_s: float = 20.0,
    total_arrival_rate: float = 100.0,
    search_rate_limit: float = 10.0,
    users_rate_limit: float = 50.0,
    route_weights: dict[str, float] | None = None,
    seed: int | None = 42,
) -> SimulationResult:
    """Run the API gateway bottleneck simulation."""
    if seed is not None:
        random.seed(seed)

    if route_weights is None:
        route_weights = {
            "/api/search": 0.6,  # Hot route: 60% of traffic
            "/api/users": 0.3,  # Moderate: 30%
            "/api/health": 0.1,  # Light: 10%
        }

    # Create backends
    search_backends = [BackendService(f"search_{i}", latency=0.05) for i in range(2)]
    user_backends = [BackendService(f"users_{i}", latency=0.02) for i in range(2)]
    health_backend = [BackendService("health", latency=0.001)]

    all_backends: dict[str, list[BackendService]] = {
        "/api/search": search_backends,
        "/api/users": user_backends,
        "/api/health": health_backend,
    }

    gateway = APIGateway(
        name="Gateway",
        routes={
            "/api/search": RouteConfig(
                name="search",
                backends=search_backends,
                rate_limit_policy=TokenBucketPolicy(
                    capacity=search_rate_limit,
                    refill_rate=search_rate_limit,
                ),
                auth_required=False,
            ),
            "/api/users": RouteConfig(
                name="users",
                backends=user_backends,
                rate_limit_policy=TokenBucketPolicy(
                    capacity=users_rate_limit,
                    refill_rate=users_rate_limit,
                ),
                auth_required=False,
            ),
            "/api/health": RouteConfig(
                name="health",
                backends=health_backend,
                auth_required=False,
            ),
        },
        auth_latency=0.0,
    )

    stop = Instant.from_seconds(duration_s)
    provider = MixedTrafficProvider(gateway, route_weights, stop_after=stop)
    profile = ConstantRateProfile(rate=total_arrival_rate)
    source = Source(
        name="Traffic",
        event_provider=provider,
        arrival_time_provider=ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch),
    )

    all_entities: list[Entity] = [gateway]
    for backends in all_backends.values():
        all_entities.extend(backends)

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=duration_s + 2.0,
        sources=[source],
        entities=all_entities,
    )
    sim.run()

    return SimulationResult(
        gateway=gateway,
        backends=all_backends,
        routes=list(route_weights.keys()),
    )


def print_summary(result: SimulationResult) -> None:
    """Print gateway statistics."""
    gw = result.gateway
    s = gw.stats

    print("\n" + "=" * 70)
    print("API GATEWAY BOTTLENECK — RESULTS")
    print("=" * 70)

    print("\nOverall:")
    print(f"  Total requests:        {s.total_requests}")
    print(f"  Routed:                {s.requests_routed}")
    print(f"  Rate limited:          {s.requests_rejected_rate_limit}")
    print(f"  No route:              {s.requests_no_route}")

    print("\nPer-Route Breakdown:")
    print(
        f"  {'Route':>15s}  {'Received':>9s}  {'Routed':>7s}  {'Limited':>8s}  {'Backend Reqs':>13s}"
    )
    print("  " + "-" * 60)

    for route in result.routes:
        received = s.per_route_requests.get(route, 0)
        backends = result.backends.get(route, [])
        backend_total = sum(b.requests_received for b in backends)
        # Rate limited = received - routed for this route
        routed = backend_total
        limited = received - routed

        print(f"  {route:>15s}  {received:>9d}  {routed:>7d}  {limited:>8d}  {backend_total:>13d}")

    if s.total_requests > 0:
        overall_rate_limited = s.requests_rejected_rate_limit / s.total_requests * 100
        print(f"\nOverall rate limited: {overall_rate_limited:.1f}%")

    print("\nBackend Load Distribution:")
    for backends in result.backends.values():
        for b in backends:
            print(f"  {b.name:>15s}: {b.requests_received} requests")

    print("=" * 70)


def visualize_results(result: SimulationResult, output_dir: Path) -> None:
    """Generate gateway visualization."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    s = result.gateway.stats

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Per-route received vs routed
    ax = axes[0]
    routes = result.routes
    received = [s.per_route_requests.get(r, 0) for r in routes]
    routed = [sum(b.requests_received for b in result.backends.get(r, [])) for r in routes]
    short_names = [r.split("/")[-1] for r in routes]

    x = range(len(routes))
    w = 0.35
    ax.bar([i - w / 2 for i in x], received, w, label="Received", color="#3498db", alpha=0.8)
    ax.bar([i + w / 2 for i in x], routed, w, label="Routed", color="#2ecc71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Requests")
    ax.set_title("Per-Route: Received vs Routed")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Rate limited breakdown
    ax = axes[1]
    limited = [r - rt for r, rt in zip(received, routed, strict=False)]
    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    ax.bar(short_names, limited, color=colors[: len(routes)], edgecolor="black", alpha=0.8)
    ax.set_ylabel("Requests Rate Limited")
    ax.set_title("Rate Limited Requests per Route")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "gateway_results.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'gateway_results.png'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="API gateway bottleneck simulation")
    parser.add_argument("--duration", type=float, default=20.0, help="Load duration (s)")
    parser.add_argument("--rate", type=float, default=100.0, help="Total arrival rate (req/s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--output", type=str, default="output/gateway", help="Output dir")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    seed = None if args.seed == -1 else args.seed

    print("Running API gateway bottleneck simulation...")
    result = run_gateway_simulation(
        duration_s=args.duration,
        total_arrival_rate=args.rate,
        seed=seed,
    )
    print_summary(result)

    if not args.no_viz:
        visualize_results(result, Path(args.output))
