"""Integration tests for load balancing components.

Tests LoadBalancer with ConsistentHash, RoundRobin, LeastConnections,
and PowerOfTwoChoices strategies in multi-component pipelines with
backend servers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.components.common import Sink
from happysimulator.components.load_balancer.load_balancer import LoadBalancer
from happysimulator.components.load_balancer.strategies import (
    ConsistentHash,
    LeastConnections,
    PowerOfTwoChoices,
    RoundRobin,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class SimpleServer(Entity):
    """Backend server that processes requests and forwards to downstream."""

    name: str
    service_time: float = 0.01
    downstream: Entity | None = None

    requests_received: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self.requests_received += 1
        self.active_connections += 1
        yield self.service_time
        self.active_connections -= 1
        if self.downstream:
            return [self.forward(event, self.downstream, event_type="Response")]
        return []


@dataclass
class CachingServer(Entity):
    """Backend server with a simple cache that tracks hit rate.

    Uses client_id from request metadata as cache key.
    """

    name: str
    service_time: float = 0.01
    cache_hit_time: float = 0.001
    downstream: Entity | None = None
    cache_capacity: int = 50

    _cache: set = field(default_factory=set, init=False)
    requests_received: int = field(default=0, init=False)
    active_connections: int = field(default=0, init=False)
    cache_hits: int = field(default=0, init=False)
    cache_misses: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self.requests_received += 1
        self.active_connections += 1

        key = event.context.get("metadata", {}).get("client_id", "")

        if key in self._cache:
            self.cache_hits += 1
            yield self.cache_hit_time
        else:
            self.cache_misses += 1
            if len(self._cache) >= self.cache_capacity:
                # Evict one random entry
                self._cache.pop()
            self._cache.add(key)
            yield self.service_time

        self.active_connections -= 1
        if self.downstream:
            return [self.forward(event, self.downstream, event_type="Response")]
        return []

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class TestConsistentHashVsRoundRobin:
    """Compare ConsistentHash vs RoundRobin for cache hit rate."""

    def test_consistent_hash_achieves_higher_cache_hit_rate(self):
        random.seed(42)
        sink = Sink("responses")
        num_servers = 3
        num_clients = 20
        num_requests = 200

        # --- ConsistentHash setup ---
        ch_servers = [
            CachingServer(name=f"ch_s{i}", downstream=sink, cache_capacity=30)
            for i in range(num_servers)
        ]
        ch_lb = LoadBalancer(
            name="ch_lb",
            backends=ch_servers,
            strategy=ConsistentHash(),
        )

        ch_sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            entities=[*ch_servers, ch_lb, sink],
        )

        # Send requests from repeating clients
        for i in range(num_requests):
            client_id = f"client_{i % num_clients}"
            ch_sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="request",
                    target=ch_lb,
                    context={"metadata": {"client_id": client_id}},
                )
            )
        ch_sim.run()

        ch_total_hits = sum(s.cache_hits for s in ch_servers)
        ch_total = sum(s.requests_received for s in ch_servers)

        # --- RoundRobin setup ---
        rr_sink = Sink("rr_responses")
        rr_servers = [
            CachingServer(name=f"rr_s{i}", downstream=rr_sink, cache_capacity=30)
            for i in range(num_servers)
        ]
        rr_lb = LoadBalancer(
            name="rr_lb",
            backends=rr_servers,
            strategy=RoundRobin(),
        )

        rr_sim = Simulation(
            start_time=Instant.Epoch,
            duration=30.0,
            entities=[*rr_servers, rr_lb, rr_sink],
        )

        for i in range(num_requests):
            client_id = f"client_{i % num_clients}"
            rr_sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="request",
                    target=rr_lb,
                    context={"metadata": {"client_id": client_id}},
                )
            )
        rr_sim.run()

        rr_total_hits = sum(s.cache_hits for s in rr_servers)
        rr_total = sum(s.requests_received for s in rr_servers)

        ch_hit_rate = ch_total_hits / ch_total if ch_total else 0
        rr_hit_rate = rr_total_hits / rr_total if rr_total else 0

        # ConsistentHash should route same client to same server → higher hit rate
        assert ch_hit_rate > rr_hit_rate

    def test_consistent_hash_reduces_misses(self):
        random.seed(42)
        num_servers = 3
        num_clients = 10
        num_requests = 100

        ch_servers = [CachingServer(name=f"ch_s{i}", cache_capacity=20) for i in range(num_servers)]
        ch_lb = LoadBalancer(
            name="ch_lb",
            backends=ch_servers,
            strategy=ConsistentHash(),
        )

        ch_sim = Simulation(
            start_time=Instant.Epoch,
            duration=15.0,
            entities=[*ch_servers, ch_lb],
        )

        for i in range(num_requests):
            client_id = f"client_{i % num_clients}"
            ch_sim.schedule(
                Event(
                    time=Instant.from_seconds(i * 0.1),
                    event_type="request",
                    target=ch_lb,
                    context={"metadata": {"client_id": client_id}},
                )
            )
        ch_sim.run()

        total_misses = sum(s.cache_misses for s in ch_servers)
        total_requests = sum(s.requests_received for s in ch_servers)

        # With consistent hashing, each client always goes to the same server
        # so misses should be at most num_clients (one cold miss per client)
        assert total_misses <= num_clients
        assert total_requests == num_requests


class TestLeastConnectionsDistribution:
    """LeastConnections distributes load evenly across servers."""

    def test_balanced_distribution(self):
        sink = Sink("responses")
        # Use slower service so connections overlap, giving LeastConnections
        # useful data to differentiate servers
        servers = [SimpleServer(name=f"s{i}", service_time=0.5, downstream=sink) for i in range(4)]
        lb = LoadBalancer(
            name="lb",
            backends=servers,
            strategy=LeastConnections(),
        )

        source = Source.constant(rate=20.0, target=lb, stop_after=5.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=8.0,
            sources=[source],
            entities=[*servers, lb, sink],
        )
        sim.run()

        counts = [s.requests_received for s in servers]
        total = sum(counts)
        assert total > 0

        # All servers should have received traffic
        for count in counts:
            assert count > 0


class TestPowerOfTwoChoicesLoadBalancing:
    """PowerOfTwoChoices provides reasonable distribution."""

    def test_reasonable_distribution(self):
        random.seed(42)
        sink = Sink("responses")
        servers = [SimpleServer(name=f"s{i}", service_time=0.05, downstream=sink) for i in range(5)]
        lb = LoadBalancer(
            name="lb",
            backends=servers,
            strategy=PowerOfTwoChoices(),
        )

        source = Source.constant(rate=20.0, target=lb, stop_after=5.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=8.0,
            sources=[source],
            entities=[*servers, lb, sink],
        )
        sim.run()

        counts = [s.requests_received for s in servers]
        total = sum(counts)
        assert total > 0

        # All servers should have received some traffic
        for count in counts:
            assert count > 0

        # Not perfectly round-robin but should be somewhat balanced
        avg = total / len(servers)
        for count in counts:
            assert count < avg * 3


class TestBackendFailureHandling:
    """Mark a backend unhealthy, assert traffic rerouted."""

    def test_traffic_rerouted_on_failure(self):
        sink = Sink("responses")
        healthy1 = SimpleServer(name="healthy1", service_time=0.01, downstream=sink)
        healthy2 = SimpleServer(name="healthy2", service_time=0.01, downstream=sink)
        failed = SimpleServer(name="failed", service_time=0.01, downstream=sink)

        lb = LoadBalancer(
            name="lb",
            backends=[healthy1, healthy2, failed],
            strategy=RoundRobin(),
        )

        # Mark one backend as unhealthy before sending requests
        lb.mark_unhealthy(failed)

        source = Source.constant(rate=10.0, target=lb, stop_after=3.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            duration=5.0,
            sources=[source],
            entities=[healthy1, healthy2, failed, lb, sink],
        )
        sim.run()

        # Failed server should receive no traffic
        assert failed.requests_received == 0
        # Healthy servers should share the load
        assert healthy1.requests_received > 0
        assert healthy2.requests_received > 0
        assert lb.stats.requests_forwarded > 0
        assert lb.stats.backends_marked_unhealthy == 1
