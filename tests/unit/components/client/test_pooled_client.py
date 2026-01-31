"""Tests for PooledClient component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.client.connection_pool import ConnectionPool
from happysimulator.components.client.pooled_client import PooledClient, PooledClientStats
from happysimulator.components.client.retry import FixedRetry, NoRetry
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency


@dataclass
class EchoServer(Entity):
    """Simple server that echoes back requests after a delay."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class SlowServer(Entity):
    """Server with configurable response time."""

    name: str
    response_time: float = 1.0

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        yield self.response_time


class TestPooledClientCreation:
    """Tests for PooledClient creation and validation."""

    def test_creates_with_defaults(self):
        """PooledClient can be created with minimal parameters."""
        server = EchoServer(name="server")
        pool = ConnectionPool(name="pool", target=server)
        client = PooledClient(name="client", connection_pool=pool)

        assert client.name == "client"
        assert client.connection_pool is pool
        assert client.timeout is None
        assert isinstance(client.retry_policy, NoRetry)
        assert client.in_flight_count == 0

    def test_creates_with_timeout(self):
        """PooledClient can be created with timeout."""
        server = EchoServer(name="server")
        pool = ConnectionPool(name="pool", target=server)
        client = PooledClient(name="client", connection_pool=pool, timeout=5.0)

        assert client.timeout == 5.0

    def test_rejects_negative_timeout(self):
        """PooledClient rejects negative timeout."""
        server = EchoServer(name="server")
        pool = ConnectionPool(name="pool", target=server)

        with pytest.raises(ValueError):
            PooledClient(name="client", connection_pool=pool, timeout=-1.0)

    def test_creates_with_retry_policy(self):
        """PooledClient can be created with custom retry policy."""
        server = EchoServer(name="server")
        pool = ConnectionPool(name="pool", target=server)
        policy = FixedRetry(max_attempts=3, delay=0.1)
        client = PooledClient(name="client", connection_pool=pool, retry_policy=policy)

        assert client.retry_policy is policy

    def test_initial_statistics_are_zero(self):
        """PooledClient starts with zero statistics."""
        server = EchoServer(name="server")
        pool = ConnectionPool(name="pool", target=server)
        client = PooledClient(name="client", connection_pool=pool)

        assert client.stats.requests_sent == 0
        assert client.stats.responses_received == 0
        assert client.stats.timeouts == 0
        assert client.stats.retries == 0
        assert client.stats.failures == 0
        assert client.stats.connection_wait_timeouts == 0


class TestPooledClientRequestSending:
    """Tests for PooledClient request sending."""

    def test_sends_single_request(self):
        """PooledClient sends a single request successfully."""
        server = EchoServer(name="server", response_time=0.010)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request(payload="test")
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.requests_sent == 1
        assert client.stats.responses_received == 1
        assert server.requests_received == 1

    def test_sends_multiple_requests(self):
        """PooledClient sends multiple requests successfully."""
        server = EchoServer(name="server", response_time=0.010)
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=5,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool, client],
        )

        for i in range(5):
            request = client.send_request(payload=f"test-{i}")
            request.time = Instant.from_seconds(i * 0.1)
            sim.schedule(request)

        sim.run()

        assert client.stats.requests_sent == 5
        assert client.stats.responses_received == 5
        assert server.requests_received == 5

    def test_reuses_connections(self):
        """PooledClient reuses connections from pool."""
        server = EchoServer(name="server", response_time=0.010)
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,  # Only one connection
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool, client],
        )

        # Send 3 sequential requests with gaps
        for i in range(3):
            request = client.send_request(payload=f"test-{i}")
            request.time = Instant.from_seconds(i * 0.1)
            sim.schedule(request)

        sim.run()

        # All should succeed with only one connection created
        assert client.stats.requests_sent == 3
        assert client.stats.responses_received == 3
        assert pool.stats.connections_created == 1  # Connection reused

    def test_tracks_response_time(self):
        """PooledClient tracks response times."""
        server = EchoServer(name="server", response_time=0.050)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        # Response time includes connection time + server response time
        assert client.average_response_time >= 0.050


class TestPooledClientTimeout:
    """Tests for PooledClient timeout handling."""

    def test_timeout_triggers_on_slow_response(self):
        """Timeout triggers when server is slower than timeout."""
        server = SlowServer(name="server", response_time=1.0)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(
            name="client",
            connection_pool=pool,
            timeout=0.1,  # 100ms timeout
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.timeouts == 1
        assert client.stats.failures == 1

    def test_no_timeout_on_fast_response(self):
        """No timeout when server responds before timeout."""
        server = EchoServer(name="server", response_time=0.010)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(
            name="client",
            connection_pool=pool,
            timeout=1.0,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.timeouts == 0
        assert client.stats.responses_received == 1


class TestPooledClientRetry:
    """Tests for PooledClient retry behavior."""

    def test_retry_on_timeout_with_fixed_policy(self):
        """PooledClient retries on timeout with FixedRetry policy."""
        server = SlowServer(name="server", response_time=0.200)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        policy = FixedRetry(max_attempts=3, delay=0.050)
        client = PooledClient(
            name="client",
            connection_pool=pool,
            timeout=0.100,
            retry_policy=policy,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        # Should have tried 3 times total (1 initial + 2 retries)
        assert client.stats.requests_sent == 3
        assert client.stats.timeouts == 3
        assert client.stats.retries == 2
        assert client.stats.failures == 1

    def test_retry_succeeds_eventually(self):
        """PooledClient retry succeeds if server becomes responsive."""
        server = EchoServer(name="server", response_time=0.050)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        policy = FixedRetry(max_attempts=3, delay=0.010)
        client = PooledClient(
            name="client",
            connection_pool=pool,
            timeout=0.100,  # Server is fast enough
            retry_policy=policy,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        # Should succeed on first try
        assert client.stats.requests_sent == 1
        assert client.stats.responses_received == 1
        assert client.stats.timeouts == 0
        assert client.stats.failures == 0


class TestPooledClientCallbacks:
    """Tests for PooledClient callbacks."""

    def test_success_callback_invoked(self):
        """on_success callback is invoked on successful response."""
        server = EchoServer(name="server", response_time=0.010)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )

        success_calls = []

        def on_success(request, response):
            success_calls.append((request, response))

        client = PooledClient(
            name="client",
            connection_pool=pool,
            on_success=on_success,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request(payload="test_payload")
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert len(success_calls) == 1
        assert success_calls[0][0].context["metadata"]["payload"] == "test_payload"

    def test_failure_callback_invoked(self):
        """on_failure callback is invoked on final failure."""
        server = SlowServer(name="server", response_time=1.0)
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )

        failure_calls = []

        def on_failure(request, reason):
            failure_calls.append((request, reason))

        client = PooledClient(
            name="client",
            connection_pool=pool,
            timeout=0.050,
            on_failure=on_failure,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool, client],
        )

        request = client.send_request(payload="test_payload")
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert len(failure_calls) == 1
        assert failure_calls[0][1] == "timeout"


class TestPooledClientConnectionWait:
    """Tests for connection pool wait behavior."""

    def test_waits_for_connection(self):
        """PooledClient waits when pool is exhausted."""
        server = EchoServer(name="server", response_time=0.100)
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=5.0,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool, client],
        )

        # Send 2 concurrent requests with only 1 connection
        for i in range(2):
            request = client.send_request(payload=f"test-{i}")
            request.time = Instant.Epoch
            sim.schedule(request)

        sim.run()

        # Both should succeed (second waits for first)
        assert client.stats.requests_sent == 2
        assert client.stats.responses_received == 2

    def test_connection_wait_timeout(self):
        """PooledClient fails when connection wait times out."""
        server = SlowServer(name="server", response_time=1.0)
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=0.1,  # Short wait timeout
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[server, pool, client],
        )

        # First request acquires the only connection and holds it
        # Second request will timeout waiting
        request1 = client.send_request(payload="test-1")
        request1.time = Instant.Epoch
        sim.schedule(request1)

        request2 = client.send_request(payload="test-2")
        request2.time = Instant.from_seconds(0.010)
        sim.schedule(request2)

        sim.run()

        # Second request should fail due to connection wait timeout
        assert client.stats.connection_wait_timeouts == 1
        assert client.stats.failures >= 1


class TestPooledClientStatistics:
    """Tests for PooledClient statistics tracking."""

    def test_tracks_response_time_percentiles(self):
        """PooledClient calculates response time percentiles."""
        server = EchoServer(name="server", response_time=0.050)
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=10,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,
        )
        client = PooledClient(name="client", connection_pool=pool)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[server, pool, client],
        )

        # Send multiple requests
        for i in range(10):
            request = client.send_request()
            request.time = Instant.from_seconds(i * 0.1)
            sim.schedule(request)

        sim.run()

        assert client.stats.responses_received == 10

        p50 = client.get_response_time_percentile(0.50)
        p99 = client.get_response_time_percentile(0.99)

        # Response times should be ~50ms + connection time
        assert p50 >= 0.050
        assert p99 >= 0.050
