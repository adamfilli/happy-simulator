"""Tests for Client component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List

import pytest

from happysimulator.components.client.client import Client, ClientStats
from happysimulator.components.client.retry import (
    NoRetry,
    FixedRetry,
    ExponentialBackoff,
)
from happysimulator.components.server.server import Server
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.source import Source


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    """Constant request rate profile."""
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


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


@dataclass
class FailingServer(Entity):
    """Server that fails after specified number of requests."""
    name: str
    fail_until: int = 2
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        yield self.response_time


class ClientRequestProvider(EventProvider):
    """Generates requests through a client."""

    def __init__(self, client: Client, stop_after: Instant | None = None):
        self.client = client
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        request = self.client.send_request(payload=f"request-{self.generated}")
        request.time = time
        return [request]


class TestClientBasics:
    """Basic Client functionality tests."""

    def test_creates_with_defaults(self):
        """Client can be created with minimal parameters."""
        server = EchoServer(name="server")
        client = Client(name="TestClient", target=server)

        assert client.name == "TestClient"
        assert client.target is server
        assert client.timeout is None
        assert isinstance(client.retry_policy, NoRetry)
        assert client.in_flight_count == 0

    def test_creates_with_timeout(self):
        """Client can be created with timeout."""
        server = EchoServer(name="server")
        client = Client(name="TestClient", target=server, timeout=5.0)

        assert client.timeout == 5.0

    def test_rejects_negative_timeout(self):
        """Client rejects negative timeout."""
        server = EchoServer(name="server")

        with pytest.raises(ValueError):
            Client(name="TestClient", target=server, timeout=-1.0)

    def test_creates_with_retry_policy(self):
        """Client can be created with custom retry policy."""
        server = EchoServer(name="server")
        policy = FixedRetry(max_attempts=3, delay=0.1)
        client = Client(name="TestClient", target=server, retry_policy=policy)

        assert client.retry_policy is policy

    def test_initial_statistics_are_zero(self):
        """Client starts with zero statistics."""
        server = EchoServer(name="server")
        client = Client(name="TestClient", target=server)

        assert client.stats.requests_sent == 0
        assert client.stats.responses_received == 0
        assert client.stats.timeouts == 0
        assert client.stats.retries == 0
        assert client.stats.failures == 0


class TestClientRequestSending:
    """Tests for Client request sending."""

    def test_sends_single_request(self):
        """Client sends a single request successfully."""
        server = EchoServer(name="server", response_time=0.010)
        client = Client(name="client", target=server)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request(payload="test")
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.requests_sent == 1
        assert client.stats.responses_received == 1
        assert server.requests_received == 1

    def test_sends_multiple_requests(self):
        """Client sends multiple requests successfully."""
        server = EchoServer(name="server", response_time=0.010)
        client = Client(name="client", target=server)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
        )

        for i in range(5):
            request = client.send_request(payload=f"test-{i}")
            request.time = Instant.from_seconds(i * 0.1)
            sim.schedule(request)

        sim.run()

        assert client.stats.requests_sent == 5
        assert client.stats.responses_received == 5
        assert server.requests_received == 5

    def test_tracks_response_time(self):
        """Client tracks response times."""
        server = EchoServer(name="server", response_time=0.050)
        client = Client(name="client", target=server)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.average_response_time == pytest.approx(0.050, rel=0.01)


class TestClientTimeout:
    """Tests for Client timeout handling."""

    def test_timeout_triggers_on_slow_response(self):
        """Timeout triggers when server is slower than timeout."""
        server = SlowServer(name="server", response_time=1.0)  # 1s response
        client = Client(name="client", target=server, timeout=0.1)  # 100ms timeout

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.timeouts == 1
        assert client.stats.failures == 1  # No retry policy

    def test_no_timeout_on_fast_response(self):
        """No timeout when server responds before timeout."""
        server = EchoServer(name="server", response_time=0.010)  # 10ms response
        client = Client(name="client", target=server, timeout=1.0)  # 1s timeout

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.timeouts == 0
        assert client.stats.responses_received == 1


class TestClientRetry:
    """Tests for Client retry behavior."""

    def test_retry_on_timeout_with_fixed_policy(self):
        """Client retries on timeout with FixedRetry policy."""
        server = SlowServer(name="server", response_time=0.200)  # 200ms response
        policy = FixedRetry(max_attempts=3, delay=0.050)
        client = Client(
            name="client",
            target=server,
            timeout=0.100,  # 100ms timeout (will timeout)
            retry_policy=policy,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        # Should have tried 3 times total (1 initial + 2 retries)
        assert client.stats.requests_sent == 3
        assert client.stats.timeouts == 3
        assert client.stats.retries == 2  # 2 retries after initial attempt
        assert client.stats.failures == 1  # Final failure

    def test_retry_succeeds_eventually(self):
        """Client retry succeeds if server becomes responsive."""
        # Server that responds fast enough for the second attempt
        server = EchoServer(name="server", response_time=0.050)
        policy = FixedRetry(max_attempts=3, delay=0.010)
        client = Client(
            name="client",
            target=server,
            timeout=0.100,  # Server is fast enough
            retry_policy=policy,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
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

    def test_exponential_backoff_retry(self):
        """Client uses exponential backoff for retries."""
        server = SlowServer(name="server", response_time=1.0)
        policy = ExponentialBackoff(
            max_attempts=3,
            initial_delay=0.100,
            max_delay=10.0,
            multiplier=2.0,
            jitter=0.0,
        )
        client = Client(
            name="client",
            target=server,
            timeout=0.050,
            retry_policy=policy,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert client.stats.requests_sent == 3
        assert client.stats.retries == 2


class TestClientCallbacks:
    """Tests for Client callbacks."""

    def test_success_callback_invoked(self):
        """on_success callback is invoked on successful response."""
        server = EchoServer(name="server", response_time=0.010)

        success_calls = []

        def on_success(request, response):
            success_calls.append((request, response))

        client = Client(
            name="client",
            target=server,
            on_success=on_success,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
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

        failure_calls = []

        def on_failure(request, reason):
            failure_calls.append((request, reason))

        client = Client(
            name="client",
            target=server,
            timeout=0.050,
            on_failure=on_failure,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
        )

        request = client.send_request(payload="test_payload")
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        assert len(failure_calls) == 1
        assert failure_calls[0][1] == "timeout"

    def test_per_request_callbacks(self):
        """Per-request callbacks override client defaults."""
        server = EchoServer(name="server", response_time=0.010)

        client_success = []
        request_success = []

        def client_on_success(request, response):
            client_success.append(request)

        def request_on_success(request, response):
            request_success.append(request)

        client = Client(
            name="client",
            target=server,
            on_success=client_on_success,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, client],
        )

        # Request with override callback
        request = client.send_request(on_success=request_on_success)
        request.time = Instant.Epoch
        sim.schedule(request)
        sim.run()

        # Should use per-request callback, not client default
        assert len(client_success) == 0
        assert len(request_success) == 1


class TestClientStatistics:
    """Tests for Client statistics tracking."""

    def test_tracks_response_time_percentiles(self):
        """Client calculates response time percentiles."""
        server = EchoServer(name="server", response_time=0.050)
        client = Client(name="client", target=server)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, client],
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

        # All responses should be ~50ms
        assert p50 == pytest.approx(0.050, rel=0.01)
        assert p99 == pytest.approx(0.050, rel=0.01)


class TestClientWithServer:
    """Integration tests for Client with Server."""

    def test_client_with_server_under_load(self):
        """Client works with Server under load."""
        server = Server(
            name="server",
            concurrency=2,
            service_time=ConstantLatency(0.050),
        )
        client = Client(name="client", target=server)

        provider = ClientRequestProvider(client, stop_after=Instant.from_seconds(1.0))
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=10.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source],
            entities=[server, client],
        )
        sim.run()

        # Should have sent and received requests
        assert client.stats.requests_sent > 0
        assert client.stats.responses_received > 0
        assert server.stats.requests_completed > 0

    def test_client_with_timeout_and_busy_server(self):
        """Client times out when server is slower than timeout."""
        # Use SlowServer which always takes longer than the timeout
        server = SlowServer(name="server", response_time=0.500)  # 500ms response
        client = Client(
            name="client",
            target=server,
            timeout=0.100,  # 100ms timeout - much faster than server
            retry_policy=FixedRetry(max_attempts=2, delay=0.050),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[],
            entities=[server, client],
        )

        # Send a single request that will definitely timeout
        request = client.send_request()
        request.time = Instant.Epoch
        sim.schedule(request)

        sim.run()

        # Request should timeout and retry
        assert client.stats.timeouts >= 1
        assert client.stats.requests_sent == 2  # Initial + 1 retry (max_attempts=2)
