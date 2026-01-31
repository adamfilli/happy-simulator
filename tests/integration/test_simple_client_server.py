"""Integration tests for SimpleClient and SimpleServer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from happysimulator.modules.client_server.simple_client import SimpleClient
from happysimulator.modules.client_server.simple_server import SimpleServer
from happysimulator.core.event import Event
from happysimulator.modules.client_server.request import Request
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.exponential import ExponentialLatency


@dataclass(frozen=True)
class ConstantRateProfile(Profile):
    rate_per_s: float

    def get_rate(self, time: Instant) -> float:
        return float(self.rate_per_s)


class RequestProvider(EventProvider):
    """Generates Request events targeting a client/server pair."""

    def __init__(
        self,
        client: SimpleClient,
        server: SimpleServer,
        network_latency=None,
        stop_after: Instant | None = None,
    ):
        self.client = client
        self.server = server
        self.network_latency = network_latency or ConstantLatency(0.01)
        self.stop_after = stop_after
        self.generated = 0

    def get_events(self, time: Instant) -> List[Event]:
        if self.stop_after and time > self.stop_after:
            return []

        self.generated += 1
        request = Request(
            time=time,
            event_type=f"Request-{self.generated}",
            client=self.client,
            server=self.server,
            network_latency=self.network_latency,
            callback=self.client.send_request,
        )
        return [request]


class TestSimpleClientServer:
    """Test basic client-server interaction."""

    def test_single_request_succeeds(self):
        """A single request should complete successfully."""
        client = SimpleClient("client")
        server = SimpleServer("server")

        provider = RequestProvider(
            client, server,
            stop_after=Instant.from_seconds(0.01),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=100.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        assert provider.generated == 1
        assert len(client.stats_successful.values) == 1
        assert client.stats_successful.values[0][1] == 1
        assert len(server.stats_requests_completed.values) == 1

    def test_multiple_requests_sequential(self):
        """Multiple requests should complete when spaced apart."""
        client = SimpleClient("client")
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.05),
        )

        # 5 requests per second, server takes 50ms = can handle ~20/s
        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.001),
            stop_after=Instant.from_seconds(0.8),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=5.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # All requests should succeed since arrival rate < service rate
        total_successful = sum(v for _, v in client.stats_successful.values)
        assert total_successful == provider.generated

    def test_server_rejects_when_busy(self):
        """Server should reject requests while processing."""
        client = SimpleClient("client")
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(1.0),
        )

        # Send 5 requests per second, server takes 1s to process
        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.001),
            stop_after=Instant.from_seconds(0.5),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=5.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # First request succeeds, rest should be rejected
        total_rejected = sum(v for _, v in server.stats_requests_rejected.values)
        client_rejected = sum(v for _, v in client.stats_rejected.values)

        assert total_rejected > 0
        assert client_rejected > 0
        assert total_rejected == client_rejected

    def test_client_timeout_detection(self):
        """Client should detect timeout when latency exceeds threshold."""
        client = SimpleClient(
            "client",
            timeout=0.1,
        )
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.5),
        )

        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.001),
            stop_after=Instant.from_seconds(0.01),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=100.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # Should have timed out
        total_timed_out = sum(v for _, v in client.stats_timed_out.values)
        assert total_timed_out > 0

    def test_client_retry_on_rejection(self):
        """Client should retry when request is rejected."""
        client = SimpleClient(
            "client",
            retries=2,
            retry_delay=0.5,
        )
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.3),
        )

        # Send 2 requests very close together - second will be rejected initially
        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.001),
            stop_after=Instant.from_seconds(0.05),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=50.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # Should have attempted retries
        total_sent = sum(v for _, v in client.stats_requests_sent.values)
        total_retries = sum(v for _, v in client.stats_retries.values)

        # If there were rejections, retries should have happened
        total_rejected = sum(v for _, v in client.stats_rejected.values)
        if total_rejected > 0:
            assert total_retries > 0
            assert total_sent > provider.generated

    def test_client_retry_on_timeout(self):
        """Client should retry when request times out."""
        client = SimpleClient(
            "client",
            timeout=0.1,
            retries=2,
            retry_delay=0.05,
        )
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.5),
        )

        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.001),
            stop_after=Instant.from_seconds(0.01),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=100.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # Should have attempted retries due to timeout
        total_sent = sum(v for _, v in client.stats_requests_sent.values)
        total_retries = sum(v for _, v in client.stats_retries.values)

        assert total_retries > 0
        assert total_sent > provider.generated

    def test_latency_tracking(self):
        """Client should track request latency."""
        client = SimpleClient("client")
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.1),
        )

        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.01),
            stop_after=Instant.from_seconds(0.5),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=2.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # Check latency was recorded
        latencies = [v for _, v in client.stats_latency.values]
        assert len(latencies) > 0

        # Latency should be ~0.12s (0.01 network + 0.1 processing + 0.01 network)
        for latency in latencies:
            assert 0.10 < latency < 0.15

    def test_exponential_latency_varies(self):
        """Network latency should vary with exponential distribution."""
        client = SimpleClient("client")
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.01),
        )

        # Use exponential network latency
        provider = RequestProvider(
            client, server,
            network_latency=ExponentialLatency(0.05),
            stop_after=Instant.from_seconds(1.0),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=10.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # Check latencies vary
        latencies = [v for _, v in client.stats_latency.values]
        assert len(latencies) > 1
        assert len(set(latencies)) > 1  # Should have variance


class TestSimpleServerStats:
    """Test server statistics tracking."""

    def test_server_tracks_processing_time(self):
        """Server should track actual processing time."""
        client = SimpleClient("client")
        server = SimpleServer(
            "server",
            processing_latency=ConstantLatency(0.1),
        )

        provider = RequestProvider(
            client, server,
            network_latency=ConstantLatency(0.001),
            stop_after=Instant.from_seconds(0.5),
        )
        arrival = ConstantArrivalTimeProvider(
            ConstantRateProfile(rate_per_s=2.0),
            start_time=Instant.Epoch,
        )
        source = Source("source", provider, arrival)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[source],
            entities=[client, server],
        )
        sim.run()

        # Check processing time was recorded
        proc_times = [v for _, v in server.stats_processing_time.values]
        assert len(proc_times) > 0

        # Processing time should be ~0.1s
        for pt in proc_times:
            assert 0.09 < pt < 0.11

    def test_has_capacity_reflects_state(self):
        """has_capacity should return False while busy."""
        server = SimpleServer("server")

        # Initially has capacity
        assert server.has_capacity() is True
        assert server.is_busy is False
