"""Tests for Hedge component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.resilience import Hedge
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass
class FastServer(Entity):
    """Server with fast response time."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class VariableServer(Entity):
    """Server with variable response time."""

    name: str
    response_times: list[float] = field(default_factory=lambda: [0.010])

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        idx = self.requests_received % len(self.response_times)
        self.requests_received += 1
        yield self.response_times[idx]


class TestHedgeCreation:
    """Tests for Hedge creation."""

    def test_creates_with_basic_parameters(self):
        """Hedge can be created with basic parameters."""
        server = FastServer(name="server")
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        assert hedge.name == "hedge"
        assert hedge.target is server
        assert hedge.hedge_delay == 0.050
        assert hedge.max_hedges == 1

    def test_creates_with_custom_max_hedges(self):
        """Hedge can be created with custom max_hedges."""
        server = FastServer(name="server")
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050, max_hedges=3)

        assert hedge.max_hedges == 3

    def test_rejects_invalid_hedge_delay(self):
        """Hedge rejects hedge_delay <= 0."""
        server = FastServer(name="server")

        with pytest.raises(ValueError):
            Hedge(name="hedge", target=server, hedge_delay=0)

        with pytest.raises(ValueError):
            Hedge(name="hedge", target=server, hedge_delay=-0.1)

    def test_rejects_invalid_max_hedges(self):
        """Hedge rejects max_hedges < 1."""
        server = FastServer(name="server")

        with pytest.raises(ValueError):
            Hedge(name="hedge", target=server, hedge_delay=0.050, max_hedges=0)

    def test_initial_statistics_are_zero(self):
        """Hedge starts with zero statistics."""
        server = FastServer(name="server")
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        assert hedge.stats.total_requests == 0
        assert hedge.stats.primary_wins == 0
        assert hedge.stats.hedge_wins == 0
        assert hedge.stats.hedges_sent == 0


class TestHedgeBehavior:
    """Tests for Hedge behavior."""

    def test_forwards_request_to_target(self):
        """Hedge forwards request to target."""
        server = FastServer(name="server")
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, hedge],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=hedge,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert server.requests_received >= 1
        assert hedge.stats.total_requests == 1

    def test_primary_wins_when_fast(self):
        """Primary wins when response is fast."""
        server = FastServer(name="server", response_time=0.010)
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, hedge],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=hedge,
            context={},
        )
        sim.schedule(request)
        sim.run()

        # Primary completes in 10ms, before hedge delay of 50ms
        assert hedge.stats.primary_wins == 1
        assert hedge.stats.hedge_wins == 0
        # Hedge was never sent because primary completed before hedge delay
        assert hedge.stats.hedges_sent == 0

    def test_sends_hedge_after_delay(self):
        """Hedge sends hedge request after delay."""
        server = VariableServer(
            name="server",
            response_times=[0.100, 0.010],  # Primary slow, hedge fast
        )
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, hedge],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=hedge,
            context={},
        )
        sim.schedule(request)
        sim.run()

        # Both primary and hedge should be sent
        assert server.requests_received == 2
        assert hedge.stats.hedges_sent == 1

    def test_hedge_wins_when_faster(self):
        """Hedge wins when it completes faster than primary."""
        server = VariableServer(
            name="server",
            response_times=[0.100, 0.010],  # Primary 100ms, hedge 10ms
        )
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, hedge],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=hedge,
            context={},
        )
        sim.schedule(request)
        sim.run()

        # Hedge at t=50ms + 10ms = 60ms
        # Primary at t=0 + 100ms = 100ms
        # Hedge should win
        assert hedge.stats.hedge_wins == 1
        assert hedge.stats.primary_wins == 0

    def test_multiple_hedges(self):
        """Hedge can send multiple hedge requests."""
        server = VariableServer(
            name="server",
            response_times=[0.200, 0.200, 0.010],  # First two slow, third fast
        )
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050, max_hedges=2)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, hedge],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=hedge,
            context={},
        )
        sim.schedule(request)
        sim.run()

        # Should send primary + 2 hedges
        assert server.requests_received == 3
        assert hedge.stats.hedges_sent == 2


class TestHedgeStatistics:
    """Tests for Hedge statistics."""

    def test_tracks_total_requests(self):
        """Hedge tracks total requests."""
        server = FastServer(name="server")
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, hedge],
        )

        for i in range(5):
            request = Event(
                time=Instant.from_seconds(i * 0.15),
                event_type="request",
                target=hedge,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert hedge.stats.total_requests == 5

    def test_tracks_hedges_sent(self):
        """Hedge tracks hedges sent."""
        # Use a slow server to ensure hedges are actually sent
        server = VariableServer(
            name="server",
            response_times=[0.100],  # 100ms response time
        )
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, hedge],
        )

        for i in range(3):
            request = Event(
                time=Instant.from_seconds(i * 0.3),
                event_type="request",
                target=hedge,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Each request should trigger one hedge
        assert hedge.stats.hedges_sent == 3


class TestHedgeProperties:
    """Tests for Hedge properties."""

    def test_in_flight_count(self):
        """in_flight_count tracks pending requests."""
        server = FastServer(name="server")
        hedge = Hedge(name="hedge", target=server, hedge_delay=0.050)

        assert hedge.in_flight_count == 0
