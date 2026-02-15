"""Tests for TimeoutWrapper component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.resilience import TimeoutWrapper
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class FastServer(Entity):
    """Server with fast response time."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class SlowServer(Entity):
    """Server with slow response time."""

    name: str
    response_time: float = 10.0

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class VariableServer(Entity):
    """Server with variable response time."""

    name: str
    response_times: list[float] = field(default_factory=lambda: [0.010])

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        idx = self.requests_received % len(self.response_times)
        self.requests_received += 1
        yield self.response_times[idx]


class TestTimeoutWrapperCreation:
    """Tests for TimeoutWrapper creation."""

    def test_creates_with_basic_parameters(self):
        """TimeoutWrapper can be created with basic parameters."""
        server = FastServer(name="server")
        tw = TimeoutWrapper(name="tw", target=server, timeout=1.0)

        assert tw.name == "tw"
        assert tw.target is server
        assert tw.timeout == 1.0

    def test_creates_with_callback(self):
        """TimeoutWrapper can be created with timeout callback."""
        server = FastServer(name="server")
        callback_events = []

        def on_timeout(event):
            callback_events.append(event)
            return

        tw = TimeoutWrapper(
            name="tw",
            target=server,
            timeout=1.0,
            on_timeout=on_timeout,
        )

        assert tw.timeout == 1.0

    def test_rejects_invalid_timeout(self):
        """TimeoutWrapper rejects timeout <= 0."""
        server = FastServer(name="server")

        with pytest.raises(ValueError):
            TimeoutWrapper(name="tw", target=server, timeout=0)

        with pytest.raises(ValueError):
            TimeoutWrapper(name="tw", target=server, timeout=-1)

    def test_initial_statistics_are_zero(self):
        """TimeoutWrapper starts with zero statistics."""
        server = FastServer(name="server")
        tw = TimeoutWrapper(name="tw", target=server, timeout=1.0)

        assert tw.stats.total_requests == 0
        assert tw.stats.successful_requests == 0
        assert tw.stats.timed_out_requests == 0


class TestTimeoutWrapperBehavior:
    """Tests for TimeoutWrapper behavior."""

    def test_forwards_requests_to_target(self):
        """TimeoutWrapper forwards requests to target."""
        server = FastServer(name="server")
        tw = TimeoutWrapper(name="tw", target=server, timeout=1.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, tw],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=tw,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert server.requests_received == 1
        assert tw.stats.total_requests == 1

    def test_counts_successful_requests(self):
        """TimeoutWrapper counts successful requests."""
        server = FastServer(name="server", response_time=0.010)
        tw = TimeoutWrapper(name="tw", target=server, timeout=1.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, tw],
        )

        for i in range(3):
            request = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=tw,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert tw.stats.successful_requests == 3
        assert tw.stats.timed_out_requests == 0

    def test_detects_timeout(self):
        """TimeoutWrapper detects timed out requests."""
        server = SlowServer(name="server", response_time=10.0)
        tw = TimeoutWrapper(name="tw", target=server, timeout=0.1)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, tw],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=tw,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert tw.stats.timed_out_requests == 1
        assert tw.stats.successful_requests == 0

    def test_timeout_callback_invoked(self):
        """TimeoutWrapper invokes callback on timeout."""
        server = SlowServer(name="server", response_time=10.0)
        timed_out_events = []

        def on_timeout(event):
            timed_out_events.append(event)
            return

        tw = TimeoutWrapper(
            name="tw",
            target=server,
            timeout=0.1,
            on_timeout=on_timeout,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, tw],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=tw,
            context={"metadata": {"id": 123}},
        )
        sim.schedule(request)
        sim.run()

        assert len(timed_out_events) == 1
        assert timed_out_events[0].context["metadata"]["id"] == 123

    def test_mixed_success_and_timeout(self):
        """TimeoutWrapper handles mix of fast and slow responses."""
        server = VariableServer(
            name="server",
            response_times=[0.010, 10.0, 0.010],
        )
        tw = TimeoutWrapper(name="tw", target=server, timeout=0.1)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, tw],
        )

        for i in range(3):
            request = Event(
                time=Instant.from_seconds(i * 0.2),
                event_type="request",
                target=tw,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert tw.stats.successful_requests == 2
        assert tw.stats.timed_out_requests == 1


class TestTimeoutWrapperProperties:
    """Tests for TimeoutWrapper properties."""

    def test_in_flight_count(self):
        """in_flight_count tracks pending requests."""
        server = FastServer(name="server")
        tw = TimeoutWrapper(name="tw", target=server, timeout=1.0)

        assert tw.in_flight_count == 0
