"""Tests for Fallback component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.resilience import Fallback
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
class SlowServer(Entity):
    """Server with slow response time."""

    name: str
    response_time: float = 10.0

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class CacheServer(Entity):
    """Simple cache server for fallback."""

    name: str
    response_time: float = 0.001

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        yield self.response_time


class TestFallbackCreation:
    """Tests for Fallback creation."""

    def test_creates_with_entity_fallback(self):
        """Fallback can be created with entity as fallback."""
        primary = FastServer(name="primary")
        fallback_server = CacheServer(name="cache")
        fb = Fallback(name="fb", primary=primary, fallback=fallback_server)

        assert fb.name == "fb"
        assert fb.primary is primary
        assert fb.fallback is fallback_server

    def test_creates_with_callable_fallback(self):
        """Fallback can be created with callable as fallback."""
        primary = FastServer(name="primary")

        def fallback_fn(event):
            return None

        fb = Fallback(name="fb", primary=primary, fallback=fallback_fn)

        assert fb.primary is primary
        assert callable(fb.fallback)

    def test_creates_with_timeout(self):
        """Fallback can be created with timeout."""
        primary = FastServer(name="primary")
        fallback_server = CacheServer(name="cache")
        fb = Fallback(
            name="fb",
            primary=primary,
            fallback=fallback_server,
            timeout=1.0,
        )

        assert fb.timeout == 1.0

    def test_rejects_invalid_timeout(self):
        """Fallback rejects timeout <= 0."""
        primary = FastServer(name="primary")
        fallback_server = CacheServer(name="cache")

        with pytest.raises(ValueError):
            Fallback(
                name="fb",
                primary=primary,
                fallback=fallback_server,
                timeout=0,
            )

    def test_initial_statistics_are_zero(self):
        """Fallback starts with zero statistics."""
        primary = FastServer(name="primary")
        fallback_server = CacheServer(name="cache")
        fb = Fallback(name="fb", primary=primary, fallback=fallback_server)

        assert fb.stats.total_requests == 0
        assert fb.stats.primary_successes == 0
        assert fb.stats.primary_failures == 0
        assert fb.stats.fallback_invocations == 0


class TestFallbackBehavior:
    """Tests for Fallback behavior."""

    def test_uses_primary_when_successful(self):
        """Fallback uses primary when it succeeds."""
        primary = FastServer(name="primary")
        fallback_server = CacheServer(name="cache")
        fb = Fallback(name="fb", primary=primary, fallback=fallback_server)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[primary, fallback_server, fb],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=fb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert primary.requests_received == 1
        assert fallback_server.requests_received == 0
        assert fb.stats.primary_successes == 1
        assert fb.stats.fallback_invocations == 0

    def test_uses_fallback_on_timeout(self):
        """Fallback uses fallback when primary times out."""
        primary = SlowServer(name="primary", response_time=10.0)
        fallback_server = CacheServer(name="cache")
        fb = Fallback(
            name="fb",
            primary=primary,
            fallback=fallback_server,
            timeout=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[primary, fallback_server, fb],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=fb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert primary.requests_received == 1
        assert fallback_server.requests_received == 1
        assert fb.stats.primary_failures == 1
        assert fb.stats.fallback_invocations == 1

    def test_callable_fallback(self):
        """Fallback invokes callable fallback."""
        primary = SlowServer(name="primary", response_time=10.0)
        fallback_calls = []

        def fallback_fn(event):
            fallback_calls.append(event)
            return None

        fb = Fallback(
            name="fb",
            primary=primary,
            fallback=fallback_fn,
            timeout=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[primary, fb],
        )

        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=fb,
            context={"metadata": {"id": 42}},
        )
        sim.schedule(request)
        sim.run()

        assert len(fallback_calls) == 1
        assert fallback_calls[0].context["metadata"]["id"] == 42


class TestFallbackStatistics:
    """Tests for Fallback statistics."""

    def test_tracks_primary_successes(self):
        """Fallback tracks primary successes."""
        primary = FastServer(name="primary")
        fallback_server = CacheServer(name="cache")
        fb = Fallback(name="fb", primary=primary, fallback=fallback_server)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[primary, fallback_server, fb],
        )

        for i in range(5):
            request = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=fb,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert fb.stats.total_requests == 5
        assert fb.stats.primary_successes == 5
        assert fb.stats.fallback_invocations == 0

    def test_tracks_fallback_successes(self):
        """Fallback tracks fallback successes."""
        primary = SlowServer(name="primary", response_time=10.0)
        fallback_server = CacheServer(name="cache")
        fb = Fallback(
            name="fb",
            primary=primary,
            fallback=fallback_server,
            timeout=0.1,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[primary, fallback_server, fb],
        )

        for i in range(3):
            request = Event(
                time=Instant.from_seconds(i * 0.2),
                event_type="request",
                target=fb,
                context={},
            )
            sim.schedule(request)

        sim.run()

        assert fb.stats.total_requests == 3
        assert fb.stats.primary_failures == 3
        assert fb.stats.fallback_invocations == 3
        assert fb.stats.fallback_successes == 3
