"""Tests for IdempotencyStore component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.microservice import (
    IdempotencyStore,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class EchoServer(Entity):
    """Server that counts requests and completes immediately."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)
    keys_seen: list[str] = field(default_factory=list, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        key = event.context.get("metadata", {}).get("_is_key")
        if key:
            self.keys_seen.append(key)
        yield self.response_time


class TestIdempotencyStoreCreation:
    """Tests for IdempotencyStore creation."""

    def test_creates_with_defaults(self):
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
        )

        assert store.name == "idem"
        assert store.target is server
        assert store.cache_size == 0
        assert store.in_flight_count == 0

    def test_initial_stats_are_zero(self):
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
        )

        assert store.stats.total_requests == 0
        assert store.stats.cache_hits == 0
        assert store.stats.cache_misses == 0
        assert store.stats.entries_stored == 0
        assert store.stats.entries_expired == 0

    def test_rejects_invalid_ttl(self):
        server = EchoServer(name="server")
        with pytest.raises(ValueError):
            IdempotencyStore(name="x", target=server, key_extractor=lambda e: "k", ttl=0)

    def test_rejects_invalid_max_entries(self):
        server = EchoServer(name="server")
        with pytest.raises(ValueError):
            IdempotencyStore(name="x", target=server, key_extractor=lambda e: "k", max_entries=0)

    def test_rejects_invalid_cleanup_interval(self):
        server = EchoServer(name="server")
        with pytest.raises(ValueError):
            IdempotencyStore(
                name="x", target=server, key_extractor=lambda e: "k", cleanup_interval=0
            )


class TestIdempotencyStoreBehavior:
    """Tests for IdempotencyStore request deduplication."""

    def test_forwards_unique_requests(self):
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, store],
        )

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=store,
                context={"metadata": {"key": f"key_{i}"}},
            )
            sim.schedule(event)

        sim.run()

        assert server.requests_received == 3
        assert store.stats.cache_misses == 3

    def test_suppresses_duplicate_requests(self):
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, store],
        )

        # Send same key twice - second should be in-flight hit
        for _i in range(2):
            event = Event(
                time=Instant.Epoch,
                event_type="request",
                target=store,
                context={"metadata": {"key": "same_key"}},
            )
            sim.schedule(event)

        sim.run()

        assert server.requests_received == 1
        assert store.stats.cache_hits == 1
        assert store.stats.cache_misses == 1

    def test_cached_duplicate_suppressed_after_completion(self):
        """After first request completes, sending same key is a cache hit."""
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, store],
        )

        # First request
        event1 = Event(
            time=Instant.Epoch,
            event_type="request",
            target=store,
            context={"metadata": {"key": "key_a"}},
        )
        sim.schedule(event1)

        # Duplicate after completion (server takes 10ms)
        event2 = Event(
            time=Instant.from_seconds(0.5),
            event_type="request",
            target=store,
            context={"metadata": {"key": "key_a"}},
        )
        sim.schedule(event2)

        sim.run()

        assert server.requests_received == 1
        assert store.stats.cache_hits == 1
        assert store.stats.entries_stored == 1

    def test_forwards_unconditionally_when_no_key(self):
        """Events with None key bypass dedup."""
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: None,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, store],
        )

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=store,
            )
            sim.schedule(event)

        sim.run()

        assert server.requests_received == 3
        assert store.stats.cache_hits == 0

    def test_evicts_oldest_when_max_entries_reached(self):
        """When cache is full, oldest entry is evicted."""
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
            max_entries=2,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, store],
        )

        # Send 3 unique requests (cache can hold 2)
        for i in range(3):
            event = Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="request",
                target=store,
                context={"metadata": {"key": f"key_{i}"}},
            )
            sim.schedule(event)

        sim.run()

        assert server.requests_received == 3
        assert store.cache_size == 2
        assert store.stats.entries_expired == 1  # oldest evicted


class TestIdempotencyStoreCleanup:
    """Tests for TTL cleanup behavior."""

    def test_cleanup_expires_old_entries(self):
        """Entries older than TTL are cleaned up."""
        server = EchoServer(name="server")
        store = IdempotencyStore(
            name="idem",
            target=server,
            key_extractor=lambda e: e.get_context("key"),
            ttl=0.5,
            cleanup_interval=0.3,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, store],
        )

        # Write entry at t=0
        event = Event(
            time=Instant.Epoch,
            event_type="request",
            target=store,
            context={"metadata": {"key": "expire_me"}},
        )
        sim.schedule(event)

        sim.run()

        # Entry should have been expired by cleanup daemon
        assert store.stats.entries_expired >= 1
