"""Tests for load balancing strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.load_balancer.strategies import (
    LoadBalancingStrategy,
    RoundRobin,
    WeightedRoundRobin,
    Random,
    LeastConnections,
    WeightedLeastConnections,
    LeastResponseTime,
    IPHash,
    ConsistentHash,
    PowerOfTwoChoices,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


@dataclass
class MockBackend(Entity):
    """Mock backend for testing."""
    name: str
    active_connections: int = 0

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        yield 0.01


def create_request(client_id: str | None = None, **metadata) -> Event:
    """Create a test request event."""
    ctx = {"metadata": metadata}
    if client_id:
        ctx["metadata"]["client_id"] = client_id
    return Event(
        time=Instant.Epoch,
        event_type="request",
        callback=lambda e: None,
        context=ctx,
    )


class TestRoundRobin:
    """Tests for RoundRobin strategy."""

    def test_cycles_through_backends(self):
        """RoundRobin cycles through backends in order."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = RoundRobin()

        selections = [strategy.select(backends, create_request()) for _ in range(6)]
        names = [b.name for b in selections]

        assert names == ["b0", "b1", "b2", "b0", "b1", "b2"]

    def test_returns_none_for_empty_list(self):
        """RoundRobin returns None for empty backend list."""
        strategy = RoundRobin()
        assert strategy.select([], create_request()) is None

    def test_handles_single_backend(self):
        """RoundRobin handles single backend."""
        backends = [MockBackend(name="only")]
        strategy = RoundRobin()

        for _ in range(5):
            assert strategy.select(backends, create_request()).name == "only"

    def test_reset_restarts_cycle(self):
        """reset() restarts the round-robin cycle."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = RoundRobin()

        strategy.select(backends, create_request())  # b0
        strategy.select(backends, create_request())  # b1
        strategy.reset()
        assert strategy.select(backends, create_request()).name == "b0"

    def test_satisfies_protocol(self):
        """RoundRobin satisfies LoadBalancingStrategy protocol."""
        strategy = RoundRobin()
        assert isinstance(strategy, LoadBalancingStrategy)


class TestWeightedRoundRobin:
    """Tests for WeightedRoundRobin strategy."""

    def test_respects_weights(self):
        """WeightedRoundRobin distributes according to weights."""
        backends = [MockBackend(name="heavy"), MockBackend(name="light")]
        strategy = WeightedRoundRobin()
        strategy.set_weight(backends[0], 3)
        strategy.set_weight(backends[1], 1)

        selections = [strategy.select(backends, create_request()).name for _ in range(8)]

        # Heavy should be selected ~3x more than light
        heavy_count = selections.count("heavy")
        light_count = selections.count("light")
        assert heavy_count > light_count

    def test_default_weight_is_one(self):
        """Backends without explicit weight default to 1."""
        backends = [MockBackend(name="a"), MockBackend(name="b")]
        strategy = WeightedRoundRobin()

        assert strategy.get_weight(backends[0]) == 1
        assert strategy.get_weight(backends[1]) == 1

    def test_rejects_invalid_weight(self):
        """set_weight rejects weight < 1."""
        strategy = WeightedRoundRobin()
        backend = MockBackend(name="test")

        with pytest.raises(ValueError):
            strategy.set_weight(backend, 0)

    def test_returns_none_for_empty_list(self):
        """WeightedRoundRobin returns None for empty backend list."""
        strategy = WeightedRoundRobin()
        assert strategy.select([], create_request()) is None


class TestRandom:
    """Tests for Random strategy."""

    def test_selects_from_backends(self):
        """Random selects from available backends."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = Random()

        random.seed(42)
        selections = {strategy.select(backends, create_request()).name for _ in range(100)}

        # With enough selections, should hit all backends
        assert len(selections) == 3

    def test_returns_none_for_empty_list(self):
        """Random returns None for empty backend list."""
        strategy = Random()
        assert strategy.select([], create_request()) is None

    def test_distribution_is_roughly_uniform(self):
        """Random distributes roughly uniformly."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = Random()

        random.seed(42)
        counts = {b.name: 0 for b in backends}
        for _ in range(3000):
            selected = strategy.select(backends, create_request())
            counts[selected.name] += 1

        # Each should be roughly 1000 +/- 100
        for count in counts.values():
            assert 800 < count < 1200


class TestLeastConnections:
    """Tests for LeastConnections strategy."""

    def test_selects_backend_with_fewest_connections(self):
        """LeastConnections selects backend with fewest connections."""
        backends = [
            MockBackend(name="busy", active_connections=10),
            MockBackend(name="idle", active_connections=0),
            MockBackend(name="medium", active_connections=5),
        ]
        strategy = LeastConnections()

        selected = strategy.select(backends, create_request())
        assert selected.name == "idle"

    def test_returns_first_on_tie(self):
        """LeastConnections returns first backend on tie."""
        backends = [
            MockBackend(name="first", active_connections=5),
            MockBackend(name="second", active_connections=5),
        ]
        strategy = LeastConnections()

        selected = strategy.select(backends, create_request())
        assert selected.name == "first"

    def test_returns_none_for_empty_list(self):
        """LeastConnections returns None for empty backend list."""
        strategy = LeastConnections()
        assert strategy.select([], create_request()) is None


class TestWeightedLeastConnections:
    """Tests for WeightedLeastConnections strategy."""

    def test_considers_weight_in_score(self):
        """WeightedLeastConnections considers weight in scoring."""
        backends = [
            MockBackend(name="light", active_connections=5),
            MockBackend(name="heavy", active_connections=5),
        ]
        strategy = WeightedLeastConnections()
        strategy.set_weight(backends[0], 1)  # score = 5/1 = 5
        strategy.set_weight(backends[1], 2)  # score = 5/2 = 2.5

        selected = strategy.select(backends, create_request())
        assert selected.name == "heavy"  # Lower score wins

    def test_rejects_invalid_weight(self):
        """set_weight rejects weight < 1."""
        strategy = WeightedLeastConnections()
        backend = MockBackend(name="test")

        with pytest.raises(ValueError):
            strategy.set_weight(backend, 0)


class TestLeastResponseTime:
    """Tests for LeastResponseTime strategy."""

    def test_selects_fastest_backend(self):
        """LeastResponseTime selects backend with lowest response time."""
        backends = [
            MockBackend(name="slow"),
            MockBackend(name="fast"),
            MockBackend(name="medium"),
        ]
        strategy = LeastResponseTime()

        # Record response times
        strategy.record_response_time(backends[0], 0.500)
        strategy.record_response_time(backends[1], 0.010)
        strategy.record_response_time(backends[2], 0.100)

        selected = strategy.select(backends, create_request())
        assert selected.name == "fast"

    def test_prioritizes_unknown_backends(self):
        """LeastResponseTime tries unknown backends first."""
        backends = [
            MockBackend(name="known"),
            MockBackend(name="unknown"),
        ]
        strategy = LeastResponseTime()
        strategy.record_response_time(backends[0], 0.010)

        random.seed(42)
        selected = strategy.select(backends, create_request())
        assert selected.name == "unknown"

    def test_uses_exponential_moving_average(self):
        """LeastResponseTime uses EMA for response times."""
        backend = MockBackend(name="test")
        strategy = LeastResponseTime(alpha=0.5)

        strategy.record_response_time(backend, 1.0)  # EMA = 1.0
        strategy.record_response_time(backend, 0.0)  # EMA = 0.5 * 0 + 0.5 * 1 = 0.5

        assert strategy.get_response_time(backend) == pytest.approx(0.5)

    def test_rejects_invalid_alpha(self):
        """LeastResponseTime rejects alpha outside (0, 1]."""
        with pytest.raises(ValueError):
            LeastResponseTime(alpha=0)
        with pytest.raises(ValueError):
            LeastResponseTime(alpha=1.5)


class TestIPHash:
    """Tests for IPHash strategy."""

    def test_same_client_same_backend(self):
        """IPHash routes same client to same backend."""
        backends = [MockBackend(name=f"b{i}") for i in range(5)]
        strategy = IPHash()

        # Multiple requests from same client
        selections = [
            strategy.select(backends, create_request(client_id="user123"))
            for _ in range(10)
        ]

        # All should go to same backend
        assert len(set(b.name for b in selections)) == 1

    def test_different_clients_may_differ(self):
        """IPHash routes different clients potentially to different backends."""
        backends = [MockBackend(name=f"b{i}") for i in range(5)]
        strategy = IPHash()

        selections = {
            strategy.select(backends, create_request(client_id=f"user{i}")).name
            for i in range(100)
        }

        # With 100 clients and 5 backends, should hit multiple backends
        assert len(selections) > 1

    def test_falls_back_without_key(self):
        """IPHash falls back to round-robin without client key."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = IPHash()

        # No client_id in request
        selections = [strategy.select(backends, create_request()) for _ in range(6)]
        names = [b.name for b in selections]

        # Should use round-robin fallback
        assert names == ["b0", "b1", "b2", "b0", "b1", "b2"]

    def test_custom_key_extractor(self):
        """IPHash uses custom key extractor."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = IPHash(get_key=lambda e: e.context.get("metadata", {}).get("custom_key"))

        request = create_request(custom_key="mykey")
        selected1 = strategy.select(backends, request)
        selected2 = strategy.select(backends, request)

        assert selected1.name == selected2.name


class TestConsistentHash:
    """Tests for ConsistentHash strategy."""

    def test_same_key_same_backend(self):
        """ConsistentHash routes same key to same backend."""
        backends = [MockBackend(name=f"b{i}") for i in range(5)]
        strategy = ConsistentHash()

        selections = [
            strategy.select(backends, create_request(client_id="user123"))
            for _ in range(10)
        ]

        assert len(set(b.name for b in selections)) == 1

    def test_minimal_remapping_on_add(self):
        """ConsistentHash minimizes remapping when backend added."""
        backends = [MockBackend(name=f"b{i}") for i in range(3)]
        strategy = ConsistentHash(virtual_nodes=50)

        # Map 100 keys
        initial_mapping = {}
        for i in range(100):
            request = create_request(client_id=f"user{i}")
            initial_mapping[i] = strategy.select(backends, request).name

        # Add a backend
        new_backend = MockBackend(name="b3")
        backends.append(new_backend)

        # Check remapping
        remapped = 0
        for i in range(100):
            request = create_request(client_id=f"user{i}")
            new_name = strategy.select(backends, request).name
            if new_name != initial_mapping[i]:
                remapped += 1

        # Should remap roughly 1/4 (since we went from 3 to 4 backends)
        # Allow some variance due to hash distribution
        assert remapped < 50  # Much less than total

    def test_rejects_invalid_virtual_nodes(self):
        """ConsistentHash rejects virtual_nodes < 1."""
        with pytest.raises(ValueError):
            ConsistentHash(virtual_nodes=0)


class TestPowerOfTwoChoices:
    """Tests for PowerOfTwoChoices strategy."""

    def test_selects_less_loaded_of_two(self):
        """PowerOfTwoChoices selects less loaded of two random choices."""
        # With deterministic setup, verify it picks the less loaded one
        backends = [
            MockBackend(name="idle", active_connections=0),
            MockBackend(name="busy", active_connections=10),
        ]
        strategy = PowerOfTwoChoices()

        # With only 2 backends, it always compares these two
        for _ in range(10):
            selected = strategy.select(backends, create_request())
            assert selected.name == "idle"

    def test_returns_none_for_empty_list(self):
        """PowerOfTwoChoices returns None for empty backend list."""
        strategy = PowerOfTwoChoices()
        assert strategy.select([], create_request()) is None

    def test_handles_single_backend(self):
        """PowerOfTwoChoices handles single backend."""
        backends = [MockBackend(name="only")]
        strategy = PowerOfTwoChoices()

        assert strategy.select(backends, create_request()).name == "only"

    def test_provides_better_distribution_than_random(self):
        """PowerOfTwoChoices provides better load balance than random."""
        backends = [MockBackend(name=f"b{i}", active_connections=0) for i in range(10)]
        strategy = PowerOfTwoChoices()

        random.seed(42)

        # Simulate 1000 requests, incrementing connections
        for _ in range(1000):
            selected = strategy.select(backends, create_request())
            selected.active_connections += 1

        # Check variance in connection counts
        connections = [b.active_connections for b in backends]
        mean = sum(connections) / len(connections)
        variance = sum((c - mean) ** 2 for c in connections) / len(connections)

        # With power of two choices, variance should be low
        # Each backend should have roughly 100 connections +/- small amount
        assert variance < 100  # Low variance indicates good balance
