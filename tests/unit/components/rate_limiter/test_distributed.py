"""Tests for DistributedRateLimiter."""

from collections.abc import Generator
from typing import Any

import pytest

from happysimulator.components.rate_limiter import DistributedRateLimiter
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class DummyDownstream(Entity):
    """Simple downstream entity for testing."""

    def __init__(self):
        super().__init__("downstream")
        self.received_events: list[Event] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.received_events.append(event)
        return []


class MockKVStore(Entity):
    """Mock KVStore for testing without simulation."""

    def __init__(self):
        super().__init__("mock_store")
        self._data: dict[str, Any] = {}
        self.read_count = 0
        self.write_count = 0

    def handle_event(self, event: Event) -> list[Event]:
        """Handle event (not used in tests)."""
        return []

    def get(self, key: str) -> Generator[float, None, Any]:
        """Get a value (yields 0 delay for testing)."""
        self.read_count += 1
        yield 0.0
        return self._data.get(key)

    def put(self, key: str, value: Any) -> Generator[float]:
        """Put a value (yields 0 delay for testing)."""
        self.write_count += 1
        yield 0.0
        self._data[key] = value


class TestDistributedCreation:
    """Tests for DistributedRateLimiter creation."""

    def test_creates_with_parameters(self):
        """Rate limiter is created with specified parameters."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=100,
            window_size=1.0,
        )

        assert limiter.name == "test"
        assert limiter.downstream is downstream
        assert limiter.backing_store is store
        assert limiter.global_limit == 100
        assert limiter.window_size == 1.0

    def test_rejects_invalid_global_limit(self):
        """Rejects global_limit < 1."""
        downstream = DummyDownstream()
        store = MockKVStore()
        with pytest.raises(ValueError):
            DistributedRateLimiter(
                name="test",
                downstream=downstream,
                backing_store=store,
                global_limit=0,
            )

    def test_rejects_invalid_window_size(self):
        """Rejects window_size <= 0."""
        downstream = DummyDownstream()
        store = MockKVStore()
        with pytest.raises(ValueError):
            DistributedRateLimiter(
                name="test",
                downstream=downstream,
                backing_store=store,
                global_limit=100,
                window_size=0,
            )

    def test_rejects_invalid_local_threshold(self):
        """Rejects local_threshold outside (0, 1]."""
        downstream = DummyDownstream()
        store = MockKVStore()
        with pytest.raises(ValueError):
            DistributedRateLimiter(
                name="test",
                downstream=downstream,
                backing_store=store,
                global_limit=100,
                local_threshold=0,
            )


class TestDistributedForwarding:
    """Tests for request forwarding behavior."""

    def test_forwards_under_limit(self):
        """Requests under the limit are forwarded."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=5,
            window_size=1.0,
        )

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            list(gen)[-1] if gen else []
            # Run the generator to completion
            try:
                while True:
                    next(gen)
            except (StopIteration, TypeError):
                pass

        assert limiter.stats.requests_forwarded == 3
        assert limiter.stats.requests_dropped == 0

    def test_drops_over_limit(self):
        """Requests over the limit are dropped."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=3,
            window_size=1.0,
        )

        results = []
        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            # Exhaust generator and capture result
            result = None
            try:
                while True:
                    result = next(gen)
            except StopIteration as e:
                result = e.value
            results.append(result)

        assert limiter.stats.requests_forwarded == 3
        assert limiter.stats.requests_dropped == 2

    def test_uses_backing_store(self):
        """Rate limiter uses backing store for coordination."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=10,
            window_size=1.0,
        )

        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
        )
        gen = limiter.handle_event(event)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert store.read_count >= 1
        assert store.write_count >= 1


class TestDistributedWindowReset:
    """Tests for window reset behavior."""

    def test_resets_on_new_window(self):
        """Counter resets when moving to a new window."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=2,
            window_size=1.0,
        )

        # Fill first window
        for i in range(2):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert limiter.stats.requests_forwarded == 2

        # Third request in first window - should be dropped
        event = Event(
            time=Instant.from_seconds(0.5),
            event_type="request",
            target=limiter,
        )
        gen = limiter.handle_event(event)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert limiter.stats.requests_dropped == 1

        # First request in second window - should be allowed
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="request",
            target=limiter,
        )
        gen = limiter.handle_event(event)
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert limiter.stats.requests_forwarded == 3


class TestDistributedMultipleInstances:
    """Tests for multiple instances sharing a store."""

    def test_shared_limit_across_instances(self):
        """Multiple instances share the global limit."""
        downstream1 = DummyDownstream()
        downstream2 = DummyDownstream()
        store = MockKVStore()  # Shared store

        limiter1 = DistributedRateLimiter(
            name="limiter1",
            downstream=downstream1,
            backing_store=store,
            global_limit=4,
            window_size=1.0,
        )
        limiter2 = DistributedRateLimiter(
            name="limiter2",
            downstream=downstream2,
            backing_store=store,
            global_limit=4,
            window_size=1.0,
        )

        # Alternate between instances
        for i in range(6):
            limiter = limiter1 if i % 2 == 0 else limiter2
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        total_forwarded = limiter1.stats.requests_forwarded + limiter2.stats.requests_forwarded
        total_dropped = limiter1.stats.requests_dropped + limiter2.stats.requests_dropped

        assert total_forwarded == 4
        assert total_dropped == 2


class TestDistributedStatistics:
    """Tests for statistics tracking."""

    def test_tracks_store_operations(self):
        """Statistics track store read/write counts."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=10,
            window_size=1.0,
        )

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert limiter.stats.store_reads == 3
        assert limiter.stats.store_writes == 3

    def test_tracks_rejection_types(self):
        """Statistics track local vs global rejections."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=2,
            window_size=1.0,
        )

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # After hitting limit, subsequent requests should be rejected
        # Either locally or globally depending on implementation
        total_rejections = limiter.stats.local_rejections + limiter.stats.global_rejections
        assert total_rejections == 3

    def test_tracks_time_series(self):
        """Time series data is recorded."""
        downstream = DummyDownstream()
        store = MockKVStore()
        limiter = DistributedRateLimiter(
            name="test",
            downstream=downstream,
            backing_store=store,
            global_limit=10,
            window_size=1.0,
        )

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            gen = limiter.handle_event(event)
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert len(limiter.received_times) == 3
        assert len(limiter.forwarded_times) == 3
