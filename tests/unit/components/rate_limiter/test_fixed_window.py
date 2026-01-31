"""Tests for FixedWindowRateLimiter."""

import pytest

from happysimulator.components.rate_limiter import FixedWindowRateLimiter
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


class TestFixedWindowCreation:
    """Tests for FixedWindowRateLimiter creation."""

    def test_creates_with_parameters(self):
        """Rate limiter is created with specified parameters."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=100,
            window_size=1.0,
        )

        assert limiter.name == "test"
        assert limiter.downstream is downstream
        assert limiter.requests_per_window == 100
        assert limiter.window_size == 1.0

    def test_rejects_zero_requests(self):
        """Rejects requests_per_window < 1."""
        downstream = DummyDownstream()
        with pytest.raises(ValueError):
            FixedWindowRateLimiter(
                name="test",
                downstream=downstream,
                requests_per_window=0,
            )

    def test_rejects_negative_window_size(self):
        """Rejects window_size <= 0."""
        downstream = DummyDownstream()
        with pytest.raises(ValueError):
            FixedWindowRateLimiter(
                name="test",
                downstream=downstream,
                requests_per_window=10,
                window_size=0,
            )

    def test_starts_with_zero_count(self):
        """Rate limiter starts with zero request count."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=10,
        )

        assert limiter.current_window_count == 0
        assert limiter.stats.requests_received == 0


class TestFixedWindowForwarding:
    """Tests for request forwarding behavior."""

    def test_forwards_under_limit(self):
        """Requests under the limit are forwarded."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=5,
            window_size=1.0,
        )

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            result = limiter.handle_event(event)
            assert len(result) == 1

        assert limiter.stats.requests_forwarded == 5
        assert limiter.stats.requests_dropped == 0

    def test_drops_over_limit(self):
        """Requests over the limit are dropped."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=3,
            window_size=1.0,
        )

        # Send 5 requests
        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert limiter.stats.requests_forwarded == 3
        assert limiter.stats.requests_dropped == 2

    def test_forward_event_targets_downstream(self):
        """Forwarded events target the downstream entity."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=10,
        )

        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)

        assert len(result) == 1
        assert result[0].target is downstream

    def test_forward_event_copies_context(self):
        """Forwarded events copy the original context."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=10,
        )

        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
            context={"user": "alice", "data": [1, 2, 3]},
        )
        result = limiter.handle_event(event)

        assert result[0].context["user"] == "alice"
        assert result[0].context["data"] == [1, 2, 3]


class TestFixedWindowReset:
    """Tests for window reset behavior."""

    def test_resets_on_new_window(self):
        """Counter resets when moving to a new window."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=2,
            window_size=1.0,
        )

        # Fill first window
        for i in range(2):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert limiter.current_window_count == 2

        # Third request in first window - dropped
        event = Event(
            time=Instant.from_seconds(0.5),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)
        assert len(result) == 0

        # First request in second window - allowed
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)
        assert len(result) == 1
        assert limiter.current_window_count == 1

    def test_tracks_windows_completed(self):
        """Statistics track completed windows."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=10,
            window_size=1.0,
        )

        # Request in window 0
        event = Event(time=Instant.from_seconds(0.5), event_type="request", target=limiter)
        limiter.handle_event(event)

        # Request in window 1
        event = Event(time=Instant.from_seconds(1.5), event_type="request", target=limiter)
        limiter.handle_event(event)

        # Request in window 2
        event = Event(time=Instant.from_seconds(2.5), event_type="request", target=limiter)
        limiter.handle_event(event)

        assert limiter.stats.windows_completed == 2


class TestFixedWindowStatistics:
    """Tests for statistics tracking."""

    def test_tracks_all_stats(self):
        """Statistics track all request outcomes."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=2,
            window_size=1.0,
        )

        for i in range(4):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert limiter.stats.requests_received == 4
        assert limiter.stats.requests_forwarded == 2
        assert limiter.stats.requests_dropped == 2

    def test_tracks_time_series(self):
        """Time series data is recorded."""
        downstream = DummyDownstream()
        limiter = FixedWindowRateLimiter(
            name="test",
            downstream=downstream,
            requests_per_window=2,
            window_size=1.0,
        )

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert len(limiter.received_times) == 3
        assert len(limiter.forwarded_times) == 2
        assert len(limiter.dropped_times) == 1
