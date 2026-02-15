"""Tests for FixedWindowPolicy + RateLimitedEntity."""

import pytest

from happysimulator.components.rate_limiter import FixedWindowPolicy, RateLimitedEntity
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class DummyDownstream(Entity):
    """Simple downstream entity for testing."""

    def __init__(self):
        super().__init__("downstream")
        self.received_events: list[Event] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.received_events.append(event)
        return []


def _make_limiter(
    requests_per_window: int = 10,
    window_size: float = 1.0,
    queue_capacity: int = 10000,
) -> tuple[RateLimitedEntity, DummyDownstream, FixedWindowPolicy]:
    downstream = DummyDownstream()
    policy = FixedWindowPolicy(
        requests_per_window=requests_per_window,
        window_size=window_size,
    )
    limiter = RateLimitedEntity(
        name="test",
        downstream=downstream,
        policy=policy,
        queue_capacity=queue_capacity,
    )
    # Create simulation for clock injection
    Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(100.0),
        sources=[],
        entities=[limiter, downstream],
    )
    return limiter, downstream, policy


class TestFixedWindowCreation:
    """Tests for FixedWindowPolicy creation."""

    def test_creates_with_parameters(self):
        """Policy is created with specified parameters."""
        policy = FixedWindowPolicy(requests_per_window=100, window_size=1.0)
        assert policy.requests_per_window == 100
        assert policy.window_size == 1.0

    def test_rejects_zero_requests(self):
        """Rejects requests_per_window < 1."""
        with pytest.raises(ValueError):
            FixedWindowPolicy(requests_per_window=0)

    def test_rejects_negative_window_size(self):
        """Rejects window_size <= 0."""
        with pytest.raises(ValueError):
            FixedWindowPolicy(requests_per_window=10, window_size=0)


class TestFixedWindowForwarding:
    """Tests for request forwarding behavior."""

    def test_forwards_under_limit(self):
        """Requests under the limit are forwarded."""
        limiter, _downstream, _policy = _make_limiter(requests_per_window=5)

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            result = limiter.handle_event(event)
            assert any(e.event_type.startswith("forward::") for e in result)

        assert limiter.stats.forwarded == 5
        assert limiter.stats.dropped == 0

    def test_queues_over_limit(self):
        """Requests over the limit are queued (not dropped)."""
        limiter, _downstream, _policy = _make_limiter(requests_per_window=3)

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert limiter.stats.forwarded == 3
        assert limiter.stats.queued == 2
        assert limiter.stats.dropped == 0

    def test_forward_event_targets_downstream(self):
        """Forwarded events target the downstream entity."""
        limiter, downstream, _policy = _make_limiter()

        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)

        forward_events = [e for e in result if e.event_type.startswith("forward::")]
        assert len(forward_events) == 1
        assert forward_events[0].target is downstream

    def test_forward_event_copies_context(self):
        """Forwarded events copy the original context."""
        limiter, _downstream, _policy = _make_limiter()

        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
            context={"user": "alice", "data": [1, 2, 3]},
        )
        result = limiter.handle_event(event)

        forward_events = [e for e in result if e.event_type.startswith("forward::")]
        assert forward_events[0].context["user"] == "alice"
        assert forward_events[0].context["data"] == [1, 2, 3]


class TestFixedWindowReset:
    """Tests for window reset behavior."""

    def test_resets_on_new_window(self):
        """Counter resets when moving to a new window."""
        limiter, _downstream, _policy = _make_limiter(requests_per_window=2)

        # Fill first window
        for i in range(2):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert limiter.stats.forwarded == 2

        # Third request in first window - queued
        event = Event(
            time=Instant.from_seconds(0.5),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)
        assert limiter.stats.queued == 1

        # First request in second window - allowed
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)
        forward_events = [e for e in result if e.event_type.startswith("forward::")]
        assert len(forward_events) == 1


class TestFixedWindowStatistics:
    """Tests for statistics tracking."""

    def test_tracks_all_stats(self):
        """Statistics track all request outcomes."""
        limiter, _downstream, _policy = _make_limiter(requests_per_window=2)

        for i in range(4):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert limiter.stats.received == 4
        assert limiter.stats.forwarded == 2
        assert limiter.stats.queued == 2
        assert limiter.stats.dropped == 0

    def test_tracks_time_series(self):
        """Time series data is recorded."""
        limiter, _downstream, _policy = _make_limiter(requests_per_window=2)

        for i in range(3):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        assert len(limiter.received_times) == 3
        assert len(limiter.forwarded_times) == 2
