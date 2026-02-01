"""Tests for AdaptiveRateLimiter."""

import pytest

from happysimulator.components.rate_limiter import (
    AdaptiveRateLimiter,
    RateAdjustmentReason,
)
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


class TestAdaptiveCreation:
    """Tests for AdaptiveRateLimiter creation."""

    def test_creates_with_parameters(self):
        """Rate limiter is created with specified parameters."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
            min_rate=10.0,
            max_rate=1000.0,
        )

        assert limiter.name == "test"
        assert limiter.downstream is downstream
        assert limiter.current_rate == 100.0
        assert limiter.min_rate == 10.0
        assert limiter.max_rate == 1000.0

    def test_rejects_invalid_min_rate(self):
        """Rejects min_rate <= 0."""
        downstream = DummyDownstream()
        with pytest.raises(ValueError):
            AdaptiveRateLimiter(
                name="test",
                downstream=downstream,
                initial_rate=100.0,
                min_rate=0,
            )

    def test_rejects_max_less_than_min(self):
        """Rejects max_rate < min_rate."""
        downstream = DummyDownstream()
        with pytest.raises(ValueError):
            AdaptiveRateLimiter(
                name="test",
                downstream=downstream,
                initial_rate=100.0,
                min_rate=100.0,
                max_rate=50.0,
            )

    def test_rejects_initial_outside_range(self):
        """Rejects initial_rate outside [min_rate, max_rate]."""
        downstream = DummyDownstream()
        with pytest.raises(ValueError):
            AdaptiveRateLimiter(
                name="test",
                downstream=downstream,
                initial_rate=5.0,
                min_rate=10.0,
                max_rate=100.0,
            )

    def test_rejects_invalid_decrease_factor(self):
        """Rejects decrease_factor outside (0, 1)."""
        downstream = DummyDownstream()
        with pytest.raises(ValueError):
            AdaptiveRateLimiter(
                name="test",
                downstream=downstream,
                initial_rate=100.0,
                decrease_factor=1.0,
            )

    def test_default_increase_step(self):
        """Default increase_step is 10% of initial_rate."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
        )

        # Record success should increase by default step (10)
        initial = limiter.current_rate
        limiter.record_success()
        assert limiter.current_rate == initial + 10.0


class TestAdaptiveRateAdjustment:
    """Tests for rate adjustment behavior."""

    def test_success_increases_rate(self):
        """Recording success increases the rate."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
            increase_step=10.0,
            max_rate=1000.0,
        )

        initial = limiter.current_rate
        limiter.record_success()

        assert limiter.current_rate == initial + 10.0
        assert limiter.stats.successes == 1
        assert limiter.stats.rate_increases == 1

    def test_failure_decreases_rate(self):
        """Recording failure decreases the rate."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
            decrease_factor=0.5,
            min_rate=10.0,
        )

        initial = limiter.current_rate
        limiter.record_failure()

        assert limiter.current_rate == initial * 0.5
        assert limiter.stats.failures == 1
        assert limiter.stats.rate_decreases == 1

    def test_timeout_decreases_rate(self):
        """Recording timeout decreases the rate."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
            decrease_factor=0.5,
        )

        initial = limiter.current_rate
        limiter.record_failure(reason=RateAdjustmentReason.TIMEOUT)

        assert limiter.current_rate == initial * 0.5
        assert limiter.stats.timeouts == 1

    def test_rate_capped_at_max(self):
        """Rate cannot exceed max_rate."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=95.0,
            increase_step=10.0,
            max_rate=100.0,
        )

        limiter.record_success()
        assert limiter.current_rate == 100.0

        limiter.record_success()
        assert limiter.current_rate == 100.0  # Still capped

    def test_rate_floored_at_min(self):
        """Rate cannot go below min_rate."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=15.0,
            decrease_factor=0.5,
            min_rate=10.0,
        )

        limiter.record_failure()
        assert limiter.current_rate == 10.0  # Floored at min

        limiter.record_failure()
        assert limiter.current_rate == 10.0  # Still floored

    def test_aimd_pattern(self):
        """Rate follows AIMD pattern (slow increase, fast decrease)."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
            increase_step=5.0,
            decrease_factor=0.5,
            min_rate=10.0,
            max_rate=200.0,
        )

        # Several successes
        for _ in range(10):
            limiter.record_success()

        after_success = limiter.current_rate  # Should be 150

        # One failure
        limiter.record_failure()

        after_failure = limiter.current_rate  # Should be 75

        assert after_success == 150.0
        assert after_failure == 75.0


class TestAdaptiveForwarding:
    """Tests for request forwarding with adaptive rate."""

    def test_forwards_with_tokens(self):
        """Requests are forwarded when tokens available."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=10.0,  # 10 requests/sec
            window_size=1.0,
        )

        # Should start with tokens
        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)

        assert len(result) == 1
        assert limiter.stats.requests_forwarded == 1

    def test_drops_without_tokens(self):
        """Requests are dropped when no tokens available."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=2.0,  # 2 requests/sec
            window_size=1.0,
        )

        # Exhaust tokens
        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.01 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        # First 2 should be forwarded, rest dropped
        assert limiter.stats.requests_forwarded == 2
        assert limiter.stats.requests_dropped == 3

    def test_tokens_refill_over_time(self):
        """Tokens refill based on current rate and elapsed time."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=10.0,
            window_size=1.0,
        )

        # Use all tokens at t=0
        for i in range(10):
            event = Event(
                time=Instant.from_seconds(0),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        # Wait 0.5 seconds - should have ~5 tokens
        event = Event(
            time=Instant.from_seconds(0.5),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)
        assert len(result) == 1  # Should have refilled


class TestAdaptiveRateHistory:
    """Tests for rate history tracking."""

    def test_records_rate_changes(self):
        """Rate history records changes with timestamps."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
            increase_step=10.0,
            decrease_factor=0.5,
        )

        now = Instant.from_seconds(1.0)
        limiter.record_success(now)
        limiter.record_failure(now)

        assert len(limiter.rate_history) == 2
        assert limiter.rate_history[0].rate == 110.0
        assert limiter.rate_history[0].reason == RateAdjustmentReason.SUCCESS
        assert limiter.rate_history[1].rate == 55.0
        assert limiter.rate_history[1].reason == RateAdjustmentReason.FAILURE


class TestAdaptiveStatistics:
    """Tests for statistics tracking."""

    def test_tracks_all_stats(self):
        """Statistics track all relevant counts."""
        downstream = DummyDownstream()
        limiter = AdaptiveRateLimiter(
            name="test",
            downstream=downstream,
            initial_rate=100.0,
        )

        # Some requests
        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        # Some feedback
        limiter.record_success()
        limiter.record_success()
        limiter.record_failure()
        limiter.record_failure(reason=RateAdjustmentReason.TIMEOUT)

        assert limiter.stats.requests_received == 5
        assert limiter.stats.successes == 2
        assert limiter.stats.failures == 1
        assert limiter.stats.timeouts == 1
        assert limiter.stats.rate_increases == 2
        assert limiter.stats.rate_decreases == 2
