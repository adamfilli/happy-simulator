"""Tests for AdaptivePolicy + RateLimitedEntity."""

import pytest

from happysimulator.components.rate_limiter import (
    AdaptivePolicy,
    RateAdjustmentReason,
    RateLimitedEntity,
)
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
    initial_rate: float = 100.0,
    min_rate: float = 1.0,
    max_rate: float = 10000.0,
    increase_step: float | None = None,
    decrease_factor: float = 0.5,
    window_size: float = 1.0,
    queue_capacity: int = 10000,
) -> tuple[RateLimitedEntity, DummyDownstream, AdaptivePolicy]:
    downstream = DummyDownstream()
    policy = AdaptivePolicy(
        initial_rate=initial_rate,
        min_rate=min_rate,
        max_rate=max_rate,
        increase_step=increase_step,
        decrease_factor=decrease_factor,
        window_size=window_size,
    )
    limiter = RateLimitedEntity(
        name="test",
        downstream=downstream,
        policy=policy,
        queue_capacity=queue_capacity,
    )
    Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(100.0),
        sources=[],
        entities=[limiter, downstream],
    )
    return limiter, downstream, policy


class TestAdaptiveCreation:
    """Tests for AdaptivePolicy creation."""

    def test_creates_with_parameters(self):
        """Policy is created with specified parameters."""
        policy = AdaptivePolicy(
            initial_rate=100.0,
            min_rate=10.0,
            max_rate=1000.0,
        )
        assert policy.current_rate == 100.0
        assert policy.min_rate == 10.0
        assert policy.max_rate == 1000.0

    def test_rejects_invalid_min_rate(self):
        """Rejects min_rate <= 0."""
        with pytest.raises(ValueError):
            AdaptivePolicy(initial_rate=100.0, min_rate=0)

    def test_rejects_max_less_than_min(self):
        """Rejects max_rate < min_rate."""
        with pytest.raises(ValueError):
            AdaptivePolicy(initial_rate=100.0, min_rate=100.0, max_rate=50.0)

    def test_rejects_initial_outside_range(self):
        """Rejects initial_rate outside [min_rate, max_rate]."""
        with pytest.raises(ValueError):
            AdaptivePolicy(initial_rate=5.0, min_rate=10.0, max_rate=100.0)

    def test_rejects_invalid_decrease_factor(self):
        """Rejects decrease_factor outside (0, 1)."""
        with pytest.raises(ValueError):
            AdaptivePolicy(initial_rate=100.0, decrease_factor=1.0)

    def test_default_increase_step(self):
        """Default increase_step is 10% of initial_rate."""
        policy = AdaptivePolicy(initial_rate=100.0)
        initial = policy.current_rate
        policy.record_success(Instant.Epoch)
        assert policy.current_rate == initial + 10.0


class TestAdaptiveRateAdjustment:
    """Tests for rate adjustment behavior."""

    def test_success_increases_rate(self):
        """Recording success increases the rate."""
        policy = AdaptivePolicy(
            initial_rate=100.0,
            increase_step=10.0,
            max_rate=1000.0,
        )
        initial = policy.current_rate
        policy.record_success(Instant.Epoch)
        assert policy.current_rate == initial + 10.0
        assert policy.successes == 1
        assert policy.rate_increases == 1

    def test_failure_decreases_rate(self):
        """Recording failure decreases the rate."""
        policy = AdaptivePolicy(
            initial_rate=100.0,
            decrease_factor=0.5,
            min_rate=10.0,
        )
        initial = policy.current_rate
        policy.record_failure(Instant.Epoch)
        assert policy.current_rate == initial * 0.5
        assert policy.failures == 1
        assert policy.rate_decreases == 1

    def test_timeout_decreases_rate(self):
        """Recording timeout decreases the rate."""
        policy = AdaptivePolicy(
            initial_rate=100.0,
            decrease_factor=0.5,
        )
        initial = policy.current_rate
        policy.record_failure(Instant.Epoch, reason=RateAdjustmentReason.TIMEOUT)
        assert policy.current_rate == initial * 0.5
        assert policy.timeouts == 1

    def test_rate_capped_at_max(self):
        """Rate cannot exceed max_rate."""
        policy = AdaptivePolicy(
            initial_rate=95.0,
            increase_step=10.0,
            max_rate=100.0,
        )
        policy.record_success(Instant.Epoch)
        assert policy.current_rate == 100.0
        policy.record_success(Instant.from_seconds(1.0))
        assert policy.current_rate == 100.0  # Still capped

    def test_rate_floored_at_min(self):
        """Rate cannot go below min_rate."""
        policy = AdaptivePolicy(
            initial_rate=15.0,
            decrease_factor=0.5,
            min_rate=10.0,
        )
        policy.record_failure(Instant.Epoch)
        assert policy.current_rate == 10.0
        policy.record_failure(Instant.from_seconds(1.0))
        assert policy.current_rate == 10.0

    def test_aimd_pattern(self):
        """Rate follows AIMD pattern (slow increase, fast decrease)."""
        policy = AdaptivePolicy(
            initial_rate=100.0,
            increase_step=5.0,
            decrease_factor=0.5,
            min_rate=10.0,
            max_rate=200.0,
        )
        for i in range(10):
            policy.record_success(Instant.from_seconds(float(i)))
        after_success = policy.current_rate  # Should be 150

        policy.record_failure(Instant.from_seconds(10.0))
        after_failure = policy.current_rate  # Should be 75

        assert after_success == 150.0
        assert after_failure == 75.0


class TestAdaptiveForwarding:
    """Tests for request forwarding with adaptive rate."""

    def test_forwards_with_tokens(self):
        """Requests are forwarded when tokens available."""
        limiter, _downstream, _policy = _make_limiter(initial_rate=10.0, window_size=1.0)

        event = Event(
            time=Instant.from_seconds(0),
            event_type="request",
            target=limiter,
        )
        result = limiter.handle_event(event)
        forward_events = [e for e in result if e.event_type.startswith("forward::")]
        assert len(forward_events) == 1
        assert limiter.stats.forwarded == 1

    def test_queues_without_tokens(self):
        """Requests are queued when no tokens available."""
        limiter, _downstream, _policy = _make_limiter(initial_rate=2.0, window_size=1.0)

        # Exhaust tokens
        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.01 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        # First 2 forwarded, rest queued
        assert limiter.stats.forwarded == 2
        assert limiter.stats.queued == 3
        assert limiter.stats.dropped == 0

    def test_tokens_refill_over_time(self):
        """Tokens refill based on current rate and elapsed time."""
        limiter, _downstream, _policy = _make_limiter(initial_rate=10.0, window_size=1.0)

        # Use all tokens at t=0
        for _i in range(10):
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
        forward_events = [e for e in result if e.event_type.startswith("forward::")]
        assert len(forward_events) == 1


class TestAdaptiveRateHistory:
    """Tests for rate history tracking."""

    def test_records_rate_changes(self):
        """Rate history records changes with timestamps."""
        policy = AdaptivePolicy(
            initial_rate=100.0,
            increase_step=10.0,
            decrease_factor=0.5,
        )
        now = Instant.from_seconds(1.0)
        policy.record_success(now)
        policy.record_failure(now)

        assert len(policy.rate_history) == 2
        assert policy.rate_history[0].rate == 110.0
        assert policy.rate_history[0].reason == RateAdjustmentReason.SUCCESS
        assert policy.rate_history[1].rate == 55.0
        assert policy.rate_history[1].reason == RateAdjustmentReason.FAILURE


class TestAdaptiveStatistics:
    """Tests for statistics tracking."""

    def test_tracks_all_stats(self):
        """Statistics track all relevant counts."""
        limiter, _downstream, policy = _make_limiter(initial_rate=100.0)

        for i in range(5):
            event = Event(
                time=Instant.from_seconds(0.1 * i),
                event_type="request",
                target=limiter,
            )
            limiter.handle_event(event)

        policy.record_success(Instant.Epoch)
        policy.record_success(Instant.Epoch)
        policy.record_failure(Instant.Epoch)
        policy.record_failure(Instant.Epoch, reason=RateAdjustmentReason.TIMEOUT)

        assert limiter.stats.received == 5
        assert policy.successes == 2
        assert policy.failures == 1
        assert policy.timeouts == 1
        assert policy.rate_increases == 2
        assert policy.rate_decreases == 2
