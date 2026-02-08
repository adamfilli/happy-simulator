"""Unit tests for rate limiter policies.

Tests each policy's try_acquire() and time_until_available() in isolation
without running a simulation.
"""

from __future__ import annotations

import pytest

from happysimulator.components.rate_limiter.policy import (
    AdaptivePolicy,
    FixedWindowPolicy,
    LeakyBucketPolicy,
    RateAdjustmentReason,
    RateLimiterPolicy,
    SlidingWindowPolicy,
    TokenBucketPolicy,
)
from happysimulator.core.temporal import Duration, Instant


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_all_policies_implement_protocol():
    """All concrete policies satisfy the RateLimiterPolicy protocol."""
    policies = [
        TokenBucketPolicy(),
        LeakyBucketPolicy(),
        SlidingWindowPolicy(),
        FixedWindowPolicy(requests_per_window=5),
        AdaptivePolicy(),
    ]
    for p in policies:
        assert isinstance(p, RateLimiterPolicy)


# ---------------------------------------------------------------------------
# TokenBucketPolicy
# ---------------------------------------------------------------------------

class TestTokenBucketPolicy:

    def test_acquire_within_capacity(self):
        p = TokenBucketPolicy(capacity=3.0, refill_rate=1.0, initial_tokens=3.0)
        t = Instant.Epoch
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is False

    def test_refill_over_time(self):
        p = TokenBucketPolicy(capacity=5.0, refill_rate=2.0, initial_tokens=0.0)
        # At t=0, no tokens
        assert p.try_acquire(Instant.Epoch) is False
        # At t=1, 2 tokens refilled
        assert p.try_acquire(Instant.from_seconds(1.0)) is True
        assert p.try_acquire(Instant.from_seconds(1.0)) is True
        assert p.try_acquire(Instant.from_seconds(1.0)) is False

    def test_time_until_available_zero_when_tokens(self):
        p = TokenBucketPolicy(capacity=5.0, refill_rate=1.0, initial_tokens=5.0)
        assert p.time_until_available(Instant.Epoch) == Duration.ZERO

    def test_time_until_available_when_empty(self):
        p = TokenBucketPolicy(capacity=5.0, refill_rate=2.0, initial_tokens=0.0)
        wait = p.time_until_available(Instant.Epoch)
        assert wait > Duration.ZERO
        # Need 1 token, refill_rate=2 => 0.5s
        assert abs(wait.to_seconds() - 0.5) < 1e-6

    def test_capacity_caps_refill(self):
        p = TokenBucketPolicy(capacity=3.0, refill_rate=100.0, initial_tokens=0.0)
        # First call initializes refill timer at t=0, no tokens yet
        assert p.try_acquire(Instant.Epoch) is False
        # After 1 second at rate=100, would be 100 tokens but cap at 3
        assert p.try_acquire(Instant.from_seconds(1.0)) is True
        assert p.try_acquire(Instant.from_seconds(1.0)) is True
        assert p.try_acquire(Instant.from_seconds(1.0)) is True
        assert p.try_acquire(Instant.from_seconds(1.0)) is False


# ---------------------------------------------------------------------------
# LeakyBucketPolicy
# ---------------------------------------------------------------------------

class TestLeakyBucketPolicy:

    def test_first_acquire_succeeds(self):
        p = LeakyBucketPolicy(leak_rate=2.0)
        assert p.try_acquire(Instant.Epoch) is True

    def test_too_fast_denied(self):
        p = LeakyBucketPolicy(leak_rate=2.0)  # 0.5s interval
        assert p.try_acquire(Instant.Epoch) is True
        assert p.try_acquire(Instant.from_seconds(0.1)) is False

    def test_after_interval_allowed(self):
        p = LeakyBucketPolicy(leak_rate=2.0)  # 0.5s interval
        assert p.try_acquire(Instant.Epoch) is True
        assert p.try_acquire(Instant.from_seconds(0.5)) is True

    def test_time_until_available_initial(self):
        p = LeakyBucketPolicy(leak_rate=2.0)
        assert p.time_until_available(Instant.Epoch) == Duration.ZERO

    def test_time_until_available_after_acquire(self):
        p = LeakyBucketPolicy(leak_rate=2.0)
        p.try_acquire(Instant.Epoch)
        wait = p.time_until_available(Instant.from_seconds(0.1))
        # 0.5 - 0.1 = 0.4s remaining
        assert abs(wait.to_seconds() - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# SlidingWindowPolicy
# ---------------------------------------------------------------------------

class TestSlidingWindowPolicy:

    def test_within_limit(self):
        p = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=3)
        t = Instant.Epoch
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is False

    def test_window_expiry(self):
        p = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=2)
        assert p.try_acquire(Instant.Epoch) is True
        assert p.try_acquire(Instant.from_seconds(0.3)) is True
        assert p.try_acquire(Instant.from_seconds(0.5)) is False
        # After window slides past first request
        assert p.try_acquire(Instant.from_seconds(1.1)) is True

    def test_time_until_available_when_full(self):
        p = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=1)
        p.try_acquire(Instant.from_seconds(2.0))
        wait = p.time_until_available(Instant.from_seconds(2.5))
        # Oldest at t=2.0 expires at t=3.0 => 0.5s wait
        assert abs(wait.to_seconds() - 0.5) < 1e-6

    def test_time_until_available_zero_when_empty(self):
        p = SlidingWindowPolicy(window_size_seconds=1.0, max_requests=5)
        assert p.time_until_available(Instant.Epoch) == Duration.ZERO


# ---------------------------------------------------------------------------
# FixedWindowPolicy
# ---------------------------------------------------------------------------

class TestFixedWindowPolicy:

    def test_validation(self):
        with pytest.raises(ValueError):
            FixedWindowPolicy(requests_per_window=0)
        with pytest.raises(ValueError):
            FixedWindowPolicy(requests_per_window=5, window_size=0)

    def test_within_window(self):
        p = FixedWindowPolicy(requests_per_window=3, window_size=1.0)
        t = Instant.from_seconds(0.5)
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is True
        assert p.try_acquire(t) is False

    def test_window_reset(self):
        p = FixedWindowPolicy(requests_per_window=2, window_size=1.0)
        assert p.try_acquire(Instant.from_seconds(0.5)) is True
        assert p.try_acquire(Instant.from_seconds(0.8)) is True
        assert p.try_acquire(Instant.from_seconds(0.9)) is False
        # New window
        assert p.try_acquire(Instant.from_seconds(1.0)) is True

    def test_time_until_available_next_window(self):
        p = FixedWindowPolicy(requests_per_window=1, window_size=1.0)
        p.try_acquire(Instant.from_seconds(0.3))
        wait = p.time_until_available(Instant.from_seconds(0.3))
        # Window [0, 1) => next window at 1.0 => 0.7s wait
        assert abs(wait.to_seconds() - 0.7) < 1e-6


# ---------------------------------------------------------------------------
# AdaptivePolicy
# ---------------------------------------------------------------------------

class TestAdaptivePolicy:

    def test_validation(self):
        with pytest.raises(ValueError):
            AdaptivePolicy(min_rate=0)
        with pytest.raises(ValueError):
            AdaptivePolicy(initial_rate=200.0, max_rate=100.0)
        with pytest.raises(ValueError):
            AdaptivePolicy(decrease_factor=0)
        with pytest.raises(ValueError):
            AdaptivePolicy(decrease_factor=1.0)

    def test_basic_acquire(self):
        p = AdaptivePolicy(initial_rate=10.0, window_size=1.0)
        # Initial tokens = 10 * 1 = 10
        for _ in range(10):
            assert p.try_acquire(Instant.Epoch) is True
        assert p.try_acquire(Instant.Epoch) is False

    def test_rate_increase_on_success(self):
        p = AdaptivePolicy(initial_rate=10.0, increase_step=5.0)
        old_rate = p.current_rate
        p.record_success(Instant.Epoch)
        assert p.current_rate == old_rate + 5.0
        assert p.successes == 1
        assert p.rate_increases == 1

    def test_rate_decrease_on_failure(self):
        p = AdaptivePolicy(initial_rate=100.0, decrease_factor=0.5)
        p.record_failure(Instant.Epoch)
        assert p.current_rate == 50.0
        assert p.failures == 1
        assert p.rate_decreases == 1

    def test_rate_bounded(self):
        p = AdaptivePolicy(initial_rate=10.0, min_rate=5.0, max_rate=20.0, increase_step=100.0)
        p.record_success(Instant.Epoch)
        assert p.current_rate == 20.0  # Capped at max

        p2 = AdaptivePolicy(initial_rate=10.0, min_rate=5.0, decrease_factor=0.01)
        p2.record_failure(Instant.Epoch)
        assert p2.current_rate == 5.0  # Floored at min

    def test_timeout_counted(self):
        p = AdaptivePolicy(initial_rate=100.0, decrease_factor=0.5)
        p.record_failure(Instant.Epoch, reason=RateAdjustmentReason.TIMEOUT)
        assert p.timeouts == 1
        assert p.failures == 0

    def test_rate_history_tracked(self):
        p = AdaptivePolicy(initial_rate=10.0, increase_step=2.0, decrease_factor=0.5)
        p.record_success(Instant.Epoch)
        p.record_failure(Instant.from_seconds(1.0))
        assert len(p.rate_history) == 2
        assert p.rate_history[0].reason == RateAdjustmentReason.SUCCESS
        assert p.rate_history[1].reason == RateAdjustmentReason.FAILURE

    def test_time_until_available(self):
        p = AdaptivePolicy(initial_rate=2.0, window_size=1.0)
        # tokens = 2 initially
        p.try_acquire(Instant.Epoch)
        p.try_acquire(Instant.Epoch)
        # Now empty
        wait = p.time_until_available(Instant.Epoch)
        assert wait > Duration.ZERO
        # Need 1 token at rate 2/s => 0.5s
        assert abs(wait.to_seconds() - 0.5) < 1e-6
