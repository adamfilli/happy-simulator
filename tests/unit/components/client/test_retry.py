"""Tests for retry policies."""

from __future__ import annotations

import random

import pytest

from happysimulator.components.client.retry import (
    RetryPolicy,
    NoRetry,
    FixedRetry,
    ExponentialBackoff,
    DecorrelatedJitter,
)


class TestNoRetry:
    """Tests for NoRetry policy."""

    def test_never_retries(self):
        """NoRetry always returns False for should_retry."""
        policy = NoRetry()

        assert policy.should_retry(1) is False
        assert policy.should_retry(2) is False
        assert policy.should_retry(100) is False

    def test_delay_is_zero(self):
        """NoRetry always returns 0 delay."""
        policy = NoRetry()

        assert policy.get_delay(1) == 0.0
        assert policy.get_delay(2) == 0.0

    def test_satisfies_protocol(self):
        """NoRetry satisfies RetryPolicy protocol."""
        policy = NoRetry()
        assert isinstance(policy, RetryPolicy)


class TestFixedRetry:
    """Tests for FixedRetry policy."""

    def test_creates_with_valid_parameters(self):
        """FixedRetry creates with valid parameters."""
        policy = FixedRetry(max_attempts=3, delay=0.5)
        assert policy.max_attempts == 3
        assert policy.delay == 0.5

    def test_rejects_invalid_max_attempts(self):
        """FixedRetry rejects max_attempts < 1."""
        with pytest.raises(ValueError):
            FixedRetry(max_attempts=0, delay=0.5)

        with pytest.raises(ValueError):
            FixedRetry(max_attempts=-1, delay=0.5)

    def test_rejects_negative_delay(self):
        """FixedRetry rejects negative delay."""
        with pytest.raises(ValueError):
            FixedRetry(max_attempts=3, delay=-0.1)

    def test_allows_zero_delay(self):
        """FixedRetry allows zero delay (immediate retry)."""
        policy = FixedRetry(max_attempts=3, delay=0.0)
        assert policy.delay == 0.0

    def test_should_retry_respects_max_attempts(self):
        """should_retry returns True until max_attempts reached."""
        policy = FixedRetry(max_attempts=3, delay=0.5)

        # Attempt 1: should retry (have 2 more attempts)
        assert policy.should_retry(1) is True
        # Attempt 2: should retry (have 1 more attempt)
        assert policy.should_retry(2) is True
        # Attempt 3: should NOT retry (at max)
        assert policy.should_retry(3) is False
        # Beyond max
        assert policy.should_retry(4) is False

    def test_delay_is_constant(self):
        """get_delay returns the same value regardless of attempt."""
        policy = FixedRetry(max_attempts=5, delay=1.5)

        assert policy.get_delay(1) == 1.5
        assert policy.get_delay(2) == 1.5
        assert policy.get_delay(5) == 1.5

    def test_satisfies_protocol(self):
        """FixedRetry satisfies RetryPolicy protocol."""
        policy = FixedRetry(max_attempts=3, delay=0.5)
        assert isinstance(policy, RetryPolicy)


class TestExponentialBackoff:
    """Tests for ExponentialBackoff policy."""

    def test_creates_with_valid_parameters(self):
        """ExponentialBackoff creates with valid parameters."""
        policy = ExponentialBackoff(
            max_attempts=5,
            initial_delay=0.1,
            max_delay=10.0,
            multiplier=2.0,
            jitter=0.05,
        )
        assert policy.max_attempts == 5
        assert policy.initial_delay == 0.1
        assert policy.max_delay == 10.0
        assert policy.multiplier == 2.0
        assert policy.jitter == 0.05

    def test_rejects_invalid_max_attempts(self):
        """ExponentialBackoff rejects max_attempts < 1."""
        with pytest.raises(ValueError):
            ExponentialBackoff(max_attempts=0, initial_delay=0.1, max_delay=10.0)

    def test_rejects_non_positive_initial_delay(self):
        """ExponentialBackoff rejects initial_delay <= 0."""
        with pytest.raises(ValueError):
            ExponentialBackoff(max_attempts=3, initial_delay=0, max_delay=10.0)

        with pytest.raises(ValueError):
            ExponentialBackoff(max_attempts=3, initial_delay=-0.1, max_delay=10.0)

    def test_rejects_max_delay_less_than_initial(self):
        """ExponentialBackoff rejects max_delay < initial_delay."""
        with pytest.raises(ValueError):
            ExponentialBackoff(max_attempts=3, initial_delay=1.0, max_delay=0.5)

    def test_rejects_multiplier_less_than_one(self):
        """ExponentialBackoff rejects multiplier < 1."""
        with pytest.raises(ValueError):
            ExponentialBackoff(
                max_attempts=3, initial_delay=0.1, max_delay=10.0, multiplier=0.5
            )

    def test_rejects_negative_jitter(self):
        """ExponentialBackoff rejects negative jitter."""
        with pytest.raises(ValueError):
            ExponentialBackoff(
                max_attempts=3, initial_delay=0.1, max_delay=10.0, jitter=-0.1
            )

    def test_should_retry_respects_max_attempts(self):
        """should_retry returns True until max_attempts reached."""
        policy = ExponentialBackoff(max_attempts=3, initial_delay=0.1, max_delay=10.0)

        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False

    def test_delay_increases_exponentially(self):
        """get_delay increases exponentially with each attempt."""
        policy = ExponentialBackoff(
            max_attempts=10,
            initial_delay=0.1,
            max_delay=100.0,
            multiplier=2.0,
            jitter=0.0,  # No jitter for predictable testing
        )

        # attempt=2 (first retry): initial_delay = 0.1
        assert policy.get_delay(2) == pytest.approx(0.1, rel=0.01)
        # attempt=3: 0.1 * 2 = 0.2
        assert policy.get_delay(3) == pytest.approx(0.2, rel=0.01)
        # attempt=4: 0.1 * 4 = 0.4
        assert policy.get_delay(4) == pytest.approx(0.4, rel=0.01)
        # attempt=5: 0.1 * 8 = 0.8
        assert policy.get_delay(5) == pytest.approx(0.8, rel=0.01)

    def test_delay_capped_at_max(self):
        """get_delay is capped at max_delay."""
        policy = ExponentialBackoff(
            max_attempts=20,
            initial_delay=1.0,
            max_delay=5.0,
            multiplier=2.0,
            jitter=0.0,
        )

        # Early attempts below cap
        assert policy.get_delay(2) == pytest.approx(1.0, rel=0.01)
        assert policy.get_delay(3) == pytest.approx(2.0, rel=0.01)
        assert policy.get_delay(4) == pytest.approx(4.0, rel=0.01)
        # Later attempts hit cap
        assert policy.get_delay(5) == pytest.approx(5.0, rel=0.01)  # Would be 8.0
        assert policy.get_delay(10) == pytest.approx(5.0, rel=0.01)  # Would be much higher

    def test_jitter_adds_randomness(self):
        """get_delay adds jitter to the calculated delay."""
        random.seed(42)

        policy = ExponentialBackoff(
            max_attempts=10,
            initial_delay=1.0,
            max_delay=100.0,
            multiplier=2.0,
            jitter=0.5,  # Add up to 0.5s jitter
        )

        # Get multiple delays for same attempt - should vary due to jitter
        delays = [policy.get_delay(2) for _ in range(10)]

        # All should be between 1.0 and 1.5 (initial + 0 to 0.5 jitter)
        for d in delays:
            assert 1.0 <= d <= 1.5

        # Should have some variation (not all identical)
        assert len(set(delays)) > 1

    def test_satisfies_protocol(self):
        """ExponentialBackoff satisfies RetryPolicy protocol."""
        policy = ExponentialBackoff(max_attempts=3, initial_delay=0.1, max_delay=10.0)
        assert isinstance(policy, RetryPolicy)


class TestDecorrelatedJitter:
    """Tests for DecorrelatedJitter policy."""

    def test_creates_with_valid_parameters(self):
        """DecorrelatedJitter creates with valid parameters."""
        policy = DecorrelatedJitter(
            max_attempts=5,
            base_delay=0.1,
            max_delay=10.0,
        )
        assert policy.max_attempts == 5
        assert policy.base_delay == 0.1
        assert policy.max_delay == 10.0

    def test_rejects_invalid_max_attempts(self):
        """DecorrelatedJitter rejects max_attempts < 1."""
        with pytest.raises(ValueError):
            DecorrelatedJitter(max_attempts=0, base_delay=0.1, max_delay=10.0)

    def test_rejects_non_positive_base_delay(self):
        """DecorrelatedJitter rejects base_delay <= 0."""
        with pytest.raises(ValueError):
            DecorrelatedJitter(max_attempts=3, base_delay=0, max_delay=10.0)

    def test_rejects_max_delay_less_than_base(self):
        """DecorrelatedJitter rejects max_delay < base_delay."""
        with pytest.raises(ValueError):
            DecorrelatedJitter(max_attempts=3, base_delay=1.0, max_delay=0.5)

    def test_should_retry_respects_max_attempts(self):
        """should_retry returns True until max_attempts reached."""
        policy = DecorrelatedJitter(max_attempts=3, base_delay=0.1, max_delay=10.0)

        assert policy.should_retry(1) is True
        assert policy.should_retry(2) is True
        assert policy.should_retry(3) is False

    def test_delay_between_base_and_max(self):
        """get_delay returns values between base_delay and max_delay."""
        random.seed(42)

        policy = DecorrelatedJitter(
            max_attempts=100,
            base_delay=0.1,
            max_delay=10.0,
        )

        # Get multiple delays
        delays = [policy.get_delay(i) for i in range(2, 20)]

        # All should be within bounds
        for d in delays:
            assert 0.1 <= d <= 10.0

    def test_delay_is_decorrelated(self):
        """get_delay produces decorrelated values."""
        random.seed(42)

        policy = DecorrelatedJitter(
            max_attempts=100,
            base_delay=0.1,
            max_delay=100.0,
        )

        # Get sequence of delays
        delays = [policy.get_delay(i) for i in range(2, 12)]

        # Should have variation (not monotonically increasing like exponential)
        # Check that delays can decrease
        has_decrease = any(delays[i] < delays[i-1] for i in range(1, len(delays)))
        assert has_decrease, "Decorrelated jitter should sometimes decrease"

    def test_reset_restores_initial_state(self):
        """reset() restores the policy to initial state."""
        random.seed(42)

        policy = DecorrelatedJitter(
            max_attempts=10,
            base_delay=0.1,
            max_delay=10.0,
        )

        # Get some delays to change internal state
        for i in range(2, 5):
            policy.get_delay(i)

        # Reset
        policy.reset()

        # First delay after reset should be between base and 3*base
        # (since previous_delay is reset to base_delay)
        random.seed(42)  # Same seed for comparison
        d = policy.get_delay(2)
        assert 0.1 <= d <= 0.3  # base to min(max, base*3)

    def test_satisfies_protocol(self):
        """DecorrelatedJitter satisfies RetryPolicy protocol."""
        policy = DecorrelatedJitter(max_attempts=3, base_delay=0.1, max_delay=10.0)
        assert isinstance(policy, RetryPolicy)
