"""Regression tests capturing expected arrival times from scipy implementation.

These values were captured from the working scipy implementation on 2026-01-31
and should remain constant after scipy removal.
"""

import pytest
from happysimulator.load.providers.constant_arrival import ConstantArrivalTimeProvider
from happysimulator.load.profile import ConstantRateProfile, LinearRampProfile, SpikeProfile
from happysimulator.core.temporal import Instant


class TestArrivalTimeRegression:
    """Golden value tests for arrival times.

    These values were captured from the scipy-based implementation and serve
    as the source of truth for verifying the scipy removal doesn't change behavior.
    """

    # Values captured from scipy implementation (2026-01-31)
    CONSTANT_RATE_50_FIRST_10 = [
        0.02,
        0.04,
        0.06,
        0.08,
        0.1,
        0.12,
        0.14,
        0.16,
        0.18,
        0.199999999,
    ]

    CONSTANT_RATE_100_FIRST_10 = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.099999999,
    ]

    LINEAR_RAMP_10_100_FIRST_10 = [
        0.095864499,
        0.184655976,
        0.267741515,
        0.346097448,
        0.420449859,
        0.49135612,
        0.55925515,
        0.624499924,
        0.687379335,
        0.748133387,
    ]

    LINEAR_RAMP_100_10_FIRST_10 = [
        0.010004504,
        0.020018032,
        0.030040609,
        0.040072259,
        0.050113007,
        0.060162878,
        0.070221897,
        0.080290089,
        0.090367479,
        0.100454092,
    ]

    SPIKE_PROFILE_FIRST_30 = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.799999999, 0.899999998, 0.999999998,
        1.099999998, 1.199999998, 1.299999998, 1.399999998, 1.499999998, 1.599999998,
        1.699999998, 1.799999998, 1.899999998, 1.999999997, 2.009999997, 2.019999996,
        2.029999996, 2.039999995, 2.049999994, 2.059999994, 2.069999993, 2.079999993,
        2.089999992, 2.099999991,
    ]

    def test_constant_rate_50_regression(self):
        """Verify constant rate=50 arrival times match captured values."""
        profile = ConstantRateProfile(rate=50.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for i, expected in enumerate(self.CONSTANT_RATE_50_FIRST_10):
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, (
                f"Event {i+1}: Expected {expected}, got {actual}"
            )

    def test_constant_rate_100_regression(self):
        """Verify constant rate=100 arrival times match captured values."""
        profile = ConstantRateProfile(rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for i, expected in enumerate(self.CONSTANT_RATE_100_FIRST_10):
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, (
                f"Event {i+1}: Expected {expected}, got {actual}"
            )

    def test_linear_ramp_up_regression(self):
        """Verify linear ramp-up (10→100) arrival times match captured values."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=10.0, end_rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for i, expected in enumerate(self.LINEAR_RAMP_10_100_FIRST_10):
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, (
                f"Event {i+1}: Expected {expected}, got {actual}"
            )

    def test_linear_ramp_down_regression(self):
        """Verify linear ramp-down (100→10) arrival times match captured values."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=100.0, end_rate=10.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for i, expected in enumerate(self.LINEAR_RAMP_100_10_FIRST_10):
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, (
                f"Event {i+1}: Expected {expected}, got {actual}"
            )

    def test_spike_profile_regression(self):
        """Verify spike profile arrival times match captured values.

        Tests a profile with:
        - baseline_rate=10 for t < 2.0s (warmup)
        - spike_rate=100 for 2.0s <= t < 3.0s (spike)
        - baseline_rate=10 for t >= 3.0s (recovery)
        """
        profile = SpikeProfile(
            baseline_rate=10.0,
            spike_rate=100.0,
            warmup_s=2.0,
            spike_duration_s=1.0
        )
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        for i, expected in enumerate(self.SPIKE_PROFILE_FIRST_30):
            actual = provider.next_arrival_time().to_seconds()
            assert abs(actual - expected) < 1e-8, (
                f"Event {i+1}: Expected {expected}, got {actual}"
            )


class TestArrivalTimeProperties:
    """Property-based tests for arrival time behavior."""

    def test_constant_rate_exact_spacing(self):
        """With constant rate r, events should be exactly 1/r apart."""
        for rate in [1.0, 10.0, 50.0, 100.0, 500.0]:
            profile = ConstantRateProfile(rate=rate)
            provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

            expected_interval = 1.0 / rate
            prev_time = 0.0

            for i in range(10):
                actual_time = provider.next_arrival_time().to_seconds()
                interval = actual_time - prev_time
                assert abs(interval - expected_interval) < 1e-8, (
                    f"Rate {rate}, Event {i+1}: Expected interval {expected_interval}, "
                    f"got {interval}"
                )
                prev_time = actual_time

    def test_ramp_up_events_accelerate(self):
        """With increasing rate, inter-arrival times should decrease."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=10.0, end_rate=100.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        times = [provider.next_arrival_time().to_seconds() for _ in range(20)]
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]

        # Each interval should be less than or equal to the previous (roughly)
        # We allow some tolerance due to numerical effects
        for i in range(1, len(intervals)):
            assert intervals[i] <= intervals[i-1] * 1.1, (
                f"Interval {i+1} ({intervals[i]}) should be <= interval {i} ({intervals[i-1]})"
            )

    def test_ramp_down_events_decelerate(self):
        """With decreasing rate, inter-arrival times should increase."""
        profile = LinearRampProfile(duration_s=10.0, start_rate=100.0, end_rate=10.0)
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        times = [provider.next_arrival_time().to_seconds() for _ in range(20)]
        intervals = [times[i] - times[i-1] for i in range(1, len(times))]

        # Each interval should be greater than or equal to the previous (roughly)
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i-1] * 0.9, (
                f"Interval {i+1} ({intervals[i]}) should be >= interval {i} ({intervals[i-1]})"
            )

    def test_spike_profile_rate_changes(self):
        """Verify spike profile shows rate changes at expected transitions."""
        profile = SpikeProfile(
            baseline_rate=10.0,
            spike_rate=100.0,
            warmup_s=2.0,
            spike_duration_s=1.0
        )
        provider = ConstantArrivalTimeProvider(profile, start_time=Instant.Epoch)

        # Collect many events - need 200+ to cover all phases
        # (20 warmup + 100 spike + 10 recovery = 130 events before t=4)
        times = []
        for _ in range(200):
            t = provider.next_arrival_time().to_seconds()
            if t > 4.0:  # Stop after spike ends and some recovery
                break
            times.append(t)

        # Count events in each phase
        warmup_events = sum(1 for t in times if t < 2.0)
        spike_events = sum(1 for t in times if 2.0 <= t < 3.0)
        recovery_events = sum(1 for t in times if 3.0 <= t < 4.0)

        # Warmup: rate=10, so ~20 events in 2s
        assert 18 <= warmup_events <= 22, f"Expected ~20 warmup events, got {warmup_events}"

        # Spike: rate=100, so ~100 events in 1s
        assert 90 <= spike_events <= 110, f"Expected ~100 spike events, got {spike_events}"

        # Recovery: rate=10, so ~10 events in 1s
        assert 8 <= recovery_events <= 12, f"Expected ~10 recovery events, got {recovery_events}"
