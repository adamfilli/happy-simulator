"""Tests for PhiAccrualDetector."""

from happysimulator.components.consensus import PhiAccrualDetector


class TestPhiComputation:
    """Tests for phi value computation."""

    def test_no_heartbeats_phi_zero(self):
        """Phi returns 0 with no heartbeat data."""
        detector = PhiAccrualDetector()

        assert detector.phi(now_s=10.0) == 0.0

    def test_single_heartbeat_phi_zero(self):
        """Phi returns 0 after a single heartbeat (no intervals yet)."""
        detector = PhiAccrualDetector()
        detector.heartbeat(1.0)

        assert detector.phi(now_s=2.0) == 0.0

    def test_regular_heartbeats_low_phi(self):
        """Regular heartbeat intervals produce a low phi value."""
        detector = PhiAccrualDetector()

        # Send heartbeats at regular 1-second intervals
        for t in range(20):
            detector.heartbeat(float(t))

        # Check phi just slightly after the last heartbeat
        phi = detector.phi(now_s=19.5)

        assert phi < 1.0

    def test_long_silence_high_phi(self):
        """A long gap after regular heartbeats produces a high phi value."""
        detector = PhiAccrualDetector()

        # Establish regular 1-second intervals
        for t in range(20):
            detector.heartbeat(float(t))

        # Check phi after a very long silence
        phi = detector.phi(now_s=100.0)

        assert phi > 8.0


class TestAvailability:
    """Tests for is_available checks."""

    def test_is_available_true(self):
        """Node is available when checked within normal heartbeat interval."""
        detector = PhiAccrualDetector(threshold=8.0)

        for t in range(10):
            detector.heartbeat(float(t))

        assert detector.is_available(now_s=9.5) is True

    def test_is_available_false(self):
        """Node is unavailable after a long silence."""
        detector = PhiAccrualDetector(threshold=8.0)

        for t in range(10):
            detector.heartbeat(float(t))

        assert detector.is_available(now_s=100.0) is False


class TestStatsAt:
    """Tests for statistics snapshots."""

    def test_stats_at(self):
        """stats_at returns a snapshot with the correct phi and metadata."""
        detector = PhiAccrualDetector(threshold=8.0)

        for t in range(10):
            detector.heartbeat(float(t))

        stats = detector.stats_at(now_s=9.5)

        assert stats.heartbeats_received == 10
        assert stats.current_phi >= 0.0
        assert stats.mean_interval > 0.0
        assert isinstance(stats.is_suspected, bool)


class TestSlidingWindow:
    """Tests for the sliding window of intervals."""

    def test_sliding_window_max_size(self):
        """Old intervals are dropped when the window exceeds max_sample_size."""
        detector = PhiAccrualDetector(max_sample_size=5)

        # Send many heartbeats -- only the last 5 intervals should be kept
        for t in range(20):
            detector.heartbeat(float(t))

        # Verify internally: the detector should have at most max_sample_size intervals
        assert len(detector._intervals) == 5


class TestConfiguration:
    """Tests for detector configuration."""

    def test_custom_threshold(self):
        """Detector uses a custom threshold for availability checks."""
        low_threshold = PhiAccrualDetector(threshold=1.0)
        high_threshold = PhiAccrualDetector(threshold=100.0)

        for t in range(10):
            low_threshold.heartbeat(float(t))
            high_threshold.heartbeat(float(t))

        # At t=11.0 (2s after last heartbeat), phi is ~23.
        # Low threshold (1.0) should suspect; high threshold (100.0) should not.
        now = 11.0
        assert low_threshold.is_available(now_s=now) is False
        assert high_threshold.is_available(now_s=now) is True

    def test_initial_interval_bootstrap(self):
        """Initial interval bootstraps the detector with a baseline interval."""
        detector = PhiAccrualDetector(initial_interval=1.0)

        # With an initial interval pre-loaded, the first heartbeat enables phi
        detector.heartbeat(0.0)

        # Now phi should be computable (we have 1 interval from bootstrap)
        phi = detector.phi(now_s=0.5)
        assert phi >= 0.0
        # Should be low since 0.5s is within the 1.0s initial interval
        assert phi < 3.0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_negative_elapsed_returns_zero(self):
        """Phi returns 0 when now_s is before the last heartbeat."""
        detector = PhiAccrualDetector(initial_interval=1.0)
        detector.heartbeat(10.0)

        phi = detector.phi(now_s=5.0)

        assert phi == 0.0
