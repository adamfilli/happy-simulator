"""Unit tests for per-node clocks and clock models."""

from happysimulator.core.clock import Clock
from happysimulator.core.node_clock import (
    ClockModel,
    FixedSkew,
    LinearDrift,
    NodeClock,
)
from happysimulator.core.temporal import Duration, Instant

import pytest


# ---------------------------------------------------------------------------
# FixedSkew
# ---------------------------------------------------------------------------


class TestFixedSkew:

    def test_positive_offset_reads_ahead(self):
        """A positive offset makes the clock read ahead of true time."""
        skew = FixedSkew(Duration.from_seconds(0.3))
        true_time = Instant.from_seconds(10.0)
        perceived = skew.read(true_time)
        assert perceived == Instant.from_seconds(10.3)

    def test_negative_offset_reads_behind(self):
        """A negative offset makes the clock read behind true time."""
        skew = FixedSkew(Duration.from_seconds(-0.3))
        true_time = Instant.from_seconds(10.0)
        perceived = skew.read(true_time)
        assert perceived == Instant.from_seconds(9.7)

    def test_zero_offset_is_identity(self):
        """Zero offset returns true time unchanged."""
        skew = FixedSkew(Duration.ZERO)
        true_time = Instant.from_seconds(5.0)
        assert skew.read(true_time) == true_time

    def test_offset_property(self):
        offset = Duration.from_seconds(0.05)
        skew = FixedSkew(offset)
        assert skew.offset == offset

    def test_offset_at_epoch(self):
        """Offset applies even at time zero."""
        skew = FixedSkew(Duration.from_seconds(0.1))
        perceived = skew.read(Instant.Epoch)
        assert perceived == Instant.from_seconds(0.1)

    def test_implements_clock_model_protocol(self):
        assert isinstance(FixedSkew(Duration.ZERO), ClockModel)


# ---------------------------------------------------------------------------
# LinearDrift
# ---------------------------------------------------------------------------


class TestLinearDrift:

    def test_positive_ppm_clock_runs_fast(self):
        """Positive ppm means the clock gains time — reads ahead."""
        drift = LinearDrift(rate_ppm=1000)  # 1ms per second
        true_time = Instant.from_seconds(10.0)
        perceived = drift.read(true_time)
        # 10s * 1000/1_000_000 = 0.01s drift
        expected = Instant.from_seconds(10.01)
        assert perceived == expected

    def test_negative_ppm_clock_runs_slow(self):
        """Negative ppm means the clock loses time — reads behind."""
        drift = LinearDrift(rate_ppm=-1000)
        true_time = Instant.from_seconds(10.0)
        perceived = drift.read(true_time)
        expected = Instant.from_seconds(9.99)
        assert perceived == expected

    def test_zero_drift_is_identity(self):
        """Zero ppm returns true time unchanged."""
        drift = LinearDrift(rate_ppm=0)
        true_time = Instant.from_seconds(42.0)
        assert drift.read(true_time) == true_time

    def test_drift_accumulates_over_time(self):
        """Drift grows linearly with elapsed time."""
        drift = LinearDrift(rate_ppm=1000)
        at_1s = drift.read(Instant.from_seconds(1.0))
        at_10s = drift.read(Instant.from_seconds(10.0))
        at_100s = drift.read(Instant.from_seconds(100.0))

        drift_1 = at_1s.to_seconds() - 1.0
        drift_10 = at_10s.to_seconds() - 10.0
        drift_100 = at_100s.to_seconds() - 100.0

        # Drift should be proportional to elapsed time
        assert abs(drift_10 / drift_1 - 10.0) < 0.01
        assert abs(drift_100 / drift_1 - 100.0) < 0.01

    def test_drift_at_epoch_is_zero(self):
        """No drift at time zero regardless of rate."""
        drift = LinearDrift(rate_ppm=5000)
        assert drift.read(Instant.Epoch) == Instant.Epoch

    def test_rate_ppm_property(self):
        drift = LinearDrift(rate_ppm=500)
        assert drift.rate_ppm == 500

    def test_implements_clock_model_protocol(self):
        assert isinstance(LinearDrift(rate_ppm=0), ClockModel)


# ---------------------------------------------------------------------------
# NodeClock
# ---------------------------------------------------------------------------


class TestNodeClock:

    def _make_clock(self, time_s: float = 0.0) -> Clock:
        """Create a Clock set to the given time."""
        clock = Clock(Instant.from_seconds(time_s))
        return clock

    def test_now_returns_transformed_time(self):
        """NodeClock.now applies the model to the base clock's time."""
        base = self._make_clock(10.0)
        nc = NodeClock(FixedSkew(Duration.from_seconds(0.5)))
        nc.set_clock(base)
        assert nc.now == Instant.from_seconds(10.5)

    def test_now_without_model_returns_true_time(self):
        """NodeClock with no model acts as identity."""
        base = self._make_clock(7.0)
        nc = NodeClock()
        nc.set_clock(base)
        assert nc.now == Instant.from_seconds(7.0)

    def test_raises_without_base_clock(self):
        """Accessing now before set_clock raises RuntimeError."""
        nc = NodeClock(FixedSkew(Duration.ZERO))
        with pytest.raises(RuntimeError, match="no base clock"):
            _ = nc.now

    def test_tracks_clock_updates(self):
        """NodeClock reflects updates to the underlying base clock."""
        base = self._make_clock(0.0)
        nc = NodeClock(FixedSkew(Duration.from_seconds(1.0)))
        nc.set_clock(base)

        assert nc.now == Instant.from_seconds(1.0)

        base.update(Instant.from_seconds(5.0))
        assert nc.now == Instant.from_seconds(6.0)

    def test_model_property(self):
        model = LinearDrift(rate_ppm=100)
        nc = NodeClock(model)
        assert nc.model is model

    def test_model_property_none(self):
        nc = NodeClock()
        assert nc.model is None
