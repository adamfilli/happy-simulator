"""Tests for Data and BucketedData classes."""

import pytest

from happysimulator.core.temporal import Instant
from happysimulator.instrumentation.data import BucketedData, Data

# === Helpers ===


def _make_data(pairs: list[tuple[float, float]]) -> Data:
    """Create Data from (time_s, value) pairs."""
    d = Data()
    for t, v in pairs:
        d.add_stat(v, Instant.from_seconds(t))
    return d


# === Basic Operations ===


class TestDataBasic:
    def test_empty_data(self):
        d = Data()
        assert d.count() == 0
        assert d.values == []
        assert len(d) == 0
        assert not d

    def test_add_stat(self):
        d = Data()
        d.add_stat(42.0, Instant.from_seconds(1.0))
        assert d.count() == 1
        assert d.values == [(1.0, 42.0)]
        assert len(d) == 1
        assert d

    def test_multiple_samples(self):
        d = _make_data([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
        assert d.count() == 3
        assert d.times() == [0.0, 1.0, 2.0]
        assert d.raw_values() == [1.0, 2.0, 3.0]


# === Slicing ===


class TestDataBetween:
    def test_between_basic(self):
        d = _make_data([(0.0, 10), (1.0, 20), (2.0, 30), (3.0, 40)])
        sliced = d.between(1.0, 3.0)
        assert sliced.count() == 2
        assert sliced.raw_values() == [20, 30]

    def test_between_inclusive_start_exclusive_end(self):
        d = _make_data([(1.0, 10), (2.0, 20), (3.0, 30)])
        sliced = d.between(1.0, 3.0)
        assert sliced.count() == 2
        assert sliced.raw_values() == [10, 20]

    def test_between_empty_range(self):
        d = _make_data([(1.0, 10), (2.0, 20)])
        sliced = d.between(5.0, 10.0)
        assert sliced.count() == 0

    def test_between_returns_new_data(self):
        d = _make_data([(0.0, 10), (1.0, 20)])
        sliced = d.between(0.0, 2.0)
        assert sliced is not d
        assert sliced.count() == 2

    def test_between_chaining(self):
        d = _make_data([(0.0, 10), (1.0, 20), (2.0, 30), (3.0, 40), (4.0, 50)])
        sliced = d.between(1.0, 4.0).between(2.0, 3.5)
        assert sliced.count() == 2
        assert sliced.raw_values() == [30, 40]


# === Aggregations ===


class TestDataAggregations:
    def test_mean(self):
        d = _make_data([(0.0, 10), (1.0, 20), (2.0, 30)])
        assert d.mean() == 20.0

    def test_mean_empty(self):
        assert Data().mean() == 0.0

    def test_min(self):
        d = _make_data([(0.0, 30), (1.0, 10), (2.0, 20)])
        assert d.min() == 10

    def test_min_empty(self):
        assert Data().min() == 0.0

    def test_max(self):
        d = _make_data([(0.0, 10), (1.0, 30), (2.0, 20)])
        assert d.max() == 30

    def test_max_empty(self):
        assert Data().max() == 0.0

    def test_sum(self):
        d = _make_data([(0.0, 10), (1.0, 20), (2.0, 30)])
        assert d.sum() == 60.0

    def test_sum_empty(self):
        assert Data().sum() == 0.0

    def test_count(self):
        d = _make_data([(0.0, 1), (1.0, 2), (2.0, 3)])
        assert d.count() == 3

    def test_std(self):
        d = _make_data(
            [(0.0, 2), (1.0, 4), (2.0, 4), (3.0, 4), (4.0, 5), (5.0, 5), (6.0, 7), (7.0, 9)]
        )
        expected = 2.0  # known pstdev for this dataset
        assert abs(d.std() - expected) < 0.01

    def test_std_single_sample(self):
        d = _make_data([(0.0, 42)])
        assert d.std() == 0.0

    def test_std_empty(self):
        assert Data().std() == 0.0

    def test_std_identical_values(self):
        d = _make_data([(0.0, 5), (1.0, 5), (2.0, 5)])
        assert d.std() == 0.0


# === Percentile ===


class TestDataPercentile:
    def test_percentile_median(self):
        d = _make_data([(i, i) for i in range(101)])
        assert d.percentile(0.5) == 50.0

    def test_percentile_p99(self):
        d = _make_data([(i, i) for i in range(101)])
        assert d.percentile(0.99) == pytest.approx(99.0, abs=0.1)

    def test_percentile_empty(self):
        assert Data().percentile(0.5) == 0.0

    def test_percentile_single_value(self):
        d = _make_data([(0.0, 42)])
        assert d.percentile(0.0) == 42.0
        assert d.percentile(0.5) == 42.0
        assert d.percentile(1.0) == 42.0

    def test_percentile_boundary_0(self):
        d = _make_data([(0.0, 10), (1.0, 20), (2.0, 30)])
        assert d.percentile(0.0) == 10.0

    def test_percentile_boundary_1(self):
        d = _make_data([(0.0, 10), (1.0, 20), (2.0, 30)])
        assert d.percentile(1.0) == 30.0

    def test_percentile_interpolation(self):
        d = _make_data([(0.0, 0), (1.0, 100)])
        assert d.percentile(0.5) == 50.0
        assert d.percentile(0.25) == 25.0
        assert d.percentile(0.75) == 75.0


# === Bucketing ===


class TestDataBucket:
    def test_bucket_basic(self):
        d = _make_data(
            [
                (0.1, 10),
                (0.5, 20),
                (0.9, 30),
                (1.1, 40),
                (1.5, 50),
                (2.3, 60),
            ]
        )
        b = d.bucket(window_s=1.0)
        assert len(b) == 3
        assert b.times() == [0.0, 1.0, 2.0]
        assert b.counts() == [3, 2, 1]
        assert b.means()[0] == pytest.approx(20.0)
        assert b.means()[1] == pytest.approx(45.0)
        assert b.means()[2] == pytest.approx(60.0)

    def test_bucket_custom_window(self):
        d = _make_data([(0.0, 1), (0.3, 2), (0.6, 3), (0.9, 4), (1.2, 5)])
        b = d.bucket(window_s=0.5)
        assert len(b) == 3
        assert b.times() == [0.0, 0.5, 1.0]
        assert b.counts() == [2, 2, 1]

    def test_bucket_empty(self):
        b = Data().bucket()
        assert len(b) == 0
        assert not b

    def test_bucket_single_sample(self):
        d = _make_data([(5.5, 42)])
        b = d.bucket(window_s=1.0)
        assert len(b) == 1
        assert b.times() == [5.0]
        assert b.means() == [42.0]

    def test_bucket_to_dict(self):
        d = _make_data([(0.5, 10), (1.5, 20)])
        b = d.bucket(window_s=1.0)
        result = b.to_dict()
        assert set(result.keys()) == {"time_s", "mean", "p50", "p99", "max", "count", "sum"}
        assert result["time_s"] == [0.0, 1.0]
        assert result["count"] == [1, 1]

    def test_bucket_maxes(self):
        d = _make_data([(0.1, 10), (0.5, 50), (0.9, 30)])
        b = d.bucket(window_s=1.0)
        assert b.maxes() == [50]

    def test_bucket_sums(self):
        d = _make_data([(0.1, 10), (0.5, 20), (0.9, 30)])
        b = d.bucket(window_s=1.0)
        assert b.sums() == [60.0]

    def test_bucket_percentiles(self):
        d = _make_data([(0.0 + i * 0.01, i) for i in range(100)])
        b = d.bucket(window_s=1.0)
        assert len(b) == 1
        assert b.p50s()[0] == pytest.approx(49.5, abs=1.0)
        assert b.p99s()[0] == pytest.approx(98.0, abs=1.0)


# === Rate ===


class TestDataRate:
    def test_rate_basic(self):
        # 10 events in first second, 5 in second
        pairs = [(0.0 + i * 0.1, 1.0) for i in range(10)]
        pairs += [(1.0 + i * 0.2, 1.0) for i in range(5)]
        d = _make_data(pairs)
        r = d.rate(window_s=1.0)
        assert r.count() == 2
        assert r.raw_values()[0] == pytest.approx(10.0)
        assert r.raw_values()[1] == pytest.approx(5.0)

    def test_rate_empty(self):
        r = Data().rate()
        assert r.count() == 0


# === Convenience ===


class TestDataConvenience:
    def test_times(self):
        d = _make_data([(1.0, 10), (2.0, 20), (3.0, 30)])
        assert d.times() == [1.0, 2.0, 3.0]

    def test_raw_values(self):
        d = _make_data([(1.0, 10), (2.0, 20), (3.0, 30)])
        assert d.raw_values() == [10, 20, 30]

    def test_combined_workflow(self):
        """Test a realistic workflow: slice, aggregate, bucket."""
        d = _make_data(
            [
                (0.5, 100),
                (1.5, 200),
                (2.5, 150),
                (10.5, 500),
                (11.5, 600),
                (12.5, 550),
            ]
        )
        # Slice to first 5 seconds
        early = d.between(0.0, 5.0)
        assert early.count() == 3
        assert early.mean() == 150.0

        # Slice to later period
        late = d.between(10.0, 15.0)
        assert late.count() == 3
        assert late.mean() == 550.0

        # Bucket all data (bucket 0: t=0-5, bucket 2: t=10-15)
        bucketed = d.bucket(window_s=5.0)
        assert len(bucketed) == 2


# === BucketedData ===


class TestBucketedData:
    def test_empty(self):
        b = BucketedData()
        assert len(b) == 0
        assert not b
        assert b.to_dict()["time_s"] == []

    def test_to_dict_returns_copies(self):
        d = _make_data([(0.5, 10)])
        b = d.bucket()
        d1 = b.to_dict()
        d2 = b.to_dict()
        assert d1 == d2
        d1["time_s"].append(999)
        assert d2["time_s"] != d1["time_s"]


# === Backward Compatibility ===


class TestBackwardCompatibility:
    def test_values_property_still_works(self):
        """Existing code uses data.values - must not break."""
        d = Data()
        d.add_stat(42.0, Instant.from_seconds(1.0))
        assert d.values == [(1.0, 42.0)]

    def test_add_stat_still_works(self):
        """Existing code uses data.add_stat(value, time)."""
        d = Data()
        d.add_stat(10, Instant.from_seconds(0.5))
        d.add_stat(20, Instant.from_seconds(1.5))
        assert len(d.values) == 2
