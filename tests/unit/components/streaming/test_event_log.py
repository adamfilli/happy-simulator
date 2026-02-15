"""Tests for EventLog."""

import pytest

from happysimulator import (
    Event,
    Instant,
    SimFuture,
    Simulation,
)
from happysimulator.components.streaming.event_log import (
    EventLog,
    Record,
    SizeRetention,
    TimeRetention,
)


def _make_log(**kwargs) -> EventLog:
    defaults = {
        "name": "test-log",
        "num_partitions": 4,
        "append_latency": 0.001,
        "read_latency": 0.0005,
    }
    defaults.update(kwargs)
    return EventLog(**defaults)


def _run_sim(entities, events, end_s=5.0):
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(end_s),
        sources=[],
        entities=entities,
    )
    for e in events:
        sim.schedule(e)
    sim.run()
    return sim


class TestEventLogCreation:
    """Tests for EventLog construction."""

    def test_creates_with_defaults(self):
        log = _make_log()
        assert log.num_partitions == 4
        assert log.total_records == 0
        assert log.stats.records_appended == 0

    def test_creates_partitions(self):
        log = _make_log(num_partitions=8)
        assert len(log.partitions) == 8
        for p in log.partitions:
            assert p.high_watermark == 0

    def test_rejects_zero_partitions(self):
        with pytest.raises(ValueError, match="num_partitions must be >= 1"):
            EventLog(name="bad", num_partitions=0)


class TestRecordDataclass:
    """Tests for the Record dataclass."""

    def test_record_is_frozen(self):
        rec = Record(offset=0, key="k", value="v", timestamp=1.0, partition=0)
        with pytest.raises(AttributeError):
            rec.offset = 1  # type: ignore

    def test_record_fields(self):
        rec = Record(offset=5, key="user-1", value={"action": "buy"}, timestamp=10.0, partition=2)
        assert rec.offset == 5
        assert rec.key == "user-1"
        assert rec.partition == 2


class TestAppend:
    """Tests for append operations."""

    def test_append_via_event(self):
        """Append via direct event handling."""
        log = _make_log(num_partitions=1)
        reply = SimFuture()

        event = Event(
            time=Instant.from_seconds(0.1),
            event_type="Append",
            target=log,
            context={"key": "k1", "value": "v1", "reply_future": reply},
        )

        _run_sim([log], [event])

        assert reply.is_resolved
        record = reply.value
        assert isinstance(record, Record)
        assert record.key == "k1"
        assert record.value == "v1"
        assert record.partition == 0
        assert record.offset == 0
        assert log.total_records == 1

    def test_multiple_appends_increment_offset(self):
        """Offsets increment monotonically within a partition."""
        log = _make_log(num_partitions=1)
        futures = []

        events = []
        for i in range(5):
            f = SimFuture()
            futures.append(f)
            events.append(
                Event(
                    time=Instant.from_seconds(0.1 + i * 0.01),
                    event_type="Append",
                    target=log,
                    context={"key": f"k{i}", "value": i, "reply_future": f},
                )
            )

        _run_sim([log], events)

        offsets = [f.value.offset for f in futures]
        assert offsets == [0, 1, 2, 3, 4]

    def test_appends_distribute_across_partitions(self):
        """Different keys route to different partitions."""
        log = _make_log(num_partitions=4)

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": f"key-{i}", "value": i},
            )
            for i in range(20)
        ]

        _run_sim([log], events)

        assert log.total_records == 20
        assert log.stats.records_appended == 20
        # At least 2 partitions should have records (hash distribution)
        non_empty = sum(1 for p in log.partitions if len(p.records) > 0)
        assert non_empty >= 2

    def test_high_watermarks(self):
        """High watermarks track per partition."""
        log = _make_log(num_partitions=1)

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "same-key", "value": i},
            )
            for i in range(3)
        ]

        _run_sim([log], events)

        assert log.high_watermark(0) == 3
        wms = log.high_watermarks()
        assert wms[0] == 3


class TestRead:
    """Tests for read operations."""

    def test_read_returns_records(self):
        """Read returns records from a partition."""
        log = _make_log(num_partitions=1)
        read_reply = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(5)
        ]

        events.append(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Read",
                target=log,
                context={
                    "partition": 0,
                    "offset": 0,
                    "max_records": 10,
                    "reply_future": read_reply,
                },
            )
        )

        _run_sim([log], events)

        assert read_reply.is_resolved
        records = read_reply.value
        assert len(records) == 5
        assert all(isinstance(r, Record) for r in records)

    def test_read_from_offset(self):
        """Read respects starting offset."""
        log = _make_log(num_partitions=1)
        read_reply = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(5)
        ]

        events.append(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Read",
                target=log,
                context={
                    "partition": 0,
                    "offset": 3,
                    "max_records": 10,
                    "reply_future": read_reply,
                },
            )
        )

        _run_sim([log], events)

        records = read_reply.value
        assert len(records) == 2
        assert records[0].offset == 3
        assert records[1].offset == 4

    def test_read_max_records_limit(self):
        """Read respects max_records."""
        log = _make_log(num_partitions=1)
        read_reply = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(10)
        ]

        events.append(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Read",
                target=log,
                context={"partition": 0, "offset": 0, "max_records": 3, "reply_future": read_reply},
            )
        )

        _run_sim([log], events)

        assert len(read_reply.value) == 3

    def test_read_empty_partition(self):
        """Read from empty partition returns empty list."""
        log = _make_log(num_partitions=4)
        read_reply = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Read",
                target=log,
                context={
                    "partition": 0,
                    "offset": 0,
                    "max_records": 10,
                    "reply_future": read_reply,
                },
            )
        ]

        _run_sim([log], events)

        assert read_reply.value == []

    def test_read_invalid_partition(self):
        """Read from invalid partition returns empty list."""
        log = _make_log(num_partitions=2)
        read_reply = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Read",
                target=log,
                context={
                    "partition": 99,
                    "offset": 0,
                    "max_records": 10,
                    "reply_future": read_reply,
                },
            )
        ]

        _run_sim([log], events)

        assert read_reply.value == []


class TestRetention:
    """Tests for retention policies."""

    def test_time_retention_policy(self):
        """TimeRetention retains records based on age."""
        policy = TimeRetention(max_age_s=10.0)
        old_record = Record(offset=0, key="k", value="v", timestamp=0.0, partition=0)
        new_record = Record(offset=1, key="k", value="v", timestamp=9.5, partition=0)

        assert not policy.should_retain(old_record, current_time_s=11.0)
        assert policy.should_retain(new_record, current_time_s=11.0)

    def test_time_retention_rejects_zero(self):
        with pytest.raises(ValueError):
            TimeRetention(max_age_s=0)

    def test_size_retention_rejects_zero(self):
        with pytest.raises(ValueError):
            SizeRetention(max_records=0)

    def test_time_retention_in_simulation(self):
        """Retention daemon removes old records."""
        log = _make_log(
            num_partitions=1,
            retention_policy=TimeRetention(max_age_s=1.0),
            retention_check_interval=0.5,
        )

        # Append records at t=0.1
        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(5)
        ]

        # Run long enough for retention to kick in
        _run_sim([log], events, end_s=3.0)

        # Records should have been expired by retention
        assert log.stats.records_expired > 0

    def test_size_retention_in_simulation(self):
        """SizeRetention limits per-partition record count."""
        log = _make_log(
            num_partitions=1,
            retention_policy=SizeRetention(max_records=3),
            retention_check_interval=0.5,
        )

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(10)
        ]

        _run_sim([log], events, end_s=3.0)

        # After retention sweep, partition should have at most max_records
        assert len(log.partitions[0].records) <= 3


class TestStats:
    """Tests for EventLogStats."""

    def test_stats_track_appends_and_reads(self):
        log = _make_log(num_partitions=1)

        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(3)
        ]

        events.append(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Read",
                target=log,
                context={
                    "partition": 0,
                    "offset": 0,
                    "max_records": 10,
                    "reply_future": SimFuture(),
                },
            )
        )

        _run_sim([log], events)

        assert log.stats.records_appended == 3
        assert log.stats.records_read == 3
        assert log.stats.avg_append_latency > 0

    def test_per_partition_appends(self):
        log = _make_log(num_partitions=1)

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Append",
                target=log,
                context={"key": "k", "value": 1},
            )
        ]

        _run_sim([log], events)

        assert log.stats.per_partition_appends[0] >= 1
