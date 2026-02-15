"""Tests for ConsumerGroup."""

from happysimulator import (
    Entity,
    Event,
    Instant,
    SimFuture,
    Simulation,
)
from happysimulator.components.streaming.consumer_group import (
    ConsumerGroup,
    RangeAssignment,
    RoundRobinAssignment,
    StickyAssignment,
)
from happysimulator.components.streaming.event_log import EventLog


class DummyConsumer(Entity):
    """Simple consumer entity for testing."""

    def __init__(self, name: str):
        super().__init__(name)
        self.received: list = []

    def handle_event(self, event):
        self.received.append(event)
        return


def _make_log(**kwargs) -> EventLog:
    defaults = {"name": "log", "num_partitions": 4, "append_latency": 0.001, "read_latency": 0.0005}
    defaults.update(kwargs)
    return EventLog(**defaults)


def _make_group(log: EventLog, **kwargs) -> ConsumerGroup:
    defaults = {"name": "group", "event_log": log, "rebalance_delay": 0.01, "poll_latency": 0.001}
    defaults.update(kwargs)
    return ConsumerGroup(**defaults)


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


class TestRangeAssignment:
    """Tests for RangeAssignment strategy."""

    def test_even_distribution(self):
        strategy = RangeAssignment()
        result = strategy.assign([0, 1, 2, 3], ["c1", "c2"])
        assert result == {"c1": [0, 1], "c2": [2, 3]}

    def test_remainder_to_early_consumers(self):
        strategy = RangeAssignment()
        result = strategy.assign([0, 1, 2], ["c1", "c2"])
        assert result == {"c1": [0, 1], "c2": [2]}

    def test_empty_consumers(self):
        strategy = RangeAssignment()
        assert strategy.assign([0, 1], []) == {}

    def test_more_consumers_than_partitions(self):
        strategy = RangeAssignment()
        result = strategy.assign([0, 1], ["c1", "c2", "c3"])
        # c1 gets 0, c2 gets 1, c3 gets nothing
        assert result["c1"] == [0]
        assert result["c2"] == [1]
        assert result["c3"] == []


class TestRoundRobinAssignment:
    """Tests for RoundRobinAssignment strategy."""

    def test_distributes_round_robin(self):
        strategy = RoundRobinAssignment()
        result = strategy.assign([0, 1, 2, 3], ["c1", "c2"])
        assert result == {"c1": [0, 2], "c2": [1, 3]}

    def test_empty_consumers(self):
        strategy = RoundRobinAssignment()
        assert strategy.assign([0, 1], []) == {}


class TestStickyAssignment:
    """Tests for StickyAssignment strategy."""

    def test_minimizes_movement(self):
        strategy = StickyAssignment()

        # Initial assignment
        r1 = strategy.assign([0, 1, 2, 3], ["c1", "c2"])
        assert set(r1["c1"] + r1["c2"]) == {0, 1, 2, 3}

        # Add consumer — existing should keep most partitions
        r2 = strategy.assign([0, 1, 2, 3], ["c1", "c2", "c3"])
        # c1 and c2 should keep at least some of their previous partitions
        kept_c1 = set(r1["c1"]) & set(r2["c1"])
        kept_c2 = set(r1["c2"]) & set(r2["c2"])
        assert len(kept_c1) + len(kept_c2) >= 2

    def test_empty_consumers_clears_state(self):
        strategy = StickyAssignment()
        strategy.assign([0, 1], ["c1"])
        result = strategy.assign([0, 1], [])
        assert result == {}


class TestConsumerGroupCreation:
    """Tests for ConsumerGroup construction."""

    def test_creates_with_defaults(self):
        log = _make_log()
        group = _make_group(log)
        assert group.consumer_count == 0
        assert group.generation == 0
        assert group.total_lag() == 0

    def test_consumers_empty_initially(self):
        log = _make_log()
        group = _make_group(log)
        assert group.consumers == []
        assert group.assignments == {}


class TestJoinAndLeave:
    """Tests for consumer join and leave."""

    def test_join_assigns_partitions(self):
        log = _make_log(num_partitions=4)
        group = _make_group(log)
        consumer = DummyConsumer("c1")

        reply = SimFuture()
        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": "c1",
                    "consumer_entity": consumer,
                    "reply_future": reply,
                },
            )
        ]

        _run_sim([log, group, consumer], events)

        assert reply.is_resolved
        assigned = reply.value
        assert len(assigned) == 4  # Single consumer gets all partitions
        assert group.consumer_count == 1
        assert group.generation == 1

    def test_join_triggers_rebalance(self):
        log = _make_log(num_partitions=4)
        group = _make_group(log)
        c1 = DummyConsumer("c1")
        c2 = DummyConsumer("c2")

        reply1 = SimFuture()
        reply2 = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Join",
                target=group,
                context={"consumer_name": "c1", "consumer_entity": c1, "reply_future": reply1},
            ),
            Event(
                time=Instant.from_seconds(0.5),
                event_type="Join",
                target=group,
                context={"consumer_name": "c2", "consumer_entity": c2, "reply_future": reply2},
            ),
        ]

        _run_sim([log, group, c1, c2], events)

        assert group.generation == 2
        assert group.consumer_count == 2
        # After second join, partitions are split
        assignments = group.assignments
        total_assigned = sum(len(v) for v in assignments.values())
        assert total_assigned == 4

    def test_leave_triggers_rebalance(self):
        log = _make_log(num_partitions=4)
        group = _make_group(log)
        c1 = DummyConsumer("c1")
        c2 = DummyConsumer("c2")

        leave_reply = SimFuture()

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Join",
                target=group,
                context={"consumer_name": "c1", "consumer_entity": c1, "reply_future": SimFuture()},
            ),
            Event(
                time=Instant.from_seconds(0.5),
                event_type="Join",
                target=group,
                context={"consumer_name": "c2", "consumer_entity": c2, "reply_future": SimFuture()},
            ),
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Leave",
                target=group,
                context={"consumer_name": "c2", "reply_future": leave_reply},
            ),
        ]

        _run_sim([log, group, c1, c2], events)

        assert group.consumer_count == 1
        assert group.generation == 3  # Join + Join + Leave
        assert leave_reply.is_resolved
        # Remaining consumer gets all partitions
        assert len(group.assignments["c1"]) == 4


class TestPollAndCommit:
    """Tests for poll and commit operations."""

    def test_poll_returns_records(self):
        log = _make_log(num_partitions=1)
        group = _make_group(log)
        consumer = DummyConsumer("c1")

        poll_reply = SimFuture()

        # Append records
        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(5)
        ]

        # Join
        events.append(
            Event(
                time=Instant.from_seconds(0.5),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": "c1",
                    "consumer_entity": consumer,
                    "reply_future": SimFuture(),
                },
            )
        )

        # Poll
        events.append(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Poll",
                target=group,
                context={"consumer_name": "c1", "max_records": 10, "reply_future": poll_reply},
            )
        )

        _run_sim([log, group, consumer], events)

        assert poll_reply.is_resolved
        records = poll_reply.value
        assert len(records) == 5

    def test_commit_advances_offset(self):
        log = _make_log(num_partitions=1)
        group = _make_group(log)
        consumer = DummyConsumer("c1")

        poll_reply = SimFuture()

        # Append records
        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(5)
        ]

        # Join
        events.append(
            Event(
                time=Instant.from_seconds(0.5),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": "c1",
                    "consumer_entity": consumer,
                    "reply_future": SimFuture(),
                },
            )
        )

        # Commit offset 3
        events.append(
            Event(
                time=Instant.from_seconds(0.8),
                event_type="Commit",
                target=group,
                context={"consumer_name": "c1", "offsets": {0: 3}},
            )
        )

        # Poll after commit
        events.append(
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Poll",
                target=group,
                context={"consumer_name": "c1", "max_records": 10, "reply_future": poll_reply},
            )
        )

        _run_sim([log, group, consumer], events)

        # Should only get records from offset 3 onwards
        records = poll_reply.value
        assert len(records) == 2
        assert records[0].offset == 3

    def test_poll_unassigned_consumer_returns_empty(self):
        log = _make_log(num_partitions=1)
        group = _make_group(log)

        poll_reply = SimFuture()
        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Poll",
                target=group,
                context={"consumer_name": "unknown", "max_records": 10, "reply_future": poll_reply},
            )
        ]

        _run_sim([log, group], events)

        assert poll_reply.is_resolved
        assert poll_reply.value == []


class TestConsumerLag:
    """Tests for consumer lag tracking."""

    def test_lag_reflects_uncommitted_records(self):
        log = _make_log(num_partitions=1)
        group = _make_group(log)
        consumer = DummyConsumer("c1")

        # Append records
        events = [
            Event(
                time=Instant.from_seconds(0.1 + i * 0.01),
                event_type="Append",
                target=log,
                context={"key": "k", "value": i},
            )
            for i in range(5)
        ]

        # Join
        events.append(
            Event(
                time=Instant.from_seconds(0.5),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": "c1",
                    "consumer_entity": consumer,
                    "reply_future": SimFuture(),
                },
            )
        )

        _run_sim([log, group, consumer], events)

        lag = group.consumer_lag("c1")
        assert lag[0] == 5  # 5 records uncommitted
        assert group.total_lag() == 5

    def test_lag_decreases_after_commit(self):
        log = _make_log(num_partitions=1)
        group = _make_group(log)
        consumer = DummyConsumer("c1")

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
                time=Instant.from_seconds(0.5),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": "c1",
                    "consumer_entity": consumer,
                    "reply_future": SimFuture(),
                },
            )
        )

        events.append(
            Event(
                time=Instant.from_seconds(0.8),
                event_type="Commit",
                target=group,
                context={"consumer_name": "c1", "offsets": {0: 3}},
            )
        )

        _run_sim([log, group, consumer], events)

        lag = group.consumer_lag("c1")
        assert lag[0] == 2  # 5 - 3 = 2

    def test_lag_unknown_consumer_returns_empty(self):
        log = _make_log()
        group = _make_group(log)
        assert group.consumer_lag("unknown") == {}


class TestConsumerGroupStats:
    """Tests for ConsumerGroupStats."""

    def test_stats_track_operations(self):
        log = _make_log(num_partitions=1)
        group = _make_group(log)
        consumer = DummyConsumer("c1")

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Join",
                target=group,
                context={
                    "consumer_name": "c1",
                    "consumer_entity": consumer,
                    "reply_future": SimFuture(),
                },
            ),
            Event(
                time=Instant.from_seconds(0.5),
                event_type="Poll",
                target=group,
                context={"consumer_name": "c1", "max_records": 10, "reply_future": SimFuture()},
            ),
            Event(
                time=Instant.from_seconds(0.8),
                event_type="Commit",
                target=group,
                context={"consumer_name": "c1", "offsets": {0: 0}},
            ),
            Event(
                time=Instant.from_seconds(1.0),
                event_type="Leave",
                target=group,
                context={"consumer_name": "c1", "reply_future": SimFuture()},
            ),
        ]

        _run_sim([log, group, consumer], events)

        assert group.stats.joins == 1
        assert group.stats.leaves == 1
        assert group.stats.polls == 1
        assert group.stats.commits == 1
        assert group.stats.rebalances == 2  # Join + Leave
