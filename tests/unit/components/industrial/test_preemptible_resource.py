"""Tests for PreemptibleResource component."""

from __future__ import annotations

import pytest
from typing import Generator

from happysimulator.components.industrial.preemptible_resource import (
    PreemptibleResource,
    PreemptibleGrant,
)
from happysimulator.components.common import Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class Acquirer(Entity):
    """Test entity that acquires from a PreemptibleResource."""

    def __init__(self, name: str, resource: PreemptibleResource, priority: float,
                 hold_time: float, downstream: Entity):
        super().__init__(name)
        self.resource = resource
        self.priority = priority
        self.hold_time = hold_time
        self.downstream = downstream
        self.acquired = False
        self.preempted = False
        self._grant = None

    def handle_event(self, event: Event) -> Generator:
        def on_preempt():
            self.preempted = True

        grant = yield self.resource.acquire(
            amount=1, priority=self.priority, on_preempt=on_preempt,
        )
        self._grant = grant
        self.acquired = True

        yield self.hold_time

        if not grant.preempted:
            grant.release()
            return [Event(time=self.now, event_type="Done", target=self.downstream,
                          context=event.context)]
        return []


class TestPreemptibleResourceBasics:

    def test_creates_with_capacity(self):
        res = PreemptibleResource("res", capacity=4)
        assert res.capacity == 4
        assert res.available == 4

    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError):
            PreemptibleResource("res", capacity=0)

    def test_immediate_acquire(self):
        res = PreemptibleResource("res", capacity=2)
        sink = Sink()
        acq = Acquirer("acq", res, priority=1.0, hold_time=0.1, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[res, acq, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Go", target=acq))
        sim.run()

        assert acq.acquired
        assert sink.events_received == 1
        stats = res.stats
        assert stats.acquisitions >= 1

    def test_preemption(self):
        """Higher-priority request preempts lower-priority holder."""
        res = PreemptibleResource("res", capacity=1)
        sink = Sink()

        # Low priority acquires first
        low = Acquirer("low", res, priority=10.0, hold_time=5.0, downstream=sink)
        # High priority arrives later
        high = Acquirer("high", res, priority=1.0, hold_time=0.1, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            entities=[res, low, high, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Go", target=low))
        sim.schedule(Event(time=Instant.from_seconds(0.5), event_type="Go", target=high))
        sim.run()

        assert low.preempted
        assert high.acquired
        assert res.stats.preemptions >= 1

    def test_no_preemption_when_same_priority(self):
        """Cannot preempt holder with same or lower priority."""
        res = PreemptibleResource("res", capacity=1)
        sink = Sink()

        first = Acquirer("first", res, priority=5.0, hold_time=2.0, downstream=sink)
        second = Acquirer("second", res, priority=5.0, hold_time=0.1, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[res, first, second, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Go", target=first))
        sim.schedule(Event(time=Instant.from_seconds(0.1), event_type="Go", target=second))
        sim.run()

        assert not first.preempted

    def test_grant_release_idempotent(self):
        res = PreemptibleResource("res", capacity=2)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            entities=[res],
        )
        sim.run()

        # Direct test of grant
        grant = PreemptibleGrant(res, 1, 1.0)
        res._available -= 1
        grant.release()
        grant.release()  # idempotent
        assert res.available == 2

    def test_stats_snapshot(self):
        res = PreemptibleResource("res", capacity=3)
        stats = res.stats
        assert stats.capacity == 3
        assert stats.available == 3
        assert stats.acquisitions == 0
        assert stats.preemptions == 0
