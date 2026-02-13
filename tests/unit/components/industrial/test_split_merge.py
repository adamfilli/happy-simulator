"""Tests for SplitMerge component."""

from __future__ import annotations

import pytest
from typing import Generator

from happysimulator.components.industrial.split_merge import SplitMerge
from happysimulator.components.common import Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


class EchoWorker(Entity):
    """Test worker that resolves reply_future after a delay."""

    def __init__(self, name: str, delay: float = 0.1, result_value: str = "ok"):
        super().__init__(name)
        self.delay = delay
        self.result_value = result_value
        self._processed = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self.delay
        self._processed += 1
        reply = event.context.get("reply_future")
        if reply is not None:
            reply.resolve({"worker": self.name, "result": self.result_value})
        return []


class TestSplitMergeBasics:

    def test_creates_with_parameters(self):
        sink = Sink()
        w1, w2 = EchoWorker("w1"), EchoWorker("w2")
        sm = SplitMerge("sm", targets=[w1, w2], downstream=sink)
        assert sm.fan_out == 2
        assert sm.stats.splits_initiated == 0

    def test_fan_out_and_merge(self):
        sink = Sink()
        w1 = EchoWorker("w1", delay=0.1, result_value="a")
        w2 = EchoWorker("w2", delay=0.2, result_value="b")
        sm = SplitMerge("sm", targets=[w1, w2], downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[sm, w1, w2, sink],
        )
        sim.schedule(
            Event(time=Instant.Epoch, event_type="Task", target=sm, context={"id": 1})
        )
        sim.run()

        assert sm.stats.splits_initiated == 1
        assert sm.stats.merges_completed == 1
        assert w1._processed == 1
        assert w2._processed == 1
        assert sink.events_received == 1

    def test_merged_event_has_sub_results(self):
        sink = Sink()
        w1 = EchoWorker("w1", result_value="x")
        w2 = EchoWorker("w2", result_value="y")
        sm = SplitMerge("sm", targets=[w1, w2], downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            entities=[sm, w1, w2, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Task", target=sm))
        sim.run()

        # The merged event should reach the sink with sub_results
        assert sink.events_received == 1

    def test_multiple_splits(self):
        sink = Sink()
        w1, w2, w3 = EchoWorker("w1"), EchoWorker("w2"), EchoWorker("w3")
        sm = SplitMerge("sm", targets=[w1, w2, w3], downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[sm, w1, w2, w3, sink],
        )
        sim.schedule(Event(time=Instant.Epoch, event_type="Task", target=sm))
        sim.schedule(Event(time=Instant.from_seconds(0.5), event_type="Task", target=sm))
        sim.run()

        assert sm.stats.splits_initiated == 2
        assert sm.stats.merges_completed == 2
        assert sink.events_received == 2

    def test_stats_snapshot(self):
        sink = Sink()
        sm = SplitMerge("sm", targets=[EchoWorker("w1")], downstream=sink)
        stats = sm.stats
        assert stats.fan_out == 1
        assert stats.splits_initiated == 0
        assert stats.merges_completed == 0
