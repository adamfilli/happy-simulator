"""Integration tests for resource management components.

Tests PooledCycleResource, PreemptibleResource, and SplitMerge
in multi-component pipelines.
"""

from __future__ import annotations

from typing import Generator

from happysimulator.components.industrial.pooled_cycle import PooledCycleResource
from happysimulator.components.industrial.preemptible_resource import (
    PreemptibleResource,
)
from happysimulator.components.industrial.split_merge import SplitMerge
from happysimulator.components.common import Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source


class EchoWorker(Entity):
    """Worker that resolves reply_future after a delay."""

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


class PriorityWorker(Entity):
    """Worker that acquires a PreemptibleResource with a given priority."""

    def __init__(self, name: str, resource: PreemptibleResource, priority: float,
                 hold_time: float, downstream: Entity):
        super().__init__(name)
        self.resource = resource
        self.priority = priority
        self.hold_time = hold_time
        self.downstream = downstream
        self.acquired = False
        self.preempted = False
        self.completed = False

    def handle_event(self, event: Event) -> Generator:
        def on_preempt():
            self.preempted = True

        grant = yield self.resource.acquire(
            amount=1, priority=self.priority, on_preempt=on_preempt,
        )
        self.acquired = True
        yield self.hold_time

        if not grant.preempted:
            grant.release()
            self.completed = True
            return [Event(time=self.now, event_type="Done", target=self.downstream,
                          context=event.context)]
        return []


class TestPooledCycleUnderLoad:
    """Source → PooledCycleResource → Sink under various load levels."""

    def test_utilization_and_queueing(self):
        sink = Sink("output")
        pool = PooledCycleResource(
            "pool", pool_size=3, cycle_time=0.5,
            downstream=sink, queue_capacity=100,
        )

        source = Source.constant(rate=10.0, target=pool, stop_after=5.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(8.0),
            sources=[source],
            entities=[pool, sink],
        )
        sim.run()

        assert pool.completed > 0
        assert sink.events_received == pool.completed
        assert pool.rejected == 0

    def test_rejection_when_queue_full(self):
        sink = Sink("output")
        pool = PooledCycleResource(
            "pool", pool_size=1, cycle_time=1.0,
            downstream=sink, queue_capacity=2,
        )

        # Very high arrival rate to overflow queue
        source = Source.constant(rate=20.0, target=pool, stop_after=3.0)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[pool, sink],
        )
        sim.run()

        assert pool.rejected > 0
        assert pool.completed > 0
        assert sink.events_received == pool.completed


class TestPreemptibleResourcePriority:
    """Multiple priority workers competing for a PreemptibleResource."""

    def test_high_priority_preempts_low(self):
        resource = PreemptibleResource("shared", capacity=1)
        sink = Sink("done")

        low = PriorityWorker("low", resource, priority=10.0, hold_time=5.0, downstream=sink)
        high = PriorityWorker("high", resource, priority=1.0, hold_time=0.1, downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(10.0),
            entities=[resource, low, high, sink],
        )
        # Low priority starts first
        sim.schedule(Event(time=Instant.Epoch, event_type="Go", target=low))
        # High priority arrives later and preempts
        sim.schedule(Event(time=Instant.from_seconds(0.5), event_type="Go", target=high))
        sim.run()

        assert low.preempted
        assert high.acquired
        assert high.completed
        assert resource.stats.preemptions >= 1

    def test_multiple_priority_levels(self):
        resource = PreemptibleResource("shared", capacity=1)
        sink = Sink("done")

        workers = [
            PriorityWorker(f"p{p}", resource, priority=float(p),
                           hold_time=2.0, downstream=sink)
            for p in [10, 8, 5, 3, 1]
        ]

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(20.0),
            entities=[resource, *workers, sink],
        )
        # Schedule all workers with slight staggering
        for i, w in enumerate(workers):
            sim.schedule(Event(
                time=Instant.from_seconds(i * 0.1),
                event_type="Go", target=w,
            ))
        sim.run()

        # Highest priority (p=1) should have completed without preemption
        highest = workers[-1]  # priority=1
        assert highest.acquired
        assert highest.completed


class TestSplitMergeFanOut:
    """Source → SplitMerge → [Worker1, Worker2, Worker3] → Sink."""

    def test_all_workers_process_and_merge(self):
        sink = Sink("merged")
        w1 = EchoWorker("w1", delay=0.1, result_value="a")
        w2 = EchoWorker("w2", delay=0.2, result_value="b")
        w3 = EchoWorker("w3", delay=0.3, result_value="c")
        sm = SplitMerge("splitter", targets=[w1, w2, w3], downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[sm, w1, w2, w3, sink],
        )
        # Send 5 tasks
        for i in range(5):
            sim.schedule(Event(
                time=Instant.from_seconds(i * 0.5),
                event_type="Task", target=sm,
                context={"task_id": i},
            ))
        sim.run()

        assert sm.stats.splits_initiated == 5
        assert sm.stats.merges_completed == 5
        assert w1._processed == 5
        assert w2._processed == 5
        assert w3._processed == 5
        assert sink.events_received == 5

    def test_merge_latency_equals_slowest_worker(self):
        sink = Sink("merged")
        fast = EchoWorker("fast", delay=0.01)
        slow = EchoWorker("slow", delay=1.0)
        sm = SplitMerge("splitter", targets=[fast, slow], downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            entities=[sm, fast, slow, sink],
        )
        sim.schedule(Event(
            time=Instant.Epoch, event_type="Task", target=sm,
            context={"created_at": Instant.Epoch},
        ))
        sim.run()

        assert sink.events_received == 1
        # Merge completes at slowest worker time (1.0s)
        assert sink.completion_times[0] == Instant.from_seconds(1.0)


class TestSplitMergeWithPooledCycleResource:
    """SplitMerge fans out to PooledCycleResource workers."""

    def test_fan_out_to_pooled_resources(self):
        sink = Sink("final")

        # Create pooled workers that resolve reply_future
        class PoolWorker(Entity):
            def __init__(self, name, cycle_time=0.1):
                super().__init__(name)
                self.cycle_time = cycle_time
                self.processed = 0

            def handle_event(self, event):
                yield self.cycle_time
                self.processed += 1
                reply = event.context.get("reply_future")
                if reply is not None:
                    reply.resolve({"worker": self.name})
                return []

        w1 = PoolWorker("pool_w1", cycle_time=0.1)
        w2 = PoolWorker("pool_w2", cycle_time=0.2)
        sm = SplitMerge("splitter", targets=[w1, w2], downstream=sink)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(5.0),
            entities=[sm, w1, w2, sink],
        )
        for i in range(3):
            sim.schedule(Event(
                time=Instant.from_seconds(i * 0.5),
                event_type="Task", target=sm,
            ))
        sim.run()

        assert sm.stats.merges_completed == 3
        assert w1.processed == 3
        assert w2.processed == 3
        assert sink.events_received == 3
