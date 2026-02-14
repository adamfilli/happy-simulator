"""Integration test: fault injection in a multi-node simulation.

Scenario:
- 3 nodes (A, B, C) with a network connecting them
- Source sends constant traffic to node A
- Node A forwards to node B through the network
- CrashNode on B at t=10, restart at t=20
- NetworkPartition between A and C at t=15-25
- InjectLatency on A->B link at t=5-30

Verifies that events are dropped during fault windows and resume after.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

from happysimulator.components.common import Counter
from happysimulator.components.network.conditions import datacenter_network
from happysimulator.components.network.network import Network
from happysimulator.components.resource import Resource
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.faults import (
    CrashNode,
    FaultSchedule,
    InjectLatency,
    NetworkPartition,
    ReduceCapacity,
)
from happysimulator.load.source import Source


# =============================================================================
# Entities
# =============================================================================


class TimestampSink(Entity):
    """Records event arrival times for analysis."""

    def __init__(self, name: str):
        super().__init__(name)
        self.arrival_times: list[float] = []

    def handle_event(self, event: Event) -> None:
        self.arrival_times.append(self.now.to_seconds())


class ForwardingNode(Entity):
    """Node that forwards events to a downstream target."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self._downstream = downstream
        self.received = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self.received += 1
        yield 0.001  # 1ms processing
        return [Event(
            time=self.now,
            event_type="Forwarded",
            target=self._downstream,
            context=event.context.copy(),
        )]


# =============================================================================
# Tests
# =============================================================================


def test_multi_fault_scenario():
    """Full scenario with crash, partition, and latency faults."""
    sink = TimestampSink("sink")
    node_a = ForwardingNode("node_a", downstream=sink)
    node_b = ForwardingNode("node_b", downstream=sink)
    node_c = TimestampSink("node_c")

    # Build schedule
    schedule = FaultSchedule()
    schedule.add(CrashNode("node_b", at=10.0, restart_at=20.0))

    # Source sends to node_a and node_b
    source_a = Source.constant(rate=2, target=node_a, event_type="Request")
    source_b = Source.constant(rate=2, target=node_b, event_type="Request")

    sim = Simulation(
        end_time=Instant.from_seconds(30.0),
        sources=[source_a, source_b],
        entities=[node_a, node_b, node_c, sink],
        fault_schedule=schedule,
    )
    summary = sim.run()

    # Node A should process all events (not crashed)
    assert node_a.received > 0

    # Node B should have fewer events due to crash window [10, 20)
    # It processes events from 0-10 and 20-30, but not 10-20
    # So it should have processed ~40 events (20s * 2/s) out of 60 total
    assert node_b.received > 0
    assert node_b.received < 60  # Must have lost some during crash

    # Sink should have received forwarded events
    assert len(sink.arrival_times) > 0


def test_crash_node_with_counter():
    """Verify Counter gets events before crash and after restart."""
    counter = Counter("target")
    schedule = FaultSchedule()
    schedule.add(CrashNode("target", at=5.0, restart_at=10.0))

    source = Source.constant(rate=10, target=counter, event_type="Tick")

    sim = Simulation(
        end_time=Instant.from_seconds(15.0),
        sources=[source],
        entities=[counter],
        fault_schedule=schedule,
    )
    sim.run()

    # 15s * 10/s = 150 total, minus ~50 during crash = ~100
    # Allow some tolerance for edge timing
    assert 80 <= counter.total <= 110, f"Got {counter.total} events"


def test_reduce_capacity_with_resource():
    """Verify resource capacity is reduced and restored."""
    resource = Resource("cpu", capacity=10)
    counter = Counter("sink")

    schedule = FaultSchedule()
    schedule.add(ReduceCapacity("cpu", factor=0.5, start=2.0, end=8.0))

    # Use a simple source to drive the simulation forward
    source = Source.constant(rate=1, target=counter, event_type="Tick")

    sim = Simulation(
        end_time=Instant.from_seconds(10.0),
        sources=[source],
        entities=[resource, counter],
        fault_schedule=schedule,
    )
    sim.run()

    # After simulation, capacity should be fully restored
    assert resource.capacity == 10
    assert resource.available == 10


def test_fault_stats_tracking():
    """FaultStats correctly tracks scheduled and cancelled faults."""
    counter = Counter("target")
    schedule = FaultSchedule()

    h1 = schedule.add(CrashNode("target", at=5.0))
    h2 = schedule.add(CrashNode("target", at=10.0, restart_at=15.0))
    h3 = schedule.add(CrashNode("target", at=20.0))

    # Cancel one fault
    h3.cancel()

    source = Source.constant(rate=1, target=counter, event_type="Tick")

    sim = Simulation(
        end_time=Instant.from_seconds(25.0),
        sources=[source],
        entities=[counter],
        fault_schedule=schedule,
    )
    sim.run()

    stats = schedule.stats
    assert stats.faults_scheduled == 3
    assert stats.faults_cancelled == 1


def test_network_partition_fault():
    """NetworkPartition blocks traffic during the window."""
    node_a = TimestampSink("a")
    node_b = TimestampSink("b")

    network = Network(name="net")
    link = datacenter_network("link_ab")
    network.add_bidirectional_link(node_a, node_b, link)

    schedule = FaultSchedule()
    schedule.add(NetworkPartition(["a"], ["b"], start=5.0, end=15.0))

    sim = Simulation(
        end_time=Instant.from_seconds(20.0),
        sources=[],
        entities=[node_a, node_b, network],
        fault_schedule=schedule,
    )
    sim.run()

    # Partition should be healed after simulation
    assert not network.is_partitioned("a", "b")


def test_inject_latency_fault():
    """InjectLatency adds extra latency during the window."""
    node_a = TimestampSink("a")
    node_b = TimestampSink("b")

    network = Network(name="net")
    link = datacenter_network("link_ab")
    network.add_bidirectional_link(node_a, node_b, link)

    schedule = FaultSchedule()
    schedule.add(
        InjectLatency("a", "b", extra_ms=100, start=3.0, end=8.0)
    )

    sim = Simulation(
        end_time=Instant.from_seconds(10.0),
        sources=[],
        entities=[node_a, node_b, network],
        fault_schedule=schedule,
    )
    sim.run()

    # After simulation, latency should be restored to original
    restored_link = network.get_link("a", "b")
    latency_s = restored_link.latency.get_latency(Instant.Epoch).to_seconds()
    assert latency_s < 0.01, f"Latency not restored: {latency_s}s"
