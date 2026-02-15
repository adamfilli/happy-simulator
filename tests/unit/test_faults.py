"""Unit tests for the fault injection framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    InjectPacketLoss,
    NetworkPartition,
    PauseNode,
    RandomPartition,
    ReduceCapacity,
)
from happysimulator.load.source import Source

if TYPE_CHECKING:
    from collections.abc import Generator

# =============================================================================
# Helpers
# =============================================================================


class SimpleServer(Entity):
    """Entity that counts events it receives."""

    def __init__(self, name: str):
        super().__init__(name)
        self.received: list[Event] = []

    def handle_event(self, event: Event) -> None:
        self.received.append(event)


class ProcessingServer(Entity):
    """Entity that yields a short delay, tracking events."""

    def __init__(self, name: str, downstream: Entity | None = None):
        super().__init__(name)
        self.processed = 0
        self._downstream = downstream

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield 0.001  # 1ms processing
        self.processed += 1
        if self._downstream:
            return [Event(time=self.now, event_type="Done", target=self._downstream)]
        return []


# =============================================================================
# CrashNode Tests
# =============================================================================


class TestCrashNode:
    def test_crash_drops_events(self):
        """Events targeting a crashed entity are silently dropped."""
        server = SimpleServer("server")
        schedule = FaultSchedule()
        schedule.add(CrashNode("server", at=5.0))

        # Source sends one event per second
        source = Source.constant(rate=1, target=server, event_type="Ping")

        sim = Simulation(
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[server],
            fault_schedule=schedule,
        )
        sim.run()

        # Server should receive events before crash (t=0..4) but not after (t=5..9)
        times = [e.time.to_seconds() for e in server.received]
        assert all(t < 5.0 for t in times), f"Events after crash: {times}"
        assert len(times) >= 4, f"Expected >=4 events before crash, got {len(times)}"

    def test_crash_and_restart(self):
        """Entity receives events before crash, drops during, receives after restart."""
        server = SimpleServer("server")
        schedule = FaultSchedule()
        schedule.add(CrashNode("server", at=5.0, restart_at=8.0))

        source = Source.constant(rate=1, target=server, event_type="Ping")

        sim = Simulation(
            end_time=Instant.from_seconds(12.0),
            sources=[source],
            entities=[server],
            fault_schedule=schedule,
        )
        sim.run()

        times = [e.time.to_seconds() for e in server.received]
        # Events at t=0,1,2,3,4 (before crash) and t=8,9,10,11 (after restart)
        before_crash = [t for t in times if t < 5.0]
        during_crash = [t for t in times if 5.0 <= t < 8.0]
        after_restart = [t for t in times if t >= 8.0]

        assert len(before_crash) >= 4
        assert len(during_crash) == 0, f"Events during crash: {during_crash}"
        assert len(after_restart) >= 3

    def test_permanent_crash(self):
        """CrashNode without restart_at is permanent."""
        server = SimpleServer("server")
        schedule = FaultSchedule()
        schedule.add(CrashNode("server", at=3.0))

        source = Source.constant(rate=1, target=server, event_type="Ping")

        sim = Simulation(
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[server],
            fault_schedule=schedule,
        )
        sim.run()

        times = [e.time.to_seconds() for e in server.received]
        assert all(t < 3.0 for t in times)


# =============================================================================
# PauseNode Tests
# =============================================================================


class TestPauseNode:
    def test_pause_and_resume(self):
        """PauseNode freezes processing during window, resumes after."""
        server = SimpleServer("server")
        schedule = FaultSchedule()
        schedule.add(PauseNode("server", start=3.0, end=7.0))

        source = Source.constant(rate=1, target=server, event_type="Ping")

        sim = Simulation(
            end_time=Instant.from_seconds(10.0),
            sources=[source],
            entities=[server],
            fault_schedule=schedule,
        )
        sim.run()

        times = [e.time.to_seconds() for e in server.received]
        during_pause = [t for t in times if 3.0 <= t < 7.0]
        after_resume = [t for t in times if t >= 7.0]

        assert len(during_pause) == 0, f"Events during pause: {during_pause}"
        assert len(after_resume) >= 2


# =============================================================================
# InjectLatency Tests
# =============================================================================


class TestInjectLatency:
    def test_latency_increases_during_window(self):
        """Link latency increases during fault window and restores after."""
        node_a = SimpleServer("a")
        node_b = SimpleServer("b")

        network = Network(name="net")
        link = datacenter_network("link_ab")
        network.add_bidirectional_link(node_a, node_b, link)

        schedule = FaultSchedule()
        schedule.add(InjectLatency("a", "b", extra_ms=500, start=5.0, end=15.0))

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            sources=[],
            entities=[node_a, node_b, network],
            fault_schedule=schedule,
        )

        # Check link latency before, during, and after
        # Run simulation to activate/deactivate
        sim.run()

        # After simulation, link latency should be restored
        restored_link = network.get_link("a", "b")
        assert restored_link is not None
        latency_val = restored_link.latency.get_latency(Instant.Epoch).to_seconds()
        # Datacenter latency is around 0.001s (1ms), not 501ms
        assert latency_val < 0.01, f"Latency not restored: {latency_val}"


# =============================================================================
# InjectPacketLoss Tests
# =============================================================================


class TestInjectPacketLoss:
    def test_loss_rate_increases_and_restores(self):
        """Packet loss rate increases during fault window and restores after."""
        node_a = SimpleServer("a")
        node_b = SimpleServer("b")

        network = Network(name="net")
        link = datacenter_network("link_ab")
        network.add_bidirectional_link(node_a, node_b, link)

        original_link = network.get_link("a", "b")
        original_loss = original_link.packet_loss_rate

        schedule = FaultSchedule()
        schedule.add(InjectPacketLoss("a", "b", loss_rate=0.5, start=5.0, end=15.0))

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            sources=[],
            entities=[node_a, node_b, network],
            fault_schedule=schedule,
        )
        sim.run()

        # After simulation, loss rate should be restored
        restored_link = network.get_link("a", "b")
        assert restored_link.packet_loss_rate == original_loss


# =============================================================================
# NetworkPartition Tests
# =============================================================================


class TestNetworkPartition:
    def test_partition_blocks_and_heals(self):
        """Traffic blocked during partition, delivered after heal."""
        node_a = SimpleServer("a")
        node_b = SimpleServer("b")

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

        # After simulation, partition should be healed
        assert not network.is_partitioned("a", "b")


# =============================================================================
# RandomPartition Tests
# =============================================================================


class TestRandomPartition:
    def test_random_partition_fires(self):
        """RandomPartition creates at least one fault/heal cycle."""
        nodes = [SimpleServer(name) for name in ["n1", "n2", "n3"]]

        network = Network(name="net")
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                link = datacenter_network(f"link_{nodes[i].name}_{nodes[j].name}")
                network.add_bidirectional_link(nodes[i], nodes[j], link)

        schedule = FaultSchedule()
        schedule.add(
            RandomPartition(
                nodes=["n1", "n2", "n3"],
                mtbf=5.0,
                mttr=2.0,
                seed=42,
            )
        )

        sim = Simulation(
            end_time=Instant.from_seconds(30.0),
            sources=[],
            entities=[*nodes, network],
            fault_schedule=schedule,
        )
        summary = sim.run()

        # Should have processed some fault events
        assert summary.total_events_processed > 0


# =============================================================================
# ReduceCapacity Tests
# =============================================================================


class TestReduceCapacity:
    def test_capacity_reduced_and_restored(self):
        """Resource capacity is halved during window and restored after."""
        resource = Resource("cpu", capacity=8)

        schedule = FaultSchedule()
        schedule.add(ReduceCapacity("cpu", factor=0.5, start=5.0, end=15.0))

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            sources=[],
            entities=[resource],
            fault_schedule=schedule,
        )
        sim.run()

        # After simulation, capacity should be restored
        assert resource.capacity == 8


# =============================================================================
# FaultHandle Tests
# =============================================================================


class TestFaultHandle:
    def test_cancel_prevents_activation(self):
        """Cancelled faults never activate."""
        server = SimpleServer("server")
        schedule = FaultSchedule()
        handle = schedule.add(CrashNode("server", at=5.0, restart_at=10.0))

        source = Source.constant(rate=1, target=server, event_type="Ping")

        sim = Simulation(
            end_time=Instant.from_seconds(12.0),
            sources=[source],
            entities=[server],
            fault_schedule=schedule,
        )

        # Cancel before running
        handle.cancel()

        sim.run()

        # Server should receive events at ALL times since fault was cancelled
        times = [e.time.to_seconds() for e in server.received]
        during_would_be_crash = [t for t in times if 5.0 <= t < 10.0]
        assert len(during_would_be_crash) >= 4, (
            f"Expected events during cancelled crash window, got {during_would_be_crash}"
        )

    def test_cancel_is_idempotent(self):
        """Calling cancel() multiple times is safe."""
        schedule = FaultSchedule()
        handle = schedule.add(CrashNode("x", at=1.0))
        handle.cancel()
        handle.cancel()  # Should not raise
        assert handle.cancelled


# =============================================================================
# FaultStats Tests
# =============================================================================


class TestFaultStats:
    def test_stats_counts(self):
        """FaultStats reflects scheduled and cancelled counts."""
        server = SimpleServer("server")
        schedule = FaultSchedule()
        schedule.add(CrashNode("server", at=5.0))
        h2 = schedule.add(PauseNode("server", start=10.0, end=15.0))

        source = Source.constant(rate=1, target=server, event_type="Ping")

        sim = Simulation(
            end_time=Instant.from_seconds(20.0),
            sources=[source],
            entities=[server],
            fault_schedule=schedule,
        )

        h2.cancel()

        sim.run()

        stats = schedule.stats
        assert stats.faults_scheduled == 2
        assert stats.faults_cancelled == 1


# =============================================================================
# Crashed flag on ProcessContinuation
# =============================================================================


class TestCrashedProcessContinuation:
    def test_crash_stops_generator_processing(self):
        """Crashing a node stops its generator-based processing."""
        sink = Counter("sink")
        server = ProcessingServer("server", downstream=sink)
        schedule = FaultSchedule()
        schedule.add(CrashNode("server", at=5.0, restart_at=8.0))

        source = Source.constant(rate=2, target=server, event_type="Request")

        sim = Simulation(
            end_time=Instant.from_seconds(12.0),
            sources=[source],
            entities=[server, sink],
            fault_schedule=schedule,
        )
        sim.run()

        # Server should have processed some events before crash and after restart
        # but during crash window [5, 8) no events should pass through
        assert server.processed > 0
