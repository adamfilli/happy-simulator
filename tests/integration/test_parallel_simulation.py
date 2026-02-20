"""Integration tests for ParallelSimulation."""

import pytest

from happysimulator.components.common import Counter, Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.load.source import Source
from happysimulator.parallel.link import PartitionLink
from happysimulator.parallel.partition import SimulationPartition
from happysimulator.parallel.simulation import ParallelSimulation


# ---------------------------------------------------------------------------
# Helper entities
# ---------------------------------------------------------------------------

class SimpleServer(Entity):
    """Server with constant service time that forwards to downstream."""

    def __init__(self, name, downstream, service_time=0.1, forward_delay=0.0):
        super().__init__(name)
        self.downstream = downstream
        self.service_time = service_time
        self.forward_delay = forward_delay
        self.events_handled = 0

    def handle_event(self, event):
        self.events_handled += 1
        yield self.service_time
        return [Event(
            time=self.now + self.forward_delay,
            event_type="Done",
            target=self.downstream,
        )]


class ReplicatingServer(Entity):
    """Server that replicates events to a remote peer after processing."""

    def __init__(self, name, downstream, peer=None, replicate_delay=0.0):
        super().__init__(name)
        self.downstream = downstream
        self.peer = peer
        self.replicate_delay = replicate_delay
        self.events_handled = 0

    def handle_event(self, event):
        self.events_handled += 1
        yield 0.01  # processing time

        results = [Event(time=self.now, event_type="Done", target=self.downstream)]
        # Only replicate primary events, not replicated ones
        if self.peer is not None and event.event_type != "Replicate":
            results.append(
                Event(
                    time=self.now + self.replicate_delay,
                    event_type="Replicate",
                    target=self.peer,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndependentPartitions:
    """Two partitions with no links — fully independent."""

    def test_two_independent_partitions(self):
        counter1 = Counter("counter1")
        counter2 = Counter("counter2")
        source1 = Source.constant(rate=10, target=counter1, event_type="Ping")
        source2 = Source.constant(rate=10, target=counter2, event_type="Ping")

        p1 = SimulationPartition(name="P1", entities=[counter1], sources=[source1])
        p2 = SimulationPartition(name="P2", entities=[counter2], sources=[source2])

        ps = ParallelSimulation(
            partitions=[p1, p2],
            duration=10.0,
        )
        summary = ps.run()

        assert summary.total_events_processed > 0
        assert len(summary.partitions) == 2
        assert "P1" in summary.partitions
        assert "P2" in summary.partitions

        # Each counter should have received ~100 events (10/s * 10s)
        assert counter1.total == 100
        assert counter2.total == 100

    def test_single_partition_matches_simulation(self):
        """Single partition should produce same results as regular Simulation."""
        counter = Counter("counter")
        source = Source.constant(rate=5, target=counter, event_type="Tick")

        p = SimulationPartition(name="Solo", entities=[counter], sources=[source])
        ps = ParallelSimulation(partitions=[p], duration=20.0)
        summary = ps.run()

        assert counter.total == 100  # 5/s * 20s
        assert summary.total_events_processed > 0

    def test_independent_summary_metrics(self):
        counter = Counter("c")
        source = Source.constant(rate=10, target=counter, event_type="X")
        p = SimulationPartition(name="P", entities=[counter], sources=[source])
        ps = ParallelSimulation(partitions=[p], duration=5.0)
        summary = ps.run()

        assert summary.duration_s == pytest.approx(5.0, abs=0.1)
        assert summary.wall_clock_seconds > 0
        assert summary.events_per_second > 0


class TestCrossPartitionReplication:
    """Two partitions connected by bidirectional links."""

    def test_bidirectional_replication(self):
        sink1 = Counter("sink1")
        sink2 = Counter("sink2")

        server1 = ReplicatingServer("server1", downstream=sink1, replicate_delay=0.05)
        server2 = ReplicatingServer("server2", downstream=sink2, replicate_delay=0.05)

        # Wire up peers after construction
        server1.peer = server2
        server2.peer = server1

        source1 = Source.constant(rate=5, target=server1, event_type="Request")

        link_ab, link_ba = PartitionLink.bidirectional("A", "B", min_latency=0.05)

        p1 = SimulationPartition(
            name="A", entities=[server1, sink1], sources=[source1]
        )
        p2 = SimulationPartition(name="B", entities=[server2, sink2])

        ps = ParallelSimulation(
            partitions=[p1, p2],
            duration=2.0,
            links=[link_ab, link_ba],
        )
        summary = ps.run()

        assert summary.total_events_processed > 0
        assert summary.total_cross_partition_events > 0
        assert summary.total_windows > 0

        # server1 handles primary requests
        assert server1.events_handled > 0
        # server2 should have received replicated events
        assert server2.events_handled > 0

    def test_unidirectional_link(self):
        """Events flow A→B but not B→A."""
        sink = Counter("sink")
        receiver = SimpleServer("receiver", downstream=sink, service_time=0.01)
        sender_sink = Counter("sender_sink")
        sender = ReplicatingServer(
            "sender", downstream=sender_sink, replicate_delay=0.05
        )
        sender.peer = receiver

        source = Source.constant(rate=5, target=sender, event_type="Request")

        # Only A→B link, no B→A
        link = PartitionLink(source_partition="A", dest_partition="B", min_latency=0.05)

        p1 = SimulationPartition(
            name="A", entities=[sender, sender_sink], sources=[source]
        )
        p2 = SimulationPartition(name="B", entities=[receiver, sink])

        ps = ParallelSimulation(
            partitions=[p1, p2],
            duration=2.0,
            links=[link],
        )
        summary = ps.run()

        assert receiver.events_handled > 0
        assert summary.total_cross_partition_events > 0


class TestMultiHop:
    """Three partitions A→B→C."""

    def test_three_partition_chain(self):
        sink = Counter("sink")
        server_c = SimpleServer("server_c", downstream=sink, service_time=0.01)
        # forward_delay >= min_latency so cross-partition events are valid
        server_b = SimpleServer(
            "server_b", downstream=server_c, service_time=0.01,
            forward_delay=0.05,
        )

        counter_a = Counter("counter_a")
        forwarder_a = ReplicatingServer(
            "forwarder_a", downstream=counter_a, replicate_delay=0.05
        )
        forwarder_a.peer = server_b

        source = Source.constant(rate=5, target=forwarder_a, event_type="Request")

        link_ab = PartitionLink(source_partition="A", dest_partition="B", min_latency=0.05)
        link_bc = PartitionLink(source_partition="B", dest_partition="C", min_latency=0.05)

        p_a = SimulationPartition(
            name="A", entities=[forwarder_a, counter_a], sources=[source]
        )
        p_b = SimulationPartition(name="B", entities=[server_b])
        p_c = SimulationPartition(name="C", entities=[server_c, sink])

        ps = ParallelSimulation(
            partitions=[p_a, p_b, p_c],
            duration=2.0,
            links=[link_ab, link_bc],
        )
        summary = ps.run()

        # Events should flow through all three partitions
        assert forwarder_a.events_handled > 0
        assert server_b.events_handled > 0
        assert server_c.events_handled > 0
        assert summary.total_cross_partition_events > 0


class TestNoLinksFallback:
    """ParallelSimulation with empty links list behaves like independent mode."""

    def test_empty_links_independent(self):
        counter = Counter("c")
        source = Source.constant(rate=10, target=counter, event_type="X")
        p = SimulationPartition(name="P", entities=[counter], sources=[source])
        ps = ParallelSimulation(partitions=[p], duration=5.0, links=[])
        summary = ps.run()

        assert counter.total == 50
        assert summary.total_windows == 0
        assert summary.total_cross_partition_events == 0


class TestCorrectnessVsSequential:
    """Compare parallel and sequential execution for deterministic setup."""

    def test_deterministic_equivalence(self):
        """Two independent partitions should produce same event counts
        as two separate Simulation runs."""
        # Sequential run 1
        counter1_seq = Counter("c1")
        source1_seq = Source.constant(rate=10, target=counter1_seq, event_type="X")
        sim1 = Simulation(
            duration=10.0, sources=[source1_seq], entities=[counter1_seq]
        )
        sim1.run()

        # Sequential run 2
        counter2_seq = Counter("c2")
        source2_seq = Source.constant(rate=10, target=counter2_seq, event_type="X")
        sim2 = Simulation(
            duration=10.0, sources=[source2_seq], entities=[counter2_seq]
        )
        sim2.run()

        # Parallel run
        counter1_par = Counter("c1")
        counter2_par = Counter("c2")
        source1_par = Source.constant(rate=10, target=counter1_par, event_type="X")
        source2_par = Source.constant(rate=10, target=counter2_par, event_type="X")

        p1 = SimulationPartition(
            name="P1", entities=[counter1_par], sources=[source1_par]
        )
        p2 = SimulationPartition(
            name="P2", entities=[counter2_par], sources=[source2_par]
        )
        ps = ParallelSimulation(partitions=[p1, p2], duration=10.0)
        ps.run()

        assert counter1_par.total == counter1_seq.total
        assert counter2_par.total == counter2_seq.total


class TestWindowBoundaryGenerators:
    """Generator spanning window boundary processes correctly."""

    def test_generator_spans_windows(self):
        """A generator that yields a delay longer than window_size should
        still process correctly across window boundaries."""
        sink = Counter("sink")
        # Service time 0.15s > window_size 0.1s
        server = SimpleServer("server", downstream=sink, service_time=0.15)

        source = Source.constant(rate=2, target=server, event_type="Request")

        link = PartitionLink(source_partition="A", dest_partition="B", min_latency=0.1)

        p1 = SimulationPartition(
            name="A", entities=[server, sink], sources=[source]
        )
        # Dummy partition B with a counter just to have links
        dummy = Counter("dummy")
        p2 = SimulationPartition(name="B", entities=[dummy])

        ps = ParallelSimulation(
            partitions=[p1, p2],
            duration=3.0,
            links=[link],
        )
        summary = ps.run()

        # Server should have processed events despite generator spanning windows
        assert server.events_handled > 0
        assert sink.total > 0


class TestSummaryFormat:
    """ParallelSimulationSummary has expected fields."""

    def test_to_dict(self):
        counter = Counter("c")
        source = Source.constant(rate=10, target=counter, event_type="X")
        p = SimulationPartition(name="P", entities=[counter], sources=[source])
        ps = ParallelSimulation(partitions=[p], duration=5.0)
        summary = ps.run()

        d = summary.to_dict()
        assert "duration_s" in d
        assert "total_events_processed" in d
        assert "partitions" in d
        assert "speedup" in d
        assert "parallelism_efficiency" in d

    def test_str(self):
        counter = Counter("c")
        source = Source.constant(rate=10, target=counter, event_type="X")
        p = SimulationPartition(name="P", entities=[counter], sources=[source])
        ps = ParallelSimulation(partitions=[p], duration=5.0)
        summary = ps.run()

        s = str(summary)
        assert "Parallel Simulation Summary" in s
        assert "Events processed" in s
