"""Integration tests for CausalGraph with real simulations."""

from typing import Generator

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.core.simulation import Simulation
from happysimulator.components.common import Counter, Sink
from happysimulator.load.source import Source
from happysimulator.instrumentation.recorder import InMemoryTraceRecorder, NullTraceRecorder
from happysimulator.analysis.causal_graph import build_causal_graph, CausalGraph


# ---------------------------------------------------------------------------
# Helper entities
# ---------------------------------------------------------------------------

class ForwardingServer(Entity):
    """Simple server that forwards events downstream after a delay."""

    def __init__(self, name: str, downstream: Entity, delay: float = 0.1):
        super().__init__(name)
        self.downstream = downstream
        self._delay = delay
        self.processed = 0

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        yield self._delay
        self.processed += 1
        return [Event(
            time=self.now,
            event_type="Forwarded",
            target=self.downstream,
            context={"created_at": event.context.get("created_at", self.now)},
        )]


class SideEffectServer(Entity):
    """Server that emits a side-effect event mid-generator."""

    def __init__(self, name: str, downstream: Entity, side_target: Entity):
        super().__init__(name)
        self.downstream = downstream
        self.side_target = side_target

    def handle_event(self, event: Event) -> Generator:
        # Emit side-effect event during yield
        yield 0.05, [Event(
            time=self.now,
            event_type="SideEffect",
            target=self.side_target,
        )]
        yield 0.05
        return [Event(
            time=self.now,
            event_type="Done",
            target=self.downstream,
        )]


# ---------------------------------------------------------------------------
# Source → Server → Sink causal chain
# ---------------------------------------------------------------------------

class TestSourceServerSinkChain:
    def test_causal_chain_exists(self):
        """Events produced by Source→Server→Sink form a causal chain."""
        recorder = InMemoryTraceRecorder()
        sink = Counter("Sink")
        server = ForwardingServer("Server", downstream=sink)
        source = Source.constant(rate=10, target=server, event_type="Request", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[server, sink],
            trace_recorder=recorder,
        )
        sim.run()

        graph = build_causal_graph(recorder)
        assert len(graph) > 0

        # There should be root events (from source, no parent)
        roots = graph.roots()
        assert len(roots) > 0

        # There should be Forwarded events that are children of something
        forwarded = [n for n in graph.nodes.values() if n.event_type == "Forwarded"]
        assert len(forwarded) > 0
        for fwd in forwarded:
            assert fwd.parent_id is not None

    def test_forwarded_events_have_server_as_parent(self):
        """Forwarded events are caused by the Request events processing."""
        recorder = InMemoryTraceRecorder()
        sink = Counter("Sink")
        server = ForwardingServer("Server", downstream=sink)
        source = Source.constant(rate=2, target=server, event_type="Request", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[server, sink],
            trace_recorder=recorder,
        )
        sim.run()

        graph = build_causal_graph(recorder)

        forwarded = [n for n in graph.nodes.values() if n.event_type == "Forwarded"]
        for fwd in forwarded:
            # Walk up ancestors — should eventually reach a Request event
            ancestors = graph.ancestors(fwd.event_id)
            ancestor_types = {a.event_type for a in ancestors}
            assert "Request" in ancestor_types, (
                f"Forwarded event {fwd.event_id} has no Request ancestor. "
                f"Ancestors: {ancestor_types}"
            )


# ---------------------------------------------------------------------------
# Generator side-effect events have correct parent
# ---------------------------------------------------------------------------

class TestGeneratorSideEffects:
    def test_side_effect_parent_is_processing_event(self):
        """Side-effect events emitted during yield should have the
        processing event as parent."""
        recorder = InMemoryTraceRecorder()
        side_counter = Counter("SideCounter")
        done_counter = Counter("DoneCounter")
        server = SideEffectServer("Server", downstream=done_counter, side_target=side_counter)

        # Schedule a single event directly
        sim = Simulation(
            end_time=Instant.from_seconds(2.0),
            entities=[server, side_counter, done_counter],
            trace_recorder=recorder,
        )
        sim.schedule(Event(
            time=Instant.from_seconds(0.1),
            event_type="Trigger",
            target=server,
        ))
        sim.run()

        assert side_counter.total >= 1
        assert done_counter.total >= 1

        graph = build_causal_graph(recorder)

        side_effects = [n for n in graph.nodes.values() if n.event_type == "SideEffect"]
        assert len(side_effects) >= 1

        for se in side_effects:
            # Side-effect should trace back to the Trigger event
            ancestors = graph.ancestors(se.event_id)
            ancestor_types = {a.event_type for a in ancestors}
            assert "Trigger" in ancestor_types


# ---------------------------------------------------------------------------
# sim.causal_graph() convenience method
# ---------------------------------------------------------------------------

class TestSimCausalGraphConvenience:
    def test_with_recorder(self):
        """sim.causal_graph() works with InMemoryTraceRecorder."""
        recorder = InMemoryTraceRecorder()
        counter = Counter("Sink")
        source = Source.constant(rate=5, target=counter, event_type="Ping", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[counter],
            trace_recorder=recorder,
        )
        sim.run()

        graph = sim.causal_graph()
        assert isinstance(graph, CausalGraph)
        assert len(graph) > 0

    def test_without_recorder_raises(self):
        """sim.causal_graph() raises TypeError without InMemoryTraceRecorder."""
        counter = Counter("Sink")
        source = Source.constant(rate=5, target=counter, event_type="Ping", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[counter],
        )
        sim.run()

        with pytest.raises(TypeError, match="InMemoryTraceRecorder"):
            sim.causal_graph()

    def test_with_null_recorder_raises(self):
        """Explicit NullTraceRecorder also raises."""
        counter = Counter("Sink")
        sim = Simulation(
            end_time=Instant.from_seconds(1.0),
            entities=[counter],
            trace_recorder=NullTraceRecorder(),
        )
        sim.run()

        with pytest.raises(TypeError, match="NullTraceRecorder"):
            sim.causal_graph()

    def test_exclude_event_types(self):
        """sim.causal_graph(exclude_event_types=...) passes through."""
        recorder = InMemoryTraceRecorder()
        counter = Counter("Sink")
        source = Source.constant(rate=5, target=counter, event_type="Ping", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[counter],
            trace_recorder=recorder,
        )
        sim.run()

        full_graph = sim.causal_graph()
        filtered_graph = sim.causal_graph(exclude_event_types={"Ping"})

        # Filtered graph should have fewer nodes
        ping_count = sum(1 for n in full_graph.nodes.values() if n.event_type == "Ping")
        assert len(filtered_graph) == len(full_graph) - ping_count


# ---------------------------------------------------------------------------
# Multiple sources → independent root families
# ---------------------------------------------------------------------------

class TestMultipleSources:
    def test_independent_sources_create_separate_roots(self):
        """Two independent sources produce events with no shared ancestry."""
        recorder = InMemoryTraceRecorder()
        counter_a = Counter("SinkA")
        counter_b = Counter("SinkB")
        source_a = Source.constant(rate=5, target=counter_a, event_type="PingA", stop_after=1.0)
        source_b = Source.constant(rate=5, target=counter_b, event_type="PingB", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source_a, source_b],
            entities=[counter_a, counter_b],
            trace_recorder=recorder,
        )
        sim.run()

        graph = sim.causal_graph()

        # Should have multiple roots
        roots = graph.roots()
        assert len(roots) >= 2

        # PingA events should not share ancestry with PingB events
        ping_a_nodes = [n for n in graph.nodes.values() if n.event_type == "PingA"]
        ping_b_nodes = [n for n in graph.nodes.values() if n.event_type == "PingB"]

        if ping_a_nodes and ping_b_nodes:
            a_node = ping_a_nodes[0]
            a_ancestor_ids = {a.event_id for a in graph.ancestors(a_node.event_id)}
            a_ancestor_ids.add(a_node.event_id)

            b_node = ping_b_nodes[0]
            b_ancestor_ids = {b.event_id for b in graph.ancestors(b_node.event_id)}
            b_ancestor_ids.add(b_node.event_id)

            # No overlap between the two families
            assert a_ancestor_ids.isdisjoint(b_ancestor_ids)


# ---------------------------------------------------------------------------
# QueuedResource internal events + filtering
# ---------------------------------------------------------------------------

class TestQueuedResourceFiltering:
    def test_filter_internal_events(self):
        """Can filter out internal event types from QueuedResource."""
        from happysimulator.components.queued_resource import QueuedResource
        from happysimulator.components.queue import FIFOQueue

        class SimpleServer(QueuedResource):
            def __init__(self, name, downstream):
                super().__init__(name, policy=FIFOQueue())
                self.downstream = downstream

            def handle_queued_event(self, event):
                yield 0.1
                return [Event(
                    time=self.now,
                    event_type="Done",
                    target=self.downstream,
                )]

        recorder = InMemoryTraceRecorder()
        sink = Counter("Sink")
        server = SimpleServer("Server", downstream=sink)
        source = Source.constant(rate=5, target=server, event_type="Job", stop_after=1.0)

        sim = Simulation(
            end_time=Instant.from_seconds(5.0),
            sources=[source],
            entities=[server, sink],
            trace_recorder=recorder,
        )
        sim.run()

        full = sim.causal_graph()
        # Filter to only Job and Done events
        filtered = full.filter(lambda n: n.event_type in ("Job", "Done"))

        # Filtered should be smaller
        assert len(filtered) <= len(full)

        # All remaining nodes should be Job or Done
        for node in filtered.nodes.values():
            assert node.event_type in ("Job", "Done")
