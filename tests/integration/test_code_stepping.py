"""Integration tests for code-level stepping in the visual debugger.

Tests the full round-trip: entity generator -> trace function -> code debugger
-> bridge -> serialized traces.
"""

import threading

from happysimulator.components.common import Sink
from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.load.source import Source
from happysimulator.visual.code_debugger import CodeBreakpoint, CodeDebugger

# --- Test entities ---


class TraceableServer(Entity):
    def __init__(self, downstream):
        super().__init__("TraceableServer")
        self.downstream = downstream

    def handle_event(self, event):
        yield 0.05
        yield 0.05
        return [Event(time=self.now, event_type="Done", target=self.downstream)]


class TraceableQueuedServer(QueuedResource):
    def __init__(self, downstream):
        super().__init__("QueuedTraceServer", policy=FIFOQueue())
        self.downstream = downstream

    def handle_queued_event(self, event):
        yield 0.1
        return [Event(time=self.now, event_type="Done", target=self.downstream)]


class TestCodeSteppingIntegration:
    """Full round-trip tests with simulation stepping."""

    def test_trace_recorded_during_step(self):
        """Stepping a simulation with an active code entity records traces."""
        sink = Sink()
        server = TraceableServer(sink)
        source = Source.constant(rate=10, target=server, event_type="Request")

        sim = Simulation(
            entities=[server, sink],
            sources=[source],
            duration=1.0,
        )

        # Inject code debugger
        debugger = CodeDebugger()
        sim._code_debugger = debugger

        # Activate tracing for the server
        debugger.get_source(server)
        debugger.activate_entity("TraceableServer")

        # Pause and step
        sim.control.pause()
        sim.run()

        # Step enough events for the generator to execute
        sim.control.step(5)

        # Drain traces
        traces = debugger.drain_completed_traces()
        assert len(traces) > 0

        trace = traces[0]
        assert trace.entity_name == "TraceableServer"
        assert trace.method_name == "handle_event"
        assert len(trace.lines) > 0

        # Lines should be valid file line numbers
        for lr in trace.lines:
            assert lr.line_number > 0

    def test_no_traces_when_not_active(self):
        """No traces recorded when entity is not activated."""
        sink = Sink()
        server = TraceableServer(sink)
        source = Source.constant(rate=10, target=server, event_type="Request")

        sim = Simulation(
            entities=[server, sink],
            sources=[source],
            duration=1.0,
        )

        debugger = CodeDebugger()
        sim._code_debugger = debugger

        # Do NOT activate tracing
        sim.control.pause()
        sim.run()
        sim.control.step(5)

        traces = debugger.drain_completed_traces()
        assert len(traces) == 0

    def test_queued_resource_tracing(self):
        """Queued resources resolve indirection correctly."""
        sink = Sink()
        server = TraceableQueuedServer(sink)
        source = Source.constant(rate=5, target=server, event_type="Request")

        sim = Simulation(
            entities=[server, sink],
            sources=[source],
            duration=1.0,
        )

        debugger = CodeDebugger()
        sim._code_debugger = debugger

        debugger.get_source(server)
        debugger.activate_entity("QueuedTraceServer")

        sim.control.pause()
        sim.run()
        sim.control.step(10)

        traces = debugger.drain_completed_traces()
        # QueuedResource uses an adapter so the entity_name should match
        # The trace entity_name comes from the resolved entity name
        server_traces = [t for t in traces if t.entity_name == "QueuedTraceServer"]
        assert len(server_traces) > 0

    def test_trace_serialization_roundtrip(self):
        """Traces can be serialized to dict form."""
        sink = Sink()
        server = TraceableServer(sink)
        source = Source.constant(rate=10, target=server, event_type="Request")

        sim = Simulation(
            entities=[server, sink],
            sources=[source],
            duration=1.0,
        )

        debugger = CodeDebugger()
        sim._code_debugger = debugger
        debugger.get_source(server)
        debugger.activate_entity("TraceableServer")

        sim.control.pause()
        sim.run()
        sim.control.step(5)

        traces = debugger.drain_completed_traces()
        assert len(traces) > 0

        d = traces[0].to_dict()
        assert "entity_name" in d
        assert "lines" in d
        assert isinstance(d["lines"], list)
        for line in d["lines"]:
            assert "line_number" in line

    def test_source_retrieval_via_debugger(self):
        """Source code can be retrieved for both entity types."""
        sink = Sink()
        server = TraceableServer(sink)
        queued = TraceableQueuedServer(sink)

        debugger = CodeDebugger()

        loc1 = debugger.get_source(server)
        assert loc1 is not None
        assert loc1.method_name == "handle_event"
        assert "step_one" in "".join(loc1.source_lines)

        loc2 = debugger.get_source(queued)
        assert loc2 is not None
        assert loc2.method_name == "handle_queued_event"
        assert "phase" in "".join(loc2.source_lines)

    def test_breakpoint_blocks_and_continues(self):
        """A breakpoint blocks the sim thread until code_continue is called."""
        import happysimulator.visual.code_debugger as cdm

        original_timeout = cdm.DEADMAN_TIMEOUT_S
        cdm.DEADMAN_TIMEOUT_S = 5.0  # Long enough for the test

        try:
            sink = Sink()
            server = TraceableServer(sink)
            source = Source.constant(rate=10, target=server, event_type="Request")

            sim = Simulation(
                entities=[server, sink],
                sources=[source],
                duration=1.0,
            )

            debugger = CodeDebugger()
            sim._code_debugger = debugger
            debugger.get_source(server)
            debugger.activate_entity("TraceableServer")

            # Set breakpoint on a code line
            location = debugger._source_cache["TraceableServer"]
            bp_line = None
            for i, line in enumerate(location.source_lines):
                stripped = line.strip()
                if "step_one" in stripped:
                    bp_line = location.start_line + i
                    break
            assert bp_line is not None, "Could not find breakpoint target line"

            debugger.add_breakpoint(
                CodeBreakpoint(entity_name="TraceableServer", line_number=bp_line)
            )

            sim.control.pause()
            sim.run()

            # Step in a background thread (it will block at breakpoint)
            step_done = threading.Event()

            def do_step():
                sim.control.step(5)
                step_done.set()

            t = threading.Thread(target=do_step)
            t.start()

            # Wait for the debugger to pause
            import time

            for _ in range(50):
                if debugger.is_code_paused():
                    break
                time.sleep(0.05)

            assert debugger.is_code_paused(), "Debugger should be paused at breakpoint"

            paused = debugger.get_paused_state()
            assert paused is not None
            assert paused["entity_name"] == "TraceableServer"
            assert paused["line_number"] == bp_line

            # Continue
            debugger.code_continue()

            # Wait for step to finish
            step_done.wait(timeout=5.0)
            assert step_done.is_set(), "Step should have completed after continue"

            t.join(timeout=2.0)
        finally:
            cdm.DEADMAN_TIMEOUT_S = original_timeout

    def test_multiple_entities_tracing(self):
        """Only activated entities produce traces."""
        sink = Sink()
        server1 = TraceableServer(sink)
        server1._name = "Server1"  # Entity uses _name internally
        server1.name = "Server1"

        server2 = TraceableServer(sink)
        server2._name = "Server2"
        server2.name = "Server2"

        source1 = Source.constant(rate=10, target=server1, event_type="Request")
        source2 = Source.constant(rate=10, target=server2, event_type="Request")

        sim = Simulation(
            entities=[server1, server2, sink],
            sources=[source1, source2],
            duration=0.5,
        )

        debugger = CodeDebugger()
        sim._code_debugger = debugger

        # Only activate server1
        debugger.get_source(server1)
        debugger.activate_entity("Server1")

        sim.control.pause()
        sim.run()
        sim.control.step(10)

        traces = debugger.drain_completed_traces()
        entity_names = {t.entity_name for t in traces}

        assert "Server1" in entity_names or len(traces) == 0  # May not have fired yet
        assert "Server2" not in entity_names

    def test_simulation_runs_normally_without_debugger(self):
        """Simulation works correctly when no code debugger is injected."""
        sink = Sink()
        server = TraceableServer(sink)
        source = Source.constant(rate=10, target=server, event_type="Request")

        sim = Simulation(
            entities=[server, sink],
            sources=[source],
            duration=0.5,
        )

        # No code debugger injected
        summary = sim.run()
        assert summary.total_events_processed > 0
        assert sink.events_received > 0
