"""Unit tests for the code debugger module."""

import contextlib
import threading
import time

from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.queued_resource import QueuedResource
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.visual.code_debugger import (
    CodeBreakpoint,
    CodeDebugger,
    CodeLocation,
    ExecutionTrace,
    LineRecord,
    _safe_serialize_locals,
    _safe_serialize_value,
)

# --- Test entities ---


class SimpleServer(Entity):
    def __init__(self):
        super().__init__("SimpleServer")

    def handle_event(self, event):
        x = 1
        yield 0.1
        y = x + 1  # noqa: F841 - intentional: test inspects source code
        return []


class SimpleQueuedServer(QueuedResource):
    def __init__(self):
        super().__init__("QueuedServer", policy=FIFOQueue())

    def handle_queued_event(self, event):
        result = "processing"
        yield 0.05
        result = "done"  # noqa: F841 - intentional: test inspects source code
        return []


class NonGeneratorEntity(Entity):
    def __init__(self):
        super().__init__("Immediate")

    def handle_event(self, event):
        return []


# --- Source retrieval ---


class TestSourceRetrieval:
    def test_get_source_for_entity(self):
        debugger = CodeDebugger()
        server = SimpleServer()
        location = debugger.get_source(server)

        assert location is not None
        assert location.entity_name == "SimpleServer"
        assert location.class_name == "SimpleServer"
        assert location.method_name == "handle_event"
        assert len(location.source_lines) > 0
        assert location.start_line > 0

    def test_get_source_for_queued_resource(self):
        debugger = CodeDebugger()
        server = SimpleQueuedServer()
        location = debugger.get_source(server)

        assert location is not None
        assert location.method_name == "handle_queued_event"
        assert "result" in "".join(location.source_lines)

    def test_source_is_cached(self):
        debugger = CodeDebugger()
        server = SimpleServer()
        loc1 = debugger.get_source(server)
        loc2 = debugger.get_source(server)

        assert loc1 is loc2

    def test_get_source_for_non_generator(self):
        debugger = CodeDebugger()
        entity = NonGeneratorEntity()
        location = debugger.get_source(entity)

        assert location is not None
        assert location.method_name == "handle_event"


class TestCodeLocation:
    def test_to_dict(self):
        loc = CodeLocation(
            entity_name="Server",
            class_name="Server",
            method_name="handle_event",
            source_lines=["    def handle_event(self, event):", "        yield 0.1"],
            start_line=10,
        )
        d = loc.to_dict()
        assert d["entity_name"] == "Server"
        assert d["start_line"] == 10
        assert len(d["source_lines"]) == 2


# --- Entity activation ---


class TestEntityActivation:
    def test_activate_deactivate(self):
        debugger = CodeDebugger()
        debugger.activate_entity("Server")
        assert debugger.is_active("Server")

        debugger.deactivate_entity("Server")
        assert not debugger.is_active("Server")

    def test_deactivate_removes_breakpoints(self):
        debugger = CodeDebugger()
        debugger.activate_entity("Server")
        debugger.add_breakpoint(CodeBreakpoint(entity_name="Server", line_number=10))
        debugger.add_breakpoint(CodeBreakpoint(entity_name="Other", line_number=20))

        debugger.deactivate_entity("Server")
        bps = debugger.list_breakpoints()
        assert len(bps) == 1
        assert bps[0].entity_name == "Other"


# --- Breakpoints ---


class TestBreakpoints:
    def test_add_remove(self):
        debugger = CodeDebugger()
        bp_id = debugger.add_breakpoint(CodeBreakpoint(entity_name="S", line_number=5))
        assert len(debugger.list_breakpoints()) == 1

        assert debugger.remove_breakpoint(bp_id)
        assert len(debugger.list_breakpoints()) == 0

    def test_remove_nonexistent(self):
        debugger = CodeDebugger()
        assert not debugger.remove_breakpoint("no-such-id")

    def test_breakpoint_to_dict(self):
        bp = CodeBreakpoint(id="bp-1", entity_name="Server", line_number=42)
        d = bp.to_dict()
        assert d["id"] == "bp-1"
        assert d["line_number"] == 42


# --- Serialization ---


class TestSerialization:
    def test_safe_serialize_primitives(self):
        result = _safe_serialize_locals({"x": 1, "name": "hello", "flag": True})
        assert result == {"x": 1, "name": "hello", "flag": True}

    def test_safe_serialize_skips_self_and_private(self):
        result = _safe_serialize_locals({"self": object(), "_internal": 1, "public": 2})
        assert "self" not in result
        assert "_internal" not in result
        assert result["public"] == 2

    def test_safe_serialize_depth_limit(self):
        nested = {"a": {"b": {"c": {"d": 1}}}}
        result = _safe_serialize_locals(nested, max_depth=2)
        # At depth 2, the innermost dict should be repr'd
        assert isinstance(result["a"]["b"]["c"], str)

    def test_safe_serialize_value_list(self):
        result = _safe_serialize_value([1, 2, 3], depth=0, max_depth=2)
        assert result == [1, 2, 3]

    def test_safe_serialize_value_complex(self):
        result = _safe_serialize_value(object(), depth=0, max_depth=2)
        assert isinstance(result, str)

    def test_safe_serialize_none(self):
        assert _safe_serialize_value(None, depth=0, max_depth=2) is None


# --- Trace recording ---


class TestTraceRecording:
    def test_install_and_remove_trace(self):
        debugger = CodeDebugger()
        server = SimpleServer()
        debugger.get_source(server)
        debugger.activate_entity("SimpleServer")

        # Create a generator
        event = Event(
            time=Instant.from_seconds(0),
            event_type="Request",
            target=server,
        )
        gen = server.handle_event(event)

        debugger.install_trace(gen, "SimpleServer")

        # Advance the generator
        with contextlib.suppress(StopIteration):
            gen.send(None)

        debugger.remove_trace(gen)

        traces = debugger.drain_completed_traces()
        assert len(traces) == 1
        assert traces[0].entity_name == "SimpleServer"
        assert len(traces[0].lines) > 0

    def test_drain_clears_traces(self):
        debugger = CodeDebugger()
        # Manually add a trace
        debugger._completed_traces.append(
            ExecutionTrace(entity_name="S", method_name="handle_event", start_line=1)
        )
        traces = debugger.drain_completed_traces()
        assert len(traces) == 1

        # Second drain should be empty
        assert len(debugger.drain_completed_traces()) == 0


# --- Line record ---


class TestLineRecord:
    def test_to_dict_no_locals(self):
        lr = LineRecord(line_number=42)
        d = lr.to_dict()
        assert d == {"line_number": 42}
        assert "locals" not in d

    def test_to_dict_with_locals(self):
        lr = LineRecord(line_number=42, locals_snapshot={"x": 1})
        d = lr.to_dict()
        assert d["locals"] == {"x": 1}


# --- Execution trace ---


class TestExecutionTrace:
    def test_to_dict(self):
        trace = ExecutionTrace(
            entity_name="Server",
            method_name="handle_event",
            start_line=10,
            lines=[LineRecord(line_number=11), LineRecord(line_number=12)],
        )
        d = trace.to_dict()
        assert d["entity_name"] == "Server"
        assert len(d["lines"]) == 2


# --- Code debug state ---


class TestCodeDebugState:
    def test_get_state(self):
        debugger = CodeDebugger()
        debugger.activate_entity("Server")
        debugger.add_breakpoint(CodeBreakpoint(entity_name="Server", line_number=10))

        state = debugger.get_state()
        assert "Server" in state["active_entities"]
        assert len(state["breakpoints"]) == 1
        assert state["is_paused"] is False

    def test_reset(self):
        debugger = CodeDebugger()
        debugger.activate_entity("Server")
        debugger.add_breakpoint(CodeBreakpoint(entity_name="Server", line_number=10))

        debugger.reset()
        state = debugger.get_state()
        assert len(state["active_entities"]) == 0
        assert len(state["breakpoints"]) == 0


# --- Blocking / pause ---


class TestCodePause:
    def test_continue_unblocks(self):
        debugger = CodeDebugger()
        # Simulate a pause
        debugger._paused_entity = "Server"
        debugger._paused_line = 42
        debugger._pause_event.clear()

        assert debugger.is_code_paused()
        paused = debugger.get_paused_state()
        assert paused["entity_name"] == "Server"

        # Continue should unblock
        debugger.code_continue()
        assert debugger._pause_event.is_set()

    def test_code_step_sets_mode(self):
        debugger = CodeDebugger()
        debugger._pause_event.clear()
        debugger.code_step()
        assert debugger._step_mode == "step"
        assert debugger._pause_event.is_set()

    def test_code_step_over_sets_mode(self):
        debugger = CodeDebugger()
        debugger._pause_event.clear()
        debugger.code_step_over()
        assert debugger._step_mode == "step_over"

    def test_code_step_out_sets_mode(self):
        debugger = CodeDebugger()
        debugger._pause_event.clear()
        debugger.code_step_out()
        assert debugger._step_mode == "step_out"

    def test_deadman_timeout(self):
        """Verify that the deadman timeout prevents infinite blocking."""
        import happysimulator.visual.code_debugger as cdm

        # Use a very short timeout for testing
        original_timeout = cdm.DEADMAN_TIMEOUT_S
        cdm.DEADMAN_TIMEOUT_S = 0.1

        try:
            debugger = CodeDebugger()
            server = SimpleServer()
            debugger.get_source(server)
            debugger.activate_entity("SimpleServer")

            event = Event(
                time=Instant.from_seconds(0),
                event_type="Request",
                target=server,
            )
            gen = server.handle_event(event)

            # Set a breakpoint on the first code line
            location = debugger._source_cache["SimpleServer"]
            # Find a line with actual code
            for i, line in enumerate(location.source_lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("def ") and not stripped.startswith("#"):
                    bp_line = location.start_line + i
                    break

            debugger.add_breakpoint(CodeBreakpoint(entity_name="SimpleServer", line_number=bp_line))

            # Run gen.send() in a thread — it should block briefly then timeout
            result_holder = []

            def run_gen():
                debugger.install_trace(gen, "SimpleServer")
                try:
                    val = gen.send(None)
                    result_holder.append(val)
                except StopIteration as e:
                    result_holder.append(e.value)
                finally:
                    debugger.remove_trace(gen)

            t = threading.Thread(target=run_gen)
            start = time.monotonic()
            t.start()
            t.join(timeout=5.0)
            elapsed = time.monotonic() - start

            # Should have unblocked within ~0.1s (+ some margin)
            assert elapsed < 2.0
            assert not t.is_alive()
        finally:
            cdm.DEADMAN_TIMEOUT_S = original_timeout
