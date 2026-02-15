"""Unit tests for SimulationControl: pause, resume, step, hooks, introspection."""

import pytest

from happysimulator import (
    Counter,
    EventCountBreakpoint,
    Instant,
    Simulation,
    SimulationState,
    Source,
    TimeBreakpoint,
)


def _make_sim(rate: int = 10, duration: float = 10.0) -> Simulation:
    """Create a simple simulation with a constant source and counter."""
    counter = Counter("sink")
    source = Source.constant(rate=rate, target=counter, event_type="Ping", name="gen")
    return Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(duration),
        sources=[source],
        entities=[counter],
    )


# -------------------------------------------------------
# Lazy creation
# -------------------------------------------------------


class TestLazyCreation:
    def test_control_not_created_until_accessed(self):
        sim = _make_sim()
        assert sim._control is None

    def test_accessing_control_creates_it(self):
        sim = _make_sim()
        ctrl = sim.control
        assert ctrl is not None
        assert sim._control is ctrl

    def test_control_same_instance_on_repeat_access(self):
        sim = _make_sim()
        assert sim.control is sim.control


# -------------------------------------------------------
# Pause / Resume
# -------------------------------------------------------


class TestPauseResume:
    def test_pause_before_first_event(self):
        sim = _make_sim()
        sim.control.pause()
        summary = sim.run()
        # Should have paused before processing any events
        assert summary.total_events_processed == 0
        state = sim.control.get_state()
        assert state.is_paused is True
        assert state.is_running is True

    def test_resume_after_pause(self):
        sim = _make_sim(rate=1, duration=5.0)
        sim.control.pause()
        sim.run()
        assert sim.control.is_paused is True

        # Resume runs to completion
        summary = sim.control.resume()
        # Source generates both user events AND self-scheduling events
        assert summary.total_events_processed > 5
        assert sim.control.is_paused is False

    def test_resume_without_pause_raises(self):
        sim = _make_sim()
        with pytest.raises(RuntimeError, match="not paused"):
            sim.control.resume()

    def test_pause_during_run_via_breakpoint(self):
        sim = _make_sim(rate=10, duration=10.0)
        sim.control.add_breakpoint(EventCountBreakpoint(count=5))
        summary = sim.run()
        # Should have paused at exactly 5 events
        assert summary.total_events_processed == 5
        assert sim.control.is_paused is True


# -------------------------------------------------------
# Step
# -------------------------------------------------------


class TestStep:
    def test_step_one(self):
        sim = _make_sim(rate=1, duration=10.0)
        # First, pause the sim
        sim.control.pause()
        sim.run()
        assert sim.control.is_paused

        # Step one event
        summary = sim.control.step(1)
        assert summary.total_events_processed == 1

    def test_step_multiple(self):
        sim = _make_sim(rate=1, duration=10.0)
        sim.control.pause()
        sim.run()

        summary = sim.control.step(3)
        assert summary.total_events_processed == 3

    def test_step_invalid_count_raises(self):
        sim = _make_sim()
        sim.control.pause()
        sim.run()
        with pytest.raises(ValueError, match="step count"):
            sim.control.step(0)

    def test_step_when_not_running_raises(self):
        sim = _make_sim()
        with pytest.raises(RuntimeError, match="not running"):
            sim.control.step(1)

    def test_step_then_resume(self):
        sim = _make_sim(rate=1, duration=5.0)
        sim.control.pause()
        sim.run()

        sim.control.step(2)
        assert sim.control.is_paused
        state = sim.control.get_state()
        assert state.events_processed == 2

        # Resume to completion — total includes source self-scheduling events
        summary = sim.control.resume()
        assert summary.total_events_processed > 5


# -------------------------------------------------------
# Get State
# -------------------------------------------------------


class TestGetState:
    def test_initial_state(self):
        sim = _make_sim()
        state = sim.control.get_state()
        assert isinstance(state, SimulationState)
        assert state.current_time == Instant.Epoch
        assert state.events_processed == 0
        assert state.is_paused is False
        assert state.is_running is False
        assert state.is_complete is False
        assert state.last_event is None

    def test_state_after_run(self):
        sim = _make_sim(rate=1, duration=5.0)
        sim.run()
        state = sim.control.get_state()
        # Source generates user events + self-scheduling events
        assert state.events_processed > 5
        assert state.is_running is False
        assert state.is_complete is True

    def test_state_while_paused(self):
        sim = _make_sim(rate=10, duration=10.0)
        sim.control.add_breakpoint(EventCountBreakpoint(count=5))
        sim.run()
        state = sim.control.get_state()
        assert state.is_paused is True
        assert state.is_running is True
        assert state.is_complete is False
        assert state.events_processed == 5
        assert state.last_event is not None


# -------------------------------------------------------
# Breakpoint management
# -------------------------------------------------------


class TestBreakpointManagement:
    def test_add_and_list_breakpoints(self):
        sim = _make_sim()
        ctrl = sim.control
        bp1 = TimeBreakpoint(Instant.from_seconds(1.0))
        bp2 = EventCountBreakpoint(10)
        id1 = ctrl.add_breakpoint(bp1)
        id2 = ctrl.add_breakpoint(bp2)

        bps = ctrl.list_breakpoints()
        assert len(bps) == 2
        ids = [b[0] for b in bps]
        assert id1 in ids
        assert id2 in ids

    def test_remove_breakpoint(self):
        sim = _make_sim()
        ctrl = sim.control
        bp_id = ctrl.add_breakpoint(TimeBreakpoint(Instant.from_seconds(1.0)))
        ctrl.remove_breakpoint(bp_id)
        assert len(ctrl.list_breakpoints()) == 0

    def test_remove_nonexistent_raises(self):
        sim = _make_sim()
        with pytest.raises(KeyError):
            sim.control.remove_breakpoint("nonexistent")

    def test_clear_breakpoints(self):
        sim = _make_sim()
        ctrl = sim.control
        ctrl.add_breakpoint(TimeBreakpoint(Instant.from_seconds(1.0)))
        ctrl.add_breakpoint(EventCountBreakpoint(10))
        ctrl.clear_breakpoints()
        assert len(ctrl.list_breakpoints()) == 0

    def test_one_shot_breakpoint_auto_removed(self):
        sim = _make_sim(rate=10, duration=10.0)
        ctrl = sim.control
        # one_shot=True by default for EventCountBreakpoint
        ctrl.add_breakpoint(EventCountBreakpoint(count=5))
        assert len(ctrl.list_breakpoints()) == 1

        sim.run()  # Triggers breakpoint -> auto-removed
        assert len(ctrl.list_breakpoints()) == 0

    def test_non_one_shot_breakpoint_persists(self):
        sim = _make_sim(rate=10, duration=10.0)
        ctrl = sim.control
        ctrl.add_breakpoint(EventCountBreakpoint(count=5, one_shot=False))

        sim.run()  # Triggers breakpoint
        assert len(ctrl.list_breakpoints()) == 1


# -------------------------------------------------------
# Event hooks
# -------------------------------------------------------


class TestEventHooks:
    def test_on_event_fires_for_each_event(self):
        sim = _make_sim(rate=1, duration=3.0)
        events_seen = []
        sim.control.on_event(lambda e: events_seen.append(e.event_type))
        sim.run()
        # Source generates Ping events plus self-scheduling events
        assert "Ping" in events_seen
        assert len(events_seen) > 0

    def test_on_time_advance_fires(self):
        sim = _make_sim(rate=1, duration=3.0)
        times_seen = []
        sim.control.on_time_advance(lambda t: times_seen.append(t))
        sim.run()
        assert len(times_seen) > 0
        # Times should be non-decreasing
        for i in range(1, len(times_seen)):
            assert times_seen[i] >= times_seen[i - 1]

    def test_remove_hook(self):
        sim = _make_sim(rate=1, duration=3.0)
        events_seen = []
        hook_id = sim.control.on_event(lambda e: events_seen.append(e))
        sim.control.remove_hook(hook_id)
        sim.run()
        assert len(events_seen) == 0

    def test_remove_nonexistent_hook_raises(self):
        sim = _make_sim()
        with pytest.raises(KeyError):
            sim.control.remove_hook("nonexistent")


# -------------------------------------------------------
# Heap introspection
# -------------------------------------------------------


class TestHeapIntrospection:
    def test_peek_next_while_paused(self):
        sim = _make_sim(rate=10, duration=5.0)
        sim.control.pause()
        sim.run()

        events = sim.control.peek_next(3)
        assert len(events) <= 3
        # Events should be in time order
        for i in range(1, len(events)):
            assert events[i].time >= events[i - 1].time

    def test_peek_next_when_not_paused_raises(self):
        sim = _make_sim()
        with pytest.raises(RuntimeError, match="only available when paused"):
            sim.control.peek_next()

    def test_find_events_while_paused(self):
        sim = _make_sim(rate=10, duration=5.0)
        # Let a few events run first so the heap has a mix
        sim.control.add_breakpoint(EventCountBreakpoint(count=5))
        sim.run()

        # The heap should have events remaining
        assert sim.control.get_state().heap_size > 0

        # Find all events (any type)
        all_events = sim.control.find_events(lambda e: True)
        assert len(all_events) > 0

    def test_find_events_when_not_paused_raises(self):
        sim = _make_sim()
        with pytest.raises(RuntimeError, match="only available when paused"):
            sim.control.find_events(lambda e: True)


# -------------------------------------------------------
# Reset
# -------------------------------------------------------


class TestReset:
    def test_reset_after_completion(self):
        sim = _make_sim(rate=1, duration=3.0)
        sim.run()
        assert sim.control.get_state().is_complete

        sim.control.reset()
        state = sim.control.get_state()
        assert state.events_processed == 0
        assert state.is_running is False
        assert state.is_paused is False
        assert state.current_time == Instant.Epoch

    def test_reset_and_rerun(self):
        sim = _make_sim(rate=1, duration=3.0)
        summary1 = sim.run()

        sim.control.reset()
        summary2 = sim.run()

        assert summary1.total_events_processed == summary2.total_events_processed

    def test_reset_while_paused(self):
        sim = _make_sim(rate=10, duration=10.0)
        sim.control.pause()
        sim.run()
        assert sim.control.is_paused

        sim.control.reset()
        state = sim.control.get_state()
        assert state.is_paused is False
        assert state.events_processed == 0

    def test_reset_clears_heap(self):
        sim = _make_sim(rate=10, duration=10.0)
        sim.control.pause()
        sim.run()

        # Heap should have events before reset
        assert sim.control.get_state().heap_size > 0

        sim.control.reset()
        # After reset, heap should be re-primed (has initial events)
        assert sim.control.get_state().heap_size > 0
