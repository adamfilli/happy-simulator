"""Integration tests for simulation control: end-to-end workflows.

These tests exercise the control surface against realistic simulation
configurations with sources, queued resources, and probes.
"""

from collections.abc import Generator

from happysimulator import (
    ConditionBreakpoint,
    Counter,
    Entity,
    Event,
    EventCountBreakpoint,
    EventTypeBreakpoint,
    FIFOQueue,
    Instant,
    MetricBreakpoint,
    QueuedResource,
    Simulation,
    Source,
    TimeBreakpoint,
)

# -------------------------------------------------------
# Helper entities
# -------------------------------------------------------


class SimpleServer(QueuedResource):
    """A queued resource with explicit concurrency limit.

    With concurrency=1 and a slow service_time, the queue will build up
    when arrivals outpace processing.
    """

    def __init__(
        self, name: str, downstream: Entity, service_time: float = 0.01, concurrency: int = 1
    ):
        super().__init__(name, policy=FIFOQueue())
        self._downstream = downstream
        self._service_time = service_time
        self._concurrency = concurrency
        self._in_flight = 0

    def has_capacity(self) -> bool:
        return self._in_flight < self._concurrency

    def handle_queued_event(self, event: Event) -> Generator[float, None, list[Event]]:
        self._in_flight += 1
        yield self._service_time
        self._in_flight -= 1
        return [self.forward(event, self._downstream, event_type="Done")]


# -------------------------------------------------------
# Pause -> inspect -> resume workflow
# -------------------------------------------------------


def test_pause_inspect_resume():
    """Pause mid-simulation, inspect state, then resume to completion."""
    counter = Counter("sink")
    source = Source.constant(rate=10, target=counter, event_type="Ping", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[source],
        entities=[counter],
    )

    # Pause after 50 events (includes source self-scheduling events)
    sim.control.add_breakpoint(EventCountBreakpoint(count=50))
    partial = sim.run()

    assert partial.total_events_processed == 50
    assert sim.control.is_paused

    state = sim.control.get_state()
    assert state.events_processed == 50
    assert state.is_running
    assert state.heap_size > 0
    assert state.last_event is not None

    # Resume to completion
    final = sim.control.resume()
    assert final.total_events_processed > 50
    assert not sim.control.is_paused


def test_step_through_events():
    """Step through events one at a time and inspect intermediate state."""
    counter = Counter("sink")
    source = Source.constant(rate=1, target=counter, event_type="Tick", name="clock")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=5.0,
        sources=[source],
        entities=[counter],
    )

    # Pause immediately
    sim.control.pause()
    sim.run()

    # Step through 3 events one at a time
    for i in range(1, 4):
        summary = sim.control.step(1)
        assert summary.total_events_processed == i

    # Resume the rest
    final = sim.control.resume()
    # Total includes source self-scheduling events
    assert final.total_events_processed > 5


def test_time_breakpoint():
    """Break at a specific simulation time."""
    counter = Counter("sink")
    source = Source.constant(rate=10, target=counter, event_type="Ping", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[source],
        entities=[counter],
    )

    sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(5.0)))
    sim.run()

    state = sim.control.get_state()
    assert state.is_paused
    assert state.current_time >= Instant.from_seconds(5.0)
    # At 10 req/s with source self-scheduling, we expect ~100 events
    # by t=5s (50 Ping + 50 self-scheduling + 1 initial)
    assert state.events_processed > 0


def test_event_type_breakpoint():
    """Break when a specific event type is processed."""
    counter = Counter("sink")
    source = Source.constant(rate=1, target=counter, event_type="Tick", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=5.0,
        sources=[source],
        entities=[counter],
    )

    sim.control.add_breakpoint(EventTypeBreakpoint(event_type="Tick"))
    sim.run()

    state = sim.control.get_state()
    assert state.is_paused
    assert state.last_event.event_type == "Tick"


def test_metric_breakpoint_on_queue_depth():
    """Break when a queued resource's queue depth exceeds a threshold.

    With concurrency=1 and service_time=1.0s, arrivals at 10/s will
    quickly build up a queue because the server can only process 1 req/s.
    """
    counter = Counter("sink")
    server = SimpleServer("Server", downstream=counter, service_time=1.0, concurrency=1)
    source = Source.constant(rate=10, target=server, event_type="Request", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[source],
        entities=[server, counter],
    )

    sim.control.add_breakpoint(
        MetricBreakpoint(
            entity_name="Server",
            attribute="depth",
            operator="gt",
            threshold=3,
        )
    )
    sim.run()

    state = sim.control.get_state()
    assert state.is_paused
    assert server.depth > 3


def test_condition_breakpoint():
    """Break when an arbitrary condition is met."""
    counter = Counter("sink")
    source = Source.constant(rate=10, target=counter, event_type="Ping", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[source],
        entities=[counter],
    )

    # Break when counter has received 20+ events
    sim.control.add_breakpoint(
        ConditionBreakpoint(
            fn=lambda ctx: counter.total >= 20,
            description="counter >= 20",
        )
    )
    sim.run()

    assert sim.control.is_paused
    assert counter.total >= 20


def test_multiple_breakpoints():
    """Multiple breakpoints: first one to trigger wins."""
    counter = Counter("sink")
    source = Source.constant(rate=10, target=counter, event_type="Ping", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[source],
        entities=[counter],
    )

    # Time breakpoint at t=3s, count breakpoint at 50 events
    sim.control.add_breakpoint(TimeBreakpoint(time=Instant.from_seconds(3.0)))
    sim.control.add_breakpoint(EventCountBreakpoint(count=50))
    sim.run()

    state = sim.control.get_state()
    assert state.is_paused
    # One of the two should have triggered
    assert state.current_time >= Instant.from_seconds(3.0) or state.events_processed >= 50


def test_event_hooks_count_events():
    """Event hooks see every processed event."""
    counter = Counter("sink")
    source = Source.constant(rate=1, target=counter, event_type="Tick", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=5.0,
        sources=[source],
        entities=[counter],
    )

    seen_types: list[str] = []
    sim.control.on_event(lambda e: seen_types.append(e.event_type))

    sim.run()

    assert "Tick" in seen_types
    # Source self-scheduling events also get counted
    assert len(seen_types) > 5


def test_peek_and_find_while_paused():
    """Heap introspection works while paused."""
    counter = Counter("sink")
    source = Source.constant(rate=10, target=counter, event_type="Ping", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=5.0,
        sources=[source],
        entities=[counter],
    )

    sim.control.add_breakpoint(EventCountBreakpoint(count=10))
    sim.run()

    # Peek at next 5 events
    upcoming = sim.control.peek_next(5)
    assert len(upcoming) >= 1
    for i in range(1, len(upcoming)):
        assert upcoming[i].time >= upcoming[i - 1].time

    # Find all events in the heap
    all_events = sim.control.find_events(lambda e: True)
    assert len(all_events) > 0


def test_run_reset_run():
    """Run a simulation, reset, run again, and get consistent results."""
    counter = Counter("sink")
    source = Source.constant(rate=1, target=counter, event_type="Tick", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=5.0,
        sources=[source],
        entities=[counter],
    )

    summary1 = sim.run()
    count1 = counter.total

    sim.control.reset()

    # Counter state is NOT reset (entities own their state)
    # but the sim re-runs from scratch
    summary2 = sim.run()

    assert summary1.total_events_processed == summary2.total_events_processed
    # Counter accumulated across both runs
    assert counter.total == count1 * 2


def test_no_overhead_without_control():
    """Verify the simulation works fine without ever accessing .control."""
    counter = Counter("sink")
    source = Source.constant(rate=10, target=counter, event_type="Ping", name="gen")

    sim = Simulation(
        start_time=Instant.Epoch,
        duration=10.0,
        sources=[source],
        entities=[counter],
    )

    summary = sim.run()
    # Source generates both Ping events AND self-scheduling events
    assert summary.total_events_processed > 100
    # _control should still be None
    assert sim._control is None
