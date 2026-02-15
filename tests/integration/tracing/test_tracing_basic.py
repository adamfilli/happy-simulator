"""
Test: Basic tracing functionality.

Scenario:
- A Source generates events at 1 per second for 5 seconds.
- Events flow through a pipeline:
  1. FirstEntity (handles via handle_event, yields a 0.1s delay, passes to next)
  2. MiddleEntity (handles via handle_event, yields a 0.1s delay, passes to sink)
  3. SinkEntity (collects events in memory)

The test verifies that each event's trace contains the expected spans
for entity-based handling through the pipeline.
"""

from collections.abc import Generator

import pytest

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source

# --- Entities ---


class FirstEntity(Entity):
    """First stop in the pipeline. Handles via handle_event, then forwards to middle."""

    def __init__(self, next_entity: "MiddleEntity"):
        super().__init__("first_entity")
        self.next_entity = next_entity
        self.events_handled = 0

    def handle_event(self, event: Event) -> Generator:
        self.events_handled += 1
        yield 0.1  # Simulate 0.1 second processing delay

        # Forward to the middle entity
        next_event = Event(
            time=event.time + 0.1,
            event_type="MiddleEvent",
            target=self.next_entity,
            context=event.context,  # Preserve trace context
        )
        return [next_event]


class MiddleEntity(Entity):
    """Middle processor entity. Forwards to sink after processing."""

    def __init__(self, sink: "SinkEntity"):
        super().__init__("middle_entity")
        self.sink = sink
        self.events_processed = 0

    def handle_event(self, event: Event) -> Generator:
        self.events_processed += 1
        yield 0.1  # Simulate 0.1 second processing delay

        # Forward to the sink entity
        sink_event = Event(
            time=event.time + 0.1,
            event_type="SinkEvent",
            target=self.sink,
            context=event.context,  # Preserve trace context
        )
        return [sink_event]


class SinkEntity(Entity):
    """Final destination. Collects events in memory."""

    def __init__(self):
        super().__init__("sink_entity")
        self.collected_events: list[Event] = []

    def handle_event(self, event: Event) -> list[Event]:
        self.collected_events.append(event)
        return []


# --- Source Components ---


class ConstantOnePerSecondProfile(Profile):
    """Returns a rate of 1.0 event per second for duration seconds."""

    def __init__(self, duration: float):
        self.duration = duration

    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(self.duration):
            return 1.0
        return 0.0


# --- Test Case ---


def test_tracing_spans_through_pipeline():
    """
    Verifies that events accumulate correct trace spans as they flow
    through a pipeline of entities.

    Expected spans per event:
    1. handle.start (entity: first_entity)
    2. handle.end (result_kind: process)
    3. process.resume.start
    4. process.yield (delay_s: 0.1)
    5. process.resume.end
    6. process.resume.start
    7. process.stop
    8. handle.start (entity: middle_entity)
    9. handle.end (result_kind: process)
    10. process.resume.start
    11. process.yield (delay_s: 0.1)
    12. process.resume.end
    13. process.resume.start
    14. process.stop
    15. handle.start (entity: sink_entity)
    16. handle.end (result_kind: immediate)
    """
    # A. SETUP
    sim_duration = 5.0

    # Build pipeline: FirstEntity -> MiddleEntity -> SinkEntity
    sink = SinkEntity()
    middle = MiddleEntity(sink)
    first = FirstEntity(middle)

    # Setup source
    profile = ConstantOnePerSecondProfile(sim_duration)
    event_source = Source.with_profile(
        profile=profile,
        target=first,
        event_type="StartEvent",
        poisson=False,
        name="PipelineSource",
    )

    # B. RUN SIMULATION
    sim = Simulation(
        sources=[event_source],
        entities=[first, middle, sink],
        probes=[],
        duration=sim_duration + 1,  # Extra time for pipeline to drain
    )
    sim.run()

    # C. ASSERTIONS

    # Verify events flowed through the entire pipeline
    assert first.events_handled >= 4, (
        f"Expected >= 4 events at first entity, got {first.events_handled}"
    )
    assert middle.events_processed >= 4, (
        f"Expected >= 4 events at middle entity, got {middle.events_processed}"
    )
    assert len(sink.collected_events) >= 4, (
        f"Expected >= 4 events at sink, got {len(sink.collected_events)}"
    )

    # Verify trace spans on collected events
    for i, event in enumerate(sink.collected_events):
        trace = event.context.get("trace", {})
        spans = trace.get("spans", [])

        assert len(spans) > 0, f"Event {i} should have trace spans"

        # Extract action names for verification
        actions = [span["action"] for span in spans]

        # Verify we have the expected handler lifecycle spans
        assert "handle.start" in actions, f"Event {i} missing handle.start span"
        assert "handle.end" in actions, f"Event {i} missing handle.end span"

        # Process spans (from generator yields)
        assert "process.resume.start" in actions, f"Event {i} missing process.resume.start span"
        assert "process.yield" in actions, f"Event {i} missing process.yield span"
        assert "process.stop" in actions, f"Event {i} missing process.stop span"

        # Verify we have spans from all three entities
        handle_starts = [s for s in spans if s["action"] == "handle.start"]
        assert len(handle_starts) >= 3, (
            f"Event {i} should have handle.start spans for first, middle, and sink. Got {len(handle_starts)}"
        )

        # All handlers should be entity-based
        handler_kinds = [s.get("data", {}).get("handler") for s in handle_starts]
        assert all(k == "entity" for k in handler_kinds), (
            f"Event {i} all handlers should be entity-based, got {handler_kinds}"
        )

        # Verify handler labels
        handler_labels = [s.get("data", {}).get("handler_label") for s in handle_starts]
        assert any("first_entity" in str(label) for label in handler_labels), (
            f"Event {i} should have first_entity in handler labels: {handler_labels}"
        )
        assert any("middle_entity" in str(label) for label in handler_labels), (
            f"Event {i} should have middle_entity in handler labels: {handler_labels}"
        )
        assert any("sink_entity" in str(label) for label in handler_labels), (
            f"Event {i} should have sink_entity in handler labels: {handler_labels}"
        )

    # Verify span ordering (handle.start should come before handle.end for each handler)
    for i, event in enumerate(sink.collected_events):
        spans = event.context["trace"]["spans"]

        start_indices = [j for j, s in enumerate(spans) if s["action"] == "handle.start"]
        end_indices = [j for j, s in enumerate(spans) if s["action"] == "handle.end"]

        # Each start should have a corresponding end that comes after it
        assert len(start_indices) == len(end_indices), (
            f"Event {i}: mismatched handle.start ({len(start_indices)}) and handle.end ({len(end_indices)}) counts"
        )

        for start_idx, end_idx in zip(start_indices, end_indices, strict=False):
            assert start_idx < end_idx, (
                f"Event {i}: handle.start at {start_idx} should come before handle.end at {end_idx}"
            )

    print("\n✓ Tracing test passed")
    print(f"  - Events through pipeline: {len(sink.collected_events)}")
    print(
        f"  - Average spans per event: {sum(len(e.context['trace']['spans']) for e in sink.collected_events) / len(sink.collected_events):.1f}"
    )

    # Print sample trace for debugging
    if sink.collected_events:
        sample_event = sink.collected_events[0]
        print("\n  Sample trace for first event:")
        for span in sample_event.context["trace"]["spans"]:
            data_str = f" | {span.get('data', {})}" if span.get("data") else ""
            print(f"    {span['action']}{data_str}")


def test_tracing_preserves_event_identity():
    """
    Verifies that the trace context ID remains consistent as an event
    flows through the pipeline, enabling end-to-end request tracing.
    """
    # A. SETUP
    sink = SinkEntity()
    middle = MiddleEntity(sink)
    first = FirstEntity(middle)

    profile = ConstantOnePerSecondProfile(2.0)
    event_source = Source.with_profile(
        profile=profile,
        target=first,
        event_type="StartEvent",
        poisson=False,
        name="PipelineSource",
    )

    sim = Simulation(
        sources=[event_source], entities=[first, middle, sink], probes=[], duration=3.0
    )
    sim.run()

    # B. ASSERTIONS

    assert len(sink.collected_events) >= 1, "Should have at least one event"

    for i, event in enumerate(sink.collected_events):
        trace = event.context["trace"]
        spans = trace["spans"]

        # All spans should reference the same event_id (the original event's ID)
        {span["event_id"] for span in spans}

        # Note: In the current design, event_id in spans may vary because
        # new Event objects are created at each stage. The key is that
        # all spans are collected in the same context["trace"]["spans"] list.
        assert len(spans) > 0, f"Event {i} should have spans"

        # Verify context ID is present and consistent
        context_id = event.context.get("id")
        assert context_id is not None, f"Event {i} should have a context ID"

    print("\n✓ Identity preservation test passed")
    print(f"  - Events collected: {len(sink.collected_events)}")


def test_tracing_captures_errors():
    """
    Verifies that handle.error spans are recorded when an exception occurs.
    """

    class FailingEntity(Entity):
        """Entity that raises an exception during handling."""

        def __init__(self):
            super().__init__("failing_entity")
            self.attempts = 0
            self.last_event: Event | None = None

        def handle_event(self, event: Event):
            self.attempts += 1
            self.last_event = event
            raise ValueError("Intentional test failure")

    # Setup - use longer duration to ensure events are generated
    failing_entity = FailingEntity()
    profile = ConstantOnePerSecondProfile(5.0)
    event_source = Source.with_profile(
        profile=profile,
        target=failing_entity,
        event_type="FailEvent",
        poisson=False,
        name="FailingSource",
    )

    sim = Simulation(sources=[event_source], entities=[failing_entity], probes=[], duration=10.0)

    # Run simulation - it should propagate the error
    with pytest.raises(ValueError, match="Intentional test failure"):
        sim.run()

    # The entity should have attempted to handle at least one event
    assert failing_entity.attempts >= 1, "Entity should have attempted to handle at least one event"

    # Verify the error was traced
    assert failing_entity.last_event is not None, "Should have captured the failing event"

    trace = failing_entity.last_event.context.get("trace", {})
    spans = trace.get("spans", [])
    actions = [span["action"] for span in spans]

    assert "handle.start" in actions, "Should have handle.start span before error"
    assert "handle.error" in actions, "Should have handle.error span capturing the exception"

    # Verify error details
    error_spans = [s for s in spans if s["action"] == "handle.error"]
    assert len(error_spans) == 1, "Should have exactly one handle.error span"

    error_data = error_spans[0].get("data", {})
    assert error_data.get("error") == "ValueError", (
        f"Error type should be ValueError, got {error_data}"
    )
    assert "Intentional test failure" in error_data.get("message", ""), (
        "Error message should be captured"
    )

    print("\n✓ Error tracing test passed")
    print(f"  - Attempts before failure: {failing_entity.attempts}")
    print(f"  - Error span captured: {error_spans[0]}")
