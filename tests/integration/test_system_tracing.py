"""
Test: System/Engine-level tracing functionality.

These tests verify the TraceRecorder infrastructure and its integration
with Simulation and EventHeap. This is separate from application-level
tracing (Event.context["trace"]) which is tested in test_tracing_basic.py.
"""

from typing import List

import pytest

from happysimulator.entities.entity import Entity
from happysimulator.event_heap import EventHeap
from happysimulator.events.event import Event
from happysimulator.load.constant_arrival_time_provider import ConstantArrivalTimeProvider
from happysimulator.load.event_provider import EventProvider
from happysimulator.load.profile import Profile
from happysimulator.load.source import Source
from happysimulator.simulation import Simulation
from happysimulator.tracing import InMemoryTraceRecorder, NullTraceRecorder
from happysimulator.utils.instant import Instant


# --- Test Fixtures ---

class SimpleEntity(Entity):
    """Simple entity that counts events and optionally yields."""
    
    def __init__(self, name: str = "simple"):
        super().__init__(name)
        self.events_handled = 0
    
    def handle_event(self, event: Event) -> List[Event]:
        self.events_handled += 1
        return []


class DelayingEntity(Entity):
    """Entity that yields a delay before completing."""
    
    def __init__(self, delay: float = 0.1):
        super().__init__("delaying")
        self.delay = delay
        self.events_handled = 0
    
    def handle_event(self, event: Event):
        self.events_handled += 1
        yield self.delay
        return []


class ConstantRateProfile(Profile):
    """Returns a constant rate for a duration."""
    
    def __init__(self, rate: float, duration: float):
        self.rate = rate
        self.duration = duration
    
    def get_rate(self, time: Instant) -> float:
        if time <= Instant.from_seconds(self.duration):
            return self.rate
        return 0.0


class SimpleEventProvider(EventProvider):
    """Provides events targeting a simple entity."""
    
    def __init__(self, target: Entity):
        super().__init__()
        self.target = target
    
    def get_events(self, time: Instant) -> List[Event]:
        return [Event(time=time, event_type="SimpleEvent", target=self.target)]


# --- TraceRecorder Unit Tests ---

class TestInMemoryTraceRecorder:
    """Tests for InMemoryTraceRecorder functionality."""
    
    def test_record_basic_span(self):
        """Verify basic span recording."""
        recorder = InMemoryTraceRecorder()
        
        recorder.record(
            time=Instant.from_seconds(1.0),
            kind="test.span",
            event_id="abc-123",
            event_type="TestEvent",
        )
        
        assert len(recorder.spans) == 1
        span = recorder.spans[0]
        assert span["time"] == Instant.from_seconds(1.0)
        assert span["kind"] == "test.span"
        assert span["event_id"] == "abc-123"
        assert span["event_type"] == "TestEvent"
    
    def test_record_with_extra_data(self):
        """Verify extra data is captured in spans."""
        recorder = InMemoryTraceRecorder()
        
        recorder.record(
            time=Instant.from_seconds(2.0),
            kind="test.data",
            event_id="xyz-456",
            event_type="DataEvent",
            heap_size=42,
            custom_field="custom_value",
        )
        
        assert len(recorder.spans) == 1
        span = recorder.spans[0]
        assert span["data"]["heap_size"] == 42
        assert span["data"]["custom_field"] == "custom_value"
    
    def test_filter_by_kind(self):
        """Verify filtering spans by kind."""
        recorder = InMemoryTraceRecorder()
        
        recorder.record(time=Instant.Epoch, kind="heap.push")
        recorder.record(time=Instant.Epoch, kind="heap.pop")
        recorder.record(time=Instant.Epoch, kind="heap.push")
        recorder.record(time=Instant.Epoch, kind="simulation.start")
        
        push_spans = recorder.filter_by_kind("heap.push")
        assert len(push_spans) == 2
        
        pop_spans = recorder.filter_by_kind("heap.pop")
        assert len(pop_spans) == 1
        
        start_spans = recorder.filter_by_kind("simulation.start")
        assert len(start_spans) == 1
    
    def test_filter_by_event(self):
        """Verify filtering spans by event ID."""
        recorder = InMemoryTraceRecorder()
        
        recorder.record(time=Instant.Epoch, kind="heap.push", event_id="event-1")
        recorder.record(time=Instant.Epoch, kind="heap.pop", event_id="event-1")
        recorder.record(time=Instant.Epoch, kind="heap.push", event_id="event-2")
        
        event1_spans = recorder.filter_by_event("event-1")
        assert len(event1_spans) == 2
        
        event2_spans = recorder.filter_by_event("event-2")
        assert len(event2_spans) == 1
    
    def test_clear(self):
        """Verify clearing all spans."""
        recorder = InMemoryTraceRecorder()
        
        recorder.record(time=Instant.Epoch, kind="test.span")
        recorder.record(time=Instant.Epoch, kind="test.span")
        assert len(recorder.spans) == 2
        
        recorder.clear()
        assert len(recorder.spans) == 0


class TestNullTraceRecorder:
    """Tests for NullTraceRecorder (no-op) functionality."""
    
    def test_does_not_store_spans(self):
        """Verify NullTraceRecorder discards all spans."""
        recorder = NullTraceRecorder()
        
        # Should not raise and should not store
        recorder.record(
            time=Instant.from_seconds(1.0),
            kind="test.span",
            event_id="abc-123",
        )
        
        # NullTraceRecorder has no spans attribute by design
        assert not hasattr(recorder, "spans") or len(getattr(recorder, "spans", [])) == 0


# --- EventHeap Tracing Tests ---

class TestEventHeapTracing:
    """Tests for EventHeap tracing integration."""
    
    def test_heap_push_traced(self):
        """Verify heap.push spans are recorded."""
        recorder = InMemoryTraceRecorder()
        heap = EventHeap(trace_recorder=recorder)
        
        entity = SimpleEntity()
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="TestEvent",
            target=entity,
        )
        
        heap.push(event)
        
        push_spans = recorder.filter_by_kind("heap.push")
        assert len(push_spans) == 1
        assert push_spans[0]["event_type"] == "TestEvent"
        assert push_spans[0]["data"]["heap_size"] == 1
    
    def test_heap_pop_traced(self):
        """Verify heap.pop spans are recorded."""
        recorder = InMemoryTraceRecorder()
        heap = EventHeap(trace_recorder=recorder)
        
        entity = SimpleEntity()
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="TestEvent",
            target=entity,
        )
        
        heap.push(event)
        popped = heap.pop()
        
        pop_spans = recorder.filter_by_kind("heap.pop")
        assert len(pop_spans) == 1
        assert pop_spans[0]["event_type"] == "TestEvent"
        assert pop_spans[0]["data"]["heap_size"] == 0  # After pop
    
    def test_heap_multiple_events_traced(self):
        """Verify tracing of multiple push/pop operations."""
        recorder = InMemoryTraceRecorder()
        heap = EventHeap(trace_recorder=recorder)
        
        entity = SimpleEntity()
        events = [
            Event(time=Instant.from_seconds(i), event_type=f"Event{i}", target=entity)
            for i in range(3)
        ]
        
        for event in events:
            heap.push(event)
        
        while heap.has_events():
            heap.pop()
        
        push_spans = recorder.filter_by_kind("heap.push")
        pop_spans = recorder.filter_by_kind("heap.pop")
        
        assert len(push_spans) == 3
        assert len(pop_spans) == 3
    
    def test_heap_without_recorder_uses_null(self):
        """Verify heap works without explicit recorder (uses NullTraceRecorder)."""
        heap = EventHeap()  # No recorder
        
        entity = SimpleEntity()
        event = Event(time=Instant.from_seconds(1.0), event_type="Test", target=entity)
        
        # Should not raise
        heap.push(event)
        heap.pop()


# --- Simulation Tracing Tests ---

class TestSimulationTracing:
    """Tests for Simulation tracing integration."""
    
    def test_simulation_lifecycle_traced(self):
        """Verify simulation lifecycle spans are recorded."""
        recorder = InMemoryTraceRecorder()
        entity = SimpleEntity()
        
        profile = ConstantRateProfile(rate=1.0, duration=2.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(3.0),
            trace_recorder=recorder,
        )
        sim.run()
        
        # Verify lifecycle spans
        init_spans = recorder.filter_by_kind("simulation.init")
        assert len(init_spans) == 1
        assert init_spans[0]["data"]["num_sources"] == 1
        assert init_spans[0]["data"]["num_entities"] == 1
        
        start_spans = recorder.filter_by_kind("simulation.start")
        assert len(start_spans) == 1
        
        end_spans = recorder.filter_by_kind("simulation.end")
        assert len(end_spans) == 1
    
    def test_simulation_dequeue_traced(self):
        """Verify simulation.dequeue spans are recorded for each event."""
        recorder = InMemoryTraceRecorder()
        entity = SimpleEntity()
        
        profile = ConstantRateProfile(rate=2.0, duration=2.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(3.0),
            trace_recorder=recorder,
        )
        sim.run()
        
        dequeue_spans = recorder.filter_by_kind("simulation.dequeue")
        # Should have dequeue spans for source events + generated events
        assert len(dequeue_spans) >= 2, f"Expected at least 2 dequeue spans, got {len(dequeue_spans)}"
    
    def test_simulation_schedule_traced(self):
        """Verify simulation.schedule spans are recorded for new events."""
        recorder = InMemoryTraceRecorder()
        entity = DelayingEntity(delay=0.1)
        
        profile = ConstantRateProfile(rate=1.0, duration=1.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(3.0),
            trace_recorder=recorder,
        )
        sim.run()
        
        schedule_spans = recorder.filter_by_kind("simulation.schedule")
        # DelayingEntity creates ProcessContinuation events
        assert len(schedule_spans) >= 1, f"Expected at least 1 schedule span, got {len(schedule_spans)}"

    def test_trace_recorder_accessible_after_run(self):
        """Verify trace_recorder property allows post-run inspection."""
        recorder = InMemoryTraceRecorder()
        entity = SimpleEntity()
        
        profile = ConstantRateProfile(rate=1.0, duration=1.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(2.0),
            trace_recorder=recorder,
        )
        sim.run()
        
        # Access via property
        assert sim.trace_recorder is recorder
        assert len(sim.trace_recorder.spans) > 0


class TestSystemTracingIntegration:
    """Integration tests combining system and application tracing."""
    
    def test_event_journey_correlates_system_and_app_traces(self):
        """Verify event IDs correlate system traces with app traces."""
        recorder = InMemoryTraceRecorder()
        entity = SimpleEntity()
        
        profile = ConstantRateProfile(rate=1.0, duration=2.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(3.0),
            trace_recorder=recorder,
        )
        sim.run()
        
        # Get all unique event IDs from system traces
        system_event_ids = set()
        for span in recorder.spans:
            if "event_id" in span and span["event_id"]:
                system_event_ids.add(span["event_id"])
        
        # Verify we captured event IDs
        assert len(system_event_ids) > 0, "Should have captured event IDs in system traces"
        
        # Each event ID that was popped should have been pushed first.
        # Note: Not all pushed events get popped - some may be scheduled past end_time.
        for event_id in system_event_ids:
            event_spans = recorder.filter_by_event(event_id)
            span_kinds = {s["kind"] for s in event_spans}
            
            # If an event was popped, it must have been pushed first
            if "heap.pop" in span_kinds:
                assert "heap.push" in span_kinds, f"Event {event_id} was popped but never pushed"
    
    def test_trace_chronological_order(self):
        """Verify system trace spans are in chronological order."""
        recorder = InMemoryTraceRecorder()
        entity = SimpleEntity()
        
        profile = ConstantRateProfile(rate=2.0, duration=3.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(4.0),
            trace_recorder=recorder,
        )
        sim.run()
        
        times = [span["time"] for span in recorder.spans]
        
        # Verify times are non-decreasing (same time is allowed)
        for i in range(1, len(times)):
            assert times[i] >= times[i-1], \
                f"Trace times should be non-decreasing: {times[i-1]} > {times[i]} at index {i}"
    
    def test_simulation_without_recorder_works(self):
        """Verify simulation works correctly without explicit trace recorder."""
        entity = SimpleEntity()
        
        profile = ConstantRateProfile(rate=1.0, duration=2.0)
        provider = SimpleEventProvider(entity)
        arrival_provider = ConstantArrivalTimeProvider(profile, Instant.Epoch)
        
        source = Source(
            name="TestSource",
            event_provider=provider,
            arrival_time_provider=arrival_provider,
        )
        
        # No trace_recorder argument
        sim = Simulation(
            sources=[source],
            entities=[entity],
            probes=[],
            end_time=Instant.from_seconds(3.0),
        )
        
        # Should run without errors
        sim.run()
        
        # Entity should have processed events
        assert entity.events_handled >= 1
