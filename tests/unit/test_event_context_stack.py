"""Tests for event context stack tracking.

The context["stack"] field should record each entity that handles an event,
building a trace of the event's journey through the system.
"""

from __future__ import annotations

from happysimulator.components.common import Sink
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.load.source import Source

# --- Test entities ---


class ForwardingEntity(Entity):
    """Immediately forwards events to a downstream entity."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self.downstream = downstream

    def handle_event(self, event: Event):
        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.downstream,
                context=event.context,
            )
        ]


class YieldingForwarder(Entity):
    """Forwards after a generator yield (simulated delay)."""

    def __init__(self, name: str, downstream: Entity):
        super().__init__(name)
        self.downstream = downstream

    def handle_event(self, event: Event):
        yield 0.1
        return [
            Event(
                time=self.now,
                event_type=event.event_type,
                target=self.downstream,
                context=event.context,
            )
        ]


class TerminalEntity(Entity):
    """Consumes events and stores the final context for inspection."""

    def __init__(self, name: str):
        super().__init__(name)
        self.received_contexts: list[dict] = []

    def handle_event(self, event: Event):
        self.received_contexts.append(dict(event.context))
        return []


# --- Tests ---


class TestEventContextStack:
    """Verify that context['stack'] tracks event handling chain."""

    def test_single_entity_adds_to_stack(self):
        """A single entity handling an event should appear in the stack."""
        terminal = TerminalEntity("Terminal")
        sim = Simulation(
            sources=[],
            entities=[terminal],
            end_time=Instant.from_seconds(1.0),
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Ping",
                target=terminal,
            )
        )
        sim.run()

        assert len(terminal.received_contexts) == 1
        assert terminal.received_contexts[0]["stack"] == ["Terminal"]

    def test_forwarded_event_accumulates_stack(self):
        """When an event is forwarded with shared context, stack accumulates."""
        terminal = TerminalEntity("Terminal")
        forwarder = ForwardingEntity("Forwarder", downstream=terminal)
        sim = Simulation(
            sources=[],
            entities=[forwarder, terminal],
            end_time=Instant.from_seconds(1.0),
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Ping",
                target=forwarder,
            )
        )
        sim.run()

        assert len(terminal.received_contexts) == 1
        assert terminal.received_contexts[0]["stack"] == ["Forwarder", "Terminal"]

    def test_three_hop_chain(self):
        """Stack tracks a three-entity chain: A -> B -> C."""
        terminal = TerminalEntity("C")
        b = ForwardingEntity("B", downstream=terminal)
        a = ForwardingEntity("A", downstream=b)
        sim = Simulation(
            sources=[],
            entities=[a, b, terminal],
            end_time=Instant.from_seconds(1.0),
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Ping",
                target=a,
            )
        )
        sim.run()

        assert len(terminal.received_contexts) == 1
        assert terminal.received_contexts[0]["stack"] == ["A", "B", "C"]

    def test_generator_entity_adds_to_stack(self):
        """Generator-based entities (yield) should also appear in the stack."""
        terminal = TerminalEntity("Terminal")
        forwarder = YieldingForwarder("Yielder", downstream=terminal)
        sim = Simulation(
            sources=[],
            entities=[forwarder, terminal],
            end_time=Instant.from_seconds(1.0),
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Ping",
                target=forwarder,
            )
        )
        sim.run()

        assert len(terminal.received_contexts) == 1
        assert terminal.received_contexts[0]["stack"] == ["Yielder", "Terminal"]

    def test_separate_events_have_independent_stacks(self):
        """Two independent events should have separate stacks."""
        terminal = TerminalEntity("Terminal")
        sim = Simulation(
            sources=[],
            entities=[terminal],
            end_time=Instant.from_seconds(1.0),
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="First",
                target=terminal,
            )
        )
        sim.schedule(
            Event(
                time=Instant.from_seconds(0.2),
                event_type="Second",
                target=terminal,
            )
        )
        sim.run()

        assert len(terminal.received_contexts) == 2
        # Each event has its own stack with just "Terminal"
        assert terminal.received_contexts[0]["stack"] == ["Terminal"]
        assert terminal.received_contexts[1]["stack"] == ["Terminal"]

    def test_source_to_sink_stack(self):
        """Full pipeline: Source -> Server -> Sink tracks the journey."""
        sink = Sink("Sink")
        server = ForwardingEntity("Server", downstream=sink)
        source = Source.constant(rate=1, target=server, name="Source")

        sim = Simulation(
            sources=[source],
            entities=[server, sink],
            end_time=Instant.from_seconds(1.5),
        )
        sim.run()

        # The Request event was handled by Server then forwarded to Sink
        # Sink doesn't expose context directly, but we can verify via
        # the events_received count that the pipeline worked
        assert sink.events_received >= 1
