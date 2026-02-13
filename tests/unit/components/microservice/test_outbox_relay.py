"""Tests for OutboxRelay component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator

import pytest

from happysimulator.components.microservice import (
    OutboxRelay,
    OutboxRelayStats,
    OutboxEntry,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant


@dataclass
class MessageCollector(Entity):
    """Collects relayed messages for testing."""

    name: str

    messages_received: int = field(default=0, init=False)
    payloads: list[dict] = field(default_factory=list, init=False)

    def handle_event(self, event: Event) -> None:
        self.messages_received += 1
        payload = event.context.get("payload")
        if payload:
            self.payloads.append(payload)


@dataclass
class OutboxWriter(Entity):
    """Entity that writes to an outbox during event handling."""

    name: str
    outbox: OutboxRelay
    payloads_to_write: list[dict] = field(default_factory=list)

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        for payload in self.payloads_to_write:
            self.outbox.write(payload)
        yield 0.001
        # Prime the outbox poll loop
        return [self.outbox.prime_poll()]


class TestOutboxRelayCreation:
    """Tests for OutboxRelay creation."""

    def test_creates_with_defaults(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(name="outbox", downstream=collector)

        assert outbox.name == "outbox"
        assert outbox.downstream is collector
        assert outbox.pending_count == 0
        assert outbox.total_entries == 0

    def test_initial_stats_are_zero(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(name="outbox", downstream=collector)

        assert outbox.stats.entries_written == 0
        assert outbox.stats.entries_relayed == 0
        assert outbox.stats.poll_cycles == 0
        assert outbox.stats.relay_lag_sum == 0.0
        assert outbox.stats.relay_lag_max == 0.0

    def test_rejects_invalid_poll_interval(self):
        collector = MessageCollector(name="downstream")
        with pytest.raises(ValueError):
            OutboxRelay(name="x", downstream=collector, poll_interval=0)

    def test_rejects_invalid_batch_size(self):
        collector = MessageCollector(name="downstream")
        with pytest.raises(ValueError):
            OutboxRelay(name="x", downstream=collector, batch_size=0)

    def test_rejects_negative_relay_latency(self):
        collector = MessageCollector(name="downstream")
        with pytest.raises(ValueError):
            OutboxRelay(name="x", downstream=collector, relay_latency=-1)


class TestOutboxRelayWriting:
    """Tests for writing entries to the outbox."""

    def test_write_increments_counter(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(name="outbox", downstream=collector)

        # Need a clock for write() to work
        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(0.01),
            sources=[],
            entities=[collector, outbox],
        )

        outbox.write({"order_id": 1})
        outbox.write({"order_id": 2})

        assert outbox.stats.entries_written == 2
        assert outbox.pending_count == 2
        assert outbox.total_entries == 2

    def test_write_returns_entry_id(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(name="outbox", downstream=collector)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(0.01),
            sources=[],
            entities=[collector, outbox],
        )

        id1 = outbox.write({"a": 1})
        id2 = outbox.write({"b": 2})

        assert id1 == 1
        assert id2 == 2


class TestOutboxRelayRelay:
    """Tests for relaying entries to downstream."""

    def test_relays_entries_on_poll(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(
            name="outbox",
            downstream=collector,
            poll_interval=0.05,
            relay_latency=0.0,
        )
        writer = OutboxWriter(
            name="writer",
            outbox=outbox,
            payloads_to_write=[{"order": 1}, {"order": 2}],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[collector, outbox, writer],
        )

        # Trigger the writer
        trigger = Event(
            time=Instant.Epoch,
            event_type="trigger",
            target=writer,
        )
        sim.schedule(trigger)
        sim.run()

        assert outbox.stats.entries_written == 2
        assert outbox.stats.entries_relayed == 2
        assert collector.messages_received == 2
        assert len(collector.payloads) == 2

    def test_batch_size_limits_per_poll(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(
            name="outbox",
            downstream=collector,
            poll_interval=0.05,
            batch_size=2,
            relay_latency=0.0,
        )
        writer = OutboxWriter(
            name="writer",
            outbox=outbox,
            payloads_to_write=[{"i": i} for i in range(5)],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[collector, outbox, writer],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="trigger",
            target=writer,
        )
        sim.schedule(trigger)
        sim.run()

        assert outbox.stats.entries_relayed == 5
        # Should take multiple poll cycles with batch_size=2
        assert outbox.stats.poll_cycles >= 3

    def test_tracks_relay_lag(self):
        collector = MessageCollector(name="downstream")
        outbox = OutboxRelay(
            name="outbox",
            downstream=collector,
            poll_interval=0.1,
            relay_latency=0.0,
        )
        writer = OutboxWriter(
            name="writer",
            outbox=outbox,
            payloads_to_write=[{"order": 1}],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[collector, outbox, writer],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="trigger",
            target=writer,
        )
        sim.schedule(trigger)
        sim.run()

        # Relay lag should be >= poll_interval (0.1s)
        assert outbox.stats.relay_lag_max >= 0.05
        assert outbox.stats.avg_relay_lag >= 0.05


class TestOutboxRelayStats:
    """Tests for OutboxRelayStats."""

    def test_avg_relay_lag_zero_when_no_relays(self):
        stats = OutboxRelayStats()
        assert stats.avg_relay_lag == 0.0

    def test_avg_relay_lag_computed(self):
        stats = OutboxRelayStats(entries_relayed=4, relay_lag_sum=2.0)
        assert stats.avg_relay_lag == 0.5
