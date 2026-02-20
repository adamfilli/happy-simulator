"""Unit tests for the event router factory."""

import pytest

from happysimulator.core.callback_entity import CallbackEntity
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.parallel.routing import make_event_router


class DummyEntity(Entity):
    def handle_event(self, event):
        return None


class TestMakeEventRouter:
    def setup_method(self):
        self.local_entity = DummyEntity("local")
        self.remote_entity = DummyEntity("remote")
        self.unlinked_entity = DummyEntity("unlinked")
        self.outbox: list = []
        self.router = make_event_router(
            partition_name="P1",
            local_entity_ids=frozenset([id(self.local_entity)]),
            linked_entity_ids=frozenset([id(self.remote_entity)]),
            outbox=self.outbox,
        )

    def test_local_event_returned(self):
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Test",
            target=self.local_entity,
        )
        result = self.router([event], Instant.from_seconds(1.0))
        assert result == [event]
        assert len(self.outbox) == 0

    def test_remote_event_goes_to_outbox(self):
        event = Event(
            time=Instant.from_seconds(2.0),
            event_type="Test",
            target=self.remote_entity,
        )
        send_time = Instant.from_seconds(1.0)
        result = self.router([event], send_time)
        assert result == []
        assert len(self.outbox) == 1
        assert self.outbox[0] == (event, send_time)

    def test_callback_entity_always_local(self):
        cb = CallbackEntity("cb", fn=lambda e: None)
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Test",
            target=cb,
        )
        result = self.router([event], Instant.from_seconds(1.0))
        assert result == [event]
        assert len(self.outbox) == 0

    def test_unlinked_entity_raises(self):
        event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Test",
            target=self.unlinked_entity,
        )
        with pytest.raises(RuntimeError, match="not reachable via any PartitionLink"):
            self.router([event], Instant.from_seconds(1.0))

    def test_mixed_events(self):
        local_event = Event(
            time=Instant.from_seconds(1.0),
            event_type="Local",
            target=self.local_entity,
        )
        remote_event = Event(
            time=Instant.from_seconds(2.0),
            event_type="Remote",
            target=self.remote_entity,
        )
        send_time = Instant.from_seconds(1.0)
        result = self.router([local_event, remote_event], send_time)
        assert result == [local_event]
        assert len(self.outbox) == 1
        assert self.outbox[0][0] is remote_event

    def test_empty_events_list(self):
        result = self.router([], Instant.from_seconds(1.0))
        assert result == []
        assert len(self.outbox) == 0
