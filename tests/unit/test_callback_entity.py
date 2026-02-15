"""Unit tests for CallbackEntity, NullEntity, and Event.once()."""

from happysimulator.core.callback_entity import CallbackEntity, NullEntity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class TestCallbackEntity:
    def test_delegates_to_function(self):
        calls = []

        def fn(event):
            calls.append(event)
            return []

        entity = CallbackEntity("test", fn=fn)
        event = Event(time=Instant.Epoch, event_type="Test", target=entity)
        result = entity.handle_event(event)

        assert len(calls) == 1
        assert calls[0] is event
        assert result == []

    def test_returns_events_from_function(self):
        sink = NullEntity()
        follow_up = Event(time=Instant.Epoch, event_type="FollowUp", target=sink)

        def fn(_event):
            return [follow_up]

        entity = CallbackEntity("test", fn=fn)
        event = Event(time=Instant.Epoch, event_type="Test", target=entity)
        result = entity.handle_event(event)

        assert result == [follow_up]

    def test_returns_none(self):
        entity = CallbackEntity("test", fn=lambda _e: None)
        event = Event(time=Instant.Epoch, event_type="Test", target=entity)
        result = entity.handle_event(event)

        assert result is None

    def test_name_preserved(self):
        entity = CallbackEntity("my_name", fn=lambda _e: None)
        assert entity.name == "my_name"


class TestNullEntity:
    def test_discards_events(self):
        entity = NullEntity()
        event = Event(time=Instant.Epoch, event_type="Test", target=entity)
        result = entity.handle_event(event)

        assert result is None

    def test_is_singleton(self):
        a = NullEntity()
        b = NullEntity()
        assert a is b

    def test_name(self):
        assert NullEntity().name == "NullEntity"


class TestEventOnce:
    def test_creates_event_with_callback_entity_target(self):
        event = Event.once(
            time=Instant.Epoch,
            event_type="MyAction",
            fn=lambda _e: None,
        )

        assert isinstance(event, Event)
        assert event.time == Instant.Epoch
        assert event.event_type == "MyAction"
        assert isinstance(event.target, CallbackEntity)
        assert event.target.name == "once:MyAction"

    def test_invokes_function_on_dispatch(self):
        calls = []

        event = Event.once(
            time=Instant.Epoch,
            event_type="Track",
            fn=lambda e: calls.append(e),
        )
        event.invoke()

        assert len(calls) == 1

    def test_daemon_flag(self):
        event = Event.once(
            time=Instant.Epoch,
            event_type="Daemon",
            fn=lambda _e: None,
            daemon=True,
        )

        assert event.daemon is True

    def test_context_passed_through(self):
        event = Event.once(
            time=Instant.Epoch,
            event_type="Ctx",
            fn=lambda _e: None,
            context={"key": "value"},
        )

        assert event.context.get("key") == "value"

    def test_returns_follow_up_events(self):
        sink = NullEntity()
        follow_up = Event(time=Instant.Epoch, event_type="Next", target=sink)

        event = Event.once(
            time=Instant.from_seconds(1.0),
            event_type="Producer",
            fn=lambda _e: [follow_up],
        )
        result = event.invoke()

        assert follow_up in result
