"""Unit tests for breakpoint implementations."""

import pytest

from happysimulator.core.control.breakpoints import (
    ConditionBreakpoint,
    EventCountBreakpoint,
    EventTypeBreakpoint,
    MetricBreakpoint,
    TimeBreakpoint,
)
from happysimulator.core.control.state import BreakpointContext
from happysimulator.core.temporal import Instant
from happysimulator.core.event import Event
from happysimulator.core.callback_entity import NullEntity


def _make_context(
    time_s: float = 0.0,
    events_processed: int = 0,
    event_type: str = "Test",
    simulation=None,
) -> BreakpointContext:
    """Helper to build a BreakpointContext for testing."""
    target = NullEntity()
    event = Event(
        time=Instant.from_seconds(time_s),
        event_type=event_type,
        target=target,
    )
    return BreakpointContext(
        current_time=Instant.from_seconds(time_s),
        events_processed=events_processed,
        last_event=event,
        simulation=simulation,
    )


# -------------------------------------------------------
# TimeBreakpoint
# -------------------------------------------------------

class TestTimeBreakpoint:
    def test_triggers_at_exact_time(self):
        bp = TimeBreakpoint(time=Instant.from_seconds(5.0))
        ctx = _make_context(time_s=5.0)
        assert bp.should_break(ctx) is True

    def test_triggers_after_time(self):
        bp = TimeBreakpoint(time=Instant.from_seconds(5.0))
        ctx = _make_context(time_s=6.0)
        assert bp.should_break(ctx) is True

    def test_does_not_trigger_before_time(self):
        bp = TimeBreakpoint(time=Instant.from_seconds(5.0))
        ctx = _make_context(time_s=4.9)
        assert bp.should_break(ctx) is False

    def test_one_shot_default_true(self):
        bp = TimeBreakpoint(time=Instant.from_seconds(1.0))
        assert bp.one_shot is True

    def test_one_shot_can_be_false(self):
        bp = TimeBreakpoint(time=Instant.from_seconds(1.0), one_shot=False)
        assert bp.one_shot is False

    def test_str_representation(self):
        bp = TimeBreakpoint(time=Instant.from_seconds(5.0))
        s = str(bp)
        assert "TimeBreakpoint" in s


# -------------------------------------------------------
# EventCountBreakpoint
# -------------------------------------------------------

class TestEventCountBreakpoint:
    def test_triggers_at_exact_count(self):
        bp = EventCountBreakpoint(count=100)
        ctx = _make_context(events_processed=100)
        assert bp.should_break(ctx) is True

    def test_triggers_above_count(self):
        bp = EventCountBreakpoint(count=100)
        ctx = _make_context(events_processed=150)
        assert bp.should_break(ctx) is True

    def test_does_not_trigger_below_count(self):
        bp = EventCountBreakpoint(count=100)
        ctx = _make_context(events_processed=99)
        assert bp.should_break(ctx) is False

    def test_one_shot_default_true(self):
        bp = EventCountBreakpoint(count=10)
        assert bp.one_shot is True


# -------------------------------------------------------
# ConditionBreakpoint
# -------------------------------------------------------

class TestConditionBreakpoint:
    def test_triggers_when_fn_returns_true(self):
        bp = ConditionBreakpoint(fn=lambda ctx: ctx.events_processed > 5)
        ctx = _make_context(events_processed=10)
        assert bp.should_break(ctx) is True

    def test_does_not_trigger_when_fn_returns_false(self):
        bp = ConditionBreakpoint(fn=lambda ctx: ctx.events_processed > 50)
        ctx = _make_context(events_processed=10)
        assert bp.should_break(ctx) is False

    def test_one_shot_default_false(self):
        bp = ConditionBreakpoint(fn=lambda ctx: True)
        assert bp.one_shot is False

    def test_description(self):
        bp = ConditionBreakpoint(fn=lambda ctx: True, description="queue full")
        s = str(bp)
        assert "queue full" in s


# -------------------------------------------------------
# MetricBreakpoint
# -------------------------------------------------------

class _FakeEntity:
    def __init__(self, name: str, **attrs):
        self.name = name
        for k, v in attrs.items():
            setattr(self, k, v)


class _FakeSim:
    def __init__(self, entities):
        self._entities = entities


class TestMetricBreakpoint:
    def test_triggers_when_threshold_crossed(self):
        entity = _FakeEntity("Server", depth=15)
        sim = _FakeSim([entity])
        bp = MetricBreakpoint(
            entity_name="Server", attribute="depth", operator="gt", threshold=10,
        )
        ctx = _make_context(simulation=sim)
        assert bp.should_break(ctx) is True

    def test_does_not_trigger_below_threshold(self):
        entity = _FakeEntity("Server", depth=5)
        sim = _FakeSim([entity])
        bp = MetricBreakpoint(
            entity_name="Server", attribute="depth", operator="gt", threshold=10,
        )
        ctx = _make_context(simulation=sim)
        assert bp.should_break(ctx) is False

    def test_all_operators(self):
        entity = _FakeEntity("X", val=10)
        sim = _FakeSim([entity])

        for op, expected in [
            ("gt", False), ("ge", True), ("lt", False),
            ("le", True), ("eq", True), ("ne", False),
        ]:
            bp = MetricBreakpoint(
                entity_name="X", attribute="val", operator=op, threshold=10,
            )
            ctx = _make_context(simulation=sim)
            assert bp.should_break(ctx) is expected, f"Failed for operator {op}"

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError, match="Unknown operator"):
            MetricBreakpoint(
                entity_name="X", attribute="val", operator="??", threshold=10,
            )

    def test_missing_entity_returns_false(self):
        sim = _FakeSim([])
        bp = MetricBreakpoint(
            entity_name="Missing", attribute="depth", operator="gt", threshold=10,
        )
        ctx = _make_context(simulation=sim)
        assert bp.should_break(ctx) is False

    def test_missing_attribute_returns_false(self):
        entity = _FakeEntity("Server")
        sim = _FakeSim([entity])
        bp = MetricBreakpoint(
            entity_name="Server", attribute="nonexistent", operator="gt", threshold=10,
        )
        ctx = _make_context(simulation=sim)
        assert bp.should_break(ctx) is False

    def test_one_shot_default_false(self):
        bp = MetricBreakpoint(
            entity_name="X", attribute="val", operator="gt", threshold=10,
        )
        assert bp.one_shot is False


# -------------------------------------------------------
# EventTypeBreakpoint
# -------------------------------------------------------

class TestEventTypeBreakpoint:
    def test_triggers_on_matching_type(self):
        bp = EventTypeBreakpoint(event_type="Error")
        ctx = _make_context(event_type="Error")
        assert bp.should_break(ctx) is True

    def test_does_not_trigger_on_different_type(self):
        bp = EventTypeBreakpoint(event_type="Error")
        ctx = _make_context(event_type="Request")
        assert bp.should_break(ctx) is False

    def test_one_shot_default_false(self):
        bp = EventTypeBreakpoint(event_type="Error")
        assert bp.one_shot is False
