"""Tests for StreamProcessor."""

import pytest

from happysimulator import (
    Entity,
    Event,
    Instant,
    Simulation,
)
from happysimulator.components.streaming.stream_processor import (
    LateEventPolicy,
    SessionWindow,
    SlidingWindow,
    StreamProcessor,
    TumblingWindow,
)


class RecordingSink(Entity):
    """Collects events for assertions."""

    def __init__(self, name: str = "sink"):
        super().__init__(name)
        self.events: list[Event] = []

    def handle_event(self, event):
        self.events.append(event)
        return


def _run_sim(entities, events, end_s=10.0):
    sim = Simulation(
        start_time=Instant.Epoch,
        end_time=Instant.from_seconds(end_s),
        sources=[],
        entities=entities,
    )
    for e in events:
        sim.schedule(e)
    sim.run()
    return sim


class TestTumblingWindow:
    """Tests for TumblingWindow."""

    def test_assigns_single_window(self):
        tw = TumblingWindow(size_s=10.0)
        windows = tw.assign_windows(5.0)
        assert windows == [(0.0, 10.0)]

    def test_assigns_correct_window_boundary(self):
        tw = TumblingWindow(size_s=10.0)
        windows = tw.assign_windows(15.0)
        assert windows == [(10.0, 20.0)]

    def test_should_close(self):
        tw = TumblingWindow(size_s=10.0)
        assert tw.should_close(10.0, watermark_s=10.0)
        assert not tw.should_close(10.0, watermark_s=9.0)

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError):
            TumblingWindow(size_s=0)


class TestSlidingWindow:
    """Tests for SlidingWindow."""

    def test_assigns_multiple_windows(self):
        sw = SlidingWindow(size_s=10.0, slide_s=5.0)
        windows = sw.assign_windows(7.0)
        # Event at t=7 belongs to windows [0,10) and [5,15)
        assert len(windows) >= 2
        starts = [w[0] for w in windows]
        assert 0.0 in starts
        assert 5.0 in starts

    def test_should_close(self):
        sw = SlidingWindow(size_s=10.0, slide_s=5.0)
        assert sw.should_close(10.0, watermark_s=10.0)
        assert not sw.should_close(10.0, watermark_s=9.0)

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError):
            SlidingWindow(size_s=0, slide_s=5.0)

    def test_rejects_zero_slide(self):
        with pytest.raises(ValueError):
            SlidingWindow(size_s=10.0, slide_s=0)


class TestSessionWindow:
    """Tests for SessionWindow."""

    def test_creates_initial_window(self):
        sw = SessionWindow(gap_s=5.0)
        windows = sw.assign_windows(10.0)
        assert windows == [(10.0, 15.0)]

    def test_should_close(self):
        sw = SessionWindow(gap_s=5.0)
        assert sw.should_close(15.0, watermark_s=15.0)
        assert not sw.should_close(15.0, watermark_s=14.0)

    def test_rejects_zero_gap(self):
        with pytest.raises(ValueError):
            SessionWindow(gap_s=0)


class TestStreamProcessorCreation:
    """Tests for StreamProcessor construction."""

    def test_creates_with_defaults(self):
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=10.0),
            aggregate_fn=len,
            downstream=sink,
        )
        assert proc.watermark_s == 0.0
        assert proc.active_windows == 0
        assert proc.total_windows_emitted == 0


class TestTumblingWindowProcessing:
    """Tests for tumbling window processing in simulation."""

    def test_emits_window_result(self):
        """Window result emitted when watermark advances past window end."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=5.0),
            aggregate_fn=lambda records: sum(r for r in records),
            downstream=sink,
            watermark_interval_s=1.0,
        )

        # Process events with event_time in window [0, 5)
        events = [
            Event(
                time=Instant.from_seconds(0.5 + i * 0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 10, "event_time_s": float(i)},
            )
            for i in range(3)
        ]

        _run_sim([proc, sink], events, end_s=15.0)

        # Should have emitted at least one window result
        window_results = [e for e in sink.events if e.event_type == "WindowResult"]
        assert len(window_results) >= 1

        result = window_results[0]
        assert result.context["key"] == "k1"
        assert result.context["result"] == 30  # 10 * 3
        assert result.context["record_count"] == 3

    def test_multiple_keys(self):
        """Events with different keys create separate windows."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=5.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=1.0,
        )

        events = []
        for key in ["k1", "k2"]:
            events.extend(
                Event(
                    time=Instant.from_seconds(0.5 + i * 0.1),
                    event_type="Process",
                    target=proc,
                    context={"key": key, "value": 1, "event_time_s": float(i)},
                )
                for i in range(2)
            )

        _run_sim([proc, sink], events, end_s=15.0)

        window_results = [e for e in sink.events if e.event_type == "WindowResult"]
        keys = {r.context["key"] for r in window_results}
        assert "k1" in keys
        assert "k2" in keys


class TestLateEventHandling:
    """Tests for late event policies."""

    def test_drop_policy(self):
        """Late events are dropped with DROP policy."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=5.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=0.5,
            late_event_policy=LateEventPolicy.DROP,
            allowed_lateness_s=0.0,
        )

        events = []
        # First event establishes watermark
        events.append(
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 10.0},
            )
        )

        # Advance watermark
        events.append(
            Event(
                time=Instant.from_seconds(5.0),
                event_type="Watermark",
                target=proc,
                context={"watermark_s": 12.0},
            )
        )

        # Late event (event_time=2.0 < watermark=12.0)
        events.append(
            Event(
                time=Instant.from_seconds(6.0),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 2.0},
            )
        )

        _run_sim([proc, sink], events, end_s=15.0)

        assert proc.stats.late_events_dropped >= 1

    def test_side_output_policy(self):
        """Late events routed to side output with SIDE_OUTPUT policy."""
        sink = RecordingSink()
        late_sink = RecordingSink("late")
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=5.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=0.5,
            late_event_policy=LateEventPolicy.SIDE_OUTPUT,
            side_output=late_sink,
            allowed_lateness_s=0.0,
        )

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 10.0},
            ),
            Event(
                time=Instant.from_seconds(5.0),
                event_type="Watermark",
                target=proc,
                context={"watermark_s": 12.0},
            ),
            Event(
                time=Instant.from_seconds(6.0),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": "late_value", "event_time_s": 2.0},
            ),
        ]

        _run_sim([proc, sink, late_sink], events, end_s=15.0)

        assert proc.stats.late_events_side_output >= 1
        late_events = [e for e in late_sink.events if e.event_type == "LateEvent"]
        assert len(late_events) >= 1

    def test_allowed_lateness(self):
        """Events within allowed lateness are not considered late."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=5.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=0.5,
            late_event_policy=LateEventPolicy.DROP,
            allowed_lateness_s=5.0,
        )

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 10.0},
            ),
            Event(
                time=Instant.from_seconds(5.0),
                event_type="Watermark",
                target=proc,
                context={"watermark_s": 12.0},
            ),
            # event_time=8.0, watermark=12.0, allowed_lateness=5.0
            # 8.0 >= 12.0 - 5.0 = 7.0, so NOT late
            Event(
                time=Instant.from_seconds(6.0),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 8.0},
            ),
        ]

        _run_sim([proc, sink], events, end_s=15.0)

        assert proc.stats.late_events == 0


class TestSessionWindowProcessing:
    """Tests for session window processing in simulation."""

    def test_session_merges_close_events(self):
        """Events within gap are merged into same session."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=SessionWindow(gap_s=2.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=1.0,
        )

        # Three events within 2s gap of each other
        events = [
            Event(
                time=Instant.from_seconds(0.5 + i * 0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": float(i)},
            )
            for i in range(3)
        ]

        _run_sim([proc, sink], events, end_s=15.0)

        # Should be a single session window (events at 0, 1, 2 are within gap=2)
        window_results = [e for e in sink.events if e.event_type == "WindowResult"]
        assert len(window_results) >= 1
        # All 3 records in one window
        total_records = sum(r.context["record_count"] for r in window_results)
        assert total_records == 3


class TestSlidingWindowProcessing:
    """Tests for sliding window processing in simulation."""

    def test_event_in_multiple_windows(self):
        """Events are assigned to overlapping windows."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=SlidingWindow(size_s=10.0, slide_s=5.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=1.0,
        )

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 7.0},
            )
        ]

        _run_sim([proc, sink], events, end_s=25.0)

        # Event at t=7 should be in windows [0,10) and [5,15)
        window_results = [e for e in sink.events if e.event_type == "WindowResult"]
        assert len(window_results) >= 2


class TestStreamProcessorStats:
    """Tests for StreamProcessorStats."""

    def test_stats_track_events(self):
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=5.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=1.0,
        )

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time_s": 1.0},
            )
        ]

        _run_sim([proc, sink], events, end_s=15.0)

        assert proc.stats.events_processed == 1
        assert proc.stats.windows_emitted >= 1


class TestEventTimeHandling:
    """Tests for event time extraction."""

    def test_event_time_from_instant(self):
        """Supports event_time as Instant."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=10.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=1.0,
        )

        events = [
            Event(
                time=Instant.from_seconds(0.1),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1, "event_time": Instant.from_seconds(5.0)},
            )
        ]

        _run_sim([proc, sink], events, end_s=15.0)

        assert proc.stats.events_processed == 1

    def test_defaults_to_sim_time(self):
        """Uses simulation time when no event_time provided."""
        sink = RecordingSink()
        proc = StreamProcessor(
            name="proc",
            window_type=TumblingWindow(size_s=10.0),
            aggregate_fn=len,
            downstream=sink,
            watermark_interval_s=1.0,
        )

        events = [
            Event(
                time=Instant.from_seconds(5.0),
                event_type="Process",
                target=proc,
                context={"key": "k1", "value": 1},
            )
        ]

        _run_sim([proc, sink], events, end_s=15.0)

        assert proc.stats.events_processed == 1
