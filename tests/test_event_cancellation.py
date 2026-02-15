"""Tests for event cancellation (lazy deletion pattern).

Covers cancel marking, idempotency, simulation-level skipping,
summary counting, process continuation cancellation, and
regression for normal event processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from happysimulator import (
    Counter,
    Entity,
    Event,
    Instant,
    Simulation,
    Source,
)
from happysimulator.core.callback_entity import NullEntity
from happysimulator.core.event import ProcessContinuation

if TYPE_CHECKING:
    from collections.abc import Generator

_null = NullEntity()


# ── Unit tests ───────────────────────────────────────────────────────────────


def test_cancel_marks_event_as_cancelled() -> None:
    event = Event(time=Instant.from_seconds(1.0), event_type="Test", target=_null)
    assert not event.cancelled

    event.cancel()
    assert event.cancelled


def test_cancel_is_idempotent() -> None:
    event = Event(time=Instant.from_seconds(1.0), event_type="Test", target=_null)
    event.cancel()
    event.cancel()  # Should not raise
    assert event.cancelled


# ── Integration tests ────────────────────────────────────────────────────────


def test_cancelled_event_skipped_in_simulation() -> None:
    """Schedule events, cancel one mid-run via Event.once(), verify the
    cancelled event's target never receives it."""
    received: list[str] = []

    class Collector(Entity):
        def handle_event(self, event: Event) -> None:
            received.append(event.event_type)

    collector = Collector("collector")

    # Three events at t=1, t=2, t=3
    e1 = Event(time=Instant.from_seconds(1.0), event_type="A", target=collector)
    e2 = Event(time=Instant.from_seconds(2.0), event_type="B", target=collector)
    e3 = Event(time=Instant.from_seconds(3.0), event_type="C", target=collector)

    # At t=1.5, cancel e2
    cancel_event = Event.once(
        time=Instant.from_seconds(1.5),
        event_type="CancelB",
        fn=lambda _: e2.cancel(),
    )

    sim = Simulation(
        end_time=Instant.from_seconds(10.0),
        entities=[collector],
    )
    sim.schedule([e1, e2, e3, cancel_event])

    summary = sim.run()

    # B should have been skipped
    assert received == ["A", "C"]
    assert summary.events_cancelled == 1


def test_cancelled_events_counted_in_summary() -> None:
    """Verify summary.events_cancelled reflects the correct count."""
    counter = Counter("sink")

    events = [
        Event(time=Instant.from_seconds(float(i)), event_type="Tick", target=counter)
        for i in range(1, 6)
    ]
    # Cancel events at t=2 and t=4
    events[1].cancel()
    events[3].cancel()

    sim = Simulation(
        end_time=Instant.from_seconds(10.0),
        entities=[counter],
    )
    sim.schedule(events)

    summary = sim.run()

    assert summary.events_cancelled == 2
    assert summary.total_events_processed == 3
    assert counter.total == 3


def test_cancel_process_continuation() -> None:
    """Cancel a multi-step process mid-yield; remaining steps must not execute."""
    steps_executed: list[int] = []

    class MultiStep(Entity):
        def handle_event(self, event: Event) -> Generator[float]:
            steps_executed.append(1)
            yield 0.1
            steps_executed.append(2)
            yield 0.1
            steps_executed.append(3)

    multi = MultiStep("multi")

    event = Event(time=Instant.from_seconds(1.0), event_type="Process", target=multi)

    sim = Simulation(
        end_time=Instant.from_seconds(10.0),
        entities=[multi],
    )
    sim.schedule(event)

    # At t=1.05 (between step 1 yield and step 2 resume), cancel the
    # continuation by scanning the heap directly.
    def cancel_continuation(_: Event) -> None:
        # By the time this fires (t=1.05), the first continuation (t=1.1)
        # is in the heap. Cancel all ProcessContinuations.
        for evt in sim._event_heap._heap:
            if isinstance(evt, ProcessContinuation):
                evt.cancel()

    cancel_event = Event.once(
        time=Instant.from_seconds(1.05),
        event_type="CancelContinuation",
        fn=cancel_continuation,
    )
    sim.schedule(cancel_event)

    sim.run()

    # Step 1 executes, but step 2 and 3 should not because the continuation
    # was cancelled after step 1 yielded.
    assert steps_executed == [1]


def test_uncancelled_events_unaffected() -> None:
    """Regression: normal events process correctly when cancellation is available."""
    counter = Counter("sink")

    source = Source.constant(rate=10, target=counter, event_type="Ping", stop_after=1.0)

    sim = Simulation(
        end_time=Instant.from_seconds(2.0),
        sources=[source],
        entities=[counter],
    )

    summary = sim.run()

    assert summary.events_cancelled == 0
    assert summary.total_events_processed > 0
    assert counter.total > 0


def test_summary_str_includes_cancelled() -> None:
    """Verify the human-readable summary includes cancelled count."""
    from happysimulator.instrumentation.summary import SimulationSummary

    summary = SimulationSummary(
        duration_s=10.0,
        total_events_processed=100,
        events_cancelled=5,
        events_per_second=10.0,
        wall_clock_seconds=0.01,
    )

    text = str(summary)
    assert "Events cancelled: 5" in text


def test_summary_to_dict_includes_cancelled() -> None:
    """Verify to_dict includes events_cancelled."""
    from happysimulator.instrumentation.summary import SimulationSummary

    summary = SimulationSummary(
        duration_s=10.0,
        total_events_processed=100,
        events_cancelled=3,
        events_per_second=10.0,
        wall_clock_seconds=0.01,
    )

    d = summary.to_dict()
    assert d["events_cancelled"] == 3
