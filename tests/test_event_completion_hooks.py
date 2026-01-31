from __future__ import annotations

from typing import Generator

from happysimulator.core.event import Event, ProcessContinuation
from happysimulator.core.temporal import Instant


def test_completion_hooks_run_for_regular_event() -> None:
    hook_times: list[Instant] = []

    def hook(finish_time: Instant) -> Event:
        hook_times.append(finish_time)
        return Event(time=finish_time, event_type="hook", callback=lambda _: None)

    event_time = Instant.from_seconds(1.0)
    event = Event(time=event_time, event_type="regular", callback=lambda _: None)
    event.add_completion_hook(hook)

    produced = event.invoke()

    assert len(hook_times) == 1
    assert hook_times[0] == event_time
    assert [e.event_type for e in produced] == ["hook"]


def test_completion_hooks_run_when_process_finishes() -> None:
    hook_times: list[Instant] = []

    def process() -> Generator[float, None, None]:
        yield 0.1

    def hook(finish_time: Instant) -> Event:
        hook_times.append(finish_time)
        return Event(time=finish_time, event_type="hook", callback=lambda _: None)

    start = Instant.from_seconds(0.0)
    event = Event(time=start, event_type="proc", callback=lambda _: process())
    event.add_completion_hook(hook)

    first = event.invoke()
    assert len(first) == 1
    assert isinstance(first[0], ProcessContinuation)

    continuation = first[0]
    assert continuation.time == Instant.from_seconds(0.1)

    finished = continuation.invoke()

    assert len(hook_times) == 1
    assert hook_times[0] == Instant.from_seconds(0.1)
    assert [e.event_type for e in finished] == ["hook"]

    # Hooks are one-shot and should not remain after process completion.
    assert event.on_complete == []
