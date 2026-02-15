"""Stateful windowed stream processor.

Models stream processing with windowed aggregation, watermark-based
progress tracking, and late event handling.

Example::

    from happysimulator.components.streaming import (
        StreamProcessor, TumblingWindow, LateEventPolicy,
    )

    processor = StreamProcessor(
        name="aggregator",
        window_type=TumblingWindow(size_s=10.0),
        aggregate_fn=lambda records: sum(r["amount"] for r in records),
        downstream=sink,
        allowed_lateness_s=5.0,
        late_event_policy=LateEventPolicy.SIDE_OUTPUT,
        side_output=late_sink,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Protocol
from abc import abstractmethod
from enum import Enum

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


class WindowType(Protocol):
    """Protocol for window assignment strategies."""

    @abstractmethod
    def assign_windows(self, event_time_s: float) -> list[tuple[float, float]]:
        """Assign an event to windows based on its event time.

        Args:
            event_time_s: Event time in seconds.

        Returns:
            List of (window_start, window_end) tuples.
        """
        ...

    @abstractmethod
    def should_close(self, window_end: float, watermark_s: float) -> bool:
        """Whether a window should close given the current watermark.

        Args:
            window_end: End time of the window.
            watermark_s: Current watermark in seconds.

        Returns:
            True if the window should be closed and emitted.
        """
        ...


class TumblingWindow:
    """Non-overlapping fixed-size windows.

    Each event belongs to exactly one window.

    Args:
        size_s: Window size in seconds.
    """

    def __init__(self, size_s: float):
        if size_s <= 0:
            raise ValueError(f"size_s must be > 0, got {size_s}")
        self._size_s = size_s

    @property
    def size_s(self) -> float:
        return self._size_s

    def assign_windows(self, event_time_s: float) -> list[tuple[float, float]]:
        start = (event_time_s // self._size_s) * self._size_s
        return [(start, start + self._size_s)]

    def should_close(self, window_end: float, watermark_s: float) -> bool:
        return watermark_s >= window_end


class SlidingWindow:
    """Overlapping windows with fixed size and slide interval.

    Each event may belong to multiple windows.

    Args:
        size_s: Window size in seconds.
        slide_s: Slide interval in seconds.
    """

    def __init__(self, size_s: float, slide_s: float):
        if size_s <= 0:
            raise ValueError(f"size_s must be > 0, got {size_s}")
        if slide_s <= 0:
            raise ValueError(f"slide_s must be > 0, got {slide_s}")
        self._size_s = size_s
        self._slide_s = slide_s

    @property
    def size_s(self) -> float:
        return self._size_s

    @property
    def slide_s(self) -> float:
        return self._slide_s

    def assign_windows(self, event_time_s: float) -> list[tuple[float, float]]:
        windows = []
        # Find the earliest window that contains this event
        last_start = (event_time_s // self._slide_s) * self._slide_s
        start = last_start
        while start + self._size_s > event_time_s:
            windows.append((start, start + self._size_s))
            start -= self._slide_s
            if start < 0:
                break
        return sorted(windows)

    def should_close(self, window_end: float, watermark_s: float) -> bool:
        return watermark_s >= window_end


class SessionWindow:
    """Dynamic windows based on activity gaps.

    Events within the gap threshold are merged into the same session.

    Args:
        gap_s: Maximum inactivity gap in seconds.
    """

    def __init__(self, gap_s: float):
        if gap_s <= 0:
            raise ValueError(f"gap_s must be > 0, got {gap_s}")
        self._gap_s = gap_s

    @property
    def gap_s(self) -> float:
        return self._gap_s

    def assign_windows(self, event_time_s: float) -> list[tuple[float, float]]:
        # Initial window for session: [event_time, event_time + gap]
        return [(event_time_s, event_time_s + self._gap_s)]

    def should_close(self, window_end: float, watermark_s: float) -> bool:
        return watermark_s >= window_end


class LateEventPolicy(Enum):
    """How to handle events that arrive after their window has closed."""

    DROP = "drop"
    UPDATE = "update"
    SIDE_OUTPUT = "side_output"


@dataclass
class WindowState:
    """State of an active window.

    Attributes:
        start: Window start time.
        end: Window end time.
        records: Records accumulated in this window.
        emitted: Whether the window result has been emitted.
    """

    start: float
    end: float
    records: list[Any] = field(default_factory=list)
    emitted: bool = False


@dataclass(frozen=True)
class StreamProcessorStats:
    """Statistics tracked by StreamProcessor.

    Attributes:
        events_processed: Total events received.
        windows_emitted: Total windows emitted.
        late_events: Total late events received.
        late_events_dropped: Late events that were dropped.
        late_events_updated: Late events applied to closed windows.
        late_events_side_output: Late events sent to side output.
    """

    events_processed: int = 0
    windows_emitted: int = 0
    late_events: int = 0
    late_events_dropped: int = 0
    late_events_updated: int = 0
    late_events_side_output: int = 0


class StreamProcessor(Entity):
    """Stateful windowed stream processor.

    Receives events with event times, assigns them to windows, and
    emits aggregated results when windows close based on watermark
    progression.

    Attributes:
        name: Entity name for identification.
    """

    def __init__(
        self,
        name: str,
        window_type: WindowType,
        aggregate_fn: Callable[[list[Any]], Any],
        downstream: Entity,
        allowed_lateness_s: float = 0.0,
        late_event_policy: LateEventPolicy = LateEventPolicy.DROP,
        side_output: Entity | None = None,
        watermark_interval_s: float = 1.0,
    ):
        """Initialize the stream processor.

        Args:
            name: Name for this processor entity.
            window_type: Window assignment strategy.
            aggregate_fn: Function to aggregate records in a window.
            downstream: Entity to receive WindowResult events.
            allowed_lateness_s: Grace period after window closes for late events.
            late_event_policy: How to handle late events.
            side_output: Entity to receive LateEvent events (for SIDE_OUTPUT policy).
            watermark_interval_s: Interval between watermark self-scheduling.
        """
        super().__init__(name)
        self._window_type = window_type
        self._aggregate_fn = aggregate_fn
        self._downstream = downstream
        self._allowed_lateness_s = allowed_lateness_s
        self._late_event_policy = late_event_policy
        self._side_output = side_output
        self._watermark_interval_s = watermark_interval_s

        # Window state: keyed by grouping key
        self._windows: dict[str, list[WindowState]] = {}
        self._watermark_s: float = 0.0
        self._min_event_time_seen: float = float('inf')
        self._watermark_scheduled: bool = False

        self._events_processed = 0
        self._windows_emitted = 0
        self._late_events = 0
        self._late_events_dropped = 0
        self._late_events_updated = 0
        self._late_events_side_output = 0

    @property
    def stats(self) -> StreamProcessorStats:
        """Return a frozen snapshot of current statistics."""
        return StreamProcessorStats(
            events_processed=self._events_processed,
            windows_emitted=self._windows_emitted,
            late_events=self._late_events,
            late_events_dropped=self._late_events_dropped,
            late_events_updated=self._late_events_updated,
            late_events_side_output=self._late_events_side_output,
        )

    @property
    def watermark_s(self) -> float:
        """Current watermark in seconds."""
        return self._watermark_s

    @property
    def active_windows(self) -> int:
        """Number of active (non-emitted) windows across all keys."""
        count = 0
        for windows in self._windows.values():
            count += sum(1 for w in windows if not w.emitted)
        return count

    @property
    def total_windows_emitted(self) -> int:
        """Total windows emitted."""
        return self._windows_emitted

    def _is_session_window(self) -> bool:
        """Check if using session windows."""
        return isinstance(self._window_type, SessionWindow)

    def _add_to_session_window(self, key: str, event_time_s: float, value: Any) -> None:
        """Handle session window merging."""
        gap = self._window_type._gap_s  # type: ignore[attr-defined]

        if key not in self._windows:
            self._windows[key] = []

        windows = self._windows[key]

        # Find or create a session that contains this event
        merged_into = None
        for w in windows:
            if w.emitted:
                continue
            # Event falls within the window or within gap of its boundaries
            if event_time_s >= w.start - gap and event_time_s <= w.end:
                w.records.append(value)
                # Extend end if event is near the boundary
                new_end = event_time_s + gap
                if new_end > w.end:
                    w.end = new_end
                merged_into = w
                break

        if merged_into is None:
            # Create new session window
            new_window = WindowState(
                start=event_time_s,
                end=event_time_s + gap,
            )
            new_window.records.append(value)
            windows.append(new_window)
            merged_into = new_window

        # Merge overlapping sessions
        self._merge_sessions(key)

    def _merge_sessions(self, key: str) -> None:
        """Merge overlapping session windows for a key."""
        windows = self._windows[key]
        active = [w for w in windows if not w.emitted]
        if len(active) <= 1:
            return

        active.sort(key=lambda w: w.start)
        merged: list[WindowState] = [active[0]]

        for w in active[1:]:
            last = merged[-1]
            if w.start <= last.end:
                # Merge
                last.end = max(last.end, w.end)
                last.records.extend(w.records)
            else:
                merged.append(w)

        emitted = [w for w in windows if w.emitted]
        self._windows[key] = emitted + merged

    def _is_late(self, event_time_s: float) -> bool:
        """Check if an event is late relative to the watermark."""
        return event_time_s < self._watermark_s - self._allowed_lateness_s

    def _emit_closed_windows(self) -> list[Event]:
        """Check all windows and emit those that should close."""
        events: list[Event] = []

        for key, windows in list(self._windows.items()):
            for window in windows:
                if window.emitted:
                    continue
                if self._window_type.should_close(window.end, self._watermark_s):
                    result = self._aggregate_fn(window.records)
                    window.emitted = True
                    self._windows_emitted += 1

                    events.append(Event(
                        time=self.now,
                        event_type="WindowResult",
                        target=self._downstream,
                        context={
                            "key": key,
                            "window_start": window.start,
                            "window_end": window.end,
                            "result": result,
                            "record_count": len(window.records),
                        },
                    ))

        return events

    def handle_event(self, event: Event) -> Generator[float, None, list[Event] | None]:
        """Handle stream processor events."""
        event_type = event.event_type

        if event_type == "Process":
            key = event.context.get("key", "default")
            value = event.context.get("value")
            # Support both float and Instant for event time
            event_time = event.context.get("event_time")
            event_time_s = event.context.get("event_time_s")

            if event_time_s is None:
                if event_time is not None:
                    if hasattr(event_time, 'to_seconds'):
                        event_time_s = event_time.to_seconds()
                    else:
                        event_time_s = float(event_time)
                else:
                    event_time_s = self.now.to_seconds()

            self._events_processed += 1

            # Track min event time for watermark advancement
            if event_time_s < self._min_event_time_seen:
                self._min_event_time_seen = event_time_s

            # Check for late event
            if self._is_late(event_time_s):
                self._late_events += 1

                if self._late_event_policy == LateEventPolicy.DROP:
                    self._late_events_dropped += 1
                    yield 0.0
                    return None

                elif self._late_event_policy == LateEventPolicy.SIDE_OUTPUT:
                    self._late_events_side_output += 1
                    if self._side_output is not None:
                        yield 0.0
                        return [Event(
                            time=self.now,
                            event_type="LateEvent",
                            target=self._side_output,
                            context={
                                "key": key,
                                "value": value,
                                "event_time_s": event_time_s,
                            },
                        )]
                    yield 0.0
                    return None

                elif self._late_event_policy == LateEventPolicy.UPDATE:
                    self._late_events_updated += 1
                    # Fall through to add to windows

            # Assign to windows
            if self._is_session_window():
                self._add_to_session_window(key, event_time_s, value)
            else:
                assigned = self._window_type.assign_windows(event_time_s)
                if key not in self._windows:
                    self._windows[key] = []

                for w_start, w_end in assigned:
                    # Find existing window or create new
                    found = False
                    for w in self._windows[key]:
                        if w.start == w_start and w.end == w_end and not w.emitted:
                            w.records.append(value)
                            found = True
                            break
                        # For UPDATE policy, allow adding to emitted windows
                        if (
                            self._late_event_policy == LateEventPolicy.UPDATE
                            and w.start == w_start
                            and w.end == w_end
                            and w.emitted
                        ):
                            w.records.append(value)
                            w.emitted = False  # Re-open for re-emission
                            found = True
                            break

                    if not found:
                        new_window = WindowState(start=w_start, end=w_end)
                        new_window.records.append(value)
                        self._windows[key].append(new_window)

            # Schedule watermark daemon on first event
            if not self._watermark_scheduled:
                self._watermark_scheduled = True
                yield 0.0
                return [Event(
                    time=Instant.from_seconds(
                        self.now.to_seconds() + self._watermark_interval_s
                    ),
                    event_type="Watermark",
                    target=self,
                    context={"watermark_s": event_time_s},
                )]

            yield 0.0
            return None

        elif event_type == "Watermark":
            watermark_s = event.context.get("watermark_s", 0.0)

            # Advance watermark to the maximum of incoming and current
            if watermark_s > self._watermark_s:
                self._watermark_s = watermark_s

            yield 0.0

            # Check for closable windows
            result_events = self._emit_closed_windows()

            # Reschedule watermark
            next_time = Instant.from_seconds(
                self.now.to_seconds() + self._watermark_interval_s
            )
            # Advance watermark based on simulation time
            next_watermark = self.now.to_seconds()
            result_events.append(Event(
                time=next_time,
                event_type="Watermark",
                target=self,
                context={"watermark_s": next_watermark},
            ))

            return result_events if result_events else None

        return None
