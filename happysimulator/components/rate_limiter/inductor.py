"""Digital Inductor: configuration-free burst suppression.

Unlike traditional rate limiters that enforce a hard throughput cap, the
Inductor resists rapid *changes* in event rate — analogous to an electrical
inductor resisting changes in current.  It has no throughput limit; instead
a single **time constant** (τ, seconds) controls how strongly the component
smooths rate fluctuations.  Higher τ = more smoothing.

Algorithm
---------
Maintains an EWMA (Exponentially Weighted Moving Average) of inter-arrival
times using a time-aware alpha:

    α = 1 − exp(−dt / τ)

where *dt* is the gap since the previous arrival.  Short gaps (bursts) get
a low α (heavy smoothing), long gaps get a high α (fast adaptation).

Events are forwarded when enough time has elapsed according to the smoothed
interval.  Excess arrivals are buffered in a bounded FIFO queue and drained
via self-scheduling poll events.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

if TYPE_CHECKING:
    from happysimulator.core.clock import Clock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InductorStats:
    """Frozen snapshot of Inductor statistics."""

    received: int = 0
    forwarded: int = 0
    queued: int = 0
    dropped: int = 0


class Inductor(Entity):
    """Entity that smooths bursty traffic using EWMA rate estimation.

    The Inductor has **no throughput cap**.  It resists rapid rate changes,
    letting sustained rates pass through while dampening spikes.

    Args:
        name: Entity name for identification.
        downstream: Entity to forward accepted events to.
        time_constant: τ in seconds — higher means more smoothing.
        queue_capacity: Maximum events buffered before dropping.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        time_constant: float,
        queue_capacity: int = 10_000,
    ):
        super().__init__(name)
        self._downstream = downstream
        self._time_constant = time_constant
        self._queue: FIFOQueue[Event] = FIFOQueue(capacity=queue_capacity)
        self._poll_scheduled = False

        # EWMA state
        self._smoothed_interval: float | None = None
        self._last_arrival_time: Instant | None = None
        self._last_output_time: Instant | None = None

        # Observability
        self._received = 0
        self._forwarded = 0
        self._queued = 0
        self._dropped = 0
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []
        self.rate_history: list[tuple[Instant, float]] = []

    # -- Properties -----------------------------------------------------------

    @property
    def downstream(self) -> Entity:
        return self._downstream

    @property
    def time_constant(self) -> float:
        return self._time_constant

    @property
    def estimated_rate(self) -> float:
        """Current estimated event rate (events/s) from the EWMA."""
        if self._smoothed_interval is None or self._smoothed_interval <= 0:
            return 0.0
        return 1.0 / self._smoothed_interval

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> InductorStats:
        """Frozen snapshot of inductor statistics."""
        return InductorStats(
            received=self._received,
            forwarded=self._forwarded,
            queued=self._queued,
            dropped=self._dropped,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to downstream."""
        super().set_clock(clock)
        self._downstream.set_clock(clock)

    # -- Event handling -------------------------------------------------------

    def handle_event(self, event: Event) -> list[Event]:
        """Dispatch arrivals vs internal poll events."""
        if event.event_type == f"inductor_poll::{self.name}":
            return self._handle_poll(event)
        return self._handle_arrival(event)

    def _handle_arrival(self, event: Event) -> list[Event]:
        now = event.time
        self._received += 1
        self.received_times.append(now)

        self._update_rate_estimate(now)
        self._last_arrival_time = now

        if self._can_forward(now):
            return self._forward(event, now)

        # Queue the event
        if self._queue.push(event):
            self._queued += 1
            logger.debug(
                "[%.3f][%s] Queued request; queue_depth=%d",
                now.to_seconds(),
                self.name,
                len(self._queue),
            )
            return self._ensure_poll_scheduled(now)

        # Queue full — drop
        self._dropped += 1
        self.dropped_times.append(now)
        logger.debug(
            "[%.3f][%s] Dropped request; queue full (%d)",
            now.to_seconds(),
            self.name,
            len(self._queue),
        )
        return []

    def _handle_poll(self, event: Event) -> list[Event]:
        now = event.time
        self._poll_scheduled = False

        if self._queue.is_empty():
            return []

        if self._can_forward(now):
            queued_event = self._queue.pop()
            assert queued_event is not None
            result = self._forward(queued_event, now)
            if not self._queue.is_empty():
                result.extend(self._ensure_poll_scheduled(now))
            return result

        # Not ready yet — reschedule
        return self._ensure_poll_scheduled(now)

    # -- EWMA logic -----------------------------------------------------------

    def _update_rate_estimate(self, now: Instant) -> None:
        """Update the EWMA of inter-arrival time."""
        if self._last_arrival_time is None:
            # First event: no interval to measure yet
            return

        dt = (now - self._last_arrival_time).to_seconds()
        if dt < 0:
            return

        tau = self._time_constant
        if self._smoothed_interval is None:
            # Seed with the first observed interval
            self._smoothed_interval = dt
        else:
            alpha = 1.0 - math.exp(-dt / tau) if tau > 0 else 1.0
            self._smoothed_interval = alpha * dt + (1.0 - alpha) * self._smoothed_interval

        self.rate_history.append((now, self.estimated_rate))

    def _can_forward(self, now: Instant) -> bool:
        """Check whether enough time has passed to forward an event."""
        # Always forward the very first event
        if self._last_output_time is None:
            return True

        # No estimate yet — allow through
        if self._smoothed_interval is None or self._smoothed_interval <= 0:
            return True

        elapsed = (now - self._last_output_time).to_seconds()
        return elapsed >= self._smoothed_interval

    def _forward(self, event: Event, now: Instant) -> list[Event]:
        self._forwarded += 1
        self.forwarded_times.append(now)
        self._last_output_time = now
        logger.debug(
            "[%.3f][%s] Forwarded request",
            now.to_seconds(),
            self.name,
        )
        forward_event = Event(
            time=now,
            event_type=f"forward::{event.event_type}",
            target=self._downstream,
            context=event.context.copy(),
        )
        return [forward_event]

    def _ensure_poll_scheduled(self, now: Instant) -> list[Event]:
        if self._poll_scheduled:
            return []
        self._poll_scheduled = True

        # Schedule poll after one smoothed interval
        wait_s = self._smoothed_interval if self._smoothed_interval else 0.01
        poll_time = now + Duration.from_seconds(wait_s)
        return [
            Event(
                time=poll_time,
                event_type=f"inductor_poll::{self.name}",
                target=self,
                daemon=True,
            )
        ]
