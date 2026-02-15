"""Generic rate-limited entity that wraps any RateLimiterPolicy.

RateLimitedEntity is an Entity that buffers incoming events in a FIFO
queue and drains them according to a pluggable RateLimiterPolicy. When
the policy denies a request, it is queued (up to ``queue_capacity``).
A self-scheduling poll event drains the queue at exactly the right time
as reported by ``policy.time_until_available()``.

Requests are only dropped when the internal queue overflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from happysimulator.components.queue_policy import FIFOQueue
from happysimulator.components.rate_limiter.policy import RateLimiterPolicy
from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RateLimitedEntityStats:
    """Frozen snapshot of RateLimitedEntity statistics."""

    received: int = 0
    forwarded: int = 0
    queued: int = 0
    dropped: int = 0


class RateLimitedEntity(Entity):
    """Entity that rate-limits incoming events using a pluggable policy.

    Incoming events are immediately forwarded if the policy allows.
    Otherwise they are enqueued in a bounded FIFO buffer and drained
    via self-scheduling poll events that fire at the exact time the
    policy reports capacity will be available.

    Args:
        name: Entity name for identification.
        downstream: Entity to forward accepted events to.
        policy: The rate limiter algorithm to use.
        queue_capacity: Maximum events that can be buffered (drops on overflow).
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        policy: RateLimiterPolicy,
        queue_capacity: int = 1000,
    ):
        super().__init__(name)
        self._downstream = downstream
        self._policy = policy
        self._queue: FIFOQueue[Event] = FIFOQueue(capacity=queue_capacity)
        self._poll_scheduled = False

        self._received = 0
        self._forwarded = 0
        self._queued = 0
        self._dropped = 0

        # Time series for visualization
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []

    @property
    def downstream(self) -> Entity:
        return self._downstream

    @property
    def policy(self) -> RateLimiterPolicy:
        return self._policy

    @property
    def queue_depth(self) -> int:
        return len(self._queue)

    @property
    def stats(self) -> RateLimitedEntityStats:
        """Frozen snapshot of rate limited entity statistics."""
        return RateLimitedEntityStats(
            received=self._received,
            forwarded=self._forwarded,
            queued=self._queued,
            dropped=self._dropped,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to downstream."""
        super().set_clock(clock)
        self._downstream.set_clock(clock)

    def handle_event(self, event: Event) -> list[Event]:
        """Handle an incoming event or internal poll event."""
        if event.event_type == f"rate_limit_poll::{self.name}":
            return self._handle_poll(event)
        return self._handle_request(event)

    def _handle_request(self, event: Event) -> list[Event]:
        now = event.time
        self._received += 1
        self.received_times.append(now)

        if self._policy.try_acquire(now):
            return self._forward(event, now)

        # Queue the event
        if self._queue.push(event):
            self._queued += 1
            logger.debug(
                "[%.3f][%s] Queued request; queue_depth=%d",
                now.to_seconds(), self.name, len(self._queue),
            )
            return self._ensure_poll_scheduled(now)

        # Queue full — drop
        self._dropped += 1
        self.dropped_times.append(now)
        logger.debug(
            "[%.3f][%s] Dropped request; queue full (%d)",
            now.to_seconds(), self.name, len(self._queue),
        )
        return []

    def _handle_poll(self, event: Event) -> list[Event]:
        now = event.time
        self._poll_scheduled = False

        if self._queue.is_empty():
            return []

        if self._policy.try_acquire(now):
            queued_event = self._queue.pop()
            assert queued_event is not None
            result = self._forward(queued_event, now)
            # If queue still has items, schedule next poll
            if not self._queue.is_empty():
                result.extend(self._ensure_poll_scheduled(now))
            return result

        # Policy still denies — reschedule
        return self._ensure_poll_scheduled(now)

    def _forward(self, event: Event, now: Instant) -> list[Event]:
        self._forwarded += 1
        self.forwarded_times.append(now)
        logger.debug(
            "[%.3f][%s] Forwarded request",
            now.to_seconds(), self.name,
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
        wait = self._policy.time_until_available(now)
        poll_time = now + wait
        return [
            Event(
                time=poll_time,
                event_type=f"rate_limit_poll::{self.name}",
                target=self,
                daemon=True,
            )
        ]
