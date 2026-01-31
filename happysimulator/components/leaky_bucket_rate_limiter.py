"""Leaky bucket rate limiter entity.

Implements the leaky bucket algorithm. Incoming requests enter a queue
(the bucket) and exit at a fixed rate (the leak). When the bucket is full,
new requests are dropped. Unlike token bucket, this enforces a strict
output rate with no bursting.

The implementation schedules "leak" events to forward queued requests
at regular intervals determined by leak_rate.
"""

import logging
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass
class LeakyBucketStats:
    """Statistics tracked by LeakyBucketRateLimiter."""
    requests_received: int = 0
    requests_forwarded: int = 0
    requests_dropped: int = 0


class LeakyBucketRateLimiter(Entity):
    """A leaky bucket rate limiter that queues and forwards requests at a fixed rate.

    The leaky bucket algorithm:
    - Incoming requests are added to a queue (bucket) if space is available.
    - Requests "leak" out of the bucket at a fixed rate (one per leak_interval).
    - If the bucket is full when a request arrives, it is dropped.
    - Unlike token bucket, leaky bucket enforces a strict output rate with no bursting.

    This implementation uses scheduled "leak" events to forward queued requests.

    Attributes:
        capacity: Maximum number of requests the bucket can hold.
        leak_rate: Requests forwarded per second.
        downstream: The entity to forward accepted requests to.
        stats: Counts of received, forwarded, and dropped requests.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        capacity: int = 10,
        leak_rate: float = 1.0,
    ):
        """Initialize the leaky bucket rate limiter.

        Args:
            name: Entity name for identification.
            downstream: Entity to forward accepted requests to.
            capacity: Maximum requests that can be queued.
            leak_rate: Requests per second leak rate.
        """
        super().__init__(name)

        self._downstream = downstream
        self._capacity = int(capacity)
        self._leak_rate = float(leak_rate)
        self._leak_interval = 1.0 / leak_rate if leak_rate > 0 else float('inf')

        # Queue state
        self._queue: list[Event] = []
        self._leak_scheduled = False

        # Statistics
        self.stats = LeakyBucketStats()

        # Time series for visualization
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []
        self.queue_depths: list[tuple[Instant, int]] = []

    @property
    def downstream(self) -> Entity:
        """The entity receiving forwarded requests."""
        return self._downstream

    @property
    def capacity(self) -> int:
        """Maximum queue capacity."""
        return self._capacity

    @property
    def leak_rate(self) -> float:
        """Leak rate (requests per second)."""
        return self._leak_rate

    @property
    def queue_depth(self) -> int:
        """Current number of queued requests."""
        return len(self._queue)

    def handle_event(self, event: Event) -> list[Event]:
        """Handle an incoming event (request or leak).

        Args:
            event: The incoming event.

        Returns:
            List of events to schedule (leak events or forwarded requests).
        """
        # Check if this is a leak event
        if event.event_type == f"leak::{self.name}":
            return self._handle_leak(event)

        # Otherwise, it's an incoming request
        return self._handle_request(event)

    def _handle_request(self, event: Event) -> list[Event]:
        """Handle an incoming request event."""
        now = event.time

        # Record received
        self.stats.requests_received += 1
        self.received_times.append(now)

        # Record queue depth before decision
        self.queue_depths.append((now, len(self._queue)))

        if len(self._queue) < self._capacity:
            # Add to queue
            self._queue.append(event)

            logger.debug(
                "[%.3f][%s] Queued request; queue_depth=%d",
                now.to_seconds(), self.name, len(self._queue)
            )

            # Schedule a leak event if not already scheduled
            return self._ensure_leak_scheduled(now)

        # Queue is full - drop the request
        self.stats.requests_dropped += 1
        self.dropped_times.append(now)

        logger.debug(
            "[%.3f][%s] Dropped request; queue_depth=%d (full)",
            now.to_seconds(), self.name, len(self._queue)
        )
        return []

    def _handle_leak(self, event: Event) -> list[Event]:
        """Handle a leak event - forward one request from the queue."""
        now = event.time
        self._leak_scheduled = False

        if not self._queue:
            # Queue is empty, nothing to leak
            return []

        # Pop the oldest request (FIFO)
        queued_event = self._queue.pop(0)

        # Record forwarded
        self.stats.requests_forwarded += 1
        self.forwarded_times.append(now)

        logger.debug(
            "[%.3f][%s] Leaked request; queue_depth=%d",
            now.to_seconds(), self.name, len(self._queue)
        )

        # Create forwarding event to downstream entity
        forward_event = Event(
            time=now,
            event_type=f"forward::{queued_event.event_type}",
            target=self._downstream,
            context=queued_event.context.copy(),
        )

        # Schedule next leak if queue is not empty
        result = [forward_event]
        if self._queue:
            result.extend(self._ensure_leak_scheduled(now))

        return result

    def _ensure_leak_scheduled(self, now: Instant) -> list[Event]:
        """Schedule a leak event if not already scheduled."""
        if self._leak_scheduled:
            return []

        self._leak_scheduled = True
        leak_time = now + self._leak_interval

        return [
            Event(
                time=leak_time,
                event_type=f"leak::{self.name}",
                target=self,
                daemon=True,
            )
        ]
