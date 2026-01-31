"""Sliding window rate limiter entity.

Implements the sliding window log algorithm. Maintains a log of request
timestamps and limits the number of requests within a rolling time window.
Old timestamps are pruned on each request arrival.

Sliding window avoids the "boundary burst" problem of fixed windows,
where 2x the limit could pass if requests cluster around window edges.
The tradeoff is memory usage proportional to request rate.
"""

import logging
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass
class SlidingWindowStats:
    """Statistics tracked by SlidingWindowRateLimiter."""
    requests_received: int = 0
    requests_forwarded: int = 0
    requests_dropped: int = 0


class SlidingWindowRateLimiter(Entity):
    """A sliding window (log-based) rate limiter.

    The sliding window algorithm:
    - Maintains a log of request timestamps within the window.
    - When a request arrives, timestamps older than (now - window_size) are pruned.
    - If the count of remaining timestamps < max_requests, the request is allowed.
    - Otherwise, the request is dropped.

    This provides smoother rate limiting than fixed windows, avoiding the
    "boundary burst" problem where requests cluster at window edges.

    Attributes:
        window_size_seconds: The size of the sliding window in seconds.
        max_requests: Maximum requests allowed within the window.
        downstream: The entity to forward accepted requests to.
        stats: Counts of received, forwarded, and dropped requests.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        window_size_seconds: float = 1.0,
        max_requests: int = 10,
    ):
        """Initialize the sliding window rate limiter.

        Args:
            name: Entity name for identification.
            downstream: Entity to forward accepted requests to.
            window_size_seconds: Size of the sliding window in seconds.
            max_requests: Maximum requests allowed in the window.
        """
        super().__init__(name)

        self._downstream = downstream
        self._window_size = window_size_seconds  # Store as float seconds
        self._window_size_seconds = float(window_size_seconds)
        self._max_requests = int(max_requests)

        # Request timestamp log (sliding window)
        self._request_log: list[Instant] = []

        # Statistics
        self.stats = SlidingWindowStats()

        # Time series for visualization
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []
        self.window_counts: list[tuple[Instant, int]] = []

    @property
    def downstream(self) -> Entity:
        """The entity receiving forwarded requests."""
        return self._downstream

    @property
    def window_size_seconds(self) -> float:
        """Size of the sliding window in seconds."""
        return self._window_size_seconds

    @property
    def max_requests(self) -> int:
        """Maximum requests allowed in the window."""
        return self._max_requests

    @property
    def current_window_count(self) -> int:
        """Current number of requests in the window (may be stale)."""
        return len(self._request_log)

    def _prune_old_requests(self, now: Instant) -> None:
        """Remove requests that have fallen outside the sliding window."""
        cutoff = now - self._window_size

        # Remove timestamps older than the cutoff
        # Since requests arrive in order, we can pop from the front
        while self._request_log and self._request_log[0] < cutoff:
            self._request_log.pop(0)

    def handle_event(self, event: Event) -> list[Event]:
        """Handle an incoming request event.

        Prunes old requests, then either forwards or drops based on window count.

        Args:
            event: The incoming request event.

        Returns:
            List containing a forwarded event to downstream, or empty if dropped.
        """
        now = event.time

        # Record received
        self.stats.requests_received += 1
        self.received_times.append(now)

        # Prune requests outside the current window
        self._prune_old_requests(now)

        # Record window count before decision
        current_count = len(self._request_log)
        self.window_counts.append((now, current_count))

        if current_count < self._max_requests:
            # Add to log and forward
            self._request_log.append(now)
            self.stats.requests_forwarded += 1
            self.forwarded_times.append(now)

            logger.debug(
                "[%.3f][%s] Forwarded request; window_count=%d/%d",
                now.to_seconds(), self.name, len(self._request_log), self._max_requests
            )

            # Create forwarding event to downstream entity
            forward_event = Event(
                time=now,
                event_type=f"forward::{event.event_type}",
                target=self._downstream,
                context=event.context.copy(),
            )
            return [forward_event]

        # Window is full - drop the request
        self.stats.requests_dropped += 1
        self.dropped_times.append(now)

        logger.debug(
            "[%.3f][%s] Dropped request; window_count=%d/%d (full)",
            now.to_seconds(), self.name, current_count, self._max_requests
        )
        return []
