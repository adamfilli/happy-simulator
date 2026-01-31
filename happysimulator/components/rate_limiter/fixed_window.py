"""Fixed window rate limiter entity.

Implements the fixed window algorithm. Divides time into discrete windows
of fixed size and limits the number of requests per window. Simple and
memory-efficient, but susceptible to boundary bursts.

Fixed window vs sliding window:
- Fixed window: Simple, O(1) space, but allows 2x burst at window boundaries
- Sliding window: Smoother limiting, but O(n) space where n = requests in window

Example:
    from happysimulator.components.rate_limiter import FixedWindowRateLimiter

    limiter = FixedWindowRateLimiter(
        name="api_limiter",
        downstream=server,
        requests_per_window=100,
        window_size=1.0,  # 1 second
    )
"""

import logging
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass
class FixedWindowStats:
    """Statistics tracked by FixedWindowRateLimiter."""

    requests_received: int = 0
    requests_forwarded: int = 0
    requests_dropped: int = 0
    windows_completed: int = 0


class FixedWindowRateLimiter(Entity):
    """A fixed window rate limiter that forwards or drops requests.

    The fixed window algorithm:
    - Time is divided into discrete windows of `window_size` seconds.
    - Each window allows up to `requests_per_window` requests.
    - When a new window starts, the counter resets.
    - Requests beyond the limit are dropped.

    This is simple and memory-efficient (O(1) space), but has the "boundary
    burst" problem: if requests cluster at window boundaries, up to 2x the
    limit could pass in a short period.

    Attributes:
        window_size: Size of each window in seconds.
        requests_per_window: Maximum requests allowed per window.
        downstream: The entity to forward accepted requests to.
        stats: Counts of received, forwarded, and dropped requests.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        requests_per_window: int,
        window_size: float = 1.0,
    ):
        """Initialize the fixed window rate limiter.

        Args:
            name: Entity name for identification.
            downstream: Entity to forward accepted requests to.
            requests_per_window: Maximum requests allowed per window.
            window_size: Size of each window in seconds.

        Raises:
            ValueError: If requests_per_window < 1 or window_size <= 0.
        """
        if requests_per_window < 1:
            raise ValueError(f"requests_per_window must be >= 1, got {requests_per_window}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        super().__init__(name)

        self._downstream = downstream
        self._requests_per_window = requests_per_window
        self._window_size = window_size

        # Current window state
        self._current_window_start: Instant | None = None
        self._current_window_count: int = 0

        # Statistics
        self.stats = FixedWindowStats()

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
    def window_size(self) -> float:
        """Size of each window in seconds."""
        return self._window_size

    @property
    def requests_per_window(self) -> int:
        """Maximum requests allowed per window."""
        return self._requests_per_window

    @property
    def current_window_count(self) -> int:
        """Current request count in the active window."""
        return self._current_window_count

    def _get_window_start(self, now: Instant) -> Instant:
        """Calculate the start of the window containing the given time."""
        now_seconds = now.to_seconds()
        window_start_seconds = (now_seconds // self._window_size) * self._window_size
        return Instant.from_seconds(window_start_seconds)

    def _maybe_reset_window(self, now: Instant) -> None:
        """Reset window counter if we've moved to a new window."""
        window_start = self._get_window_start(now)

        if self._current_window_start is None:
            self._current_window_start = window_start
            self._current_window_count = 0
        elif window_start > self._current_window_start:
            # New window
            self.stats.windows_completed += 1
            self._current_window_start = window_start
            self._current_window_count = 0

    def handle_event(self, event: Event) -> list[Event]:
        """Handle an incoming request event.

        Checks the current window count and either forwards or drops.

        Args:
            event: The incoming request event.

        Returns:
            List containing a forwarded event to downstream, or empty if dropped.
        """
        now = event.time

        # Record received
        self.stats.requests_received += 1
        self.received_times.append(now)

        # Check/reset window
        self._maybe_reset_window(now)

        # Record window count before decision
        self.window_counts.append((now, self._current_window_count))

        if self._current_window_count < self._requests_per_window:
            # Allow request
            self._current_window_count += 1
            self.stats.requests_forwarded += 1
            self.forwarded_times.append(now)

            logger.debug(
                "[%.3f][%s] Forwarded request; window_count=%d/%d",
                now.to_seconds(),
                self.name,
                self._current_window_count,
                self._requests_per_window,
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
            now.to_seconds(),
            self.name,
            self._current_window_count,
            self._requests_per_window,
        )
        return []
