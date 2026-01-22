"""Token bucket rate limiter entity.

Implements the classic token bucket algorithm. Tokens refill at a constant
rate up to a maximum capacity. Each request consumes one token if available;
otherwise the request is dropped. This allows controlled bursting up to
the bucket capacity.

Token bucket vs leaky bucket:
- Token bucket allows bursts (requests pass immediately while tokens exist)
- Leaky bucket enforces strict output rate (no bursting)
"""

import logging
from dataclasses import dataclass
from typing import Optional

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.instant import Instant

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterStats:
    """Statistics tracked by TokenBucketRateLimiter."""
    requests_received: int = 0
    requests_forwarded: int = 0
    requests_dropped: int = 0


class TokenBucketRateLimiter(Entity):
    """A token bucket rate limiter that forwards or drops requests.

    The token bucket algorithm:
    - Tokens are added at `refill_rate` tokens per second.
    - The bucket holds at most `capacity` tokens.
    - Each request consumes one token if available; otherwise it's dropped.

    This is a "drop on empty" rate limiter (no queuing).

    Attributes:
        capacity: Maximum number of tokens the bucket can hold.
        refill_rate: Tokens added per second.
        downstream: The entity to forward accepted requests to.
        stats: Counts of received, forwarded, and dropped requests.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        capacity: float = 10.0,
        refill_rate: float = 1.0,
        initial_tokens: Optional[float] = None,
    ):
        """Initialize the token bucket rate limiter.

        Args:
            name: Entity name for identification.
            downstream: Entity to forward accepted requests to.
            capacity: Maximum tokens in the bucket.
            refill_rate: Tokens per second refill rate.
            initial_tokens: Starting tokens (defaults to capacity if None).
        """
        super().__init__(name)

        self._downstream = downstream
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)

        # Token bucket state
        self._tokens = self._capacity if initial_tokens is None else float(initial_tokens)
        self._last_refill_time: Optional[Instant] = None

        # Statistics
        self.stats = RateLimiterStats()

        # Time series for visualization
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []
        self.token_levels: list[tuple[Instant, float]] = []

    @property
    def downstream(self) -> Entity:
        """The entity receiving forwarded requests."""
        return self._downstream

    @property
    def tokens(self) -> float:
        """Current token count (may be stale until next refill)."""
        return self._tokens

    @property
    def capacity(self) -> float:
        """Maximum token capacity."""
        return self._capacity

    @property
    def refill_rate(self) -> float:
        """Token refill rate (tokens per second)."""
        return self._refill_rate

    def _refill(self, now: Instant) -> None:
        """Refill tokens based on elapsed time since last refill."""
        if self._last_refill_time is None:
            self._last_refill_time = now
            return

        elapsed_seconds = (now - self._last_refill_time).to_seconds()
        if elapsed_seconds <= 0:
            return

        tokens_to_add = elapsed_seconds * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + tokens_to_add)
        self._last_refill_time = now

    def handle_event(self, event: Event) -> list[Event]:
        """Handle an incoming request event.

        Refills tokens, then either forwards or drops the request.

        Args:
            event: The incoming request event.

        Returns:
            List containing a forwarded event to downstream, or empty if dropped.
        """
        now = event.time

        # Record received
        self.stats.requests_received += 1
        self.received_times.append(now)

        # Refill tokens up to current time
        self._refill(now)

        # Record token level before decision
        self.token_levels.append((now, self._tokens))

        if self._tokens >= 1.0:
            # Consume a token and forward
            self._tokens -= 1.0
            self.stats.requests_forwarded += 1
            self.forwarded_times.append(now)

            logger.debug(
                "[%.3f][%s] Forwarded request; tokens=%.2f",
                now.to_seconds(), self.name, self._tokens
            )

            # Create forwarding event to downstream entity
            forward_event = Event(
                time=now,
                event_type=f"forward::{event.event_type}",
                target=self._downstream,
                context=event.context.copy(),
            )
            return [forward_event]

        # No tokens available - drop the request
        self.stats.requests_dropped += 1
        self.dropped_times.append(now)

        logger.debug(
            "[%.3f][%s] Dropped request; tokens=%.2f",
            now.to_seconds(), self.name, self._tokens
        )
        return []
