"""Distributed rate limiter entity.

Implements a rate limiter that coordinates across multiple instances using
a shared backing store (like Redis). This allows rate limiting across a
cluster of servers while maintaining a global limit.

The implementation uses a sliding window counter stored in the backing
store, with each instance checking and incrementing the counter atomically.

Example:
    from happysimulator.components.rate_limiter import DistributedRateLimiter
    from happysimulator.components.datastore import KVStore

    # Shared state store (simulates Redis)
    redis = KVStore(name="redis", read_latency=0.001, write_latency=0.001)

    # Multiple limiter instances sharing state
    limiter1 = DistributedRateLimiter(
        name="limiter_node1",
        downstream=server1,
        backing_store=redis,
        global_limit=1000,
        window_size=1.0,
    )
    limiter2 = DistributedRateLimiter(
        name="limiter_node2",
        downstream=server2,
        backing_store=redis,
        global_limit=1000,
        window_size=1.0,
    )
"""

import logging
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistributedRateLimiterStats:
    """Frozen snapshot of DistributedRateLimiter statistics."""

    requests_received: int = 0
    requests_forwarded: int = 0
    requests_dropped: int = 0
    store_reads: int = 0
    store_writes: int = 0
    local_rejections: int = 0
    global_rejections: int = 0


@dataclass
class WindowCounter:
    """Counter state for a time window."""

    window_id: int
    count: int


class DistributedRateLimiter(Entity):
    """A distributed rate limiter using a shared backing store.

    This rate limiter coordinates across multiple instances by storing
    the request count in a shared backing store. It uses a fixed window
    approach with the window ID derived from the current time.

    To reduce load on the backing store, instances maintain a local
    counter and only sync with the store periodically or when the
    local estimate approaches the limit.

    Algorithm:
    1. Calculate current window ID from time
    2. Read current global count from store
    3. If count < limit, increment and write back
    4. Forward or reject based on result

    Note: In a real system, this would use atomic operations (INCR in Redis).
    The simulation approximates this with read-modify-write.

    Attributes:
        global_limit: Maximum requests across all instances per window.
        window_size: Size of each window in seconds.
        downstream: The entity to forward accepted requests to.
        backing_store: Shared storage for coordination.
        stats: Request and store operation counts.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        backing_store: Entity,
        global_limit: int,
        window_size: float = 1.0,
        key_prefix: str = "ratelimit",
        local_threshold: float = 0.8,
    ):
        """Initialize the distributed rate limiter.

        Args:
            name: Entity name for identification.
            downstream: Entity to forward accepted requests to.
            backing_store: Shared KVStore for coordination.
            global_limit: Maximum requests across all instances per window.
            window_size: Size of each window in seconds.
            key_prefix: Prefix for keys in the backing store.
            local_threshold: Fraction of limit before syncing with store.

        Raises:
            ValueError: If parameters are invalid.
        """
        if global_limit < 1:
            raise ValueError(f"global_limit must be >= 1, got {global_limit}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if local_threshold <= 0 or local_threshold > 1:
            raise ValueError(f"local_threshold must be in (0, 1], got {local_threshold}")

        super().__init__(name)

        self._downstream = downstream
        self._backing_store = backing_store
        self._global_limit = global_limit
        self._window_size = window_size
        self._key_prefix = key_prefix
        self._local_threshold = local_threshold

        # Local state for reducing store calls
        self._local_window_id: int | None = None
        self._local_count: int = 0
        self._last_known_global_count: int = 0

        # Statistics
        self._requests_received = 0
        self._requests_forwarded = 0
        self._requests_dropped = 0
        self._store_reads = 0
        self._store_writes = 0
        self._local_rejections = 0
        self._global_rejections = 0

        # Time series for visualization
        self.received_times: list[Instant] = []
        self.forwarded_times: list[Instant] = []
        self.dropped_times: list[Instant] = []
        self.global_counts: list[tuple[Instant, int]] = []

    @property
    def downstream(self) -> Entity:
        """The entity receiving forwarded requests."""
        return self._downstream

    @property
    def backing_store(self) -> Entity:
        """The shared backing store."""
        return self._backing_store

    @property
    def global_limit(self) -> int:
        """Maximum requests across all instances per window."""
        return self._global_limit

    @property
    def window_size(self) -> float:
        """Size of each window in seconds."""
        return self._window_size

    @property
    def local_count(self) -> int:
        """Local request count for current window."""
        return self._local_count

    @property
    def stats(self) -> DistributedRateLimiterStats:
        """Frozen snapshot of distributed rate limiter statistics."""
        return DistributedRateLimiterStats(
            requests_received=self._requests_received,
            requests_forwarded=self._requests_forwarded,
            requests_dropped=self._requests_dropped,
            store_reads=self._store_reads,
            store_writes=self._store_writes,
            local_rejections=self._local_rejections,
            global_rejections=self._global_rejections,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to downstream and backing store."""
        super().set_clock(clock)
        self._downstream.set_clock(clock)
        self._backing_store.set_clock(clock)

    def _get_window_id(self, now: Instant) -> int:
        """Calculate the window ID for the given time."""
        return int(now.to_seconds() // self._window_size)

    def _get_counter_key(self, window_id: int) -> str:
        """Generate the key for storing the window counter."""
        return f"{self._key_prefix}:window:{window_id}"

    def _should_sync(self) -> bool:
        """Check if we should sync with the backing store."""
        # Sync if local count exceeds threshold of what we think is available
        estimated_available = self._global_limit - self._last_known_global_count
        return self._local_count >= estimated_available * self._local_threshold

    def check_and_increment(self, now: Instant) -> Generator[float, None, bool]:
        """Check rate limit and increment counter if allowed.

        This is a generator that yields while accessing the backing store.

        Args:
            now: Current time.

        Yields:
            Delays for store operations.

        Returns:
            True if request is allowed, False if rate limited.
        """
        window_id = self._get_window_id(now)

        # Check for window change
        if self._local_window_id != window_id:
            self._local_window_id = window_id
            self._local_count = 0
            self._last_known_global_count = 0

        # Quick local rejection if we know we're over limit
        # Use only last_known_global_count since that includes all requests
        if self._last_known_global_count >= self._global_limit:
            self._local_rejections += 1
            return False

        # Sync with backing store
        key = self._get_counter_key(window_id)

        # Read current global count
        self._store_reads += 1
        read_gen = self._backing_store.get(key)
        try:
            while True:
                delay = next(read_gen)
                yield delay
        except StopIteration as e:
            current_count = e.value if e.value is not None else 0

        self._last_known_global_count = current_count

        # Check if over limit
        if current_count >= self._global_limit:
            self._global_rejections += 1
            self.global_counts.append((now, current_count))
            return False

        # Increment counter
        new_count = current_count + 1
        self._store_writes += 1
        write_gen = self._backing_store.put(key, new_count)
        try:
            while True:
                delay = next(write_gen)
                yield delay
        except StopIteration:
            pass

        self._local_count += 1
        self._last_known_global_count = new_count
        self.global_counts.append((now, new_count))

        return True

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]]:
        """Handle an incoming request event.

        This is a generator that yields while coordinating with the store.

        Args:
            event: The incoming request event.

        Yields:
            Delays for store operations.

        Returns:
            List containing a forwarded event to downstream, or empty if dropped.
        """
        now = event.time

        # Record received
        self._requests_received += 1
        self.received_times.append(now)

        # Check rate limit (this yields for store access)
        allowed = yield from self.check_and_increment(now)

        if allowed:
            self._requests_forwarded += 1
            self.forwarded_times.append(now)

            logger.debug(
                "[%.3f][%s] Forwarded request; global_count=%d/%d",
                now.to_seconds(),
                self.name,
                self._last_known_global_count,
                self._global_limit,
            )

            # Create forwarding event to downstream entity
            forward_event = Event(
                time=now,
                event_type=f"forward::{event.event_type}",
                target=self._downstream,
                context=event.context.copy(),
            )
            return [forward_event]

        # Rate limited
        self._requests_dropped += 1
        self.dropped_times.append(now)

        logger.debug(
            "[%.3f][%s] Dropped request; global_count=%d/%d (limit reached)",
            now.to_seconds(),
            self.name,
            self._last_known_global_count,
            self._global_limit,
        )
        return []
