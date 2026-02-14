"""Idempotency store for duplicate request suppression.

Wraps a target entity and deduplicates requests based on an idempotency
key extracted from each event. Cached responses prevent duplicate processing
while a TTL-based cleanup prevents unbounded memory growth.

Example:
    from happysimulator.components.microservice import IdempotencyStore

    store = IdempotencyStore(
        name="idem",
        target=payment_service,
        key_extractor=lambda e: e.get_context("idempotency_key"),
        ttl=300.0,
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass
class _CachedResponse:
    """Internal record for a cached idempotency result."""

    key: str
    cached_at: Instant
    ttl: float


@dataclass
class IdempotencyStoreStats:
    """Statistics tracked by IdempotencyStore."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    entries_expired: int = 0
    entries_stored: int = 0


class IdempotencyStore(Entity):
    """Deduplicates requests based on an idempotency key.

    Incoming events have their idempotency key extracted via ``key_extractor``.
    If the key is already cached, the event is silently dropped (duplicate).
    Otherwise, the event is forwarded to the target and the key is cached
    upon completion.

    A periodic cleanup daemon expires entries older than ``ttl`` seconds.

    Attributes:
        name: Store identifier.
        stats: Accumulated statistics.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        key_extractor: Callable[[Event], str | None],
        ttl: float = 300.0,
        max_entries: int = 10_000,
        cleanup_interval: float = 60.0,
    ):
        """Initialize the idempotency store.

        Args:
            name: Store identifier.
            target: Downstream entity to forward unique requests to.
            key_extractor: Function that extracts an idempotency key from an
                event. Return None to forward unconditionally (no dedup).
            ttl: Time-to-live in seconds for cached entries.
            max_entries: Maximum cache size. Oldest entries evicted on overflow.
            cleanup_interval: Seconds between TTL cleanup sweeps.

        Raises:
            ValueError: If ttl, max_entries, or cleanup_interval are invalid.
        """
        super().__init__(name)

        if ttl <= 0:
            raise ValueError(f"ttl must be > 0, got {ttl}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        if cleanup_interval <= 0:
            raise ValueError(f"cleanup_interval must be > 0, got {cleanup_interval}")

        self._target = target
        self._key_extractor = key_extractor
        self._ttl = ttl
        self._max_entries = max_entries
        self._cleanup_interval = cleanup_interval

        self._cache: dict[str, _CachedResponse] = {}
        self._in_flight: set[str] = set()

        self.stats = IdempotencyStoreStats()

        logger.debug(
            "[%s] IdempotencyStore initialized: target=%s, ttl=%.1fs, max_entries=%d",
            name,
            target.name,
            ttl,
            max_entries,
        )

    @property
    def target(self) -> Entity:
        """The protected target entity."""
        return self._target

    @property
    def cache_size(self) -> int:
        """Current number of cached entries."""
        return len(self._cache)

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently being processed."""
        return len(self._in_flight)

    def handle_event(self, event: Event) -> list[Event] | None:
        """Route events: cleanup daemon vs request dedup.

        Args:
            event: The incoming event.

        Returns:
            Events to schedule, or None for duplicates.
        """
        if event.event_type == f"_is_cleanup::{self.name}":
            return self._handle_cleanup(event)

        if event.event_type == "_is_response":
            return self._handle_response(event)

        return self._handle_request(event)

    def _handle_request(self, event: Event) -> list[Event] | None:
        """Check idempotency and forward or suppress."""
        self.stats.total_requests += 1

        key = self._key_extractor(event)
        if key is None:
            # No idempotency key — forward unconditionally
            return self._forward(event, key=None)

        # Check cache
        if key in self._cache:
            self.stats.cache_hits += 1
            logger.debug("[%s] Cache hit for key=%s (duplicate suppressed)", self.name, key)
            return None

        # Check if already in-flight
        if key in self._in_flight:
            self.stats.cache_hits += 1
            logger.debug("[%s] In-flight hit for key=%s (duplicate suppressed)", self.name, key)
            return None

        self.stats.cache_misses += 1
        return self._forward(event, key=key)

    def _forward(self, event: Event, *, key: str | None) -> list[Event]:
        """Forward event to target with completion tracking."""
        if key is not None:
            self._in_flight.add(key)

        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=self._target,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_is_key": key,
                    "_is_name": self.name,
                },
            },
        )

        # Completion hook to cache the result
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_is_response",
                target=self,
                context={"metadata": {"key": key}},
            )

        forwarded.add_completion_hook(on_complete)

        # Copy original completion hooks
        for hook in event.on_complete:
            forwarded.add_completion_hook(hook)

        result: list[Event] = [forwarded]

        # Schedule first cleanup if this is the first entry
        if len(self._cache) == 0 and len(self._in_flight) <= 1:
            result.append(self._schedule_cleanup())

        return result

    def _handle_response(self, event: Event) -> None:
        """Cache the completed key."""
        metadata = event.context.get("metadata", {})
        key = metadata.get("key")

        if key is None:
            return

        self._in_flight.discard(key)

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats.entries_expired += 1

        self._cache[key] = _CachedResponse(
            key=key,
            cached_at=self.now,
            ttl=self._ttl,
        )
        self.stats.entries_stored += 1

        logger.debug("[%s] Cached key=%s", self.name, key)

    def _handle_cleanup(self, event: Event) -> list[Event]:
        """Remove expired entries and reschedule."""
        now = self.now
        expired_keys = [
            key
            for key, entry in self._cache.items()
            if (now - entry.cached_at).to_seconds() >= entry.ttl
        ]

        for key in expired_keys:
            del self._cache[key]
            self.stats.entries_expired += 1

        if expired_keys:
            logger.debug(
                "[%s] Cleanup: expired %d entries, %d remaining",
                self.name,
                len(expired_keys),
                len(self._cache),
            )

        # Reschedule if there are still entries to manage
        if self._cache or self._in_flight:
            return [self._schedule_cleanup()]
        return []

    def _schedule_cleanup(self) -> Event:
        """Create a daemon cleanup event."""
        return Event(
            time=self.now + Duration.from_seconds(self._cleanup_interval),
            event_type=f"_is_cleanup::{self.name}",
            target=self,
            daemon=True,
        )
