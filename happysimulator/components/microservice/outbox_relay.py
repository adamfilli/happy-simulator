"""Transactional outbox pattern with poll-based relay.

Implements the outbox pattern where business events are written to an
in-memory outbox atomically with business logic, and a self-scheduling
poll loop relays entries to a downstream entity (e.g., MessageQueue).

Example:
    from happysimulator.components.microservice import OutboxRelay

    outbox = OutboxRelay(
        name="order_outbox",
        downstream=message_queue,
        poll_interval=0.1,
        batch_size=50,
    )

    # In your entity's handle_event:
    outbox.write({"order_id": 123, "status": "created"})
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass
class OutboxEntry:
    """A single entry in the outbox."""

    entry_id: int
    payload: dict[str, Any]
    written_at: Instant
    relayed: bool = False


@dataclass(frozen=True)
class OutboxRelayStats:
    """Statistics tracked by OutboxRelay."""

    entries_written: int = 0
    entries_relayed: int = 0
    relay_failures: int = 0
    poll_cycles: int = 0
    relay_lag_sum: float = 0.0
    relay_lag_max: float = 0.0

    @property
    def avg_relay_lag(self) -> float:
        """Average relay lag in seconds."""
        if self.entries_relayed == 0:
            return 0.0
        return self.relay_lag_sum / self.entries_relayed


class OutboxRelay(Entity):
    """Transactional outbox with poll-based relay to downstream.

    Events are written to an in-memory outbox via ``write()``. A
    self-scheduling poll daemon reads unrelayed entries in batches
    and forwards them to the downstream entity.

    The relay lag (time between write and relay) is tracked for
    observability.

    Attributes:
        name: Outbox identifier.
        stats: Accumulated statistics.
    """

    def __init__(
        self,
        name: str,
        downstream: Entity,
        poll_interval: float = 0.1,
        batch_size: int = 100,
        relay_latency: float = 0.001,
    ):
        """Initialize the outbox relay.

        Args:
            name: Outbox identifier.
            downstream: Entity to relay entries to.
            poll_interval: Seconds between poll cycles.
            batch_size: Maximum entries relayed per poll cycle.
            relay_latency: Simulated per-entry relay latency in seconds.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if poll_interval <= 0:
            raise ValueError(f"poll_interval must be > 0, got {poll_interval}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if relay_latency < 0:
            raise ValueError(f"relay_latency must be >= 0, got {relay_latency}")

        self._downstream = downstream
        self._poll_interval = poll_interval
        self._batch_size = batch_size
        self._relay_latency = relay_latency

        self._entries: list[OutboxEntry] = []
        self._next_entry_id = 0
        self._poll_scheduled = False

        self._entries_written = 0
        self._entries_relayed = 0
        self._relay_failures = 0
        self._poll_cycles = 0
        self._relay_lag_sum = 0.0
        self._relay_lag_max = 0.0

        logger.debug(
            "[%s] OutboxRelay initialized: downstream=%s, poll_interval=%.3fs, batch_size=%d",
            name,
            downstream.name,
            poll_interval,
            batch_size,
        )

    @property
    def stats(self) -> OutboxRelayStats:
        """Return a frozen snapshot of current statistics."""
        return OutboxRelayStats(
            entries_written=self._entries_written,
            entries_relayed=self._entries_relayed,
            relay_failures=self._relay_failures,
            poll_cycles=self._poll_cycles,
            relay_lag_sum=self._relay_lag_sum,
            relay_lag_max=self._relay_lag_max,
        )

    @property
    def downstream(self) -> Entity:
        """The relay target entity."""
        return self._downstream

    @property
    def pending_count(self) -> int:
        """Number of entries waiting to be relayed."""
        return sum(1 for e in self._entries if not e.relayed)

    @property
    def total_entries(self) -> int:
        """Total entries in the outbox (including relayed)."""
        return len(self._entries)

    def write(self, payload: dict[str, Any]) -> int:
        """Write an entry to the outbox.

        This is a regular method (not a generator) intended to be called
        by user entities during their ``handle_event()`` processing.

        Args:
            payload: The event payload to relay downstream.

        Returns:
            The entry ID for tracking.
        """
        self._next_entry_id += 1
        entry_id = self._next_entry_id

        entry = OutboxEntry(
            entry_id=entry_id,
            payload=payload,
            written_at=self.now,
        )
        self._entries.append(entry)
        self._entries_written += 1

        logger.debug("[%s] Entry written: id=%d", self.name, entry_id)
        return entry_id

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]] | list[Event]:
        """Handle poll events and prime the poll loop.

        Args:
            event: The incoming event.

        Returns:
            Events to schedule.
        """
        if event.event_type == f"_outbox_poll::{self.name}":
            return self._handle_poll(event)

        # Any non-poll event primes the poll loop if needed
        if not self._poll_scheduled:
            return [self._schedule_poll()]
        return []

    def _handle_poll(self, event: Event) -> Generator[float, None, list[Event]]:
        """Process a batch of pending outbox entries."""
        self._poll_scheduled = False
        self._poll_cycles += 1

        # Collect pending entries up to batch_size
        pending = [e for e in self._entries if not e.relayed][: self._batch_size]

        relay_events: list[Event] = []
        for entry in pending:
            entry.relayed = True
            self._entries_relayed += 1

            # Track relay lag
            lag = (self.now - entry.written_at).to_seconds()
            self._relay_lag_sum += lag
            if lag > self._relay_lag_max:
                self._relay_lag_max = lag

            relay_events.append(
                Event(
                    time=self.now,
                    event_type="outbox_relay",
                    target=self._downstream,
                    context={
                        "metadata": {
                            "outbox_name": self.name,
                            "entry_id": entry.entry_id,
                        },
                        "payload": entry.payload,
                    },
                )
            )

            # Simulate relay latency between entries
            if self._relay_latency > 0:
                yield self._relay_latency

        logger.debug(
            "[%s] Poll cycle: relayed %d entries, %d remaining",
            self.name,
            len(pending),
            self.pending_count,
        )

        # Reschedule if there are more pending entries or keep polling
        result = relay_events
        if self.pending_count > 0 or self._entries_written > 0:
            result.append(self._schedule_poll())

        return result

    def prime_poll(self) -> Event:
        """Create the initial poll event to start the relay loop.

        Call this and schedule the returned event to begin relaying.
        Alternatively, sending any event to the outbox will auto-prime.

        Returns:
            The initial poll event.
        """
        return self._schedule_poll()

    def _schedule_poll(self) -> Event:
        """Create a daemon poll event."""
        self._poll_scheduled = True
        return Event(
            time=self.now + Duration.from_seconds(self._poll_interval),
            event_type=f"_outbox_poll::{self.name}",
            target=self,
            daemon=True,
        )
