"""Hedged request pattern implementation.

Sends redundant requests after a delay to reduce tail latency.
The first successful response is used and other in-flight requests
are cancelled (or ignored).

Example:
    from happysimulator.components.resilience import Hedge

    hedge = Hedge(
        name="api_hedge",
        target=backend_service,
        hedge_delay=0.050,  # 50ms
        max_hedges=1,
    )

    # Requests go through the hedge wrapper
    request = Event(time=now, event_type="request", target=hedge, ...)
    sim.schedule(request)
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HedgeStats:
    """Frozen snapshot of Hedge statistics."""

    total_requests: int = 0
    primary_wins: int = 0
    hedge_wins: int = 0
    hedges_sent: int = 0
    hedges_cancelled: int = 0


class Hedge(Entity):
    """Sends redundant requests to reduce tail latency.

    When a request doesn't complete within the hedge delay, a second
    (hedge) request is sent to the same target. The first response
    is used and the other request is effectively cancelled.

    This is useful when:
    - Tail latency is significantly higher than median latency
    - The cost of extra requests is acceptable
    - The target service is idempotent

    Attributes:
        name: Hedge wrapper identifier.
        target: The service to send requests to.
        hedge_delay: Time to wait before sending hedge request.
        max_hedges: Maximum number of hedge requests per original.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        hedge_delay: float,
        max_hedges: int = 1,
    ):
        """Initialize the hedge wrapper.

        Args:
            name: Hedge wrapper identifier.
            target: The downstream entity to send requests to.
            hedge_delay: Seconds to wait before sending hedge request.
            max_hedges: Maximum number of hedge requests per original.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if hedge_delay <= 0:
            raise ValueError(f"hedge_delay must be > 0, got {hedge_delay}")
        if max_hedges < 1:
            raise ValueError(f"max_hedges must be >= 1, got {max_hedges}")

        self._target = target
        self._hedge_delay = hedge_delay
        self._max_hedges = max_hedges

        # In-flight request tracking
        # Maps original request_id -> request info
        self._in_flight: dict[int, dict] = {}
        self._next_request_id = 0

        # Statistics
        self._total_requests = 0
        self._primary_wins = 0
        self._hedge_wins = 0
        self._hedges_sent = 0
        self._hedges_cancelled = 0

        logger.debug(
            "[%s] Hedge initialized: target=%s, delay=%.3fs, max_hedges=%d",
            name,
            target.name,
            hedge_delay,
            max_hedges,
        )

    @property
    def target(self) -> Entity:
        """The target entity."""
        return self._target

    @property
    def hedge_delay(self) -> float:
        """Delay before sending hedge request."""
        return self._hedge_delay

    @property
    def max_hedges(self) -> int:
        """Maximum number of hedge requests."""
        return self._max_hedges

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently in flight."""
        return len(self._in_flight)

    @property
    def stats(self) -> HedgeStats:
        """Frozen snapshot of hedge statistics."""
        return HedgeStats(
            total_requests=self._total_requests,
            primary_wins=self._primary_wins,
            hedge_wins=self._hedge_wins,
            hedges_sent=self._hedges_sent,
            hedges_cancelled=self._hedges_cancelled,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to target."""
        super().set_clock(clock)
        self._target.set_clock(clock)

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle incoming events.

        Forwards requests with hedge scheduling.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule.
        """
        event_type = event.event_type

        if event_type == "_hg_response":
            return self._handle_response(event)

        if event_type == "_hg_send_hedge":
            return self._send_hedge(event)

        self._total_requests += 1
        return self._forward_primary(event)

    def _forward_primary(self, event: Event) -> list[Event]:
        """Forward the primary request and schedule hedge."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._in_flight[request_id] = {
            "start_time": self.now,
            "original_event": event,
            "completed": False,
            "hedges_sent": 0,
            "responses_received": 0,
        }

        logger.debug(
            "[%s] Forwarding primary request %d to %s",
            self.name,
            request_id,
            self._target.name,
        )

        events = []

        # Send primary request
        primary = self._create_request_event(request_id, event, is_hedge=False)
        events.append(primary)

        # Schedule first hedge
        if self._max_hedges >= 1:
            hedge_trigger = Event(
                time=self.now + Duration.from_seconds(self._hedge_delay),
                event_type="_hg_send_hedge",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "hedge_number": 1,
                    },
                },
            )
            events.append(hedge_trigger)

        return events

    def _create_request_event(
        self, request_id: int, original_event: Event, is_hedge: bool
    ) -> Event:
        """Create a request event to send to target."""
        forwarded = Event(
            time=self.now,
            event_type=original_event.event_type,
            target=self._target,
            context={
                **original_event.context,
                "metadata": {
                    **original_event.context.get("metadata", {}),
                    "_hg_request_id": request_id,
                    "_hg_name": self.name,
                    "_hg_is_hedge": is_hedge,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_hg_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "is_hedge": is_hedge,
                    },
                },
            )

        forwarded.add_completion_hook(on_complete)

        # Copy completion hooks from original event (only for primary)
        if not is_hedge:
            for hook in original_event.on_complete:
                forwarded.add_completion_hook(hook)

        return forwarded

    def _send_hedge(self, event: Event) -> list[Event] | None:
        """Send a hedge request."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        hedge_number = metadata.get("hedge_number", 1)

        if request_id not in self._in_flight:
            return None

        request_info = self._in_flight[request_id]

        # Check if already completed
        if request_info["completed"]:
            self._hedges_cancelled += 1
            return None

        # Check if we've sent max hedges
        if request_info["hedges_sent"] >= self._max_hedges:
            return None

        # Send hedge
        request_info["hedges_sent"] += 1
        self._hedges_sent += 1

        original_event = request_info["original_event"]

        logger.debug(
            "[%s] Sending hedge %d for request %d",
            self.name,
            hedge_number,
            request_id,
        )

        events = []

        # Create and send hedge request
        hedge_request = self._create_request_event(request_id, original_event, is_hedge=True)
        events.append(hedge_request)

        # Schedule next hedge if applicable
        if hedge_number < self._max_hedges:
            next_hedge = Event(
                time=self.now + Duration.from_seconds(self._hedge_delay),
                event_type="_hg_send_hedge",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "hedge_number": hedge_number + 1,
                    },
                },
            )
            events.append(next_hedge)

        return events

    def _handle_response(self, event: Event) -> None:
        """Handle a response from the target."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        is_hedge = metadata.get("is_hedge", False)

        if request_id not in self._in_flight:
            logger.debug(
                "[%s] Response for unknown request: id=%s",
                self.name,
                request_id,
            )
            return

        request_info = self._in_flight[request_id]
        request_info["responses_received"] += 1

        # Check if this is the first response (winner)
        if not request_info["completed"]:
            request_info["completed"] = True

            response_time = (self.now - request_info["start_time"]).to_seconds()

            if is_hedge:
                self._hedge_wins += 1
                logger.debug(
                    "[%s] Request %d completed by hedge in %.3fs",
                    self.name,
                    request_id,
                    response_time,
                )
            else:
                self._primary_wins += 1
                logger.debug(
                    "[%s] Request %d completed by primary in %.3fs",
                    self.name,
                    request_id,
                    response_time,
                )

        # Clean up if all expected responses received
        expected = 1 + request_info["hedges_sent"]
        if request_info["responses_received"] >= expected:
            del self._in_flight[request_id]

        return
