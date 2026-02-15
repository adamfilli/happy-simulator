"""Timeout wrapper implementation.

Wraps a target entity with timeout handling, ensuring requests
don't wait indefinitely for responses.

Example:
    from happysimulator.components.resilience import TimeoutWrapper

    timeout_wrapper = TimeoutWrapper(
        name="api_timeout",
        target=slow_service,
        timeout=5.0,
    )

    # Requests go through the timeout wrapper
    request = Event(time=now, event_type="request", target=timeout_wrapper, ...)
    sim.schedule(request)
"""

import logging
from dataclasses import dataclass
from typing import Callable, Generator

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeoutStats:
    """Frozen snapshot of TimeoutWrapper statistics."""

    total_requests: int = 0
    successful_requests: int = 0
    timed_out_requests: int = 0


class TimeoutWrapper(Entity):
    """Wraps a target entity with timeout handling.

    Forwards requests to the target and tracks their completion.
    If a request doesn't complete within the timeout period, it
    is considered failed and an optional callback is invoked.

    Attributes:
        name: Timeout wrapper identifier.
        target: The service being wrapped.
        timeout: Maximum time to wait for response.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        timeout: float,
        on_timeout: Callable[[Event], Event | None] | None = None,
    ):
        """Initialize the timeout wrapper.

        Args:
            name: Timeout wrapper identifier.
            target: The downstream entity to wrap.
            timeout: Maximum time in seconds to wait for response.
            on_timeout: Optional callback when timeout occurs.
                       Receives the original request event.
                       Can return an event to schedule (e.g., fallback).

        Raises:
            ValueError: If timeout is invalid.
        """
        super().__init__(name)

        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")

        self._target = target
        self._timeout = timeout
        self._on_timeout = on_timeout

        # In-flight request tracking
        self._in_flight: dict[int, dict] = {}
        self._next_request_id = 0

        # Statistics
        self._total_requests = 0
        self._successful_requests = 0
        self._timed_out_requests = 0

        logger.debug(
            "[%s] TimeoutWrapper initialized: target=%s, timeout=%.3fs",
            name,
            target.name,
            timeout,
        )

    @property
    def target(self) -> Entity:
        """The wrapped target entity."""
        return self._target

    @property
    def timeout(self) -> float:
        """Timeout in seconds."""
        return self._timeout

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently in flight."""
        return len(self._in_flight)

    @property
    def stats(self) -> TimeoutStats:
        """Frozen snapshot of timeout wrapper statistics."""
        return TimeoutStats(
            total_requests=self._total_requests,
            successful_requests=self._successful_requests,
            timed_out_requests=self._timed_out_requests,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to target."""
        super().set_clock(clock)
        self._target.set_clock(clock)

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle incoming events.

        Forwards requests to target with timeout tracking.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule.
        """
        event_type = event.event_type

        if event_type == "_tw_response":
            return self._handle_response(event)

        if event_type == "_tw_timeout":
            return self._handle_timeout(event)

        self._total_requests += 1
        return self._forward_request(event)

    def _forward_request(self, event: Event) -> list[Event]:
        """Forward a request to the target with timeout tracking."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._in_flight[request_id] = {
            "start_time": self.now,
            "original_event": event,
            "completed": False,
        }

        logger.debug(
            "[%s] Forwarding request %d to %s (timeout=%.3fs)",
            self.name,
            request_id,
            self._target.name,
            self._timeout,
        )

        # Create forwarded event
        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=self._target,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_tw_request_id": request_id,
                    "_tw_name": self.name,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_tw_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                    },
                },
            )

        forwarded.add_completion_hook(on_complete)

        # Copy completion hooks from original event
        for hook in event.on_complete:
            forwarded.add_completion_hook(hook)

        # Schedule timeout event
        timeout_event = Event(
            time=self.now + Duration.from_seconds(self._timeout),
            event_type="_tw_timeout",
            target=self,
            context={
                "metadata": {
                    "request_id": request_id,
                },
            },
        )

        return [forwarded, timeout_event]

    def _handle_response(self, event: Event) -> None:
        """Handle a response from the target."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            logger.debug(
                "[%s] Response for unknown request: id=%s",
                self.name,
                request_id,
            )
            return None

        request_info = self._in_flight[request_id]

        # Check if already timed out
        if request_info["completed"]:
            logger.debug(
                "[%s] Response for already completed request: id=%s",
                self.name,
                request_id,
            )
            return None

        # Mark as completed
        request_info["completed"] = True
        del self._in_flight[request_id]

        self._successful_requests += 1

        response_time = (self.now - request_info["start_time"]).to_seconds()
        logger.debug(
            "[%s] Request %d completed in %.3fs",
            self.name,
            request_id,
            response_time,
        )

        return None

    def _handle_timeout(self, event: Event) -> list[Event] | Event | None:
        """Handle a timeout event."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            # Request already completed or timed out
            return None

        request_info = self._in_flight[request_id]

        # Check if already completed
        if request_info["completed"]:
            return None

        # Mark as timed out
        request_info["completed"] = True
        del self._in_flight[request_id]

        self._timed_out_requests += 1

        logger.debug(
            "[%s] Request %d timed out after %.3fs",
            self.name,
            request_id,
            self._timeout,
        )

        # Invoke timeout callback if provided
        if self._on_timeout:
            original_event = request_info["original_event"]
            return self._on_timeout(original_event)

        return None
