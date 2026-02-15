"""Fallback pattern implementation.

Provides fallback behavior when the primary service fails,
ensuring graceful degradation.

Example:
    from happysimulator.components.resilience import Fallback

    fallback = Fallback(
        name="api_fallback",
        primary=main_service,
        fallback=cache_service,
    )

    # Requests go through the fallback wrapper
    request = Event(time=now, event_type="request", target=fallback, ...)
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
class FallbackStats:
    """Frozen snapshot of Fallback statistics."""

    total_requests: int = 0
    primary_successes: int = 0
    primary_failures: int = 0
    fallback_invocations: int = 0
    fallback_successes: int = 0
    fallback_failures: int = 0


class Fallback(Entity):
    """Provides fallback behavior on primary failure.

    Forwards requests to the primary service and monitors for failures.
    When a failure is detected (via timeout, exception, or predicate),
    the request is retried with the fallback service.

    Attributes:
        name: Fallback wrapper identifier.
        primary: The primary service to try first.
        fallback: The fallback service or function.
    """

    def __init__(
        self,
        name: str,
        primary: Entity,
        fallback: Entity | Callable[[Event], Event | None],
        failure_predicate: Callable[[Event], bool] | None = None,
        timeout: float | None = None,
    ):
        """Initialize the fallback wrapper.

        Args:
            name: Fallback wrapper identifier.
            primary: The primary entity to forward requests to.
            fallback: The fallback entity or callable.
                     If callable, receives the original event and returns
                     a fallback event to schedule (or None).
            failure_predicate: Optional function to detect failures.
                              Returns True if the response indicates failure.
            timeout: Optional timeout before triggering fallback.
                    If None, only failure_predicate triggers fallback.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if timeout is not None and timeout <= 0:
            raise ValueError(f"timeout must be > 0 or None, got {timeout}")

        self._primary = primary
        self._fallback = fallback
        self._failure_predicate = failure_predicate
        self._timeout = timeout

        # In-flight request tracking
        self._in_flight: dict[int, dict] = {}
        self._next_request_id = 0

        # Statistics
        self._total_requests = 0
        self._primary_successes = 0
        self._primary_failures = 0
        self._fallback_invocations = 0
        self._fallback_successes = 0
        self._fallback_failures = 0

        fallback_name = fallback.name if isinstance(fallback, Entity) else "callback"
        logger.debug(
            "[%s] Fallback initialized: primary=%s, fallback=%s",
            name,
            primary.name,
            fallback_name,
        )

    @property
    def primary(self) -> Entity:
        """The primary entity."""
        return self._primary

    @property
    def fallback(self) -> Entity | Callable[[Event], Event | None]:
        """The fallback entity or callable."""
        return self._fallback

    @property
    def timeout(self) -> float | None:
        """Timeout before triggering fallback."""
        return self._timeout

    @property
    def stats(self) -> FallbackStats:
        """Frozen snapshot of fallback statistics."""
        return FallbackStats(
            total_requests=self._total_requests,
            primary_successes=self._primary_successes,
            primary_failures=self._primary_failures,
            fallback_invocations=self._fallback_invocations,
            fallback_successes=self._fallback_successes,
            fallback_failures=self._fallback_failures,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to primary and fallback."""
        super().set_clock(clock)
        self._primary.set_clock(clock)
        if isinstance(self._fallback, Entity):
            self._fallback.set_clock(clock)

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle incoming events.

        Forwards requests to primary with fallback handling.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule.
        """
        event_type = event.event_type

        if event_type == "_fb_primary_response":
            return self._handle_primary_response(event)

        if event_type == "_fb_timeout":
            return self._handle_timeout(event)

        if event_type == "_fb_fallback_response":
            return self._handle_fallback_response(event)

        self._total_requests += 1
        return self._forward_to_primary(event)

    def _forward_to_primary(self, event: Event) -> list[Event]:
        """Forward a request to the primary service."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._in_flight[request_id] = {
            "start_time": self.now,
            "original_event": event,
            "state": "primary",  # "primary", "fallback", "completed"
        }

        logger.debug(
            "[%s] Forwarding request %d to primary (%s)",
            self.name,
            request_id,
            self._primary.name,
        )

        # Create forwarded event
        forwarded = Event(
            time=self.now,
            event_type=event.event_type,
            target=self._primary,
            context={
                **event.context,
                "metadata": {
                    **event.context.get("metadata", {}),
                    "_fb_request_id": request_id,
                    "_fb_name": self.name,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_fb_primary_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "success": True,
                    },
                },
            )

        forwarded.add_completion_hook(on_complete)

        # Copy completion hooks from original event
        for hook in event.on_complete:
            forwarded.add_completion_hook(hook)

        events = [forwarded]

        # Schedule timeout if configured
        if self._timeout is not None:
            timeout_event = Event(
                time=self.now + Duration.from_seconds(self._timeout),
                event_type="_fb_timeout",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                    },
                },
            )
            events.append(timeout_event)

        return events

    def _forward_to_fallback(self, request_id: int, original_event: Event) -> list[Event] | Event | None:
        """Forward a request to the fallback service."""
        self._fallback_invocations += 1

        if callable(self._fallback) and not isinstance(self._fallback, Entity):
            # Fallback is a callable
            fallback_event = self._fallback(original_event)
            if fallback_event:
                logger.debug(
                    "[%s] Request %d using fallback callback",
                    self.name,
                    request_id,
                )
                return fallback_event
            return None

        # Fallback is an entity
        fallback_entity = self._fallback
        logger.debug(
            "[%s] Forwarding request %d to fallback (%s)",
            self.name,
            request_id,
            fallback_entity.name,
        )

        # Create forwarded event
        forwarded = Event(
            time=self.now,
            event_type=original_event.event_type,
            target=fallback_entity,
            context={
                **original_event.context,
                "metadata": {
                    **original_event.context.get("metadata", {}),
                    "_fb_request_id": request_id,
                    "_fb_name": self.name,
                    "_fb_is_fallback": True,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_fb_fallback_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                    },
                },
            )

        forwarded.add_completion_hook(on_complete)

        # Copy completion hooks from original event
        for hook in original_event.on_complete:
            forwarded.add_completion_hook(hook)

        return [forwarded]

    def _handle_primary_response(self, event: Event) -> list[Event] | Event | None:
        """Handle a response from the primary service."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        success = metadata.get("success", True)

        if request_id not in self._in_flight:
            logger.debug(
                "[%s] Response for unknown request: id=%s",
                self.name,
                request_id,
            )
            return None

        request_info = self._in_flight[request_id]

        # Check if already handled (timed out)
        if request_info["state"] != "primary":
            return None

        # Check failure predicate
        if self._failure_predicate:
            original_event = request_info["original_event"]
            if self._failure_predicate(original_event):
                success = False

        if success:
            # Primary succeeded
            request_info["state"] = "completed"
            del self._in_flight[request_id]
            self._primary_successes += 1
            logger.debug(
                "[%s] Request %d succeeded on primary",
                self.name,
                request_id,
            )
            return None

        # Primary failed, try fallback
        self._primary_failures += 1
        request_info["state"] = "fallback"
        return self._forward_to_fallback(request_id, request_info["original_event"])

    def _handle_timeout(self, event: Event) -> list[Event] | Event | None:
        """Handle a timeout event."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            return None

        request_info = self._in_flight[request_id]

        # Only timeout if still waiting for primary
        if request_info["state"] != "primary":
            return None

        self._primary_failures += 1
        request_info["state"] = "fallback"

        logger.debug(
            "[%s] Request %d timed out, trying fallback",
            self.name,
            request_id,
        )

        return self._forward_to_fallback(request_id, request_info["original_event"])

    def _handle_fallback_response(self, event: Event) -> None:
        """Handle a response from the fallback service."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        if request_id not in self._in_flight:
            logger.debug(
                "[%s] Fallback response for unknown request: id=%s",
                self.name,
                request_id,
            )
            return None

        request_info = self._in_flight[request_id]

        if request_info["state"] != "fallback":
            return None

        request_info["state"] = "completed"
        del self._in_flight[request_id]
        self._fallback_successes += 1

        logger.debug(
            "[%s] Request %d succeeded on fallback",
            self.name,
            request_id,
        )

        return None
