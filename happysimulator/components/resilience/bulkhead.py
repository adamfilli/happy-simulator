"""Bulkhead pattern implementation.

Provides isolation between different parts of a system by limiting
concurrent access to resources, preventing cascade failures.

Example:
    from happysimulator.components.resilience import Bulkhead

    bulkhead = Bulkhead(
        name="api_bulkhead",
        target=backend_server,
        max_concurrent=10,
        max_wait_queue=50,
        max_wait_time=5.0,
    )

    # Requests go through the bulkhead
    request = Event(time=now, event_type="request", target=bulkhead, ...)
    sim.schedule(request)
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Generator

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BulkheadStats:
    """Frozen snapshot of Bulkhead statistics."""

    total_requests: int = 0
    accepted_requests: int = 0
    rejected_requests: int = 0
    timed_out_requests: int = 0
    queued_requests: int = 0
    peak_concurrent: int = 0
    peak_queue_depth: int = 0


@dataclass
class WaitingRequest:
    """A request waiting in the bulkhead queue."""

    event: Event
    enqueue_time: Instant
    request_id: int


class Bulkhead(Entity):
    """Isolates resources by limiting concurrent access.

    The bulkhead limits the number of concurrent requests to a target
    service. Additional requests can optionally wait in a queue with
    a timeout. When both concurrent and queue limits are reached,
    requests are rejected immediately.

    This prevents a slow or failing service from consuming all resources
    and affecting other parts of the system.

    Attributes:
        name: Bulkhead identifier.
        target: The service being protected.
        max_concurrent: Maximum concurrent requests allowed.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        max_concurrent: int,
        max_wait_queue: int = 0,
        max_wait_time: float | None = None,
    ):
        """Initialize the bulkhead.

        Args:
            name: Bulkhead identifier.
            target: The downstream entity to protect.
            max_concurrent: Maximum concurrent requests to target.
            max_wait_queue: Maximum requests that can wait in queue.
                           0 means no queuing (immediate reject when full).
            max_wait_time: Maximum time a request can wait in queue.
                          None means no timeout (wait indefinitely).

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        if max_wait_queue < 0:
            raise ValueError(f"max_wait_queue must be >= 0, got {max_wait_queue}")
        if max_wait_time is not None and max_wait_time <= 0:
            raise ValueError(f"max_wait_time must be > 0 or None, got {max_wait_time}")

        self._target = target
        self._max_concurrent = max_concurrent
        self._max_wait_queue = max_wait_queue
        self._max_wait_time = max_wait_time

        # Concurrency tracking
        self._active_count = 0
        self._wait_queue: deque[WaitingRequest] = deque()
        self._next_request_id = 0

        # In-flight request tracking
        self._in_flight: dict[int, dict] = {}

        # Statistics
        self._total_requests = 0
        self._accepted_requests = 0
        self._rejected_requests = 0
        self._timed_out_requests = 0
        self._queued_requests = 0
        self._peak_concurrent = 0
        self._peak_queue_depth = 0

        logger.debug(
            "[%s] Bulkhead initialized: target=%s, max_concurrent=%d, max_queue=%d",
            name,
            target.name,
            max_concurrent,
            max_wait_queue,
        )

    @property
    def target(self) -> Entity:
        """The protected target entity."""
        return self._target

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent requests allowed."""
        return self._max_concurrent

    @property
    def max_wait_queue(self) -> int:
        """Maximum requests that can wait in queue."""
        return self._max_wait_queue

    @property
    def max_wait_time(self) -> float | None:
        """Maximum wait time in queue."""
        return self._max_wait_time

    @property
    def active_count(self) -> int:
        """Number of currently active requests."""
        return self._active_count

    @property
    def queue_depth(self) -> int:
        """Number of requests waiting in queue."""
        return len(self._wait_queue)

    @property
    def available_permits(self) -> int:
        """Number of available concurrent slots."""
        return max(0, self._max_concurrent - self._active_count)

    @property
    def stats(self) -> BulkheadStats:
        """Frozen snapshot of bulkhead statistics."""
        return BulkheadStats(
            total_requests=self._total_requests,
            accepted_requests=self._accepted_requests,
            rejected_requests=self._rejected_requests,
            timed_out_requests=self._timed_out_requests,
            queued_requests=self._queued_requests,
            peak_concurrent=self._peak_concurrent,
            peak_queue_depth=self._peak_queue_depth,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to target."""
        super().set_clock(clock)
        self._target.set_clock(clock)

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle incoming events.

        Routes requests through the bulkhead logic and handles
        responses from the target.

        Args:
            event: The event to handle.

        Returns:
            Events to schedule, or None if rejected.
        """
        event_type = event.event_type

        if event_type == "_bh_response":
            return self._handle_response(event)

        if event_type == "_bh_timeout":
            return self._handle_timeout(event)

        self._total_requests += 1

        # Check if we have capacity
        if self._active_count < self._max_concurrent:
            return self._forward_request(event)

        # Check if we can queue
        if len(self._wait_queue) < self._max_wait_queue:
            return self._enqueue_request(event)

        # Reject - no capacity and queue is full
        self._rejected_requests += 1
        logger.debug(
            "[%s] Request rejected (no capacity, queue full)",
            self.name,
        )
        return None

    def _forward_request(self, event: Event) -> list[Event]:
        """Forward a request to the target."""
        self._next_request_id += 1
        request_id = self._next_request_id

        self._active_count += 1
        self._accepted_requests += 1

        # Track peak
        if self._active_count > self._peak_concurrent:
            self._peak_concurrent = self._active_count

        self._in_flight[request_id] = {
            "start_time": self.now,
            "original_event": event,
        }

        logger.debug(
            "[%s] Forwarding request %d to %s (active=%d/%d)",
            self.name,
            request_id,
            self._target.name,
            self._active_count,
            self._max_concurrent,
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
                    "_bh_request_id": request_id,
                    "_bh_name": self.name,
                },
            },
        )

        # Add completion hook for response tracking
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_bh_response",
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

        return [forwarded]

    def _enqueue_request(self, event: Event) -> list[Event]:
        """Enqueue a request to wait for capacity."""
        self._next_request_id += 1
        request_id = self._next_request_id

        waiting = WaitingRequest(
            event=event,
            enqueue_time=self.now,
            request_id=request_id,
        )
        self._wait_queue.append(waiting)
        self._queued_requests += 1

        # Track peak queue depth
        queue_depth = len(self._wait_queue)
        if queue_depth > self._peak_queue_depth:
            self._peak_queue_depth = queue_depth

        logger.debug(
            "[%s] Request %d queued (queue_depth=%d/%d)",
            self.name,
            request_id,
            queue_depth,
            self._max_wait_queue,
        )

        events: list[Event] = []

        # Schedule timeout if configured
        if self._max_wait_time is not None:
            timeout_event = Event(
                time=self.now + Duration.from_seconds(self._max_wait_time),
                event_type="_bh_timeout",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                    },
                },
            )
            events.append(timeout_event)

        return events if events else []

    def _handle_response(self, event: Event) -> list[Event] | None:
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

        self._in_flight.pop(request_id)
        self._active_count = max(0, self._active_count - 1)

        logger.debug(
            "[%s] Request %d completed (active=%d/%d)",
            self.name,
            request_id,
            self._active_count,
            self._max_concurrent,
        )

        # Try to process next queued request
        return self._try_process_queued()

    def _handle_timeout(self, event: Event) -> None:
        """Handle a queue timeout."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")

        # Find and remove the request from queue
        for i, waiting in enumerate(self._wait_queue):
            if waiting.request_id == request_id:
                del self._wait_queue[i]
                self._timed_out_requests += 1
                logger.debug(
                    "[%s] Request %d timed out in queue",
                    self.name,
                    request_id,
                )
                return

        # Request already processed or removed
        return None

    def _try_process_queued(self) -> list[Event] | None:
        """Try to process the next queued request."""
        if not self._wait_queue:
            return None

        if self._active_count >= self._max_concurrent:
            return None

        # Get next waiting request
        waiting = self._wait_queue.popleft()

        # Check if it's expired (if timeout is configured)
        if self._max_wait_time is not None:
            wait_time = (self.now - waiting.enqueue_time).to_seconds()
            if wait_time > self._max_wait_time:
                # Skip expired request, try next
                self._timed_out_requests += 1
                return self._try_process_queued()

        logger.debug(
            "[%s] Processing queued request %d",
            self.name,
            waiting.request_id,
        )

        return self._forward_request(waiting.event)
