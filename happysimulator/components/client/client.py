"""Client for sending requests to target entities.

Client provides a high-level abstraction for making requests to servers
or other entities, with support for timeouts, retries, and response handling.

Example:
    from happysimulator.components.client import Client, ExponentialBackoff

    client = Client(
        name="api_client",
        target=server,
        timeout=5.0,
        retry_policy=ExponentialBackoff(max_attempts=3, initial_delay=0.1, max_delay=10.0),
    )

    # Send a request
    request = client.send_request(payload={"action": "get_user"})
    sim.schedule(request)
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Generator

from happysimulator.components.client.retry import NoRetry, RetryPolicy
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientStats:
    """Frozen snapshot of Client statistics."""

    requests_sent: int = 0
    responses_received: int = 0
    timeouts: int = 0
    retries: int = 0
    failures: int = 0


class Client(Entity):
    """Client that sends requests and handles responses.

    Provides a clean interface for making requests to target entities with
    support for timeouts and retry policies. Tracks in-flight requests and
    collects statistics for analysis.

    The client handles the request lifecycle:
    1. Send request to target
    2. Wait for response (with optional timeout)
    3. On success: invoke success callback, record stats
    4. On timeout: retry according to policy, or invoke failure callback

    Attributes:
        name: Client identifier for logging.
        target: The entity to send requests to.
        timeout: Request timeout in seconds (None = no timeout).
        retry_policy: Policy for retry behavior.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
        on_success: Callable[[Event, Event], None] | None = None,
        on_failure: Callable[[Event, str], None] | None = None,
    ):
        """Initialize the client.

        Args:
            name: Client identifier.
            target: Entity to send requests to.
            timeout: Request timeout in seconds (None = no timeout).
            retry_policy: Retry policy for failed requests (default NoRetry).
            on_success: Callback(request, response) on successful response.
            on_failure: Callback(request, reason) on final failure.

        Raises:
            ValueError: If timeout is negative.
        """
        super().__init__(name)

        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be >= 0, got {timeout}")

        self._target = target
        self._timeout = timeout
        self._retry_policy = retry_policy or NoRetry()
        self._on_success = on_success
        self._on_failure = on_failure

        # Track in-flight requests: request_id -> request info
        # We use a composite key: (request_id, attempt) to avoid confusion
        # when retries are in flight at the same time as original requests
        self._in_flight: dict[tuple[int, int], dict] = {}
        self._next_request_id = 0

        # Statistics
        self._requests_sent = 0
        self._responses_received = 0
        self._timeouts = 0
        self._retries = 0
        self._failures = 0

        # Response times for analysis
        self._response_times: list[float] = []

        logger.debug(
            "[%s] Client initialized: target=%s, timeout=%s, retry_policy=%s",
            name,
            target.name if hasattr(target, 'name') else str(target),
            timeout,
            type(self._retry_policy).__name__,
        )

    @property
    def target(self) -> Entity:
        """The target entity for requests."""
        return self._target

    @property
    def timeout(self) -> float | None:
        """Request timeout in seconds."""
        return self._timeout

    @property
    def retry_policy(self) -> RetryPolicy:
        """The retry policy for failed requests."""
        return self._retry_policy

    @property
    def in_flight_count(self) -> int:
        """Number of requests currently in flight."""
        return len(self._in_flight)

    @property
    def average_response_time(self) -> float:
        """Average response time in seconds."""
        if not self._response_times:
            return 0.0
        return sum(self._response_times) / len(self._response_times)

    @property
    def stats(self) -> ClientStats:
        """Frozen snapshot of client statistics."""
        return ClientStats(
            requests_sent=self._requests_sent,
            responses_received=self._responses_received,
            timeouts=self._timeouts,
            retries=self._retries,
            failures=self._failures,
        )

    def send_request(
        self,
        payload: Any = None,
        event_type: str = "request",
        on_success: Callable[[Event, Event], None] | None = None,
        on_failure: Callable[[Event, str], None] | None = None,
    ) -> Event:
        """Create a request event to send to the target.

        Creates an event that, when scheduled, will send a request to the
        target entity. The event is ready to be scheduled with sim.schedule().

        Args:
            payload: Optional payload data for the request.
            event_type: Type string for the request event.
            on_success: Override success callback for this request.
            on_failure: Override failure callback for this request.

        Returns:
            Event ready to be scheduled.
        """
        self._next_request_id += 1
        request_id = self._next_request_id

        # Create the request event
        event = Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type=event_type,
            target=self,  # Route through client first
            context={
                "metadata": {
                    "request_id": request_id,
                    "payload": payload,
                    "attempt": 1,
                    "client_name": self.name,
                },
                "_on_success": on_success or self._on_success,
                "_on_failure": on_failure or self._on_failure,
            },
        )

        return event

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle client events.

        Processes:
        - Outgoing requests: forward to target with tracking
        - Timeout events: handle timeout and retry
        - Response events: complete the request

        Args:
            event: The event to handle.

        Returns:
            Events to schedule or generator for async processing.
        """
        event_type = event.event_type

        if event_type == "_client_timeout":
            return self._handle_timeout(event)

        if event_type == "_client_response":
            return self._handle_response(event)

        # New outgoing request
        return self._send_request(event)

    def _send_request(self, event: Event) -> list[Event]:
        """Send a request to the target."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        attempt = metadata.get("attempt", 1)

        # Use composite key (request_id, attempt) to track each attempt separately
        # This prevents a late response from an earlier attempt from being
        # incorrectly matched to a retry
        flight_key = (request_id, attempt)

        # Track the in-flight request
        self._in_flight[flight_key] = {
            "start_time": self.now,
            "attempt": attempt,
            "event": event,
            "on_success": event.context.get("_on_success"),
            "on_failure": event.context.get("_on_failure"),
            "request_id": request_id,
        }

        # Update stats
        self._requests_sent += 1
        if attempt > 1:
            self._retries += 1

        logger.debug(
            "[%s] Sending request: id=%s attempt=%d target=%s",
            self.name,
            request_id,
            attempt,
            self._target.name if hasattr(self._target, 'name') else str(self._target),
        )

        # Create the actual request to target
        target_event = Event(
            time=self.now,
            event_type=event.event_type if event.event_type != "request" else f"{self.name}_request",
            target=self._target,
            context={
                "metadata": {
                    "request_id": request_id,
                    "payload": metadata.get("payload"),
                    "client": self,
                    "attempt": attempt,
                },
            },
        )

        # Add completion hook to receive response
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_client_response",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "attempt": attempt,
                    },
                },
            )

        target_event.add_completion_hook(on_complete)

        result_events = [target_event]

        # Schedule timeout if configured
        if self._timeout is not None:
            timeout_event = Event(
                time=self.now + Duration.from_seconds(self._timeout),
                event_type="_client_timeout",
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "attempt": attempt,
                    },
                },
            )
            result_events.append(timeout_event)

        return result_events

    def _handle_response(self, event: Event) -> list[Event] | None:
        """Handle a response from the target."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        attempt = metadata.get("attempt", 1)

        # Use composite key to match exact attempt
        flight_key = (request_id, attempt)

        # Check if this specific attempt is still in flight (not timed out)
        if flight_key not in self._in_flight:
            logger.debug(
                "[%s] Received response for unknown/completed request: id=%s attempt=%d",
                self.name,
                request_id,
                attempt,
            )
            return None

        # Get request info and remove from in-flight
        request_info = self._in_flight.pop(flight_key)
        start_time = request_info["start_time"]
        original_event = request_info["event"]

        # Calculate response time
        response_time = (self.now - start_time).to_seconds()
        self._response_times.append(response_time)

        # Update stats
        self._responses_received += 1

        logger.debug(
            "[%s] Received response: id=%s response_time=%.4fs",
            self.name,
            request_id,
            response_time,
        )

        # Invoke success callback
        on_success = request_info.get("on_success")
        if on_success is not None:
            on_success(original_event, event)

        return None

    def _handle_timeout(self, event: Event) -> list[Event] | None:
        """Handle a request timeout."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        attempt = metadata.get("attempt", 1)

        # Use composite key to match exact attempt
        flight_key = (request_id, attempt)

        # Check if this specific attempt is still in flight
        if flight_key not in self._in_flight:
            # Already completed, ignore timeout
            return None

        # Get request info
        request_info = self._in_flight[flight_key]
        original_event = request_info["event"]

        # Update stats
        self._timeouts += 1

        logger.debug(
            "[%s] Request timeout: id=%s attempt=%d",
            self.name,
            request_id,
            attempt,
        )

        # Check if should retry
        if self._retry_policy.should_retry(attempt, error=TimeoutError("Request timeout")):
            # Remove from in-flight (will be re-added on retry)
            self._in_flight.pop(flight_key)

            # Get retry delay
            delay = self._retry_policy.get_delay(attempt)

            logger.debug(
                "[%s] Scheduling retry: id=%s attempt=%d delay=%.3fs",
                self.name,
                request_id,
                attempt + 1,
                delay,
            )

            # Create retry event
            retry_event = Event(
                time=self.now + Duration.from_seconds(delay),
                event_type=original_event.event_type,
                target=self,
                context={
                    "metadata": {
                        "request_id": request_id,
                        "payload": original_event.context.get("metadata", {}).get("payload"),
                        "attempt": attempt + 1,
                        "client_name": self.name,
                    },
                    "_on_success": request_info.get("on_success"),
                    "_on_failure": request_info.get("on_failure"),
                },
            )

            return [retry_event]

        # No more retries - fail the request
        self._in_flight.pop(flight_key)
        self._failures += 1

        logger.debug(
            "[%s] Request failed (max retries): id=%s attempts=%d",
            self.name,
            request_id,
            attempt,
        )

        # Invoke failure callback
        on_failure = request_info.get("on_failure")
        if on_failure is not None:
            on_failure(original_event, "timeout")

        return None

    def get_response_time_percentile(self, percentile: float) -> float:
        """Calculate a percentile of observed response times.

        Args:
            percentile: Percentile value between 0 and 1.

        Returns:
            Response time at the given percentile, or 0 if no data.
        """
        if not self._response_times:
            return 0.0

        sorted_times = sorted(self._response_times)
        n = len(sorted_times)

        if percentile <= 0:
            return sorted_times[0]
        if percentile >= 1:
            return sorted_times[-1]

        pos = percentile * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        frac = pos - lo

        return sorted_times[lo] * (1.0 - frac) + sorted_times[hi] * frac
