"""Pooled client that uses connection pooling for requests.

Provides a client that acquires connections from a pool before sending
requests, improving performance by reusing established connections.

Example:
    from happysimulator.components.client import PooledClient, ConnectionPool

    pool = ConnectionPool(
        name="db_pool",
        target=database_server,
        min_connections=2,
        max_connections=10,
    )

    client = PooledClient(
        name="db_client",
        connection_pool=pool,
        timeout=5.0,
        retry_policy=ExponentialBackoff(max_attempts=3, initial_delay=0.1, max_delay=10.0),
    )

    # Send a request - connection is acquired from pool automatically
    request = client.send_request(payload={"query": "SELECT * FROM users"})
    sim.schedule(request)
"""

import logging
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any

from happysimulator.components.client.connection_pool import ConnectionPool
from happysimulator.components.client.retry import NoRetry, RetryPolicy
from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PooledClientStats:
    """Frozen snapshot of PooledClient statistics."""

    requests_sent: int = 0
    responses_received: int = 0
    timeouts: int = 0
    retries: int = 0
    failures: int = 0
    connection_wait_timeouts: int = 0


class PooledClient(Entity):
    """Client that uses connection pooling for requests.

    Acquires a connection from the pool before each request, ensuring
    connections are reused efficiently. Supports the same timeout and
    retry features as the basic Client.

    Attributes:
        name: Client identifier for logging.
        connection_pool: Pool to acquire connections from.
        timeout: Request timeout in seconds (None = no timeout).
        retry_policy: Policy for retry behavior.
    """

    def __init__(
        self,
        name: str,
        connection_pool: ConnectionPool,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
        on_success: Callable[[Event, Event], None] | None = None,
        on_failure: Callable[[Event, str], None] | None = None,
    ):
        """Initialize the pooled client.

        Args:
            name: Client identifier.
            connection_pool: Pool to acquire connections from.
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

        self._pool = connection_pool
        self._timeout = timeout
        self._retry_policy = retry_policy or NoRetry()
        self._on_success = on_success
        self._on_failure = on_failure

        # Track in-flight requests: (request_id, attempt) -> request info
        self._in_flight: dict[tuple[int, int], dict] = {}
        self._next_request_id = 0

        # Statistics
        self._requests_sent = 0
        self._responses_received = 0
        self._timeouts = 0
        self._retries = 0
        self._failures = 0
        self._connection_wait_timeouts = 0

        # Response times for analysis
        self._response_times: list[float] = []

        logger.debug(
            "[%s] PooledClient initialized: pool=%s, timeout=%s, retry_policy=%s",
            name,
            connection_pool.name,
            timeout,
            type(self._retry_policy).__name__,
        )

    @property
    def connection_pool(self) -> ConnectionPool:
        """The connection pool used for requests."""
        return self._pool

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
    def stats(self) -> PooledClientStats:
        """Frozen snapshot of pooled client statistics."""
        return PooledClientStats(
            requests_sent=self._requests_sent,
            responses_received=self._responses_received,
            timeouts=self._timeouts,
            retries=self._retries,
            failures=self._failures,
            connection_wait_timeouts=self._connection_wait_timeouts,
        )

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to connection pool."""
        super().set_clock(clock)
        self._pool.set_clock(clock)

    def send_request(
        self,
        payload: Any = None,
        event_type: str = "request",
        on_success: Callable[[Event, Event], None] | None = None,
        on_failure: Callable[[Event, str], None] | None = None,
    ) -> Event:
        """Create a request event to send through the pool.

        Creates an event that, when scheduled, will acquire a connection
        from the pool and send a request to the target. The event is
        ready to be scheduled with sim.schedule().

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
        - Outgoing requests: acquire connection, forward to target
        - Timeout events: handle timeout and retry
        - Response events: complete the request, release connection

        Args:
            event: The event to handle.

        Returns:
            Events to schedule or generator for async processing.
        """
        event_type = event.event_type

        if event_type == "_pooled_client_timeout":
            return self._handle_timeout(event)

        if event_type == "_pooled_client_response":
            return self._handle_response(event)

        # New outgoing request
        return self._send_request(event)

    def _send_request(self, event: Event) -> Generator[float, None, list[Event] | None]:
        """Send a request using a pooled connection."""
        metadata = event.context.get("metadata", {})
        request_id = metadata.get("request_id")
        attempt = metadata.get("attempt", 1)

        logger.debug(
            "[%s] Sending request: id=%s attempt=%d",
            self.name,
            request_id,
            attempt,
        )

        # Acquire a connection from the pool
        try:
            connection = yield from self._pool.acquire()
        except TimeoutError:
            # Failed to acquire connection
            self._connection_wait_timeouts += 1
            self._failures += 1

            logger.warning(
                "[%s] Connection acquisition timeout: request_id=%s",
                self.name,
                request_id,
            )

            # Invoke failure callback
            on_failure = event.context.get("_on_failure")
            if on_failure is not None:
                on_failure(event, "connection_timeout")

            return None

        # Use composite key (request_id, attempt) to track each attempt
        flight_key = (request_id, attempt)

        # Track the in-flight request
        self._in_flight[flight_key] = {
            "start_time": self.now,
            "attempt": attempt,
            "event": event,
            "connection": connection,
            "on_success": event.context.get("_on_success"),
            "on_failure": event.context.get("_on_failure"),
            "request_id": request_id,
        }

        # Update stats
        self._requests_sent += 1
        if attempt > 1:
            self._retries += 1

        # Create the actual request to target
        target = self._pool.target
        target_event = Event(
            time=self.now,
            event_type=f"{self.name}_request",
            target=target,
            context={
                "metadata": {
                    "request_id": request_id,
                    "payload": metadata.get("payload"),
                    "client": self,
                    "attempt": attempt,
                    "connection_id": connection.id,
                },
            },
        )

        # Add completion hook to receive response
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_pooled_client_response",
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
                event_type="_pooled_client_timeout",
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
        connection = request_info["connection"]

        # Release connection back to pool
        release_events = self._pool.release(connection)

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

        return release_events if release_events else None

    def _handle_timeout(self, event: Event) -> Generator[float, None, list[Event] | None]:
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
        connection = request_info["connection"]

        # Release connection back to pool
        release_events = self._pool.release(connection)

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

            # Wait for retry delay
            yield delay

            # Create retry event
            retry_event = Event(
                time=self.now,
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

            all_events = [retry_event]
            if release_events:
                all_events.extend(release_events)
            return all_events

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

        return release_events if release_events else None

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
