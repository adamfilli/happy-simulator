"""Server entity that processes requests with configurable latency and failure.

Models a single-threaded server that processes one request at a time.
Incoming requests while busy are rejected (no internal queue). For queued
behavior, wrap this server with QueuedResource.
"""

import logging
import random
from typing import Generator

from happysimulator.data.data import Data
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from happysimulator.events.request import Request, ResponseStatus
from happysimulator.load.profile import Profile
from happysimulator.utils.instant import Instant

from happysimulator.math.latency_distribution import LatencyDistribution
from happysimulator.math.constant_latency import ConstantLatency

logger = logging.getLogger(__name__)


class _ZeroFailureProfile(Profile):
    """Default failure profile that always returns 0 (no failures)."""

    def get_rate(self, time: Instant) -> float:
        return 0.0


class SimpleServer(Entity):
    """Single-threaded server with configurable processing latency and failure rate.

    Processes one request at a time using generator-based simulation.
    When a request arrives while the server is busy, it is immediately
    rejected with ResponseStatus.REJECTED. The server does not maintain
    an internal queue.

    Processing time is sampled from the processing_latency distribution.
    After processing, the server may fail the request based on the
    failure_rate profile.

    Statistics collected:
    - requests_received: Total requests that arrived
    - requests_rejected: Requests rejected due to capacity
    - requests_completed: Successfully processed requests
    - requests_failed: Requests that failed after processing
    - processing_time: Server-side processing duration

    Args:
        name: Server identifier for logging.
        processing_latency: Distribution for processing duration. Defaults to 100ms.
        failure_rate: Time-varying failure probability. Defaults to 0 (no failures).
    """

    def __init__(
        self,
        name: str,
        processing_latency: LatencyDistribution | None = None,
        failure_rate: Profile | None = None,
    ):
        super().__init__(name)

        self._processing_latency = processing_latency or ConstantLatency(
            Instant.from_seconds(0.1)
        )
        self._failure_rate = failure_rate or _ZeroFailureProfile()

        # State
        self._busy = False

        # Statistics
        self._requests_received = Data()
        self._requests_rejected = Data()
        self._requests_completed = Data()
        self._requests_failed = Data()
        self._processing_time = Data()

    def has_capacity(self) -> bool:
        """Return True if server can accept a new request."""
        return not self._busy

    def handle_event(self, event: Event) -> Generator[float, None, list[Event]] | list[Event] | None:
        """Route events to appropriate handler."""
        if isinstance(event, Request):
            return self._process_request(event)
        return None

    def receive_request(self, event: Event) -> Generator[float, None, list[Event]] | list[Event]:
        """Handle incoming request (called via callback)."""
        if isinstance(event, Request):
            return self._process_request(event)
        return []

    def _process_request(self, request: Request) -> Generator[float, None, list[Event]] | list[Event]:
        """Process a request through its lifecycle.

        Yields to simulate processing time, then sends response.
        """
        request.server_receive_time = request.time
        self._requests_received.add_stat(1, request.time)

        logger.debug(
            "[%s][%s] Server received request (attempt %d)",
            request.time, self.name, request.attempt
        )

        # Check capacity - reject if busy
        if self._busy:
            logger.debug("[%s][%s] Server busy, rejecting request", request.time, self.name)
            self._requests_rejected.add_stat(1, request.time)
            request.response_status = ResponseStatus.REJECTED
            return self._send_response(request)

        # Mark as busy and process with generator
        self._busy = True
        return self._process_with_latency(request)

    def _process_with_latency(self, request: Request) -> Generator[float, None, list[Event]]:
        """Generator that yields processing delay then sends response."""
        # Calculate and yield processing time
        processing_time = self._processing_latency.get_latency(request.time).to_seconds()
        yield processing_time

        # Determine success/failure
        if random.random() < self._failure_rate.get_rate(self.now):
            request.response_status = ResponseStatus.FAIL
            self._requests_failed.add_stat(1, self.now)
        else:
            request.response_status = ResponseStatus.SUCCESS
            self._requests_completed.add_stat(1, self.now)

        # Record server-side processing time
        server_latency = (self.now - request.server_receive_time).to_seconds()
        self._processing_time.add_stat(server_latency, self.now)

        # Mark as available
        self._busy = False

        request.server_send_time = self.now

        logger.debug(
            "[%s][%s] Server processed request in %.3fs, status=%s",
            self.now, self.name, server_latency, request.response_status.value
        )

        return self._send_response(request)

    def _send_response(self, request: Request) -> list[Event]:
        """Route response back to client with network latency."""
        # Add network latency for response
        response_latency = request.network_latency.get_latency(self.now).to_seconds()
        request.time = self.now + response_latency
        request.callback = request.client.receive_response

        return [request]

    # --- Statistics accessors ---

    @property
    def stats_requests_received(self) -> Data:
        return self._requests_received

    @property
    def stats_requests_rejected(self) -> Data:
        return self._requests_rejected

    @property
    def stats_requests_completed(self) -> Data:
        return self._requests_completed

    @property
    def stats_requests_failed(self) -> Data:
        return self._requests_failed

    @property
    def stats_processing_time(self) -> Data:
        return self._processing_time

    @property
    def is_busy(self) -> bool:
        return self._busy
