"""Client entity for sending requests and receiving responses.

Models a client that sends Request events to a server, receives responses,
and optionally retries on failure or timeout. Collects statistics for
post-simulation analysis.
"""

import logging
from typing import Optional

from happysimulator.data.data import Data
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from happysimulator.events.request import Request, ResponseStatus
from happysimulator.utils.instant import Instant

logger = logging.getLogger(__name__)


class SimpleClient(Entity):
    """Client entity with configurable timeout and retry behavior.

    Tracks timing on Request events to compute round-trip latency.
    Requests that exceed the timeout threshold are marked as timed out.
    Failed requests (timeout, rejection, or server failure) can be retried
    up to a configurable number of times.

    Statistics collected:
    - requests_sent: Total requests initiated (including retries)
    - responses_received: Total responses received
    - successful: Requests that completed successfully
    - failed: Requests that failed for any reason
    - timed_out: Requests that exceeded timeout
    - rejected: Requests rejected by busy server
    - retries: Number of retry attempts made
    - latency: Round-trip time measurements

    Args:
        name: Client identifier for logging.
        timeout: Maximum round-trip time before declaring timeout. None disables.
        retries: Number of additional attempts after initial failure.
        retry_delay: Wait time before retrying. Defaults to immediate (Epoch).
    """

    def __init__(
        self,
        name: str,
        timeout: Optional[Instant] = None,
        retries: int = 0,
        retry_delay: Optional[Instant] = None,
    ):
        super().__init__(name)

        self._timeout = timeout
        self._max_retries = retries
        self._retry_delay = retry_delay or Instant.Epoch

        # Statistics
        self._requests_sent = Data()
        self._responses_received = Data()
        self._successful_requests = Data()
        self._failed_requests = Data()
        self._timed_out_requests = Data()
        self._rejected_requests = Data()
        self._retries_attempted = Data()

        # Latency tracking
        self._total_latency = Data()
        self._successful_latency = Data()
        self._failed_latency = Data()

    def handle_event(self, event: Event) -> list[Event] | None:
        """Entry point - delegates based on callback."""
        if isinstance(event, Request):
            if event.callback == self.send_request:
                return self.send_request(event)
            elif event.callback == self.receive_response:
                return self.receive_response(event)
        return None

    def send_request(self, event: Event) -> list[Event]:
        """Send a request to the target server.

        Called when:
        1. Initially triggered by a Source
        2. On retry after failure/timeout
        """
        if not isinstance(event, Request):
            return []

        request = event
        request.client_send_time = request.time
        self._requests_sent.add_stat(1, request.time)

        logger.debug(
            "[%s][%s] Client sending request to %s (attempt %d)",
            request.time, self.name, request.server.name, request.attempt
        )

        # Add network latency for the request
        send_latency = request.network_latency.get_latency(request.time).to_seconds()
        request.time = request.time + send_latency
        request.callback = request.server.receive_request

        return [request]

    def receive_response(self, event: Event) -> list[Event]:
        """Handle response from server.

        Processes timeout detection, failure handling, and retry logic.
        """
        if not isinstance(event, Request):
            return []

        request = event
        request.client_receive_time = request.time
        self._responses_received.add_stat(1, request.time)

        # Calculate round-trip latency
        latency = (request.time - request.client_send_time).to_seconds()
        self._total_latency.add_stat(latency, request.time)

        logger.debug(
            "[%s][%s] Client received response: status=%s, latency=%.3fs",
            request.time, self.name,
            request.response_status.value if request.response_status else "None",
            latency
        )

        # Check for timeout
        timed_out = False
        if self._timeout is not None:
            timed_out = latency > self._timeout.to_seconds()
            if timed_out:
                self._timed_out_requests.add_stat(1, request.time)
                logger.debug(
                    "[%s][%s] Request timed out (%.3fs > %.3fs)",
                    request.time, self.name, latency, self._timeout.to_seconds()
                )

        # Check for rejection
        rejected = request.response_status == ResponseStatus.REJECTED
        if rejected:
            self._rejected_requests.add_stat(1, request.time)

        # Check for server failure
        server_failed = request.response_status == ResponseStatus.FAIL

        # Determine overall success
        request_failed = timed_out or rejected or server_failed

        if request_failed:
            self._failed_requests.add_stat(1, request.time)
            self._failed_latency.add_stat(latency, request.time)
            return self._maybe_retry(request)
        else:
            self._successful_requests.add_stat(1, request.time)
            self._successful_latency.add_stat(latency, request.time)
            return []

    def _maybe_retry(self, request: Request) -> list[Event]:
        """Retry the request if retries remain."""
        if request.attempt <= self._max_retries:
            request.attempt += 1
            self._retries_attempted.add_stat(1, request.time)

            logger.debug(
                "[%s][%s] Retrying request (attempt %d of %d)",
                request.time, self.name, request.attempt, self._max_retries + 1
            )

            # Schedule retry after delay
            retry_time = request.time + self._retry_delay.to_seconds()
            request.time = retry_time
            request.callback = self.send_request

            # Reset timing for new attempt
            request.client_send_time = None
            request.server_receive_time = None
            request.server_send_time = None
            request.client_receive_time = None
            request.response_status = None

            return [request]

        logger.debug(
            "[%s][%s] Request failed after %d attempts",
            request.time, self.name, request.attempt
        )
        return []

    # --- Statistics accessors ---

    @property
    def stats_requests_sent(self) -> Data:
        return self._requests_sent

    @property
    def stats_responses_received(self) -> Data:
        return self._responses_received

    @property
    def stats_successful(self) -> Data:
        return self._successful_requests

    @property
    def stats_failed(self) -> Data:
        return self._failed_requests

    @property
    def stats_timed_out(self) -> Data:
        return self._timed_out_requests

    @property
    def stats_rejected(self) -> Data:
        return self._rejected_requests

    @property
    def stats_retries(self) -> Data:
        return self._retries_attempted

    @property
    def stats_latency(self) -> Data:
        return self._total_latency

    @property
    def stats_successful_latency(self) -> Data:
        return self._successful_latency

    @property
    def stats_failed_latency(self) -> Data:
        return self._failed_latency
