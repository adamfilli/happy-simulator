import logging
from typing import Callable, Optional

from happysimulator.data.data import Data
from happysimulator.entities.entity import Entity
from happysimulator.events.event import Event
from archive.client_server_request_event import Request
from archive.measurement_event import MeasurementEvent
from happysimulator.utils.instant import Instant

logger = logging.getLogger(__name__)


class RateLimiter(Entity):
    """A simple token-bucket based rate limiter.

    Behavior:
    - Maintains a token bucket with a given capacity and refill rate (tokens/sec).
    - On request arrival at time T, tokens are refilled based on elapsed time since last refill.
    - If at least 1 token is available the request is forwarded immediately and 1 token is consumed.
    - Otherwise the request is delayed until enough tokens are available; when that delay elapses
      the rate limiter forwards the request (consuming the token at that time).
    """

    def __init__(self, name: str, capacity: float = 1.0, refill_rate: float = 1.0, initial_tokens: Optional[float] = None):
        super().__init__(name)

        # config
        self._capacity = float(capacity)
        self._refill_rate = float(refill_rate)  # tokens per second

        # token bucket state
        self._tokens = self._capacity if initial_tokens is None else float(initial_tokens)
        self._last_refill_time = Instant.from_seconds(0)

        # stats
        self._queued_requests = Data()
        self._available_tokens = Data()

    def _refill(self, now: Instant):
        # Refill tokens based on time elapsed since last refill
        elapsed = (now - self._last_refill_time).to_seconds()
        if elapsed <= 0:
            return

        added = elapsed * self._refill_rate
        self._tokens = min(self._capacity, self._tokens + added)
        self._last_refill_time = now

    def start_request(self, request: Event) -> list[Event]:
        """Handle incoming request event. May forward immediately or schedule a delayed forward."""
        assert isinstance(request, Request)

        logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] RateLimiter received request")

        # Refill tokens up to current time
        self._refill(request.time)

        # Report token count for measurements
        self._available_tokens.add_stat(self._tokens, request.time)

        if self._tokens >= 1.0:
            # Consume a token and forward immediately
            self._tokens -= 1.0
            logger.debug(f"[{request.time.to_seconds()}][{self.name}] Forwarding immediately; tokens remaining={self._tokens}")

            # Determine downstream server (allow explicit downstream to be set on the request)
            downstream = getattr(request, "_downstream_server", request.server)
            # Set callback to downstream server and return the request for immediate scheduling
            request.callback = downstream.start_request
            return [request]

        # Need to wait until a token is available
        if self._refill_rate <= 0:
            # No refill: drop or block forever; for now we delay infinitely (practically schedule at very large time)
            wait_seconds = float('inf')
        else:
            wait_seconds = (1.0 - self._tokens) / self._refill_rate

        logger.debug(f"[{request.time.to_seconds()}][{self.name}] No tokens; delaying request by {wait_seconds} seconds")

        # Track queued request stat
        self._queued_requests.add_stat(1, request.time)

        # Reserve the downstream callback and schedule a forwarding event at the computed time
        downstream = getattr(request, "_downstream_server", request.server)
        request._rl_original_callback = downstream.start_request
        request.callback = self.forward_request
        request.time = request.time + Instant.from_seconds(wait_seconds)

        return [request]

    def forward_request(self, request: Event) -> list[Event]:
        """Called when a delayed request's wait time expires; consume a token and forward to downstream."""
        assert isinstance(request, Request)

        logger.info(f"[{request.time.to_seconds()}][{self.name}][{request.name}] RateLimiter forwarding delayed request")

        # Refill tokens up to current time and consume one
        self._refill(request.time)

        # In rare numerical edge-cases tokens may still be <1; force consume if possible
        if self._tokens < 1.0:
            # If refill_rate is zero and tokens <1, we cannot forward; drop request silently by returning []
            if self._refill_rate <= 0:
                logger.warning(f"[{request.time.to_seconds()}][{self.name}] Cannot forward request - no refill configured")
                return []
            # Otherwise allow token to go negative slightly and proceed
        self._tokens -= 1.0
        logger.debug(f"[{request.time.to_seconds()}][{self.name}] Token consumed on forward; tokens remaining={self._tokens}")

        # Restore original downstream callback and forward request
        request.callback = request._rl_original_callback
        # Update token stat for measurements
        self._available_tokens.add_stat(self._tokens, request.time)

        return [request]

    # Measurement handlers
    def queued_requests_count(self, event: MeasurementEvent) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for queued_requests_count")
        self.sink_data(self._queued_requests, event)
        return []

    def available_tokens(self, event: MeasurementEvent) -> list[Event]:
        logger.debug(f"[{event.time.to_seconds()}][{self.name}][{event.name}] Received measurement event for available_tokens")
        self.sink_data(self._available_tokens, event)
        return []
