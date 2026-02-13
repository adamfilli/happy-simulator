"""Configurable server with concurrency control and service time distribution.

Server extends QueuedResource to provide a complete server abstraction with:
- Configurable concurrency (max simultaneous requests or ConcurrencyModel)
- Pluggable service time distribution
- Queue buffering for excess requests
- Statistics tracking for analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Generator

from happysimulator.components.queued_resource import QueuedResource
from happysimulator.components.queue_policy import FIFOQueue, QueuePolicy
from happysimulator.components.server.concurrency import (
    ConcurrencyModel,
    FixedConcurrency,
)
from happysimulator.core.event import Event
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


@dataclass
class ServerStats:
    """Statistics tracked by Server."""

    requests_completed: int = 0
    requests_rejected: int = 0
    total_service_time: float = 0.0


class Server(QueuedResource):
    """Server with configurable concurrency and service time distribution.

    Processes requests with simulated service time drawn from a distribution.
    Supports multiple concurrent requests up to the configured limit.
    Excess requests are queued according to the configured policy.

    Concurrency can be specified as:
    - An integer (wrapped in FixedConcurrency)
    - A ConcurrencyModel instance for advanced control (Dynamic, Weighted, etc.)

    The server uses completion hooks internally to signal when a request
    slot becomes available, allowing the queue driver to dispatch the
    next waiting request.

    Attributes:
        name: Identifier for logging and debugging.
        concurrency_model: The concurrency control strategy.
        service_time: Distribution for request processing time.
    """

    def __init__(
        self,
        name: str,
        concurrency: int | ConcurrencyModel = 1,
        service_time: LatencyDistribution | None = None,
        queue_policy: QueuePolicy | None = None,
        queue_capacity: int | None = None,
        downstream: 'Entity | None' = None,
    ):
        """Initialize the server.

        Args:
            name: Server identifier.
            concurrency: Maximum concurrent requests as int, or a ConcurrencyModel
                instance for advanced control (default 1).
            service_time: Service time distribution (default 10ms constant).
            queue_policy: Queue ordering policy (default FIFO).
            queue_capacity: Maximum queue size (default unlimited).
            downstream: Optional entity to forward completed requests to.
                When set, completed events are automatically sent to this entity
                with the original context preserved.
        """
        # Create queue policy with capacity if specified
        if queue_policy is None:
            policy = FIFOQueue(
                capacity=queue_capacity if queue_capacity is not None else float("inf")
            )
        elif queue_capacity is not None:
            # Need to respect capacity even with custom policy
            # For now, assume the policy handles its own capacity
            policy = queue_policy
        else:
            policy = queue_policy

        super().__init__(name, policy=policy)

        # Wrap int in FixedConcurrency for uniform handling
        if isinstance(concurrency, int):
            self._concurrency_model = FixedConcurrency(concurrency)
        else:
            self._concurrency_model = concurrency

        self._service_time = service_time or ConstantLatency(0.01)
        self._downstream = downstream

        # Statistics
        self.stats = ServerStats()

        # Time series for analysis
        self._service_times: list[float] = []

        logger.debug(
            "[%s] Server initialized: concurrency=%s, service_time=%s",
            name,
            self._concurrency_model,
            self._service_time,
        )

    @property
    def downstream(self) -> 'Entity | None':
        """Optional downstream entity for event forwarding."""
        return self._downstream

    @downstream.setter
    def downstream(self, target: 'Entity | None') -> None:
        self._downstream = target

    @property
    def concurrency_model(self) -> ConcurrencyModel:
        """The concurrency control model."""
        return self._concurrency_model

    @property
    def concurrency(self) -> int:
        """Maximum concurrent requests (current limit)."""
        return self._concurrency_model.limit

    @property
    def service_time(self) -> LatencyDistribution:
        """Service time distribution."""
        return self._service_time

    @property
    def active_requests(self) -> int:
        """Number of requests currently being processed."""
        return self._concurrency_model.active

    @property
    def available_capacity(self) -> int:
        """Number of available processing slots."""
        return self._concurrency_model.available

    @property
    def utilization(self) -> float:
        """Current utilization as fraction of concurrency.

        Returns:
            Value between 0.0 and 1.0 representing active/concurrency.
        """
        limit = self._concurrency_model.limit
        if limit == 0:
            return 0.0
        return self._concurrency_model.active / limit

    @property
    def average_service_time(self) -> float:
        """Average observed service time in seconds."""
        if not self._service_times:
            return 0.0
        return sum(self._service_times) / len(self._service_times)

    def has_capacity(self, weight: int = 1) -> bool:
        """Check if server can accept another request.

        Args:
            weight: Capacity weight for weighted concurrency models (default 1).

        Returns:
            True if sufficient capacity is available.
        """
        return self._concurrency_model.has_capacity(weight)

    def handle_queued_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None]:
        """Process a request from the queue.

        Simulates request processing by yielding the service time.
        Updates statistics on completion.

        Args:
            event: The request event to process.

        Yields:
            Service time delay.

        Returns:
            Any completion events (typically None).
        """
        # Get weight from event context (for weighted concurrency)
        weight = event.context.get("metadata", {}).get("weight", 1)

        # Acquire processing capacity
        acquired = self._concurrency_model.acquire(weight)
        if not acquired:
            # This shouldn't happen if queue driver checks has_capacity,
            # but handle gracefully
            logger.warning(
                "[%s] Failed to acquire capacity for request: type=%s weight=%d",
                self.name,
                event.event_type,
                weight,
            )
            self.stats.requests_rejected += 1
            return None

        logger.debug(
            "[%s] Processing request: type=%s active=%d/%d weight=%d",
            self.name,
            event.event_type,
            self._concurrency_model.active,
            self._concurrency_model.limit,
            weight,
        )

        # Sample service time from distribution
        service_duration = self._service_time.get_latency(self.now)
        service_time_s = service_duration.to_seconds()

        # Record for statistics
        self._service_times.append(service_time_s)

        # Simulate processing
        yield service_time_s

        # Release processing capacity
        self._concurrency_model.release(weight)

        # Update statistics
        self.stats.requests_completed += 1
        self.stats.total_service_time += service_time_s

        logger.debug(
            "[%s] Request completed: type=%s service_time=%.4fs active=%d/%d",
            self.name,
            event.event_type,
            service_time_s,
            self._concurrency_model.active,
            self._concurrency_model.limit,
        )

        if self._downstream is not None:
            return [self.forward(event, self._downstream)]
        return None

    def get_service_time_percentile(self, percentile: float) -> float:
        """Calculate a percentile of observed service times.

        Args:
            percentile: Percentile value between 0 and 1 (e.g., 0.99 for p99).

        Returns:
            Service time at the given percentile, or 0 if no data.
        """
        if not self._service_times:
            return 0.0

        sorted_times = sorted(self._service_times)
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
