"""Async server modeling event-loop style request handling.

AsyncServer simulates single-threaded async servers like Node.js or Python asyncio.
The key characteristics are:

1. High concurrency for I/O-bound work (many simultaneous connections)
2. CPU-bound work blocks the event loop (processes sequentially)
3. While waiting for I/O (external services), other requests can proceed

This is different from a thread-pool server where each worker handles one
request at a time. Here, one "thread" handles many connections, but CPU
work serializes while I/O work overlaps.

Example:
    # Node.js-style server handling many concurrent connections
    server = AsyncServer(
        name="api_server",
        max_connections=10000,
        cpu_work_distribution=ConstantLatency(0.001),  # 1ms CPU per request
    )

    # Requests do 1ms CPU work then wait for downstream I/O
    # CPU work is serialized, but I/O waits overlap
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Callable, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncServerStats:
    """Statistics tracked by AsyncServer."""

    requests_completed: int = 0
    requests_rejected: int = 0
    total_cpu_time: float = 0.0
    total_io_time: float = 0.0


class AsyncServer(Entity):
    """Non-blocking server that multiplexes many connections on single thread.

    Models event-loop style servers where:
    - Many connections can be active simultaneously (up to max_connections)
    - CPU-bound work blocks the event loop (serialized processing)
    - I/O-bound work (waiting for responses) is non-blocking

    The server processes requests in two phases:
    1. CPU phase: Serialized, one request at a time (blocks event loop)
    2. I/O phase: Concurrent, requests wait for external responses

    Attributes:
        name: Server identifier for logging.
        max_connections: Maximum simultaneous connections.
        cpu_work_distribution: Distribution for CPU-bound work time.
    """

    def __init__(
        self,
        name: str,
        max_connections: int = 10000,
        cpu_work_distribution: LatencyDistribution | None = None,
        io_handler: Callable[[Event], Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None] | None = None,
    ):
        """Initialize the async server.

        Args:
            name: Server identifier.
            max_connections: Maximum concurrent connections (default 10000).
            cpu_work_distribution: Distribution for CPU work time per request.
                Default is 0 (no CPU work, pure I/O).
            io_handler: Optional handler for I/O phase. If provided, called after
                CPU work completes. Can return events or yield delays for I/O waits.
        """
        super().__init__(name)

        if max_connections < 1:
            raise ValueError(f"max_connections must be >= 1, got {max_connections}")

        self._max_connections = max_connections
        self._cpu_work = cpu_work_distribution or ConstantLatency(0)
        self._io_handler = io_handler

        # Connection tracking
        self._active_connections = 0
        self._peak_connections = 0

        # CPU work queue (simple deque, only one CPU task at a time)
        self._cpu_queue: deque[Event] = deque()
        self._cpu_busy = False

        # Statistics
        self._requests_completed = 0
        self._requests_rejected = 0
        self._total_cpu_time = 0.0
        self._total_io_time = 0.0

        # Time series for analysis
        self._cpu_times: list[float] = []
        self._io_times: list[float] = []

        logger.debug(
            "[%s] AsyncServer initialized: max_connections=%d",
            name,
            max_connections,
        )

    @property
    def max_connections(self) -> int:
        """Maximum allowed concurrent connections."""
        return self._max_connections

    @property
    def active_connections(self) -> int:
        """Number of currently active connections."""
        return self._active_connections

    @property
    def peak_connections(self) -> int:
        """Peak number of concurrent connections observed."""
        return self._peak_connections

    @property
    def cpu_queue_depth(self) -> int:
        """Number of requests waiting for CPU time."""
        return len(self._cpu_queue)

    @property
    def is_cpu_busy(self) -> bool:
        """Whether CPU is currently processing a request."""
        return self._cpu_busy

    @property
    def stats(self) -> AsyncServerStats:
        """Frozen snapshot of current statistics."""
        return AsyncServerStats(
            requests_completed=self._requests_completed,
            requests_rejected=self._requests_rejected,
            total_cpu_time=self._total_cpu_time,
            total_io_time=self._total_io_time,
        )

    @property
    def utilization(self) -> float:
        """Connection utilization as fraction of max_connections."""
        if self._max_connections == 0:
            return 0.0
        return self._active_connections / self._max_connections

    @property
    def average_cpu_time(self) -> float:
        """Average observed CPU time per request."""
        if not self._cpu_times:
            return 0.0
        return sum(self._cpu_times) / len(self._cpu_times)

    def has_capacity(self) -> bool:
        """Check if server can accept another connection.

        Returns:
            True if under max_connections limit.
        """
        return self._active_connections < self._max_connections

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle an incoming request.

        Requests go through:
        1. Connection acceptance (checked against max_connections)
        2. CPU work phase (serialized via queue)
        3. Optional I/O phase (concurrent)

        Args:
            event: The incoming request event.

        Returns:
            Events or generator for async processing.
        """
        # Check for internal CPU processing events
        if event.event_type == "_cpu_complete":
            return self._on_cpu_complete(event)

        if event.event_type == "_process_cpu_queue":
            return self._process_next_cpu_task()

        # New connection request
        if not self.has_capacity():
            logger.debug(
                "[%s] Connection rejected (at capacity): type=%s active=%d/%d",
                self.name,
                event.event_type,
                self._active_connections,
                self._max_connections,
            )
            self._requests_rejected += 1
            return None

        # Accept connection
        self._active_connections += 1
        self._peak_connections = max(self._peak_connections, self._active_connections)

        logger.debug(
            "[%s] Connection accepted: type=%s active=%d/%d",
            self.name,
            event.event_type,
            self._active_connections,
            self._max_connections,
        )

        # Queue for CPU processing
        return self._queue_for_cpu(event)

    def _queue_for_cpu(self, event: Event) -> list[Event] | None:
        """Queue a request for CPU processing.

        If CPU is idle, process immediately. Otherwise, queue the request.
        """
        if not self._cpu_busy and len(self._cpu_queue) == 0:
            # CPU is free, process immediately
            return self._start_cpu_processing(event)
        else:
            # Queue for later processing
            self._cpu_queue.append(event)
            logger.debug(
                "[%s] Request queued for CPU: type=%s queue_depth=%d",
                self.name,
                event.event_type,
                len(self._cpu_queue),
            )
            return None

    def _start_cpu_processing(self, event: Event) -> list[Event]:
        """Start CPU processing for a request.

        Returns a CPU completion event scheduled after the CPU work duration.
        """
        self._cpu_busy = True

        # Sample CPU work time
        cpu_duration = self._cpu_work.get_latency(self.now)
        cpu_time_s = cpu_duration.to_seconds()

        logger.debug(
            "[%s] Starting CPU work: type=%s duration=%.4fs",
            self.name,
            event.event_type,
            cpu_time_s,
        )

        # Schedule CPU completion
        completion = Event(
            time=self.now + cpu_duration,
            event_type="_cpu_complete",
            target=self,
            context={
                "_original_event": event,
                "_cpu_time": cpu_time_s,
            },
        )

        return [completion]

    def _on_cpu_complete(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle CPU work completion.

        Records statistics, triggers I/O phase if configured, and processes
        the next queued request.
        """
        original_event = event.context.get("_original_event")
        cpu_time = event.context.get("_cpu_time", 0.0)

        # Record CPU time
        self._cpu_times.append(cpu_time)
        self._total_cpu_time += cpu_time

        logger.debug(
            "[%s] CPU work complete: type=%s cpu_time=%.4fs",
            self.name,
            original_event.event_type if original_event else "unknown",
            cpu_time,
        )

        # Mark CPU as free
        self._cpu_busy = False

        # Process next queued request (if any)
        result_events = []
        if len(self._cpu_queue) > 0:
            next_event = Event(
                time=self.now,
                event_type="_process_cpu_queue",
                target=self,
            )
            result_events.append(next_event)

        # Handle I/O phase
        if self._io_handler is not None and original_event is not None:
            io_result = self._io_handler(original_event)

            if io_result is not None:
                # Generator-based I/O handling
                if hasattr(io_result, "__next__"):
                    # Return generator for I/O processing
                    def io_wrapper():
                        io_start = self.now.to_seconds()
                        result = yield from io_result
                        io_time = self.now.to_seconds() - io_start
                        self._io_times.append(io_time)
                        self._total_io_time += io_time

                        # Complete the request
                        self._complete_request(original_event)

                        # Return any events from I/O handler plus queue processing
                        if result is None:
                            return result_events if result_events else None
                        elif isinstance(result, list):
                            return result + result_events
                        else:
                            return [result] + result_events

                    return io_wrapper()

                # Immediate events from I/O handler
                self._complete_request(original_event)
                if isinstance(io_result, list):
                    return io_result + result_events
                else:
                    return [io_result] + result_events

        # No I/O handler - complete request immediately
        if original_event is not None:
            self._complete_request(original_event)

        return result_events if result_events else None

    def _process_next_cpu_task(self) -> list[Event] | None:
        """Process the next request from the CPU queue."""
        if self._cpu_busy or len(self._cpu_queue) == 0:
            return None

        # Pop from queue
        next_request = self._cpu_queue.popleft()

        return self._start_cpu_processing(next_request)

    def _complete_request(self, event: Event) -> None:
        """Mark a request as complete and release the connection."""
        self._active_connections = max(0, self._active_connections - 1)
        self._requests_completed += 1

        logger.debug(
            "[%s] Request completed: type=%s active=%d/%d",
            self.name,
            event.event_type,
            self._active_connections,
            self._max_connections,
        )

    def get_cpu_time_percentile(self, percentile: float) -> float:
        """Calculate a percentile of observed CPU times.

        Args:
            percentile: Percentile value between 0 and 1.

        Returns:
            CPU time at the given percentile, or 0 if no data.
        """
        if not self._cpu_times:
            return 0.0

        sorted_times = sorted(self._cpu_times)
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
