"""Connection pool for managing reusable connections.

Provides a pool of connections to a target entity, with support for
minimum and maximum pool sizes, idle timeout, and connection acquisition
queuing when the pool is exhausted.

Example:
    from happysimulator.components.client import ConnectionPool
    from happysimulator.distributions import ConstantLatency

    pool = ConnectionPool(
        name="db_pool",
        target=database_server,
        min_connections=2,
        max_connections=10,
        connection_timeout=5.0,
        idle_timeout=60.0,
        connection_latency=ConstantLatency(0.01),
    )

    # Acquire a connection (may yield while waiting)
    connection = yield from pool.acquire()

    # Use the connection...
    # Release when done
    pool.release(connection)
"""

import logging
from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant
from happysimulator.distributions.constant import ConstantLatency
from happysimulator.distributions.latency_distribution import LatencyDistribution

logger = logging.getLogger(__name__)


@dataclass
class Connection:
    """Represents a connection from the pool.

    Attributes:
        id: Unique connection identifier.
        created_at: When the connection was created.
        last_used_at: When the connection was last used.
        is_active: Whether the connection is currently in use.
    """

    id: int
    created_at: Instant
    last_used_at: Instant
    is_active: bool = False


@dataclass(frozen=True)
class ConnectionPoolStats:
    """Frozen snapshot of ConnectionPool statistics."""

    connections_created: int = 0
    connections_closed: int = 0
    acquisitions: int = 0
    releases: int = 0
    timeouts: int = 0
    total_wait_time: float = 0.0


class ConnectionPool(Entity):
    """Manages a pool of reusable connections to a target.

    The connection pool maintains warm connections that can be reused
    across requests, reducing the overhead of establishing new connections.
    When the pool is exhausted, new connections are created up to the
    maximum limit. Requests block when at capacity until a connection
    is released.

    Attributes:
        name: Pool identifier for logging.
        target: The entity to connect to.
        min_connections: Minimum number of warm connections to maintain.
        max_connections: Maximum number of total connections.
        connection_timeout: Time to wait for a connection before failing.
        idle_timeout: Time before idle connections are closed.
        connection_latency: Time to establish a new connection.
    """

    def __init__(
        self,
        name: str,
        target: Entity,
        min_connections: int = 0,
        max_connections: int = 10,
        connection_timeout: float = 5.0,
        idle_timeout: float = 60.0,
        connection_latency: LatencyDistribution | None = None,
        on_acquire: Callable[[Connection], None] | None = None,
        on_release: Callable[[Connection], None] | None = None,
        on_timeout: Callable[[], None] | None = None,
    ):
        """Initialize the connection pool.

        Args:
            name: Pool identifier.
            target: Entity to connect to.
            min_connections: Minimum pool size (default 0).
            max_connections: Maximum pool size (default 10).
            connection_timeout: Wait timeout in seconds (default 5.0).
            idle_timeout: Idle connection timeout in seconds (default 60.0).
            connection_latency: Time to create a new connection.
            on_acquire: Callback when connection is acquired.
            on_release: Callback when connection is released.
            on_timeout: Callback when acquisition times out.

        Raises:
            ValueError: If parameters are invalid.
        """
        super().__init__(name)

        if min_connections < 0:
            raise ValueError(f"min_connections must be >= 0, got {min_connections}")
        if max_connections < 1:
            raise ValueError(f"max_connections must be >= 1, got {max_connections}")
        if max_connections < min_connections:
            raise ValueError(
                f"max_connections ({max_connections}) must be >= "
                f"min_connections ({min_connections})"
            )
        if connection_timeout <= 0:
            raise ValueError(f"connection_timeout must be > 0, got {connection_timeout}")
        if idle_timeout <= 0:
            raise ValueError(f"idle_timeout must be > 0, got {idle_timeout}")

        self._target = target
        self._min_connections = min_connections
        self._max_connections = max_connections
        self._connection_timeout = connection_timeout
        self._idle_timeout = idle_timeout
        self._connection_latency = connection_latency or ConstantLatency(0.01)

        self._on_acquire = on_acquire
        self._on_release = on_release
        self._on_timeout = on_timeout

        # Pool state
        self._idle_connections: deque[Connection] = deque()
        self._active_connections: dict[int, Connection] = {}
        self._next_connection_id = 0
        self._total_connections = 0

        # Waiters: list of (waiter_id, request_time, callback)
        self._waiters: deque[tuple[int, Instant, Callable[[Connection | None], None]]] = deque()
        self._next_waiter_id = 0

        # Statistics
        self._connections_created = 0
        self._connections_closed = 0
        self._acquisitions = 0
        self._releases = 0
        self._timeouts = 0
        self._total_wait_time = 0.0

        logger.debug(
            "[%s] ConnectionPool initialized: target=%s, min=%d, max=%d, timeout=%.1fs, idle=%.1fs",
            name,
            target.name if hasattr(target, "name") else str(target),
            min_connections,
            max_connections,
            connection_timeout,
            idle_timeout,
        )

    @property
    def target(self) -> Entity:
        """The target entity for connections."""
        return self._target

    @property
    def min_connections(self) -> int:
        """Minimum number of connections to maintain."""
        return self._min_connections

    @property
    def max_connections(self) -> int:
        """Maximum number of connections allowed."""
        return self._max_connections

    @property
    def connection_timeout(self) -> float:
        """Timeout in seconds for acquiring a connection."""
        return self._connection_timeout

    @property
    def idle_timeout(self) -> float:
        """Timeout in seconds before idle connections are closed."""
        return self._idle_timeout

    @property
    def active_connections(self) -> int:
        """Number of connections currently in use."""
        return len(self._active_connections)

    @property
    def idle_connections(self) -> int:
        """Number of idle connections in the pool."""
        return len(self._idle_connections)

    @property
    def total_connections(self) -> int:
        """Total number of open connections."""
        return self._total_connections

    @property
    def pending_requests(self) -> int:
        """Number of requests waiting for a connection."""
        return len(self._waiters)

    @property
    def stats(self) -> ConnectionPoolStats:
        """Frozen snapshot of connection pool statistics."""
        return ConnectionPoolStats(
            connections_created=self._connections_created,
            connections_closed=self._connections_closed,
            acquisitions=self._acquisitions,
            releases=self._releases,
            timeouts=self._timeouts,
            total_wait_time=self._total_wait_time,
        )

    @property
    def average_wait_time(self) -> float:
        """Average time spent waiting for a connection."""
        if self._acquisitions == 0:
            return 0.0
        return self._total_wait_time / self._acquisitions

    def acquire(self) -> Generator[float, None, Connection]:
        """Acquire a connection from the pool.

        This generator yields while waiting for a connection to become
        available or while creating a new connection. It returns a
        Connection object that must be released with release() when done.

        Yields:
            Time to wait (for connection creation or waiting in queue).

        Returns:
            A Connection object.

        Raises:
            TimeoutError: If no connection becomes available within timeout.
        """
        start_time = self.now
        self._acquisitions += 1

        logger.debug(
            "[%s] Acquire requested: idle=%d, active=%d, total=%d, max=%d",
            self.name,
            len(self._idle_connections),
            len(self._active_connections),
            self._total_connections,
            self._max_connections,
        )

        # Try to get an idle connection first
        connection = self._try_get_idle_connection()
        if connection is not None:
            self._activate_connection(connection)
            logger.debug(
                "[%s] Acquired idle connection: id=%d",
                self.name,
                connection.id,
            )
            return connection

        # Can we create a new connection?
        if self._total_connections < self._max_connections:
            connection = yield from self._create_connection()
            self._activate_connection(connection)
            logger.debug(
                "[%s] Created and acquired new connection: id=%d",
                self.name,
                connection.id,
            )
            return connection

        # Must wait for a connection to become available
        logger.debug(
            "[%s] Pool exhausted, waiting for connection",
            self.name,
        )

        # Create a result holder for the callback pattern
        result: list[Connection | None] = [None]
        received = [False]

        def on_connection_available(conn: Connection | None):
            result[0] = conn
            received[0] = True

        self._next_waiter_id += 1
        waiter_id = self._next_waiter_id
        self._waiters.append((waiter_id, start_time, on_connection_available))

        # Wait up to timeout, polling periodically
        # In a real discrete-event simulation, we'd use events for this
        # But for the generator pattern, we'll yield small delays
        poll_interval = min(0.1, self._connection_timeout / 10)
        elapsed = 0.0

        while elapsed < self._connection_timeout:
            yield poll_interval
            elapsed += poll_interval

            if received[0]:
                connection = result[0]
                if connection is not None:
                    wait_time = (self.now - start_time).to_seconds()
                    self._total_wait_time += wait_time
                    logger.debug(
                        "[%s] Acquired connection after wait: id=%d, wait=%.3fs",
                        self.name,
                        connection.id,
                        wait_time,
                    )
                    return connection
                else:
                    # Connection was None (shouldn't happen normally)
                    break

        # Timeout - remove ourselves from waiters
        self._remove_waiter(waiter_id)
        self._timeouts += 1

        if self._on_timeout is not None:
            self._on_timeout()

        logger.warning(
            "[%s] Connection acquisition timeout after %.1fs",
            self.name,
            self._connection_timeout,
        )

        raise TimeoutError(
            f"Failed to acquire connection from {self.name} within {self._connection_timeout}s"
        )

    def release(self, connection: Connection) -> list[Event]:
        """Release a connection back to the pool.

        The connection becomes available for reuse. If there are waiters,
        the connection is immediately given to the next waiter.

        Args:
            connection: The connection to release.

        Returns:
            Events to schedule (for idle timeout checks).
        """
        if connection.id not in self._active_connections:
            logger.warning(
                "[%s] Attempted to release unknown connection: id=%d",
                self.name,
                connection.id,
            )
            return []

        # Remove from active
        del self._active_connections[connection.id]
        connection.is_active = False
        connection.last_used_at = self.now

        self._releases += 1

        logger.debug(
            "[%s] Released connection: id=%d",
            self.name,
            connection.id,
        )

        if self._on_release is not None:
            self._on_release(connection)

        # Check if there are waiters
        if self._waiters:
            waiter_id, _request_time, callback = self._waiters.popleft()

            # Reactivate connection for waiter
            self._activate_connection(connection)
            callback(connection)

            logger.debug(
                "[%s] Gave connection to waiter: conn_id=%d, waiter_id=%d",
                self.name,
                connection.id,
                waiter_id,
            )
            return []

        # No waiters - return to idle pool
        self._idle_connections.append(connection)

        # Schedule idle timeout check
        timeout_event = Event(
            time=self.now + Duration.from_seconds(self._idle_timeout),
            event_type="_pool_idle_timeout",
            target=self,
            context={
                "metadata": {
                    "connection_id": connection.id,
                    "expected_last_used": connection.last_used_at,
                },
            },
        )

        return [timeout_event]

    def handle_event(
        self, event: Event
    ) -> Generator[float, None, list[Event] | Event | None] | list[Event] | Event | None:
        """Handle pool events.

        Processes:
        - Idle timeout events: close connections that have been idle too long

        Args:
            event: The event to handle.

        Returns:
            Events to schedule or None.
        """
        event_type = event.event_type

        if event_type == "_pool_idle_timeout":
            return self._handle_idle_timeout(event)

        if event_type == "_pool_warmup":
            return self._handle_warmup(event)

        # Unknown event type
        logger.debug(
            "[%s] Ignoring unknown event type: %s",
            self.name,
            event_type,
        )
        return None

    def warmup(self) -> Event:
        """Create minimum connections during simulation startup.

        Returns an event that, when scheduled, will create the minimum
        number of connections specified in min_connections.

        Returns:
            Event to schedule for warmup.
        """
        return Event(
            time=self.now if self._clock is not None else Instant.Epoch,
            event_type="_pool_warmup",
            target=self,
            context={},
        )

    def _handle_warmup(self, event: Event) -> Generator[float, None, list[Event] | None]:
        """Create minimum connections."""
        events = []

        while self._total_connections < self._min_connections:
            connection = yield from self._create_connection()
            self._idle_connections.append(connection)

            # Schedule idle timeout check
            timeout_event = Event(
                time=self.now + Duration.from_seconds(self._idle_timeout),
                event_type="_pool_idle_timeout",
                target=self,
                context={
                    "metadata": {
                        "connection_id": connection.id,
                        "expected_last_used": connection.last_used_at,
                    },
                },
            )
            events.append(timeout_event)

        logger.debug(
            "[%s] Warmup complete: created %d connections",
            self.name,
            self._min_connections,
        )

        return events if events else None

    def _handle_idle_timeout(self, event: Event) -> list[Event] | None:
        """Handle idle timeout for a connection."""
        metadata = event.context.get("metadata", {})
        connection_id = metadata.get("connection_id")
        expected_last_used = metadata.get("expected_last_used")

        # Find the connection in idle pool
        for i, conn in enumerate(self._idle_connections):
            if conn.id == connection_id:
                # Check if it's still the same "idle session"
                if conn.last_used_at == expected_last_used:
                    # Check if we should maintain minimum connections
                    if self._total_connections > self._min_connections:
                        # Remove and close the connection
                        del self._idle_connections[i]
                        self._close_connection(conn)
                        logger.debug(
                            "[%s] Closed idle connection: id=%d",
                            self.name,
                            connection_id,
                        )
                    else:
                        # Reschedule the timeout check
                        return [
                            Event(
                                time=self.now + Duration.from_seconds(self._idle_timeout),
                                event_type="_pool_idle_timeout",
                                target=self,
                                context={
                                    "metadata": {
                                        "connection_id": connection_id,
                                        "expected_last_used": conn.last_used_at,
                                    },
                                },
                            )
                        ]
                break

        return None

    def _try_get_idle_connection(self) -> Connection | None:
        """Try to get an idle connection from the pool."""
        while self._idle_connections:
            connection = self._idle_connections.popleft()
            # Connection is valid, return it
            return connection
        return None

    def _create_connection(self) -> Generator[float, None, Connection]:
        """Create a new connection to the target."""
        # Simulate connection establishment time
        latency = self._connection_latency.get_latency(self.now)
        yield latency.to_seconds()

        self._next_connection_id += 1
        connection = Connection(
            id=self._next_connection_id,
            created_at=self.now,
            last_used_at=self.now,
            is_active=False,
        )
        self._total_connections += 1
        self._connections_created += 1

        logger.debug(
            "[%s] Created connection: id=%d, total=%d",
            self.name,
            connection.id,
            self._total_connections,
        )

        return connection

    def _activate_connection(self, connection: Connection) -> None:
        """Mark a connection as active."""
        connection.is_active = True
        connection.last_used_at = self.now
        self._active_connections[connection.id] = connection

        if self._on_acquire is not None:
            self._on_acquire(connection)

    def _close_connection(self, connection: Connection) -> None:
        """Close a connection."""
        self._total_connections -= 1
        self._connections_closed += 1

        logger.debug(
            "[%s] Closed connection: id=%d, total=%d",
            self.name,
            connection.id,
            self._total_connections,
        )

    def _remove_waiter(self, waiter_id: int) -> None:
        """Remove a waiter from the queue."""
        self._waiters = deque(
            (wid, time, cb) for wid, time, cb in self._waiters if wid != waiter_id
        )

    def close_all(self) -> None:
        """Close all connections in the pool.

        This should be called during simulation teardown.
        """
        # Close idle connections
        for conn in self._idle_connections:
            self._close_connection(conn)
        self._idle_connections.clear()

        # Close active connections
        for conn in self._active_connections.values():
            self._close_connection(conn)
        self._active_connections.clear()

        # Clear waiters
        for _waiter_id, _request_time, callback in self._waiters:
            callback(None)
        self._waiters.clear()

        logger.info(
            "[%s] Pool closed: created=%d, closed=%d",
            self.name,
            self._connections_created,
            self._connections_closed,
        )
