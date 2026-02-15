"""Database implementation.

Provides a relational database simulation with connection pool,
query execution, and transaction support.

Example:
    from happysimulator.components.datastore import Database

    db = Database(
        name="postgres",
        max_connections=100,
        query_latency=0.005,
    )

    def handle_event(self, event):
        # Simple query
        result = yield from db.execute("SELECT * FROM users WHERE id = 1")

        # Transaction
        tx = yield from db.begin_transaction()
        yield from tx.execute("UPDATE accounts SET balance = 100 WHERE id = 1")
        yield from tx.execute("UPDATE accounts SET balance = 200 WHERE id = 2")
        yield from tx.commit()
"""

from collections import deque
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant


class TransactionState(Enum):
    """State of a database transaction."""

    ACTIVE = "active"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class DatabaseStats:
    """Statistics tracked by Database."""

    queries_executed: int = 0
    transactions_started: int = 0
    transactions_committed: int = 0
    transactions_rolled_back: int = 0
    connections_created: int = 0
    connections_closed: int = 0
    connection_wait_count: int = 0
    connection_wait_time_total: float = 0.0
    query_latencies: list[float] = field(default_factory=list)

    @property
    def avg_query_latency(self) -> float:
        """Average query latency."""
        if not self.query_latencies:
            return 0.0
        return sum(self.query_latencies) / len(self.query_latencies)

    @property
    def query_latency_p95(self) -> float:
        """95th percentile query latency."""
        if not self.query_latencies:
            return 0.0
        sorted_latencies = sorted(self.query_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


@dataclass
class Connection:
    """Represents a database connection."""

    id: int
    created_at: Instant
    in_transaction: bool = False
    transaction_id: int | None = None


class Transaction:
    """Represents a database transaction."""

    def __init__(
        self,
        transaction_id: int,
        database: "Database",
        connection: Connection,
    ):
        """Initialize the transaction.

        Args:
            transaction_id: Unique transaction identifier.
            database: The database this transaction belongs to.
            connection: The connection used by this transaction.
        """
        self._id = transaction_id
        self._database = database
        self._connection = connection
        self._state = TransactionState.ACTIVE
        self._statements: list[str] = []

    @property
    def id(self) -> int:
        """Transaction identifier."""
        return self._id

    @property
    def state(self) -> TransactionState:
        """Current transaction state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Whether the transaction is still active."""
        return self._state == TransactionState.ACTIVE

    def execute(self, query: str) -> Generator[float, None, Any]:
        """Execute a query within this transaction.

        Args:
            query: The SQL query to execute.

        Yields:
            Query latency.

        Returns:
            Query result.

        Raises:
            RuntimeError: If transaction is not active.
        """
        if not self.is_active:
            raise RuntimeError(f"Transaction {self._id} is not active")

        self._statements.append(query)
        result = yield from self._database._execute_query(query)
        return result

    def commit(self) -> Generator[float]:
        """Commit the transaction.

        Yields:
            Commit latency.

        Raises:
            RuntimeError: If transaction is not active.
        """
        if not self.is_active:
            raise RuntimeError(f"Transaction {self._id} is not active")

        # Simulate commit latency (typically slower than regular query)
        yield self._database._commit_latency

        self._state = TransactionState.COMMITTED
        self._database._end_transaction(self)

    def rollback(self) -> Generator[float]:
        """Roll back the transaction.

        Yields:
            Rollback latency.

        Raises:
            RuntimeError: If transaction is not active.
        """
        if not self.is_active:
            raise RuntimeError(f"Transaction {self._id} is not active")

        yield self._database._rollback_latency

        self._state = TransactionState.ROLLED_BACK
        self._database._end_transaction(self)


class Database(Entity):
    """Relational database with connection pool and transactions.

    Simulates a database with connection pooling, query execution,
    and transaction support. Query latency can be fixed or depend
    on the query type.

    Attributes:
        name: Entity name for identification.
        max_connections: Maximum concurrent connections.
        active_connections: Currently active connections.
        available_connections: Connections available for use.
    """

    def __init__(
        self,
        name: str,
        max_connections: int = 100,
        query_latency: float | Callable[[str], float] = 0.005,
        connection_latency: float = 0.010,
        commit_latency: float = 0.010,
        rollback_latency: float = 0.005,
    ):
        """Initialize the database.

        Args:
            name: Name for this database entity.
            max_connections: Maximum number of concurrent connections.
            query_latency: Latency per query (fixed or function of query).
            connection_latency: Latency to acquire a connection.
            commit_latency: Latency for transaction commit.
            rollback_latency: Latency for transaction rollback.

        Raises:
            ValueError: If parameters are invalid.
        """
        if max_connections < 1:
            raise ValueError(f"max_connections must be >= 1, got {max_connections}")

        super().__init__(name)
        self._max_connections = max_connections
        self._query_latency = query_latency
        self._connection_latency = connection_latency
        self._commit_latency = commit_latency
        self._rollback_latency = rollback_latency

        # Connection pool state
        self._connections: dict[int, Connection] = {}
        self._available_connections: deque[int] = deque()
        self._next_connection_id = 0
        self._next_transaction_id = 0

        # Waiters for connections
        self._connection_waiters: deque[Callable[[], None]] = deque()

        # In-memory data store (simplified)
        self._tables: dict[str, list[dict]] = {}

        # Statistics
        self._queries_executed = 0
        self._transactions_started = 0
        self._transactions_committed = 0
        self._transactions_rolled_back = 0
        self._connections_created = 0
        self._connections_closed = 0
        self._connection_wait_count = 0
        self._connection_wait_time_total = 0.0
        self._query_latencies: list[float] = []

    @property
    def stats(self) -> DatabaseStats:
        """Frozen snapshot of database statistics."""
        return DatabaseStats(
            queries_executed=self._queries_executed,
            transactions_started=self._transactions_started,
            transactions_committed=self._transactions_committed,
            transactions_rolled_back=self._transactions_rolled_back,
            connections_created=self._connections_created,
            connections_closed=self._connections_closed,
            connection_wait_count=self._connection_wait_count,
            connection_wait_time_total=self._connection_wait_time_total,
            query_latencies=list(self._query_latencies),
        )

    @property
    def max_connections(self) -> int:
        """Maximum concurrent connections."""
        return self._max_connections

    @property
    def active_connections(self) -> int:
        """Number of currently active connections."""
        return len(self._connections)

    @property
    def available_connections(self) -> int:
        """Number of available connections."""
        return len(self._available_connections)

    @property
    def pending_waiters(self) -> int:
        """Number of waiters for connections."""
        return len(self._connection_waiters)

    def _get_query_latency(self, query: str) -> float:
        """Get latency for a specific query."""
        if callable(self._query_latency):
            return self._query_latency(query)
        return self._query_latency

    def _create_connection(self) -> Connection:
        """Create a new connection."""
        conn_id = self._next_connection_id
        self._next_connection_id += 1

        now = self._clock.now if self._clock else Instant.Epoch
        conn = Connection(id=conn_id, created_at=now)
        self._connections[conn_id] = conn
        self._connections_created += 1

        return conn

    def _acquire_connection(self) -> Generator[float, None, Connection]:
        """Acquire a connection from the pool.

        Yields:
            Connection latency.

        Returns:
            An available connection.
        """
        # Check if we have an available connection
        if self._available_connections:
            conn_id = self._available_connections.popleft()
            yield self._connection_latency
            return self._connections[conn_id]

        # Check if we can create a new connection
        if len(self._connections) < self._max_connections:
            yield self._connection_latency
            conn = self._create_connection()
            return conn

        # Must wait for a connection
        self._connection_wait_count += 1
        wait_start = self._clock.now if self._clock else Instant.Epoch

        acquired = [False]
        acquired_conn: list[Connection | None] = [None]

        def on_available():
            acquired[0] = True
            if self._available_connections:
                conn_id = self._available_connections.popleft()
                acquired_conn[0] = self._connections[conn_id]

        self._connection_waiters.append(on_available)

        while not acquired[0]:
            yield 0.01  # Poll interval

        if self._clock:
            wait_time = (self._clock.now - wait_start).to_seconds()
            self._connection_wait_time_total += wait_time

        yield self._connection_latency
        return acquired_conn[0]

    def _release_connection(self, conn: Connection) -> None:
        """Release a connection back to the pool."""
        if conn.id not in self._connections:
            return

        conn.in_transaction = False
        conn.transaction_id = None

        # Wake a waiter if any
        if self._connection_waiters:
            waiter = self._connection_waiters.popleft()
            self._available_connections.append(conn.id)
            waiter()
        else:
            self._available_connections.append(conn.id)

    def _execute_query(self, query: str) -> Generator[float, None, Any]:
        """Execute a query (internal).

        Args:
            query: The SQL query.

        Yields:
            Query latency.

        Returns:
            Query result.
        """
        latency = self._get_query_latency(query)
        yield latency

        self._queries_executed += 1
        self._query_latencies.append(latency)

        # Simple query parsing for simulation
        query_upper = query.upper().strip()

        if query_upper.startswith("SELECT"):
            # Return empty result set
            return []
        elif query_upper.startswith(("INSERT", "UPDATE", "DELETE")):
            return {"affected_rows": 1}
        else:
            return None

    def execute(self, query: str) -> Generator[float, None, Any]:
        """Execute a query using a temporary connection.

        Acquires a connection, executes the query, and releases.

        Args:
            query: The SQL query to execute.

        Yields:
            Connection and query latency.

        Returns:
            Query result.
        """
        conn = yield from self._acquire_connection()

        try:
            result = yield from self._execute_query(query)
            return result
        finally:
            self._release_connection(conn)

    def begin_transaction(self) -> Generator[float, None, Transaction]:
        """Begin a new transaction.

        Acquires a connection and starts a transaction.

        Yields:
            Connection latency.

        Returns:
            The new transaction.
        """
        conn = yield from self._acquire_connection()

        tx_id = self._next_transaction_id
        self._next_transaction_id += 1

        conn.in_transaction = True
        conn.transaction_id = tx_id

        self._transactions_started += 1

        return Transaction(tx_id, self, conn)

    def _end_transaction(self, tx: Transaction) -> None:
        """End a transaction and release its connection."""
        if tx.state == TransactionState.COMMITTED:
            self._transactions_committed += 1
        elif tx.state == TransactionState.ROLLED_BACK:
            self._transactions_rolled_back += 1

        self._release_connection(tx._connection)

    def create_table(self, name: str) -> None:
        """Create a table (for simulation).

        Args:
            name: Table name.
        """
        self._tables[name] = []

    def get_table_names(self) -> list[str]:
        """Get all table names.

        Returns:
            List of table names.
        """
        return list(self._tables.keys())

    def handle_event(self, event: Event) -> None:
        """Database can handle events for query execution."""
