"""Tests for Database."""

import pytest

from happysimulator.components.datastore import (
    Database,
    Transaction,
    TransactionState,
)


class TestDatabaseCreation:
    """Tests for Database creation."""

    def test_creates_with_defaults(self):
        """Database is created with default values."""
        db = Database(name="postgres")

        assert db.name == "postgres"
        assert db.max_connections == 100
        assert db.active_connections == 0
        assert db.available_connections == 0

    def test_creates_with_custom_settings(self):
        """Database is created with custom settings."""
        db = Database(
            name="mysql",
            max_connections=50,
            query_latency=0.010,
            connection_latency=0.020,
        )

        assert db.name == "mysql"
        assert db.max_connections == 50

    def test_rejects_zero_connections(self):
        """Rejects max_connections of zero."""
        with pytest.raises(ValueError):
            Database(name="db", max_connections=0)

    def test_rejects_negative_connections(self):
        """Rejects negative max_connections."""
        with pytest.raises(ValueError):
            Database(name="db", max_connections=-1)


class TestDatabaseQuery:
    """Tests for database query execution."""

    def test_execute_select_query(self):
        """Execute a SELECT query."""
        db = Database(name="db", query_latency=0.005)

        gen = db.execute("SELECT * FROM users WHERE id = 1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == []  # Empty result set
        assert db.stats.queries_executed == 1

    def test_execute_insert_query(self):
        """Execute an INSERT query."""
        db = Database(name="db", query_latency=0.005)

        gen = db.execute("INSERT INTO users (name) VALUES ('test')")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == {"affected_rows": 1}
        assert db.stats.queries_executed == 1

    def test_execute_update_query(self):
        """Execute an UPDATE query."""
        db = Database(name="db", query_latency=0.005)

        gen = db.execute("UPDATE users SET name = 'new' WHERE id = 1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == {"affected_rows": 1}

    def test_execute_delete_query(self):
        """Execute a DELETE query."""
        db = Database(name="db", query_latency=0.005)

        gen = db.execute("DELETE FROM users WHERE id = 1")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == {"affected_rows": 1}

    def test_callable_query_latency(self):
        """Uses callable for query-dependent latency."""

        def get_latency(query: str) -> float:
            if "SELECT" in query.upper():
                return 0.001
            return 0.010

        db = Database(name="db", query_latency=get_latency)

        # SELECT should be fast
        gen = db.execute("SELECT * FROM users")
        latencies = []
        try:
            while True:
                latencies.append(next(gen))
        except StopIteration:
            pass

        # INSERT should be slower
        gen = db.execute("INSERT INTO users (name) VALUES ('x')")
        try:
            while True:
                latencies.append(next(gen))
        except StopIteration:
            pass

        # The latencies should differ
        assert db.stats.queries_executed == 2


class TestDatabaseConnectionPool:
    """Tests for connection pool behavior."""

    def test_creates_connections_on_demand(self):
        """Creates connections when needed."""
        db = Database(name="db", max_connections=10)

        gen = db.execute("SELECT 1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert db.stats.connections_created == 1

    def test_reuses_connections(self):
        """Reuses available connections."""
        db = Database(name="db", max_connections=10)

        # First query
        gen = db.execute("SELECT 1")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Connection should be available
        assert db.available_connections == 1

        # Second query reuses
        gen = db.execute("SELECT 2")
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Still only one connection created
        assert db.stats.connections_created == 1


class TestTransaction:
    """Tests for database transactions."""

    def test_begin_transaction(self):
        """Begin a transaction."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        assert tx is not None
        assert isinstance(tx, Transaction)
        assert tx.state == TransactionState.ACTIVE
        assert tx.is_active is True
        assert db.stats.transactions_started == 1

    def test_transaction_execute(self):
        """Execute queries within transaction."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # Execute within transaction
        gen = tx.execute("INSERT INTO users (name) VALUES ('test')")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            result = e.value

        assert result == {"affected_rows": 1}
        assert db.stats.queries_executed == 1

    def test_transaction_commit(self):
        """Commit a transaction."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # Commit
        gen = tx.commit()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert tx.state == TransactionState.COMMITTED
        assert tx.is_active is False
        assert db.stats.transactions_committed == 1

    def test_transaction_rollback(self):
        """Roll back a transaction."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # Rollback
        gen = tx.rollback()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        assert tx.state == TransactionState.ROLLED_BACK
        assert tx.is_active is False
        assert db.stats.transactions_rolled_back == 1

    def test_execute_after_commit_raises(self):
        """Cannot execute after commit."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # Commit
        gen = tx.commit()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Try to execute after commit
        with pytest.raises(RuntimeError):
            gen = tx.execute("SELECT 1")
            next(gen)

    def test_execute_after_rollback_raises(self):
        """Cannot execute after rollback."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # Rollback
        gen = tx.rollback()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Try to execute after rollback
        with pytest.raises(RuntimeError):
            gen = tx.execute("SELECT 1")
            next(gen)

    def test_double_commit_raises(self):
        """Cannot commit twice."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # First commit
        gen = tx.commit()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Second commit should raise
        with pytest.raises(RuntimeError):
            gen = tx.commit()
            next(gen)

    def test_double_rollback_raises(self):
        """Cannot rollback twice."""
        db = Database(name="db")

        gen = db.begin_transaction()
        tx = None
        try:
            while True:
                next(gen)
        except StopIteration as e:
            tx = e.value

        # First rollback
        gen = tx.rollback()
        try:
            while True:
                next(gen)
        except StopIteration:
            pass

        # Second rollback should raise
        with pytest.raises(RuntimeError):
            gen = tx.rollback()
            next(gen)


class TestDatabaseTables:
    """Tests for table management."""

    def test_create_table(self):
        """Can create tables."""
        db = Database(name="db")

        db.create_table("users")
        db.create_table("orders")

        assert "users" in db.get_table_names()
        assert "orders" in db.get_table_names()

    def test_get_table_names_empty(self):
        """Returns empty list when no tables."""
        db = Database(name="db")

        assert db.get_table_names() == []


class TestDatabaseStatistics:
    """Tests for database statistics."""

    def test_tracks_query_latencies(self):
        """Statistics track query latencies."""
        db = Database(name="db", query_latency=0.005)

        for _ in range(5):
            gen = db.execute("SELECT 1")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        assert len(db.stats.query_latencies) == 5
        assert db.stats.avg_query_latency == pytest.approx(0.005)

    def test_query_latency_p95(self):
        """Can get p95 query latency."""
        db = Database(name="db", query_latency=0.005)

        for _ in range(100):
            gen = db.execute("SELECT 1")
            try:
                while True:
                    next(gen)
            except StopIteration:
                pass

        # With constant latency, p95 should equal the latency
        assert db.stats.query_latency_p95 == pytest.approx(0.005)

    def test_empty_latency_stats(self):
        """Empty latency stats return zero."""
        db = Database(name="db")

        assert db.stats.avg_query_latency == 0.0
        assert db.stats.query_latency_p95 == 0.0
