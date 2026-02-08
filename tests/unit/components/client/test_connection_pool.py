"""Tests for ConnectionPool component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, List

import pytest

from happysimulator.components.client.connection_pool import (
    Connection,
    ConnectionPool,
    ConnectionPoolStats,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant
from happysimulator.core.callback_entity import NullEntity
from happysimulator.distributions.constant import ConstantLatency

_null = NullEntity()


@dataclass
class MockServer(Entity):
    """Simple server for testing."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float, None, None]:
        self.requests_received += 1
        yield self.response_time


class TestConnectionPoolCreation:
    """Tests for ConnectionPool creation and validation."""

    def test_creates_with_defaults(self):
        """Pool can be created with minimal parameters."""
        server = MockServer(name="server")
        pool = ConnectionPool(name="pool", target=server)

        assert pool.name == "pool"
        assert pool.target is server
        assert pool.min_connections == 0
        assert pool.max_connections == 10
        assert pool.connection_timeout == 5.0
        assert pool.idle_timeout == 60.0

    def test_creates_with_custom_parameters(self):
        """Pool can be created with custom parameters."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            min_connections=2,
            max_connections=20,
            connection_timeout=10.0,
            idle_timeout=120.0,
        )

        assert pool.min_connections == 2
        assert pool.max_connections == 20
        assert pool.connection_timeout == 10.0
        assert pool.idle_timeout == 120.0

    def test_rejects_negative_min_connections(self):
        """Pool rejects negative min_connections."""
        server = MockServer(name="server")

        with pytest.raises(ValueError):
            ConnectionPool(name="pool", target=server, min_connections=-1)

    def test_rejects_zero_max_connections(self):
        """Pool rejects max_connections < 1."""
        server = MockServer(name="server")

        with pytest.raises(ValueError):
            ConnectionPool(name="pool", target=server, max_connections=0)

    def test_rejects_max_less_than_min(self):
        """Pool rejects max_connections < min_connections."""
        server = MockServer(name="server")

        with pytest.raises(ValueError):
            ConnectionPool(
                name="pool",
                target=server,
                min_connections=10,
                max_connections=5,
            )

    def test_rejects_non_positive_connection_timeout(self):
        """Pool rejects connection_timeout <= 0."""
        server = MockServer(name="server")

        with pytest.raises(ValueError):
            ConnectionPool(name="pool", target=server, connection_timeout=0)

        with pytest.raises(ValueError):
            ConnectionPool(name="pool", target=server, connection_timeout=-1)

    def test_rejects_non_positive_idle_timeout(self):
        """Pool rejects idle_timeout <= 0."""
        server = MockServer(name="server")

        with pytest.raises(ValueError):
            ConnectionPool(name="pool", target=server, idle_timeout=0)

    def test_initial_statistics_are_zero(self):
        """Pool starts with zero statistics."""
        server = MockServer(name="server")
        pool = ConnectionPool(name="pool", target=server)

        assert pool.stats.connections_created == 0
        assert pool.stats.connections_closed == 0
        assert pool.stats.acquisitions == 0
        assert pool.stats.releases == 0
        assert pool.stats.timeouts == 0

    def test_initial_pool_state(self):
        """Pool starts empty."""
        server = MockServer(name="server")
        pool = ConnectionPool(name="pool", target=server)

        assert pool.active_connections == 0
        assert pool.idle_connections == 0
        assert pool.total_connections == 0
        assert pool.pending_requests == 0


class TestConnectionPoolAcquireRelease:
    """Tests for connection acquire and release."""

    def test_acquire_creates_connection(self):
        """Acquiring from empty pool creates new connection."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        result = {"connection": None}

        def acquire_connection():
            conn = yield from pool.acquire()
            result["connection"] = conn

        # Create an event that acquires a connection
        event = Event.once(
            time=Instant.Epoch,
            event_type="acquire",
            fn=lambda e: acquire_connection(),
        )
        sim.schedule(event)
        sim.run()

        assert result["connection"] is not None
        assert isinstance(result["connection"], Connection)
        assert pool.stats.connections_created == 1
        assert pool.active_connections == 1

    def test_release_returns_to_pool(self):
        """Released connection goes to idle pool."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            idle_timeout=300.0,  # Long timeout - we're not testing idle behavior here
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        def acquire_and_release():
            conn = yield from pool.acquire()
            yield 0.010  # Use connection
            pool.release(conn)
            # Note: not returning idle timeout events to avoid simulation
            # end_time boundary issue (sim processes events past end_time)

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: acquire_and_release(),
        )
        sim.schedule(event)
        sim.run()

        assert pool.active_connections == 0
        assert pool.idle_connections == 1
        assert pool.stats.releases == 1

    def test_acquire_reuses_idle_connection(self):
        """Acquiring reuses idle connection instead of creating new one."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        result = {"first_id": None, "second_id": None}

        def two_acquisitions():
            # First acquisition - creates new connection
            conn1 = yield from pool.acquire()
            result["first_id"] = conn1.id
            yield 0.010
            pool.release(conn1)

            yield 0.010

            # Second acquisition - should reuse
            conn2 = yield from pool.acquire()
            result["second_id"] = conn2.id

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: two_acquisitions(),
        )
        sim.schedule(event)
        sim.run()

        # Should have reused the same connection
        assert result["first_id"] == result["second_id"]
        assert pool.stats.connections_created == 1  # Only one created


class TestConnectionPoolLimits:
    """Tests for pool size limits."""

    def test_respects_max_connections(self):
        """Pool doesn't exceed max_connections."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=2,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        connections = []

        def acquire_many():
            # Acquire 2 connections (should succeed)
            for _ in range(2):
                conn = yield from pool.acquire()
                connections.append(conn)

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: acquire_many(),
        )
        sim.schedule(event)
        sim.run()

        assert len(connections) == 2
        assert pool.total_connections == 2
        assert pool.active_connections == 2

    def test_waits_when_pool_exhausted(self):
        """Requests wait when pool is exhausted."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=5.0,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool],
        )

        result = {"second_acquired_at": None, "first_released_at": None}

        def first_holder():
            conn = yield from pool.acquire()
            yield 0.500  # Hold for 500ms
            result["first_released_at"] = pool.now
            pool.release(conn)

        def second_requester():
            yield 0.100  # Start slightly after first
            conn = yield from pool.acquire()
            result["second_acquired_at"] = pool.now
            pool.release(conn)

        event1 = Event.once(
            time=Instant.Epoch,
            event_type="holder",
            fn=lambda e: first_holder(),
        )
        event2 = Event.once(
            time=Instant.Epoch,
            event_type="requester",
            fn=lambda e: second_requester(),
        )
        sim.schedule(event1)
        sim.schedule(event2)
        sim.run()

        # Second should acquire after first releases
        assert result["second_acquired_at"] is not None
        assert result["first_released_at"] is not None
        # Allow for some timing slack due to poll interval
        assert result["second_acquired_at"] >= result["first_released_at"]


class TestConnectionPoolTimeout:
    """Tests for connection acquisition timeout."""

    def test_timeout_when_pool_exhausted(self):
        """Acquisition times out when pool stays exhausted."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=0.5,  # 500ms timeout
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool],
        )

        result = {"timed_out": False}

        def holder():
            conn = yield from pool.acquire()
            yield 1.0  # Hold for 1s (longer than timeout)
            pool.release(conn)

        def requester():
            yield 0.100  # Start slightly after holder
            try:
                conn = yield from pool.acquire()
                pool.release(conn)
            except TimeoutError:
                result["timed_out"] = True

        event1 = Event.once(
            time=Instant.Epoch,
            event_type="holder",
            fn=lambda e: holder(),
        )
        event2 = Event.once(
            time=Instant.Epoch,
            event_type="requester",
            fn=lambda e: requester(),
        )
        sim.schedule(event1)
        sim.schedule(event2)
        sim.run()

        assert result["timed_out"]
        assert pool.stats.timeouts == 1

    def test_timeout_callback_invoked(self):
        """on_timeout callback is invoked on timeout."""
        server = MockServer(name="server")
        timeout_calls = []

        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=0.3,
            connection_latency=ConstantLatency(0.001),
            on_timeout=lambda: timeout_calls.append(1),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool],
        )

        def holder():
            conn = yield from pool.acquire()
            yield 1.0
            pool.release(conn)

        def requester():
            yield 0.050
            try:
                yield from pool.acquire()
            except TimeoutError:
                pass

        event1 = Event.once(
            time=Instant.Epoch,
            event_type="holder",
            fn=lambda e: holder(),
        )
        event2 = Event.once(
            time=Instant.Epoch,
            event_type="requester",
            fn=lambda e: requester(),
        )
        sim.schedule(event1)
        sim.schedule(event2)
        sim.run()

        assert len(timeout_calls) == 1


class TestConnectionPoolIdleTimeout:
    """Tests for idle connection timeout."""

    def test_idle_connections_closed(self):
        """Idle connections are closed after timeout."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            min_connections=0,
            idle_timeout=0.2,  # 200ms idle timeout
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        def use_and_release():
            conn = yield from pool.acquire()
            yield 0.010
            events = pool.release(conn)
            return events

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: use_and_release(),
        )
        sim.schedule(event)
        sim.run()

        # After idle timeout, connection should be closed
        assert pool.stats.connections_closed == 1
        assert pool.idle_connections == 0
        assert pool.total_connections == 0

    def test_min_connections_maintained(self):
        """Minimum connections are maintained even after idle timeout."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            min_connections=1,
            idle_timeout=0.2,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        def use_and_release():
            conn = yield from pool.acquire()
            yield 0.010
            events = pool.release(conn)
            return events

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: use_and_release(),
        )
        sim.schedule(event)
        sim.run()

        # Connection should NOT be closed due to min_connections
        assert pool.stats.connections_closed == 0
        assert pool.total_connections == 1


class TestConnectionPoolCallbacks:
    """Tests for pool callbacks."""

    def test_acquire_callback_invoked(self):
        """on_acquire callback is invoked when connection acquired."""
        server = MockServer(name="server")
        acquired = []

        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            on_acquire=lambda conn: acquired.append(conn),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        def acquire():
            conn = yield from pool.acquire()
            pool.release(conn)

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: acquire(),
        )
        sim.schedule(event)
        sim.run()

        assert len(acquired) == 1
        assert isinstance(acquired[0], Connection)

    def test_release_callback_invoked(self):
        """on_release callback is invoked when connection released."""
        server = MockServer(name="server")
        released = []

        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
            on_release=lambda conn: released.append(conn),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        def acquire_release():
            conn = yield from pool.acquire()
            pool.release(conn)

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: acquire_release(),
        )
        sim.schedule(event)
        sim.run()

        assert len(released) == 1
        assert isinstance(released[0], Connection)


class TestConnectionPoolWarmup:
    """Tests for pool warmup."""

    def test_warmup_creates_min_connections(self):
        """Warmup creates min_connections."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            min_connections=3,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        warmup_event = pool.warmup()
        warmup_event.time = Instant.Epoch
        sim.schedule(warmup_event)
        sim.run()

        assert pool.stats.connections_created == 3
        assert pool.idle_connections == 3
        assert pool.total_connections == 3


class TestConnectionPoolConcurrency:
    """Tests for concurrent connection usage."""

    def test_multiple_concurrent_acquisitions(self):
        """Multiple connections can be acquired concurrently."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=5,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        connections = []
        released = []

        def worker(worker_id: int):
            conn = yield from pool.acquire()
            connections.append((worker_id, conn))
            yield 0.100  # Work
            pool.release(conn)
            released.append(worker_id)

        # Start 3 workers concurrently
        for i in range(3):
            event = Event.once(
                time=Instant.Epoch,
                event_type=f"worker_{i}",
                fn=lambda e, wid=i: worker(wid),
            )
            sim.schedule(event)

        sim.run()

        # All should have acquired and released
        assert len(connections) == 3
        assert len(released) == 3
        assert pool.stats.connections_created == 3
        assert pool.stats.releases == 3

    def test_waiter_gets_released_connection(self):
        """When pool is exhausted, waiter gets released connection."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=5.0,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool],
        )

        result = {"worker1_conn": None, "worker2_conn": None}

        def worker1():
            conn = yield from pool.acquire()
            result["worker1_conn"] = conn.id
            yield 0.200  # Hold for 200ms
            pool.release(conn)

        def worker2():
            yield 0.050  # Start after worker1 has acquired
            conn = yield from pool.acquire()
            result["worker2_conn"] = conn.id
            pool.release(conn)

        event1 = Event.once(
            time=Instant.Epoch,
            event_type="worker1",
            fn=lambda e: worker1(),
        )
        event2 = Event.once(
            time=Instant.Epoch,
            event_type="worker2",
            fn=lambda e: worker2(),
        )
        sim.schedule(event1)
        sim.schedule(event2)
        sim.run()

        # Both should have got the same connection (since max is 1)
        assert result["worker1_conn"] == result["worker2_conn"]
        assert pool.stats.connections_created == 1


class TestConnectionPoolStatistics:
    """Tests for pool statistics."""

    def test_average_wait_time(self):
        """Pool tracks average wait time."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            max_connections=1,
            connection_timeout=5.0,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, pool],
        )

        def holder():
            conn = yield from pool.acquire()
            yield 0.300  # Hold for 300ms
            pool.release(conn)

        def waiter():
            yield 0.050  # Start after holder
            yield from pool.acquire()

        event1 = Event.once(
            time=Instant.Epoch,
            event_type="holder",
            fn=lambda e: holder(),
        )
        event2 = Event.once(
            time=Instant.Epoch,
            event_type="waiter",
            fn=lambda e: waiter(),
        )
        sim.schedule(event1)
        sim.schedule(event2)
        sim.run()

        # Waiter should have waited ~250ms
        # Average includes both (immediate and waited)
        assert pool.stats.acquisitions == 2
        assert pool.stats.total_wait_time > 0
        assert pool.average_wait_time > 0


class TestConnectionPoolCloseAll:
    """Tests for closing all connections."""

    def test_close_all_clears_pool(self):
        """close_all closes all connections."""
        server = MockServer(name="server")
        pool = ConnectionPool(
            name="pool",
            target=server,
            connection_latency=ConstantLatency(0.001),
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, pool],
        )

        connections = []

        def acquire_multiple():
            for _ in range(3):
                conn = yield from pool.acquire()
                connections.append(conn)
            # Release one
            pool.release(connections[0])

        event = Event.once(
            time=Instant.Epoch,
            event_type="test",
            fn=lambda e: acquire_multiple(),
        )
        sim.schedule(event)
        sim.run()

        # Now close all
        pool.close_all()

        assert pool.active_connections == 0
        assert pool.idle_connections == 0
        assert pool.total_connections == 0
        assert pool.stats.connections_closed == 3
