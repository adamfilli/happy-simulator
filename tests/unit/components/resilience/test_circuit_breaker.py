"""Tests for CircuitBreaker component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.resilience import (
    CircuitBreaker,
    CircuitState,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class ReliableServer(Entity):
    """Server that always succeeds."""

    name: str
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


@dataclass
class FailingServer(Entity):
    """Server that fails after N requests."""

    name: str
    fail_after: int = 0
    response_time: float = 0.010

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        if self.requests_received > self.fail_after:
            # Simulate failure by not completing normally
            # In real usage, this would throw or return error
            pass
        yield self.response_time


@dataclass
class SlowServer(Entity):
    """Server with very slow response time."""

    name: str
    response_time: float = 10.0

    requests_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.requests_received += 1
        yield self.response_time


class TestCircuitBreakerCreation:
    """Tests for CircuitBreaker creation."""

    def test_creates_with_defaults(self):
        """CircuitBreaker can be created with minimal parameters."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server)

        assert cb.name == "cb"
        assert cb.target is server
        assert cb.failure_threshold == 5
        assert cb.success_threshold == 2
        assert cb.timeout == 30.0
        assert cb.state == CircuitState.CLOSED

    def test_creates_with_custom_parameters(self):
        """CircuitBreaker can be created with custom parameters."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(
            name="cb",
            target=server,
            failure_threshold=3,
            success_threshold=1,
            timeout=10.0,
            half_open_max_requests=2,
        )

        assert cb.failure_threshold == 3
        assert cb.success_threshold == 1
        assert cb.timeout == 10.0

    def test_rejects_invalid_failure_threshold(self):
        """CircuitBreaker rejects failure_threshold < 1."""
        server = ReliableServer(name="server")

        with pytest.raises(ValueError):
            CircuitBreaker(name="cb", target=server, failure_threshold=0)

    def test_rejects_invalid_success_threshold(self):
        """CircuitBreaker rejects success_threshold < 1."""
        server = ReliableServer(name="server")

        with pytest.raises(ValueError):
            CircuitBreaker(name="cb", target=server, success_threshold=0)

    def test_rejects_invalid_timeout(self):
        """CircuitBreaker rejects timeout <= 0."""
        server = ReliableServer(name="server")

        with pytest.raises(ValueError):
            CircuitBreaker(name="cb", target=server, timeout=0)

        with pytest.raises(ValueError):
            CircuitBreaker(name="cb", target=server, timeout=-1)

    def test_initial_statistics_are_zero(self):
        """CircuitBreaker starts with zero statistics."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server)

        assert cb.stats.total_requests == 0
        assert cb.stats.successful_requests == 0
        assert cb.stats.failed_requests == 0
        assert cb.stats.rejected_requests == 0
        assert cb.stats.state_changes == 0


class TestCircuitBreakerStates:
    """Tests for CircuitBreaker state transitions."""

    def test_starts_in_closed_state(self):
        """CircuitBreaker starts in CLOSED state."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server)

        assert cb.state == CircuitState.CLOSED

    def test_forwards_requests_when_closed(self):
        """CircuitBreaker forwards requests when CLOSED."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server, failure_threshold=3)

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, cb],
        )

        # Send request through circuit breaker
        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=cb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        assert server.requests_received == 1
        assert cb.stats.total_requests == 1

    def test_opens_after_failure_threshold(self):
        """CircuitBreaker opens after failure_threshold failures."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server, failure_threshold=3)

        # Manually record failures
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.stats.times_opened == 1

    def test_rejects_requests_when_open(self):
        """CircuitBreaker rejects requests when OPEN."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server, failure_threshold=3)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[server, cb],
        )

        # Try to send request
        request = Event(
            time=Instant.Epoch,
            event_type="request",
            target=cb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        # Request should be rejected
        assert server.requests_received == 0
        assert cb.stats.rejected_requests == 1

    def test_transitions_to_half_open_after_timeout(self):
        """CircuitBreaker transitions to HALF_OPEN after timeout."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(
            name="cb",
            target=server,
            failure_threshold=3,
            timeout=0.5,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[server, cb],
        )

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

        # Send request after timeout
        request = Event(
            time=Instant.from_seconds(1.0),
            event_type="request",
            target=cb,
            context={},
        )
        sim.schedule(request)
        sim.run()

        # Circuit should transition to HALF_OPEN and allow request
        assert server.requests_received == 1

    def test_closes_after_success_threshold_in_half_open(self):
        """CircuitBreaker closes after success_threshold in HALF_OPEN."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(
            name="cb",
            target=server,
            failure_threshold=3,
            success_threshold=2,
            timeout=0.5,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(3.0),
            sources=[],
            entities=[server, cb],
        )

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        # Send requests after timeout to transition to HALF_OPEN
        for i in range(2):
            request = Event(
                time=Instant.from_seconds(1.0 + i * 0.2),
                event_type="request",
                target=cb,
                context={},
            )
            sim.schedule(request)

        sim.run()

        # Circuit should be closed after 2 successes
        assert cb.state == CircuitState.CLOSED
        assert cb.stats.times_closed == 1


class TestCircuitBreakerForceControl:
    """Tests for manual circuit control."""

    def test_force_open(self):
        """force_open() opens the circuit."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server)

        assert cb.state == CircuitState.CLOSED

        cb.force_open()

        assert cb.state == CircuitState.OPEN

    def test_force_close(self):
        """force_close() closes the circuit."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server, failure_threshold=3)

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

        cb.force_close()

        assert cb.state == CircuitState.CLOSED

    def test_reset(self):
        """reset() returns to initial state."""
        server = ReliableServer(name="server")
        cb = CircuitBreaker(name="cb", target=server, failure_threshold=3)

        # Open the circuit and record some stats
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count == 3

        cb.reset()

        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCircuitBreakerStateCallback:
    """Tests for state change callbacks."""

    def test_on_state_change_called(self):
        """on_state_change callback is invoked on transitions."""
        server = ReliableServer(name="server")
        state_changes = []

        def on_change(old_state, new_state):
            state_changes.append((old_state, new_state))

        cb = CircuitBreaker(
            name="cb",
            target=server,
            failure_threshold=3,
            on_state_change=on_change,
        )

        # Open the circuit
        for _ in range(3):
            cb.record_failure()

        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)
