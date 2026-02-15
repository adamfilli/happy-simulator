"""Tests for Saga component."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from happysimulator.components.microservice import (
    Saga,
    SagaState,
    SagaStep,
)
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.simulation import Simulation
from happysimulator.core.temporal import Instant

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass
class StepService(Entity):
    """Service that processes saga step actions/compensations."""

    name: str
    response_time: float = 0.010

    actions_received: int = field(default=0, init=False)
    compensations_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        metadata = event.context.get("metadata", {})
        if metadata.get("_saga_compensation"):
            self.compensations_received += 1
        else:
            self.actions_received += 1
        yield self.response_time


@dataclass
class SlowStepService(Entity):
    """Service that takes too long (for timeout testing)."""

    name: str
    response_time: float = 100.0

    actions_received: int = field(default=0, init=False)

    def handle_event(self, event: Event) -> Generator[float]:
        self.actions_received += 1
        yield self.response_time


class TestSagaCreation:
    """Tests for Saga creation."""

    def test_creates_with_steps(self):
        svc = StepService(name="svc")
        steps = [
            SagaStep("step1", svc, "do1", svc, "undo1"),
        ]
        saga = Saga(name="saga", steps=steps)

        assert saga.name == "saga"
        assert len(saga.steps) == 1
        assert saga.active_instances == 0

    def test_initial_stats_are_zero(self):
        svc = StepService(name="svc")
        saga = Saga(name="saga", steps=[SagaStep("s", svc, "do", svc, "undo")])

        assert saga.stats.sagas_started == 0
        assert saga.stats.sagas_completed == 0
        assert saga.stats.sagas_compensated == 0
        assert saga.stats.steps_executed == 0

    def test_rejects_empty_steps(self):
        with pytest.raises(ValueError):
            Saga(name="saga", steps=[])


class TestSagaExecution:
    """Tests for Saga forward execution."""

    def test_executes_single_step_saga(self):
        svc = StepService(name="svc")
        saga = Saga(
            name="saga",
            steps=[SagaStep("reserve", svc, "reserve", svc, "release")],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[svc, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start_order",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert saga.stats.sagas_started == 1
        assert saga.stats.sagas_completed == 1
        assert saga.stats.steps_executed == 1
        assert svc.actions_received == 1

    def test_executes_multi_step_saga(self):
        svc1 = StepService(name="inventory")
        svc2 = StepService(name="payment")
        svc3 = StepService(name="shipping")

        saga = Saga(
            name="order_saga",
            steps=[
                SagaStep("reserve", svc1, "reserve", svc1, "release"),
                SagaStep("charge", svc2, "charge", svc2, "refund"),
                SagaStep("ship", svc3, "ship", svc3, "cancel"),
            ],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[svc1, svc2, svc3, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start_order",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert saga.stats.sagas_completed == 1
        assert saga.stats.steps_executed == 3
        assert svc1.actions_received == 1
        assert svc2.actions_received == 1
        assert svc3.actions_received == 1

    def test_concurrent_saga_instances(self):
        svc = StepService(name="svc")
        saga = Saga(
            name="saga",
            steps=[SagaStep("action", svc, "do", svc, "undo")],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[svc, saga],
        )

        for i in range(3):
            trigger = Event(
                time=Instant.Epoch,
                event_type="start",
                target=saga,
                context={"payload": {"id": i}},
            )
            sim.schedule(trigger)

        sim.run()

        assert saga.stats.sagas_started == 3
        assert saga.stats.sagas_completed == 3


class TestSagaCompensation:
    """Tests for Saga compensation on failure."""

    def test_compensates_on_step_timeout(self):
        """When a step times out, completed steps are compensated."""
        fast_svc = StepService(name="fast_svc")
        slow_svc = SlowStepService(name="slow_svc")

        saga = Saga(
            name="saga",
            steps=[
                SagaStep("step1", fast_svc, "action1", fast_svc, "comp1"),
                SagaStep("step2", slow_svc, "action2", fast_svc, "comp2", timeout=0.05),
            ],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[fast_svc, slow_svc, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert saga.stats.sagas_compensated == 1
        assert saga.stats.steps_failed == 1
        assert saga.stats.compensations_executed == 1
        # step1 was compensated
        assert fast_svc.compensations_received == 1

    def test_compensates_multiple_steps_in_reverse(self):
        """Compensation runs in reverse order."""
        svc1 = StepService(name="svc1")
        svc2 = StepService(name="svc2")
        slow_svc = SlowStepService(name="slow_svc")

        saga = Saga(
            name="saga",
            steps=[
                SagaStep("step1", svc1, "a1", svc1, "c1"),
                SagaStep("step2", svc2, "a2", svc2, "c2"),
                SagaStep("step3", slow_svc, "a3", slow_svc, "c3", timeout=0.05),
            ],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[svc1, svc2, slow_svc, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert saga.stats.sagas_compensated == 1
        assert saga.stats.compensations_executed == 2
        assert svc1.compensations_received == 1
        assert svc2.compensations_received == 1

    def test_no_compensation_when_first_step_fails(self):
        """If first step fails, nothing to compensate."""
        slow_svc = SlowStepService(name="slow_svc")
        fast_svc = StepService(name="fast_svc")

        saga = Saga(
            name="saga",
            steps=[
                SagaStep("step1", slow_svc, "a1", fast_svc, "c1", timeout=0.05),
                SagaStep("step2", fast_svc, "a2", fast_svc, "c2"),
            ],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(2.0),
            sources=[],
            entities=[slow_svc, fast_svc, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert saga.stats.sagas_compensated == 1
        assert saga.stats.compensations_executed == 0


class TestSagaCallbacks:
    """Tests for Saga callbacks."""

    def test_on_complete_callback_fires(self):
        svc = StepService(name="svc")
        results = []

        def on_complete(saga_id, state, step_results):
            results.append((saga_id, state))

        saga = Saga(
            name="saga",
            steps=[SagaStep("s", svc, "do", svc, "undo")],
            on_complete=on_complete,
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[svc, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert len(results) == 1
        assert results[0] == (1, SagaState.COMPLETED)

    def test_get_instance_state(self):
        svc = StepService(name="svc")
        saga = Saga(
            name="saga",
            steps=[SagaStep("s", svc, "do", svc, "undo")],
        )

        sim = Simulation(
            start_time=Instant.Epoch,
            end_time=Instant.from_seconds(1.0),
            sources=[],
            entities=[svc, saga],
        )

        trigger = Event(
            time=Instant.Epoch,
            event_type="start",
            target=saga,
        )
        sim.schedule(trigger)
        sim.run()

        assert saga.get_instance_state(1) == SagaState.COMPLETED
        assert saga.get_instance_state(999) is None
