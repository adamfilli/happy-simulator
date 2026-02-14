"""Saga orchestrator for distributed transactions.

Executes a sequence of steps forward; on failure (timeout), compensates
completed steps in reverse order. Supports concurrent saga instances.

Example:
    from happysimulator.components.microservice import Saga, SagaStep

    saga = Saga(
        name="order_saga",
        steps=[
            SagaStep("reserve_inventory", inventory_svc, "reserve",
                     inventory_svc, "release", timeout=2.0),
            SagaStep("charge_payment", payment_svc, "charge",
                     payment_svc, "refund", timeout=5.0),
            SagaStep("ship_order", shipping_svc, "ship",
                     shipping_svc, "cancel_shipment", timeout=10.0),
        ],
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Generator

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Duration, Instant

logger = logging.getLogger(__name__)


class SagaState(Enum):
    """State of a saga instance."""

    PENDING = "pending"
    RUNNING = "running"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    COMPENSATED = "compensated"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Definition of a single saga step.

    Args:
        name: Human-readable step name.
        action_target: Entity to send the forward action to.
        action_event_type: Event type for the forward action.
        compensation_target: Entity to send compensation to on rollback.
        compensation_event_type: Event type for compensation.
        timeout: Seconds to wait for step completion. None for no timeout.
    """

    name: str
    action_target: Entity
    action_event_type: str
    compensation_target: Entity
    compensation_event_type: str
    timeout: float | None = None


@dataclass
class SagaStepResult:
    """Result of an individual saga step execution."""

    step_name: str
    success: bool
    started_at: Instant | None = None
    completed_at: Instant | None = None


@dataclass
class SagaStats:
    """Statistics tracked by Saga."""

    sagas_started: int = 0
    sagas_completed: int = 0
    sagas_compensated: int = 0
    sagas_failed: int = 0
    steps_executed: int = 0
    steps_failed: int = 0
    compensations_executed: int = 0


@dataclass
class _SagaInstance:
    """Internal state for a running saga instance."""

    saga_id: int
    state: SagaState
    current_step: int
    step_results: list[SagaStepResult]
    original_event: Event
    started_at: Instant


class Saga(Entity):
    """Distributed transaction orchestrator using the saga pattern.

    Executes steps in sequence. Each step sends an action event to its
    target and waits for completion (via completion hooks). If a step
    times out, the saga enters compensating state and executes
    compensation for all completed steps in reverse order.

    The same Saga entity can orchestrate multiple concurrent saga
    instances, each identified by a unique saga_id.

    Attributes:
        name: Saga identifier.
        stats: Accumulated statistics.
    """

    def __init__(
        self,
        name: str,
        steps: list[SagaStep],
        on_complete: Callable[[int, SagaState, list[SagaStepResult]], None] | None = None,
    ):
        """Initialize the saga orchestrator.

        Args:
            name: Saga identifier.
            steps: Ordered list of saga steps.
            on_complete: Optional callback fired when a saga finishes.
                        Receives (saga_id, final_state, step_results).

        Raises:
            ValueError: If steps list is empty.
        """
        super().__init__(name)

        if not steps:
            raise ValueError("Saga must have at least one step")

        self._steps = list(steps)
        self._on_complete = on_complete

        self._instances: dict[int, _SagaInstance] = {}
        self._next_saga_id = 0

        self.stats = SagaStats()

        logger.debug(
            "[%s] Saga initialized with %d steps: %s",
            name,
            len(steps),
            [s.name for s in steps],
        )

    @property
    def steps(self) -> list[SagaStep]:
        """The saga step definitions."""
        return list(self._steps)

    @property
    def active_instances(self) -> int:
        """Number of currently active saga instances."""
        return sum(
            1 for inst in self._instances.values()
            if inst.state in (SagaState.RUNNING, SagaState.COMPENSATING)
        )

    def handle_event(self, event: Event) -> list[Event] | None:
        """Route events to the appropriate saga handler.

        Args:
            event: The incoming event.

        Returns:
            Events to schedule.
        """
        event_type = event.event_type

        if event_type == "_saga_step_complete":
            return self._handle_step_complete(event)

        if event_type == "_saga_step_timeout":
            return self._handle_step_timeout(event)

        if event_type == "_saga_comp_complete":
            return self._handle_compensation_complete(event)

        # Start a new saga instance
        return self._start_saga(event)

    def _start_saga(self, event: Event) -> list[Event]:
        """Begin a new saga instance."""
        self._next_saga_id += 1
        saga_id = self._next_saga_id

        instance = _SagaInstance(
            saga_id=saga_id,
            state=SagaState.RUNNING,
            current_step=0,
            step_results=[],
            original_event=event,
            started_at=self.now,
        )
        self._instances[saga_id] = instance
        self.stats.sagas_started += 1

        logger.info("[%s] Saga %d started", self.name, saga_id)

        return self._execute_step(instance)

    def _execute_step(self, instance: _SagaInstance) -> list[Event]:
        """Execute the current step of a saga instance."""
        step_idx = instance.current_step
        step = self._steps[step_idx]

        self.stats.steps_executed += 1

        result = SagaStepResult(
            step_name=step.name,
            success=False,
            started_at=self.now,
        )
        instance.step_results.append(result)

        logger.debug(
            "[%s] Saga %d: executing step %d (%s)",
            self.name,
            instance.saga_id,
            step_idx,
            step.name,
        )

        # Create action event
        action_event = Event(
            time=self.now,
            event_type=step.action_event_type,
            target=step.action_target,
            context={
                "metadata": {
                    "_saga_id": instance.saga_id,
                    "_saga_step": step_idx,
                    "_saga_name": self.name,
                },
                "payload": instance.original_event.context.get("payload", {}),
            },
        )

        # Completion hook to notify saga of step success
        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_saga_step_complete",
                target=self,
                context={
                    "metadata": {
                        "saga_id": instance.saga_id,
                        "step_idx": step_idx,
                    },
                },
            )

        action_event.add_completion_hook(on_complete)

        events: list[Event] = [action_event]

        # Schedule timeout if configured
        if step.timeout is not None:
            timeout_event = Event(
                time=self.now + Duration.from_seconds(step.timeout),
                event_type="_saga_step_timeout",
                target=self,
                context={
                    "metadata": {
                        "saga_id": instance.saga_id,
                        "step_idx": step_idx,
                    },
                },
            )
            events.append(timeout_event)

        return events

    def _handle_step_complete(self, event: Event) -> list[Event] | None:
        """Handle successful completion of a saga step."""
        metadata = event.context.get("metadata", {})
        saga_id = metadata.get("saga_id")
        step_idx = metadata.get("step_idx")

        instance = self._instances.get(saga_id)
        if instance is None:
            return None

        # Ignore if saga is no longer running (e.g., already compensating)
        if instance.state != SagaState.RUNNING:
            return None

        # Ignore if this is a stale completion (step already advanced)
        if step_idx != instance.current_step:
            return None

        # Mark step as successful
        instance.step_results[step_idx].success = True
        instance.step_results[step_idx].completed_at = self.now

        logger.debug(
            "[%s] Saga %d: step %d (%s) completed",
            self.name,
            saga_id,
            step_idx,
            self._steps[step_idx].name,
        )

        # Advance to next step
        instance.current_step += 1

        if instance.current_step >= len(self._steps):
            # All steps completed successfully
            return self._complete_saga(instance, SagaState.COMPLETED)

        return self._execute_step(instance)

    def _handle_step_timeout(self, event: Event) -> list[Event] | None:
        """Handle step timeout — begin compensation."""
        metadata = event.context.get("metadata", {})
        saga_id = metadata.get("saga_id")
        step_idx = metadata.get("step_idx")

        instance = self._instances.get(saga_id)
        if instance is None:
            return None

        # Ignore if saga is no longer running or step already completed
        if instance.state != SagaState.RUNNING:
            return None
        if step_idx != instance.current_step:
            return None

        self.stats.steps_failed += 1

        logger.info(
            "[%s] Saga %d: step %d (%s) timed out, compensating",
            self.name,
            saga_id,
            step_idx,
            self._steps[step_idx].name,
        )

        # Begin compensation from the last completed step
        instance.state = SagaState.COMPENSATING
        # current_step points to the failed step; compensate steps 0..current_step-1
        instance.current_step = step_idx - 1

        if instance.current_step < 0:
            # No steps to compensate
            return self._complete_saga(instance, SagaState.COMPENSATED)

        return self._execute_compensation(instance)

    def _execute_compensation(self, instance: _SagaInstance) -> list[Event]:
        """Execute compensation for the current step (reverse order)."""
        step_idx = instance.current_step
        step = self._steps[step_idx]

        self.stats.compensations_executed += 1

        logger.debug(
            "[%s] Saga %d: compensating step %d (%s)",
            self.name,
            instance.saga_id,
            step_idx,
            step.name,
        )

        comp_event = Event(
            time=self.now,
            event_type=step.compensation_event_type,
            target=step.compensation_target,
            context={
                "metadata": {
                    "_saga_id": instance.saga_id,
                    "_saga_step": step_idx,
                    "_saga_name": self.name,
                    "_saga_compensation": True,
                },
                "payload": instance.original_event.context.get("payload", {}),
            },
        )

        def on_complete(finish_time: Instant):
            return Event(
                time=finish_time,
                event_type="_saga_comp_complete",
                target=self,
                context={
                    "metadata": {
                        "saga_id": instance.saga_id,
                        "step_idx": step_idx,
                    },
                },
            )

        comp_event.add_completion_hook(on_complete)
        return [comp_event]

    def _handle_compensation_complete(self, event: Event) -> list[Event] | None:
        """Handle compensation step completion — continue reverse."""
        metadata = event.context.get("metadata", {})
        saga_id = metadata.get("saga_id")
        step_idx = metadata.get("step_idx")

        instance = self._instances.get(saga_id)
        if instance is None:
            return None

        if instance.state != SagaState.COMPENSATING:
            return None

        logger.debug(
            "[%s] Saga %d: compensation for step %d complete",
            self.name,
            saga_id,
            step_idx,
        )

        # Move to previous step
        instance.current_step -= 1

        if instance.current_step < 0:
            # All compensations done
            return self._complete_saga(instance, SagaState.COMPENSATED)

        return self._execute_compensation(instance)

    def _complete_saga(self, instance: _SagaInstance, final_state: SagaState) -> list[Event]:
        """Finalize a saga instance."""
        instance.state = final_state

        if final_state == SagaState.COMPLETED:
            self.stats.sagas_completed += 1
            logger.info("[%s] Saga %d completed successfully", self.name, instance.saga_id)
        elif final_state == SagaState.COMPENSATED:
            self.stats.sagas_compensated += 1
            logger.info("[%s] Saga %d compensated", self.name, instance.saga_id)
        else:
            self.stats.sagas_failed += 1
            logger.warning("[%s] Saga %d failed", self.name, instance.saga_id)

        if self._on_complete:
            self._on_complete(instance.saga_id, final_state, instance.step_results)

        # Fire original event's completion hooks on success
        result: list[Event] = []
        if final_state == SagaState.COMPLETED:
            for hook in instance.original_event.on_complete:
                hook_result = hook(self.now)
                if hook_result is not None:
                    if isinstance(hook_result, list):
                        result.extend(hook_result)
                    else:
                        result.append(hook_result)

        return result

    def get_instance_state(self, saga_id: int) -> SagaState | None:
        """Get the state of a saga instance.

        Args:
            saga_id: The saga instance ID.

        Returns:
            The saga state, or None if not found.
        """
        instance = self._instances.get(saga_id)
        return instance.state if instance else None
