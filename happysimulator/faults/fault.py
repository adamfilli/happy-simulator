"""Fault protocol, context, handle, and stats for the fault injection framework.

Defines the contract that all fault types implement (``Fault`` protocol),
the resolution context passed during event generation, the ``FaultHandle``
for manual cancellation, and ``FaultStats`` for observability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from happysimulator.components.network.network import Network
    from happysimulator.components.resource import Resource
    from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


@dataclass
class FaultContext:
    """Resolution context passed to faults during event generation.

    Built by ``FaultSchedule.start()`` from the simulation's registered
    entities, networks, and resources.

    Attributes:
        entities: Name-to-Entity lookup (all registered entities).
        networks: Name-to-Network lookup.
        resources: Name-to-Resource lookup.
        start_time: Simulation start time.
    """

    entities: dict[str, Entity]
    networks: dict[str, Network]
    resources: dict[str, Resource]
    start_time: Instant


@runtime_checkable
class Fault(Protocol):
    """Protocol that all fault types implement."""

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        """Generate activation/deactivation events for this fault.

        Args:
            ctx: Resolution context with entity/network/resource lookups.

        Returns:
            Events to schedule for fault activation and deactivation.
        """
        ...


class FaultHandle:
    """Handle returned by ``FaultSchedule.add()`` for manual cancellation.

    Cancelling a handle marks all its pending fault events as cancelled
    so they are skipped by the simulation loop.

    Attributes:
        fault: The fault this handle controls.
    """

    def __init__(self, fault: Fault) -> None:
        self.fault = fault
        self._events: list[Event] = []
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        """Whether this fault has been cancelled."""
        return self._cancelled

    def cancel(self) -> None:
        """Cancel all pending events for this fault."""
        if self._cancelled:
            return
        self._cancelled = True
        for event in self._events:
            event.cancel()
        logger.info("FaultHandle cancelled: %d event(s)", len(self._events))


@dataclass(frozen=True)
class FaultStats:
    """Summary of fault injection activity.

    Attributes:
        faults_scheduled: Number of faults added to the schedule.
        faults_activated: Number of fault activations that fired.
        faults_deactivated: Number of fault deactivations that fired.
        faults_cancelled: Number of faults cancelled before activation.
    """

    faults_scheduled: int
    faults_activated: int
    faults_deactivated: int
    faults_cancelled: int


@dataclass
class _MutableFaultStats:
    """Internal mutable stats tracker."""

    faults_scheduled: int = 0
    faults_activated: int = 0
    faults_deactivated: int = 0
    faults_cancelled: int = 0

    def freeze(self) -> FaultStats:
        return FaultStats(
            faults_scheduled=self.faults_scheduled,
            faults_activated=self.faults_activated,
            faults_deactivated=self.faults_deactivated,
            faults_cancelled=self.faults_cancelled,
        )
