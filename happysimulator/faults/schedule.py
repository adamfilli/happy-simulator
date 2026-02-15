"""FaultSchedule entity that orchestrates fault injection during simulation.

FaultSchedule collects ``Fault`` objects and generates their activation/
deactivation events when bootstrapped by the simulation. It follows the
same bootstrap pattern as ``Source``: the simulation calls ``start()``
during initialization, which returns initial events for the heap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.faults.fault import (
    Fault,
    FaultContext,
    FaultHandle,
    FaultStats,
    _MutableFaultStats,
)

if TYPE_CHECKING:
    from happysimulator.core.event import Event
    from happysimulator.core.simulation import Simulation
    from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)


class FaultSchedule(Entity):
    """Orchestrates fault injection during simulation.

    Collects faults via ``add()`` and generates their events during
    ``start()``, which is called by the Simulation during initialization.

    Example::

        schedule = FaultSchedule()
        schedule.add(CrashNode("server", at=30.0, restart_at=45.0))
        sim = Simulation(sources=[...], entities=[...], fault_schedule=schedule)

    Args:
        name: Identifier for logging. Defaults to ``"FaultSchedule"``.
    """

    def __init__(self, name: str = "FaultSchedule") -> None:
        super().__init__(name)
        self._faults: list[Fault] = []
        self._handles: list[FaultHandle] = []
        self._stats = _MutableFaultStats()

    def add(self, fault: Fault) -> FaultHandle:
        """Register a fault for injection.

        Args:
            fault: The fault to schedule.

        Returns:
            A handle that can be used to cancel the fault before activation.
        """
        handle = FaultHandle(fault)
        self._faults.append(fault)
        self._handles.append(handle)
        self._stats.faults_scheduled += 1
        return handle

    def start(self, start_time: Instant, sim: Simulation) -> list[Event]:
        """Generate fault events by resolving entity/network/resource references.

        Called by ``Simulation.__init__()`` during bootstrap.

        Args:
            start_time: The simulation's start time.
            sim: The simulation instance (used to resolve names).

        Returns:
            All fault events to push onto the heap.
        """
        ctx = self._build_context(start_time, sim)
        all_events: list[Event] = []

        for fault, handle in zip(self._faults, self._handles, strict=False):
            fault_events = fault.generate_events(ctx)
            handle._events = fault_events
            all_events.extend(fault_events)
            logger.debug(
                "[%s] Fault %s generated %d event(s)",
                self.name,
                type(fault).__name__,
                len(fault_events),
            )

        logger.info(
            "[%s] Started with %d fault(s), %d total event(s)",
            self.name,
            len(self._faults),
            len(all_events),
        )
        return all_events

    @property
    def stats(self) -> FaultStats:
        """Frozen snapshot of fault injection statistics."""
        # Update cancelled count from handles
        self._stats.faults_cancelled = sum(1 for h in self._handles if h.cancelled)
        return self._stats.freeze()

    def handle_event(self, event: Event) -> None:
        """FaultSchedule does not process events itself."""

    def _build_context(self, start_time: Instant, sim: Simulation) -> FaultContext:
        """Build a FaultContext from the simulation's registered components."""
        from happysimulator.components.network.network import Network
        from happysimulator.components.resource import Resource

        entities: dict[str, Entity] = {}
        networks: dict[str, Network] = {}
        resources: dict[str, Resource] = {}

        all_components = list(sim._entities) + list(sim._sources) + list(sim._probes)
        for component in all_components:
            if isinstance(component, Entity):
                entities[component.name] = component
            if isinstance(component, Network):
                networks[component.name] = component
            if isinstance(component, Resource):
                resources[component.name] = component

        return FaultContext(
            entities=entities,
            networks=networks,
            resources=resources,
            start_time=start_time,
        )
