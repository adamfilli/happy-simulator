"""Resource-level fault injection.

``ReduceCapacity`` temporarily reduces a resource's capacity for a time
window, simulating degraded hardware, throttling, or partial failures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.faults.fault import FaultContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReduceCapacity:
    """Temporarily reduce a resource's capacity.

    At ``start``, multiplies the resource's capacity by ``factor``
    (e.g., 0.5 = halve). At ``end``, restores the original capacity.

    Attributes:
        resource_name: Name of the resource to degrade.
        factor: Capacity multiplier (0 < factor < 1 to reduce).
        start: Fault activation time in seconds.
        end: Fault deactivation time in seconds.
    """

    resource_name: str
    factor: float
    start: float
    end: float

    def generate_events(self, ctx: FaultContext) -> list[Event]:
        resource = ctx.resources[self.resource_name]
        resource_name = self.resource_name
        factor = self.factor
        original_capacity = resource._capacity

        def activate(e: Event) -> None:
            new_capacity = original_capacity * factor
            reduction = resource._capacity - new_capacity
            resource._capacity = new_capacity
            # Clamp available to not exceed new capacity
            if resource._available > new_capacity:
                resource._available = new_capacity
            logger.info(
                "[FaultInjection] Reduced '%s' capacity to %.1f (factor=%.2f) at %s",
                resource_name, new_capacity, factor, e.time,
            )

        def deactivate(e: Event) -> None:
            capacity_increase = original_capacity - resource._capacity
            resource._capacity = original_capacity
            # Restore available by the same amount capacity increased
            resource._available += capacity_increase
            logger.info(
                "[FaultInjection] Restored '%s' capacity to %.1f at %s",
                resource_name, original_capacity, e.time,
            )

        return [
            Event.once(
                time=Instant.from_seconds(self.start),
                event_type=f"fault.capacity.reduce:{resource_name}",
                fn=activate,
                daemon=True,
            ),
            Event.once(
                time=Instant.from_seconds(self.end),
                event_type=f"fault.capacity.restore:{resource_name}",
                fn=deactivate,
                daemon=True,
            ),
        ]
