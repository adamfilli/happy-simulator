"""EventProvider that samples event context fields from distributions.

DistributedFieldProvider creates events where specified context fields
are sampled from ValueDistribution instances. This enables generating
realistic load patterns where properties like customer_id, region, or
endpoint follow distributions like Zipf (power-law).

Example:
    # Generate requests with Zipf-distributed customer IDs
    provider = DistributedFieldProvider(
        target=server,
        event_type="Request",
        field_distributions={
            "customer_id": ZipfDistribution(range(1000), s=1.0),
            "region": UniformDistribution(["us-east", "us-west", "eu"]),
        },
    )
"""

from typing import Any

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.temporal import Instant
from happysimulator.distributions.value_distribution import ValueDistribution
from happysimulator.load.event_provider import EventProvider


class DistributedFieldProvider(EventProvider):
    """EventProvider that samples event context fields from distributions.

    Creates events where specified context fields are dynamically sampled
    from ValueDistribution instances at each tick. Supports mixing distributed
    fields with static constant values.

    Args:
        target: Target entity for generated events.
        event_type: Type string for generated events.
        field_distributions: Dict mapping field names to ValueDistributions.
                            Each field will be sampled independently.
        static_fields: Dict of constant field values (optional).
        stop_after: Stop generating events after this time (optional).

    Attributes:
        generated: Count of events generated.

    Example:
        provider = DistributedFieldProvider(
            target=router,
            event_type="Request",
            field_distributions={
                "customer_id": ZipfDistribution(range(1000), s=1.0),
                "region": UniformDistribution(["us-east", "us-west", "eu"]),
            },
            static_fields={
                "api_version": "v2",
            },
        )
    """

    def __init__(
        self,
        target: Entity,
        event_type: str,
        field_distributions: dict[str, ValueDistribution],
        static_fields: dict[str, Any] | None = None,
        stop_after: Instant | None = None,
    ):
        """Initialize the provider.

        Args:
            target: Entity to receive generated events.
            event_type: Event type string.
            field_distributions: Mapping of field names to distributions.
            static_fields: Constant values to include in context.
            stop_after: Optional time after which to stop generating.
        """
        self._target = target
        self._event_type = event_type
        self._field_dists = field_distributions
        self._static_fields = static_fields or {}
        self._stop_after = stop_after
        self.generated: int = 0

    def get_events(self, time: Instant) -> list[Event]:
        """Generate an event with sampled field values.

        Args:
            time: Current simulation time.

        Returns:
            List containing a single event, or empty list if past stop_after.
        """
        if self._stop_after is not None and time > self._stop_after:
            return []

        self.generated += 1

        # Build context with static fields first
        context: dict[str, Any] = dict(self._static_fields)
        context["created_at"] = time

        # Sample each distributed field
        for field_name, distribution in self._field_dists.items():
            context[field_name] = distribution.sample()

        return [
            Event(
                time=time,
                event_type=self._event_type,
                target=self._target,
                context=context,
            )
        ]

    @property
    def target(self) -> Entity:
        """Return the target entity."""
        return self._target

    @property
    def event_type(self) -> str:
        """Return the event type string."""
        return self._event_type

    def __repr__(self) -> str:
        fields = list(self._field_dists.keys())
        return f"DistributedFieldProvider(event_type={self._event_type!r}, fields={fields})"
