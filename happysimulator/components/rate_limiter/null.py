"""Pass-through rate limiter that forwards all events.

NullRateLimiter implements the rate limiter interface but performs no
rate limiting. Use it as a placeholder in pipelines where rate limiting
may be optionally added, or as a baseline for comparison.
"""

from happysimulator.core.clock import Clock
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class NullRateLimiter(Entity):
    """Pass-through rate limiter that forwards all events immediately.

    This is a no-op rate limiter useful for:
    - Providing a default when rate limiting is optional
    - Baseline comparisons in experiments
    - Simplifying pipeline construction

    All events are forwarded to the downstream entity without delay.
    """

    def __init__(self, name: str, downstream: Entity):
        """Initialize the pass-through rate limiter.

        Args:
            name: Name of this entity.
            downstream: Entity to forward events to.
        """
        super().__init__(name)
        self._downstream = downstream

    @property
    def downstream(self) -> Entity:
        """The downstream entity receiving forwarded events."""
        return self._downstream

    def set_clock(self, clock: Clock) -> None:
        """Inject clock and propagate to downstream."""
        super().set_clock(clock)
        self._downstream.set_clock(clock)

    def handle_event(self, event: Event) -> list[Event]:
        """Forward the event to downstream immediately."""
        return [
            Event(
                time=event.time,
                event_type=event.event_type,
                target=self._downstream,
                context=event.context,
            )
        ]
