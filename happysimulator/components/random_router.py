"""Random router component for distributing requests across targets."""

import random
from collections import defaultdict

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class RandomRouter(Entity):
    """Routes requests randomly across a set of target entities.

    A simple load balancer that distributes incoming events uniformly
    at random to one of the configured targets.

    Args:
        name: Entity name for identification.
        targets: List of entities to route to.
    """

    def __init__(self, name: str, *, targets: list[Entity]):
        super().__init__(name)
        self.targets = targets

        # Stats
        self.stats_routed: int = 0
        self.target_counts: dict[int, int] = defaultdict(int)

    def handle_event(self, event: Event) -> list[Event]:
        """Route event to a randomly selected target."""
        self.stats_routed += 1

        idx = random.randint(0, len(self.targets) - 1)
        self.target_counts[idx] += 1

        routed_event = Event(
            time=self.now,
            event_type=event.event_type,
            target=self.targets[idx],
            context=event.context,
        )
        return [routed_event]
