"""Network topology manager for routing events through links.

Network manages a collection of NetworkLinks forming a topology between
entities. It routes events through the appropriate links based on source
and destination, supports network partitions, and provides a default link
for unconfigured routes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.components.network.link import NetworkLink
    from happysimulator.core.clock import Clock

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkStats:
    """Per-connection-pair traffic statistics.

    Attributes:
        source: Name of the source entity.
        destination: Name of the destination entity.
        packets_sent: Number of packets successfully transmitted.
        packets_dropped: Number of packets dropped (loss).
        bytes_transmitted: Total bytes transmitted.
    """

    source: str = ""
    destination: str = ""
    packets_sent: int = 0
    packets_dropped: int = 0
    bytes_transmitted: int = 0


@dataclass
class Partition:
    """Handle for a network partition, enabling selective healing.

    Returned by ``Network.partition()``. Call ``heal()`` to remove only this
    partition while leaving other partitions intact.

    Attributes:
        pairs: The frozenset pairs (bidirectional) belonging to this partition.
        directed_pairs: The directed tuple pairs (asymmetric) belonging to this partition.
    """

    pairs: frozenset[frozenset[str]]
    directed_pairs: frozenset[tuple[str, str]]
    _network: Network

    @property
    def is_active(self) -> bool:
        """True if any of this partition's pairs are still active."""
        if self.pairs & self._network._partitioned_pairs:
            return True
        return bool(self.directed_pairs & self._network._directed_partitions)

    def heal(self) -> None:
        """Remove only this partition's pairs, leaving others intact."""
        self._network._partitioned_pairs -= self.pairs
        self._network._directed_partitions -= self.directed_pairs
        logger.info(
            "[%s] Selective partition healed: %d bidirectional + %d directed pairs",
            self._network.name,
            len(self.pairs),
            len(self.directed_pairs),
        )


@dataclass
class Network(Entity):
    """Routes events through configured network topology.

    Network acts as a routing layer that directs events through appropriate
    NetworkLinks based on source and destination entities. Events must have
    'source' and 'destination' in their context metadata to be routed.

    Supports network partitions where events between partitioned groups are
    dropped, simulating network failures or split-brain scenarios.

    Attributes:
        name: Identifier for logging and debugging.
        default_link: Link used when no specific route is configured.
    """

    name: str
    default_link: NetworkLink | None = None

    # Routing table: (source_name, dest_name) -> NetworkLink
    _routes: dict[tuple[str, str], NetworkLink] = field(default_factory=dict, init=False)

    # Partition state: set of frozenset pairs (bidirectional)
    _partitioned_pairs: set[frozenset[str]] = field(default_factory=set, init=False)

    # Directed partition state: set of (source, dest) tuples (asymmetric)
    _directed_partitions: set[tuple[str, str]] = field(default_factory=set, init=False)

    # Track all known entities for partition validation
    _known_entities: dict[str, Entity] = field(default_factory=dict, init=False)

    # Statistics
    events_routed: int = field(default=0, init=False)
    events_dropped_no_route: int = field(default=0, init=False)
    events_dropped_partition: int = field(default=0, init=False)

    def set_clock(self, clock: Clock) -> None:
        """Inject the simulation clock and propagate to all links."""
        super().set_clock(clock)

        # Propagate to default link
        if self.default_link is not None:
            self.default_link.set_clock(clock)

        # Propagate to all configured links
        for link in self._routes.values():
            link.set_clock(clock)

    def add_link(self, source: Entity, dest: Entity, link: NetworkLink) -> None:
        """Configure a unidirectional link between two entities.

        Args:
            source: The sending entity.
            dest: The receiving entity.
            link: The NetworkLink to use for this route.
        """
        self._known_entities[source.name] = source
        self._known_entities[dest.name] = dest

        # Configure the link's egress to be the destination
        link.egress = dest

        self._routes[(source.name, dest.name)] = link
        logger.debug(
            "[%s] Added link: %s -> %s via %s",
            self.name,
            source.name,
            dest.name,
            link.name,
        )

    def add_bidirectional_link(self, a: Entity, b: Entity, link: NetworkLink) -> None:
        """Configure a bidirectional link between two entities.

        Creates two routes (a->b and b->a) using the same link characteristics.
        Note: This creates a copy of the link for the reverse direction to
        allow independent statistics tracking.

        Args:
            a: First entity.
            b: Second entity.
            link: The NetworkLink to use (copied for reverse direction).
        """
        import copy

        # Forward direction: a -> b
        forward_link = link
        forward_link.egress = b
        self._routes[(a.name, b.name)] = forward_link

        # Reverse direction: b -> a (create a copy for independent stats)
        reverse_link = copy.copy(link)
        reverse_link.name = f"{link.name}_reverse"
        reverse_link.egress = a
        # Reset stats for the copy
        reverse_link.bytes_transmitted = 0
        reverse_link.packets_sent = 0
        reverse_link.packets_dropped = 0
        reverse_link._bytes_in_flight = 0
        self._routes[(b.name, a.name)] = reverse_link

        self._known_entities[a.name] = a
        self._known_entities[b.name] = b

        logger.debug(
            "[%s] Added bidirectional link: %s <-> %s via %s",
            self.name,
            a.name,
            b.name,
            link.name,
        )

    def partition(
        self,
        group_a: list[Entity],
        group_b: list[Entity],
        *,
        asymmetric: bool = False,
    ) -> Partition:
        """Create a network partition between two groups.

        Events between entities in group_a and entities in group_b will be
        dropped. Events within the same group are unaffected.

        Args:
            group_a: First group of entities.
            group_b: Second group of entities.
            asymmetric: If True, only block traffic from group_a to group_b
                (group_b can still send to group_a). Default is False
                (bidirectional partition).

        Returns:
            A Partition handle that can be used to selectively heal this
            partition without affecting others.
        """
        bidirectional_pairs: set[frozenset[str]] = set()
        directed_pairs: set[tuple[str, str]] = set()

        for entity_a in group_a:
            self._known_entities[entity_a.name] = entity_a
            for entity_b in group_b:
                self._known_entities[entity_b.name] = entity_b
                if asymmetric:
                    directed_pairs.add((entity_a.name, entity_b.name))
                    self._directed_partitions.add((entity_a.name, entity_b.name))
                else:
                    pair = frozenset([entity_a.name, entity_b.name])
                    bidirectional_pairs.add(pair)
                    self._partitioned_pairs.add(pair)

        if asymmetric:
            logger.info(
                "[%s] Asymmetric partition created: %s -X-> %s",
                self.name,
                [e.name for e in group_a],
                [e.name for e in group_b],
            )
        else:
            logger.info(
                "[%s] Network partition created: %s <-X-> %s",
                self.name,
                [e.name for e in group_a],
                [e.name for e in group_b],
            )

        return Partition(
            pairs=frozenset(bidirectional_pairs),
            directed_pairs=frozenset(directed_pairs),
            _network=self,
        )

    def heal_partition(self) -> None:
        """Remove all network partitions, restoring full connectivity."""
        num_bidir = len(self._partitioned_pairs)
        num_directed = len(self._directed_partitions)
        self._partitioned_pairs.clear()
        self._directed_partitions.clear()
        logger.info(
            "[%s] All partitions healed: %d bidirectional + %d directed pairs restored",
            self.name,
            num_bidir,
            num_directed,
        )

    def is_partitioned(self, source_name: str, dest_name: str) -> bool:
        """Check if two entities are separated by a partition.

        Checks both bidirectional (frozenset) and directed (tuple) partitions.

        Args:
            source_name: Name of the source entity.
            dest_name: Name of the destination entity.

        Returns:
            True if the entities are partitioned in the given direction.
        """
        pair = frozenset([source_name, dest_name])
        if pair in self._partitioned_pairs:
            return True
        return (source_name, dest_name) in self._directed_partitions

    def get_link(self, source_name: str, dest_name: str) -> NetworkLink | None:
        """Get the link for a source-destination pair.

        Args:
            source_name: Name of the source entity.
            dest_name: Name of the destination entity.

        Returns:
            The configured link, or the default link if no specific route exists.
        """
        return self._routes.get((source_name, dest_name), self.default_link)

    def traffic_matrix(self) -> list[LinkStats]:
        """Per-connection-pair traffic statistics.

        Iterates all configured routes and collects each link's stats into
        a list of LinkStats.

        Returns:
            A list of LinkStats, one per configured route.
        """
        result: list[LinkStats] = []
        for (src, dst), link in self._routes.items():
            result.append(
                LinkStats(
                    source=src,
                    destination=dst,
                    packets_sent=link.packets_sent,
                    packets_dropped=link.packets_dropped,
                    bytes_transmitted=link.bytes_transmitted,
                )
            )
        return result

    def handle_event(self, event: Event) -> Generator[float, None, list[Event] | Event | None]:
        """Route an event through the appropriate network link.

        The event must have 'source' and 'destination' in its context metadata
        to be routed. If the route is partitioned, the event is dropped.

        Args:
            event: The event to route.

        Yields:
            Delay from the underlying network link.

        Returns:
            Forwarded event(s) or None if dropped.
        """
        metadata = event.context.get("metadata", {})
        source_name = metadata.get("source")
        dest_name = metadata.get("destination")

        # Validate routing metadata
        if source_name is None or dest_name is None:
            logger.warning(
                "[%s] Event missing source/destination metadata: type=%s",
                self.name,
                event.event_type,
            )
            self.events_dropped_no_route += 1
            return None

        # Check for partition
        if self.is_partitioned(source_name, dest_name):
            logger.debug(
                "[%s] Event dropped (partition): %s -> %s, type=%s",
                self.name,
                source_name,
                dest_name,
                event.event_type,
            )
            self.events_dropped_partition += 1
            return None

        # Get the appropriate link
        link = self.get_link(source_name, dest_name)
        if link is None:
            logger.warning(
                "[%s] No route found: %s -> %s, type=%s",
                self.name,
                source_name,
                dest_name,
                event.event_type,
            )
            self.events_dropped_no_route += 1
            return None

        # Ensure the link has a clock
        if link._clock is None and self._clock is not None:
            link.set_clock(self._clock)

        # Route through the link
        logger.debug(
            "[%s] Routing: %s -> %s via %s, type=%s",
            self.name,
            source_name,
            dest_name,
            link.name,
            event.event_type,
        )

        self.events_routed += 1

        # Delegate to the link's handle_event (which is a generator)
        result = link.handle_event(event)

        # If the link returns a generator, yield from it
        if hasattr(result, "__next__"):
            return (yield from result)
        else:
            return result

    def send(
        self,
        source: Entity,
        destination: Entity,
        event_type: str,
        payload: dict | None = None,
        daemon: bool = False,
    ) -> Event:
        """Create an event ready to be sent through the network.

        Convenience method that creates an event with proper routing metadata.

        Args:
            source: The sending entity.
            destination: The receiving entity.
            event_type: Type label for the event.
            payload: Optional additional metadata.
            daemon: Whether the event is a daemon event.

        Returns:
            An Event configured for routing through this network.
        """
        event = Event(
            time=self.now,
            event_type=event_type,
            target=self,
            daemon=daemon,
        )
        event.context["metadata"]["source"] = source.name
        event.context["metadata"]["destination"] = destination.name
        if payload:
            event.context["metadata"].update(payload)
        return event
