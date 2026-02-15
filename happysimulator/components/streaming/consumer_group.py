"""Consumer group with partition assignment and rebalancing.

Models a Kafka-inspired consumer group that coordinates partition
ownership across consumers, tracks committed offsets, and triggers
rebalancing on join/leave.

Example::

    from happysimulator.components.streaming import (
        EventLog,
        ConsumerGroup,
        RoundRobinAssignment,
    )

    log = EventLog(name="events", num_partitions=4)
    group = ConsumerGroup(
        name="my-group",
        event_log=log,
        assignment_strategy=RoundRobinAssignment(),
    )

    # In entity handle_event:
    partitions = yield from group.join("consumer-1", self)
    records = yield from group.poll("consumer-1", max_records=10)
    yield from group.commit("consumer-1", {0: 5, 1: 3})
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture

if TYPE_CHECKING:
    from collections.abc import Generator

    from happysimulator.components.streaming.event_log import EventLog, Record

logger = logging.getLogger(__name__)


class PartitionAssignment(Protocol):
    """Protocol for partition assignment strategies."""

    @abstractmethod
    def assign(self, partitions: list[int], consumers: list[str]) -> dict[str, list[int]]:
        """Assign partitions to consumers.

        Args:
            partitions: Available partition IDs.
            consumers: Consumer names (sorted).

        Returns:
            Mapping of consumer name to assigned partition IDs.
        """
        ...


class RangeAssignment:
    """Contiguous range assignment.

    Divides partitions into contiguous ranges. Remainder partitions
    go to the earliest consumers.
    """

    def assign(self, partitions: list[int], consumers: list[str]) -> dict[str, list[int]]:
        if not consumers:
            return {}

        sorted_parts = sorted(partitions)
        sorted_consumers = sorted(consumers)
        n = len(sorted_parts)
        c = len(sorted_consumers)

        result: dict[str, list[int]] = {name: [] for name in sorted_consumers}
        base = n // c
        remainder = n % c

        idx = 0
        for i, name in enumerate(sorted_consumers):
            count = base + (1 if i < remainder else 0)
            result[name] = sorted_parts[idx : idx + count]
            idx += count

        return result


class RoundRobinAssignment:
    """Round-robin partition assignment.

    Distributes partitions in a round-robin fashion across consumers.
    """

    def assign(self, partitions: list[int], consumers: list[str]) -> dict[str, list[int]]:
        if not consumers:
            return {}

        sorted_parts = sorted(partitions)
        sorted_consumers = sorted(consumers)

        result: dict[str, list[int]] = {name: [] for name in sorted_consumers}
        for i, pid in enumerate(sorted_parts):
            consumer = sorted_consumers[i % len(sorted_consumers)]
            result[consumer].append(pid)

        return result


class StickyAssignment:
    """Sticky partition assignment.

    Tracks the previous assignment and minimizes partition movement
    when consumers join or leave.
    """

    def __init__(self):
        self._previous: dict[str, list[int]] = {}

    def assign(self, partitions: list[int], consumers: list[str]) -> dict[str, list[int]]:
        if not consumers:
            self._previous = {}
            return {}

        sorted_consumers = sorted(consumers)
        all_parts = set(partitions)
        result: dict[str, list[int]] = {name: [] for name in sorted_consumers}

        # Keep existing assignments for consumers that are still present
        assigned: set[int] = set()
        for name in sorted_consumers:
            if name in self._previous:
                kept = [p for p in self._previous[name] if p in all_parts]
                result[name] = kept
                assigned.update(kept)

        # Distribute unassigned partitions round-robin to least-loaded consumers
        unassigned = sorted(all_parts - assigned)
        for pid in unassigned:
            # Find consumer with fewest partitions
            target = min(sorted_consumers, key=lambda n: len(result[n]))
            result[target].append(pid)

        # Sort each consumer's partitions
        for name in result:
            result[name] = sorted(result[name])

        self._previous = {k: list(v) for k, v in result.items()}
        return result


class ConsumerState(Enum):
    """State of a consumer within the group."""

    ACTIVE = "active"
    REBALANCING = "rebalancing"


@dataclass(frozen=True)
class ConsumerGroupStats:
    """Statistics tracked by ConsumerGroup.

    Attributes:
        joins: Number of consumer joins.
        leaves: Number of consumer leaves.
        rebalances: Number of rebalance events.
        polls: Number of poll operations.
        commits: Number of commit operations.
        records_polled: Total records returned by polls.
    """

    joins: int = 0
    leaves: int = 0
    rebalances: int = 0
    polls: int = 0
    commits: int = 0
    records_polled: int = 0


class ConsumerGroup(Entity):
    """Coordinated consumer group over an EventLog.

    Manages partition assignment, offset tracking, and rebalancing
    when consumers join or leave.

    Attributes:
        name: Entity name for identification.
    """

    def __init__(
        self,
        name: str,
        event_log: EventLog,
        assignment_strategy: PartitionAssignment | None = None,
        rebalance_delay: float = 0.5,
        poll_latency: float = 0.001,
        session_timeout: float | None = None,
    ):
        """Initialize the consumer group.

        Args:
            name: Name for this group entity.
            event_log: The EventLog to consume from.
            assignment_strategy: Strategy for assigning partitions.
                Defaults to RangeAssignment.
            rebalance_delay: Simulated rebalance latency in seconds.
            poll_latency: Simulated poll latency in seconds.
            session_timeout: Optional session timeout (not enforced in
                this implementation, reserved for future use).
        """
        super().__init__(name)
        self._event_log = event_log
        self._strategy = assignment_strategy or RangeAssignment()
        self._rebalance_delay = rebalance_delay
        self._poll_latency = poll_latency
        self._session_timeout = session_timeout

        # Consumer state
        self._consumers: dict[str, Entity] = {}  # name -> entity
        self._assignments: dict[str, list[int]] = {}  # name -> partition IDs
        self._committed_offsets: dict[str, dict[int, int]] = {}  # name -> {pid: offset}
        self._generation: int = 0

        self._joins = 0
        self._leaves = 0
        self._rebalances = 0
        self._polls = 0
        self._commits = 0
        self._records_polled = 0

    @property
    def stats(self) -> ConsumerGroupStats:
        """Return a frozen snapshot of current statistics."""
        return ConsumerGroupStats(
            joins=self._joins,
            leaves=self._leaves,
            rebalances=self._rebalances,
            polls=self._polls,
            commits=self._commits,
            records_polled=self._records_polled,
        )

    @property
    def consumer_count(self) -> int:
        """Number of active consumers."""
        return len(self._consumers)

    @property
    def consumers(self) -> list[str]:
        """Names of active consumers."""
        return sorted(self._consumers.keys())

    @property
    def assignments(self) -> dict[str, list[int]]:
        """Current partition assignments."""
        return {k: list(v) for k, v in self._assignments.items()}

    @property
    def generation(self) -> int:
        """Current group generation (incremented on each rebalance)."""
        return self._generation

    def consumer_lag(self, consumer_name: str) -> dict[int, int]:
        """Per-partition lag for a consumer.

        Lag = event_log high watermark - committed offset.

        Args:
            consumer_name: The consumer to check.

        Returns:
            Mapping of partition ID to lag.
        """
        if consumer_name not in self._assignments:
            return {}

        offsets = self._committed_offsets.get(consumer_name, {})
        lag: dict[int, int] = {}
        for pid in self._assignments[consumer_name]:
            hw = self._event_log.high_watermark(pid)
            committed = offsets.get(pid, 0)
            lag[pid] = hw - committed
        return lag

    def total_lag(self) -> int:
        """Total lag across all consumers and partitions."""
        total = 0
        for name in self._consumers:
            lag = self.consumer_lag(name)
            total += sum(lag.values())
        return total

    def _rebalance(self) -> None:
        """Recalculate partition assignments."""
        self._generation += 1
        partitions = list(range(self._event_log.num_partitions))
        consumer_names = sorted(self._consumers.keys())
        self._assignments = self._strategy.assign(partitions, consumer_names)
        self._rebalances += 1

    # Convenience generators for yield-from composition

    def join(
        self, consumer_name: str, consumer_entity: Entity
    ) -> Generator[float | SimFuture | tuple[float, list[Event]], None, list[int]]:
        """Join a consumer to the group.

        Args:
            consumer_name: Unique name for the consumer.
            consumer_entity: The entity to receive polled records.

        Yields:
            Rebalance delay.

        Returns:
            List of assigned partition IDs.
        """
        reply = SimFuture()
        event = Event(
            time=self.now,
            event_type="Join",
            target=self,
            context={
                "consumer_name": consumer_name,
                "consumer_entity": consumer_entity,
                "reply_future": reply,
            },
        )
        yield 0.0, [event]
        result = yield reply
        return result

    def leave(self, consumer_name: str) -> Generator[float | SimFuture | tuple[float, list[Event]]]:
        """Remove a consumer from the group.

        Args:
            consumer_name: Name of the consumer to remove.

        Yields:
            Rebalance delay.
        """
        reply = SimFuture()
        event = Event(
            time=self.now,
            event_type="Leave",
            target=self,
            context={
                "consumer_name": consumer_name,
                "reply_future": reply,
            },
        )
        yield 0.0, [event]
        yield reply

    def poll(
        self, consumer_name: str, max_records: int = 100
    ) -> Generator[float | SimFuture | tuple[float, list[Event]], None, list[Record]]:
        """Poll for records from assigned partitions.

        Args:
            consumer_name: The polling consumer.
            max_records: Maximum records to return.

        Yields:
            Poll latency.

        Returns:
            List of records from assigned partitions.
        """
        reply = SimFuture()
        event = Event(
            time=self.now,
            event_type="Poll",
            target=self,
            context={
                "consumer_name": consumer_name,
                "max_records": max_records,
                "reply_future": reply,
            },
        )
        yield 0.0, [event]
        result = yield reply
        return result

    def commit(
        self, consumer_name: str, offsets: dict[int, int]
    ) -> Generator[float | tuple[float, list[Event]]]:
        """Commit offsets for a consumer.

        Args:
            consumer_name: The committing consumer.
            offsets: Mapping of partition ID to committed offset.

        Yields:
            Minimal latency.
        """
        event = Event(
            time=self.now,
            event_type="Commit",
            target=self,
            context={
                "consumer_name": consumer_name,
                "offsets": offsets,
            },
        )
        yield 0.0, [event]

    def handle_event(self, event: Event) -> Generator[float, None, list[Event] | None]:
        """Handle consumer group events."""
        event_type = event.event_type

        if event_type == "Join":
            consumer_name = event.context.get("consumer_name", "")
            consumer_entity = event.context.get("consumer_entity")
            reply_future: SimFuture | None = event.context.get("reply_future")

            self._consumers[consumer_name] = consumer_entity
            if consumer_name not in self._committed_offsets:
                self._committed_offsets[consumer_name] = {}
            self._joins += 1

            yield self._rebalance_delay
            self._rebalance()

            assigned = self._assignments.get(consumer_name, [])
            if reply_future is not None:
                reply_future.resolve(assigned)

            return None

        elif event_type == "Leave":
            consumer_name = event.context.get("consumer_name", "")
            reply_future = event.context.get("reply_future")

            self._consumers.pop(consumer_name, None)
            self._assignments.pop(consumer_name, None)
            # Keep committed offsets for potential rejoin
            self._leaves += 1

            yield self._rebalance_delay
            self._rebalance()

            if reply_future is not None:
                reply_future.resolve(None)

            return None

        elif event_type == "Poll":
            consumer_name = event.context.get("consumer_name", "")
            max_records = event.context.get("max_records", 100)
            reply_future = event.context.get("reply_future")

            yield self._poll_latency

            assigned = self._assignments.get(consumer_name, [])
            offsets = self._committed_offsets.get(consumer_name, {})
            records: list[Record] = []

            for pid in assigned:
                offset = offsets.get(pid, 0)
                remaining = max_records - len(records)
                if remaining <= 0:
                    break
                partition_records = self._event_log._do_read(pid, offset, remaining)
                records.extend(partition_records)

            self._polls += 1
            self._records_polled += len(records)

            if reply_future is not None:
                reply_future.resolve(records)

            return None

        elif event_type == "Commit":
            consumer_name = event.context.get("consumer_name", "")
            offsets = event.context.get("offsets", {})

            if consumer_name not in self._committed_offsets:
                self._committed_offsets[consumer_name] = {}

            for pid, offset in offsets.items():
                self._committed_offsets[consumer_name][pid] = offset

            self._commits += 1
            return None

        return None
