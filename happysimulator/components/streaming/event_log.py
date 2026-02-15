"""Append-only partitioned event log.

Models a Kafka-inspired partitioned log where records are appended to
partitions via a configurable sharding strategy. Supports retention
policies (time-based and size-based) and provides convenience generators
for ``yield from`` composition.

Example::

    from happysimulator.components.streaming import EventLog, TimeRetention

    log = EventLog(
        name="orders",
        num_partitions=4,
        retention_policy=TimeRetention(max_age_s=300.0),
    )

    # In an entity's handle_event:
    record = yield from log.append("user-42", {"action": "buy"})
    records = yield from log.read(partition_id=0, offset=0, max_records=10)
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from happysimulator.components.datastore.sharded_store import HashSharding
from happysimulator.core.entity import Entity
from happysimulator.core.event import Event
from happysimulator.core.sim_future import SimFuture

if TYPE_CHECKING:
    from collections.abc import Generator

logger = logging.getLogger(__name__)


class RetentionPolicy(Protocol):
    """Protocol for log retention policies."""

    @abstractmethod
    def should_retain(self, record: Record, current_time_s: float) -> bool:
        """Whether a record should be kept.

        Args:
            record: The record to evaluate.
            current_time_s: Current simulation time in seconds.

        Returns:
            True if the record should be retained.
        """
        ...


@dataclass(frozen=True)
class Record:
    """An immutable record in the event log.

    Attributes:
        offset: Position within the partition.
        key: Routing key used for partition assignment.
        value: The record payload.
        timestamp: Simulation time when the record was appended.
        partition: Partition ID this record belongs to.
    """

    offset: int
    key: str
    value: Any
    timestamp: float
    partition: int


@dataclass
class Partition:
    """Internal state of a single partition.

    Attributes:
        id: Partition identifier.
        records: Ordered list of records.
        high_watermark: Offset of next record to be written.
    """

    id: int
    records: list[Record] = field(default_factory=list)
    high_watermark: int = 0


class TimeRetention:
    """Retain records younger than a maximum age.

    Args:
        max_age_s: Maximum record age in seconds.
    """

    def __init__(self, max_age_s: float):
        if max_age_s <= 0:
            raise ValueError(f"max_age_s must be > 0, got {max_age_s}")
        self._max_age_s = max_age_s

    @property
    def max_age_s(self) -> float:
        return self._max_age_s

    def should_retain(self, record: Record, current_time_s: float) -> bool:
        return (current_time_s - record.timestamp) < self._max_age_s


class SizeRetention:
    """Retain at most *max_records* per partition.

    When a partition exceeds this limit, the oldest records are expired
    during the retention check cycle.

    Args:
        max_records: Maximum number of records per partition.
    """

    def __init__(self, max_records: int):
        if max_records <= 0:
            raise ValueError(f"max_records must be > 0, got {max_records}")
        self._max_records = max_records

    @property
    def max_records(self) -> int:
        return self._max_records

    def should_retain(self, record: Record, current_time_s: float) -> bool:
        # Evaluated per-partition by EventLog; not standalone.
        # The EventLog trims from oldest when count exceeds max.
        return True  # pragma: no cover — trimming handled in EventLog


@dataclass(frozen=True)
class EventLogStats:
    """Statistics tracked by EventLog.

    Attributes:
        records_appended: Total records appended across all partitions.
        records_read: Total records returned by read operations.
        records_expired: Total records removed by retention.
        per_partition_appends: Append count per partition.
        append_latencies: Tuple of individual append latencies.
    """

    records_appended: int = 0
    records_read: int = 0
    records_expired: int = 0
    per_partition_appends: dict[int, int] = field(default_factory=dict)
    append_latencies: tuple[float, ...] = ()

    @property
    def avg_append_latency(self) -> float:
        if not self.append_latencies:
            return 0.0
        return sum(self.append_latencies) / len(self.append_latencies)


class EventLog(Entity):
    """Append-only partitioned event log.

    Records are routed to partitions via a sharding strategy (default:
    hash-based). Each partition maintains an ordered list of records and
    a monotonically increasing high-watermark offset.

    Attributes:
        name: Entity name for identification.
    """

    def __init__(
        self,
        name: str,
        num_partitions: int = 4,
        sharding_strategy: Any | None = None,
        retention_policy: RetentionPolicy | None = None,
        append_latency: float = 0.001,
        read_latency: float = 0.0005,
        retention_check_interval: float = 60.0,
    ):
        """Initialize the event log.

        Args:
            name: Name for this log entity.
            num_partitions: Number of partitions.
            sharding_strategy: Strategy for routing keys to partitions.
                Defaults to HashSharding.
            retention_policy: Optional policy for expiring old records.
            append_latency: Simulated append latency in seconds.
            read_latency: Simulated read latency in seconds.
            retention_check_interval: Seconds between retention sweeps.

        Raises:
            ValueError: If num_partitions < 1.
        """
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")

        super().__init__(name)
        self._num_partitions = num_partitions
        self._sharding = sharding_strategy or HashSharding()
        self._retention_policy = retention_policy
        self._append_latency = append_latency
        self._read_latency = read_latency
        self._retention_check_interval = retention_check_interval

        self._partitions: list[Partition] = [Partition(id=i) for i in range(num_partitions)]

        self._retention_scheduled = False

        self._records_appended = 0
        self._records_read = 0
        self._records_expired = 0
        self._per_partition_appends: dict[int, int] = {}
        self._append_latencies: list[float] = []
        for i in range(num_partitions):
            self._per_partition_appends[i] = 0

    @property
    def stats(self) -> EventLogStats:
        """Return a frozen snapshot of current statistics."""
        return EventLogStats(
            records_appended=self._records_appended,
            records_read=self._records_read,
            records_expired=self._records_expired,
            per_partition_appends=dict(self._per_partition_appends),
            append_latencies=tuple(self._append_latencies),
        )

    @property
    def num_partitions(self) -> int:
        """Number of partitions."""
        return self._num_partitions

    @property
    def partitions(self) -> list[Partition]:
        """The partition objects."""
        return list(self._partitions)

    def high_watermark(self, partition_id: int) -> int:
        """High watermark for a partition.

        Args:
            partition_id: Partition index.

        Returns:
            The next offset that will be assigned.
        """
        return self._partitions[partition_id].high_watermark

    def high_watermarks(self) -> dict[int, int]:
        """High watermarks for all partitions."""
        return {p.id: p.high_watermark for p in self._partitions}

    @property
    def total_records(self) -> int:
        """Total records across all partitions."""
        return sum(len(p.records) for p in self._partitions)

    def _get_partition_for_key(self, key: str) -> int:
        """Route a key to a partition index."""
        return self._sharding.get_shard(key, self._num_partitions)

    def append(
        self, key: str, value: Any
    ) -> Generator[float | SimFuture | tuple[float, list[Event]], None, Record]:
        """Append a record to the log.

        Convenience generator for ``yield from`` in entity handlers.

        Args:
            key: Routing key.
            value: Record payload.

        Yields:
            Append latency.

        Returns:
            The appended Record.
        """
        reply = SimFuture()
        event = Event(
            time=self.now,
            event_type="Append",
            target=self,
            context={"key": key, "value": value, "reply_future": reply},
        )
        yield 0.0, [event]
        result = yield reply
        return result

    def read(
        self, partition_id: int, offset: int = 0, max_records: int = 100
    ) -> Generator[float | SimFuture | tuple[float, list[Event]], None, list[Record]]:
        """Read records from a partition.

        Convenience generator for ``yield from`` in entity handlers.

        Args:
            partition_id: Partition to read from.
            offset: Starting offset.
            max_records: Maximum records to return.

        Yields:
            Read latency.

        Returns:
            List of records.
        """
        reply = SimFuture()
        event = Event(
            time=self.now,
            event_type="Read",
            target=self,
            context={
                "partition": partition_id,
                "offset": offset,
                "max_records": max_records,
                "reply_future": reply,
            },
        )
        yield 0.0, [event]
        result = yield reply
        return result

    def _do_append(self, key: str, value: Any) -> Record:
        """Internal append without latency."""
        pid = self._get_partition_for_key(key)
        partition = self._partitions[pid]
        now_s = self.now.to_seconds()

        record = Record(
            offset=partition.high_watermark,
            key=key,
            value=value,
            timestamp=now_s,
            partition=pid,
        )
        partition.records.append(record)
        partition.high_watermark += 1

        self._records_appended += 1
        self._per_partition_appends[pid] = self._per_partition_appends.get(pid, 0) + 1

        return record

    def _do_read(self, partition_id: int, offset: int, max_records: int) -> list[Record]:
        """Internal read without latency."""
        if partition_id < 0 or partition_id >= self._num_partitions:
            return []

        partition = self._partitions[partition_id]
        result = []
        for rec in partition.records:
            if rec.offset >= offset:
                result.append(rec)
                if len(result) >= max_records:
                    break

        self._records_read += len(result)
        return result

    def _apply_retention(self) -> int:
        """Apply retention policy across all partitions.

        Returns:
            Number of records expired.
        """
        if self._retention_policy is None:
            return 0

        now_s = self.now.to_seconds()
        total_expired = 0

        if isinstance(self._retention_policy, SizeRetention):
            for partition in self._partitions:
                excess = len(partition.records) - self._retention_policy.max_records
                if excess > 0:
                    partition.records = partition.records[excess:]
                    total_expired += excess
        else:
            for partition in self._partitions:
                before = len(partition.records)
                partition.records = [
                    r for r in partition.records if self._retention_policy.should_retain(r, now_s)
                ]
                total_expired += before - len(partition.records)

        self._records_expired += total_expired
        return total_expired

    def handle_event(self, event: Event) -> Generator[float, None, list[Event] | None]:
        """Handle log events."""
        event_type = event.event_type

        if event_type == "Append":
            key = event.context.get("key", "")
            value = event.context.get("value")
            reply_future: SimFuture | None = event.context.get("reply_future")

            yield self._append_latency
            self._append_latencies.append(self._append_latency)

            record = self._do_append(key, value)

            if reply_future is not None:
                reply_future.resolve(record)

            # Schedule retention daemon on first append
            if not self._retention_scheduled and self._retention_policy is not None:
                self._retention_scheduled = True
                return [
                    Event(
                        time=self.now,
                        event_type="RetentionCheck",
                        target=self,
                    )
                ]

            return None

        elif event_type == "Read":
            partition_id = event.context.get("partition", 0)
            offset = event.context.get("offset", 0)
            max_records = event.context.get("max_records", 100)
            reply_future = event.context.get("reply_future")

            yield self._read_latency

            records = self._do_read(partition_id, offset, max_records)

            if reply_future is not None:
                reply_future.resolve(records)

            return None

        elif event_type == "RetentionCheck":
            self._apply_retention()

            # Reschedule
            from happysimulator.core.temporal import Instant

            next_time = Instant.from_seconds(self.now.to_seconds() + self._retention_check_interval)
            return [
                Event(
                    time=next_time,
                    event_type="RetentionCheck",
                    target=self,
                )
            ]

        return None
