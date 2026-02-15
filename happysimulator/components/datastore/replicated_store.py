"""Replicated store implementation.

Provides a distributed key-value store with configurable consistency levels.
Replicates data across multiple replicas and uses quorum-based operations.

Example:
    from happysimulator.components.datastore import (
        KVStore, ReplicatedStore, ConsistencyLevel
    )

    # Create replica nodes
    replicas = [
        KVStore(name=f"node{i}", read_latency=0.005, write_latency=0.010)
        for i in range(3)
    ]

    # Create replicated store with quorum consistency
    store = ReplicatedStore(
        name="distributed_db",
        replicas=replicas,
        read_consistency=ConsistencyLevel.QUORUM,
        write_consistency=ConsistencyLevel.QUORUM,
    )
"""

from dataclasses import dataclass, field
from typing import Any, Generator, Optional
from enum import Enum
import heapq

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class ConsistencyLevel(Enum):
    """Consistency level for distributed operations."""

    ONE = "one"  # Only one replica required
    QUORUM = "quorum"  # Majority of replicas required
    ALL = "all"  # All replicas required


@dataclass(frozen=True)
class ReplicatedStoreStats:
    """Statistics tracked by ReplicatedStore."""

    reads: int = 0
    writes: int = 0
    read_successes: int = 0
    read_failures: int = 0
    write_successes: int = 0
    write_failures: int = 0
    replica_timeouts: int = 0
    read_latencies: list[float] = field(default_factory=list)
    write_latencies: list[float] = field(default_factory=list)

    @property
    def read_latency_p50(self) -> float:
        """50th percentile read latency."""
        if not self.read_latencies:
            return 0.0
        sorted_latencies = sorted(self.read_latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def read_latency_p99(self) -> float:
        """99th percentile read latency."""
        if not self.read_latencies:
            return 0.0
        sorted_latencies = sorted(self.read_latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def write_latency_p50(self) -> float:
        """50th percentile write latency."""
        if not self.write_latencies:
            return 0.0
        sorted_latencies = sorted(self.write_latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    @property
    def write_latency_p99(self) -> float:
        """99th percentile write latency."""
        if not self.write_latencies:
            return 0.0
        sorted_latencies = sorted(self.write_latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


class ReplicatedStore(Entity):
    """Distributed key-value store with replication.

    Replicates data across multiple nodes and uses configurable
    consistency levels for read and write operations.

    Quorum = floor(n/2) + 1, where n is the number of replicas.

    For strong consistency, use QUORUM for both reads and writes
    (R + W > N guarantees seeing most recent write).

    Attributes:
        name: Entity name for identification.
        num_replicas: Number of replica nodes.
        read_consistency: Consistency level for reads.
        write_consistency: Consistency level for writes.
    """

    def __init__(
        self,
        name: str,
        replicas: list[Entity],
        read_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
        write_consistency: ConsistencyLevel = ConsistencyLevel.QUORUM,
        read_timeout: float = 1.0,
        write_timeout: float = 2.0,
    ):
        """Initialize the replicated store.

        Args:
            name: Name for this store entity.
            replicas: List of replica KVStore nodes.
            read_consistency: Consistency level for read operations.
            write_consistency: Consistency level for write operations.
            read_timeout: Timeout for read operations in seconds.
            write_timeout: Timeout for write operations in seconds.

        Raises:
            ValueError: If insufficient replicas for consistency level.
        """
        if not replicas:
            raise ValueError("At least one replica is required")

        super().__init__(name)
        self._replicas = replicas
        self._read_consistency = read_consistency
        self._write_consistency = write_consistency
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout

        # Validate consistency levels
        self._validate_consistency()

        # Statistics
        self._reads = 0
        self._writes = 0
        self._read_successes = 0
        self._read_failures = 0
        self._write_successes = 0
        self._write_failures = 0
        self._replica_timeouts = 0
        self._read_latencies: list[float] = []
        self._write_latencies: list[float] = []

    @property
    def stats(self) -> ReplicatedStoreStats:
        """Frozen snapshot of replicated store statistics."""
        return ReplicatedStoreStats(
            reads=self._reads,
            writes=self._writes,
            read_successes=self._read_successes,
            read_failures=self._read_failures,
            write_successes=self._write_successes,
            write_failures=self._write_failures,
            replica_timeouts=self._replica_timeouts,
            read_latencies=list(self._read_latencies),
            write_latencies=list(self._write_latencies),
        )

    @property
    def num_replicas(self) -> int:
        """Number of replica nodes."""
        return len(self._replicas)

    @property
    def replicas(self) -> list[Entity]:
        """The replica nodes."""
        return self._replicas

    @property
    def read_consistency(self) -> ConsistencyLevel:
        """Consistency level for reads."""
        return self._read_consistency

    @property
    def write_consistency(self) -> ConsistencyLevel:
        """Consistency level for writes."""
        return self._write_consistency

    @property
    def quorum_size(self) -> int:
        """Number of replicas needed for quorum."""
        return len(self._replicas) // 2 + 1

    def _validate_consistency(self) -> None:
        """Validate that consistency levels are achievable."""
        n = len(self._replicas)
        if self._read_consistency == ConsistencyLevel.ALL and n == 0:
            raise ValueError("Cannot use ALL consistency with zero replicas")
        if self._write_consistency == ConsistencyLevel.ALL and n == 0:
            raise ValueError("Cannot use ALL consistency with zero replicas")

    def _required_responses(self, consistency: ConsistencyLevel) -> int:
        """Get number of responses required for given consistency."""
        if consistency == ConsistencyLevel.ONE:
            return 1
        elif consistency == ConsistencyLevel.QUORUM:
            return self.quorum_size
        else:  # ALL
            return len(self._replicas)

    def get(self, key: str) -> Generator[float, None, Optional[Any]]:
        """Get a value with configured read consistency.

        Reads from multiple replicas in parallel and returns the
        value once enough responses are received.

        Args:
            key: The key to look up.

        Yields:
            Read latency delays.

        Returns:
            The value if found, None otherwise.
        """
        self._reads += 1
        required = self._required_responses(self._read_consistency)
        start_time = 0.0
        total_latency = 0.0

        # Collect responses from replicas
        responses: list[Any] = []
        latencies: list[float] = []

        for replica in self._replicas:
            try:
                gen = replica.get(key)
                value = None
                replica_latency = 0.0
                try:
                    while True:
                        delay = next(gen)
                        replica_latency += delay
                        yield delay
                except StopIteration as e:
                    value = e.value

                latencies.append(replica_latency)
                responses.append(value)

                # Check if we have enough responses
                if len(responses) >= required:
                    # Use the first non-None value (simple conflict resolution)
                    for resp in responses:
                        if resp is not None:
                            self._read_successes += 1
                            total_latency = min(latencies[:required])
                            self._read_latencies.append(total_latency)
                            return resp

            except Exception:
                self._replica_timeouts += 1
                continue

        # Check if we met the consistency requirement
        if len(responses) >= required:
            self._read_successes += 1
            if latencies:
                self._read_latencies.append(min(latencies))
            # All responses were None
            return None
        else:
            self._read_failures += 1
            return None

    def put(self, key: str, value: Any) -> Generator[float, None, bool]:
        """Store a value with configured write consistency.

        Writes to multiple replicas in parallel and returns success
        once enough acknowledgments are received.

        Args:
            key: The key to store under.
            value: The value to store.

        Yields:
            Write latency delays.

        Returns:
            True if write met consistency requirement.
        """
        self._writes += 1
        required = self._required_responses(self._write_consistency)

        # Write to all replicas
        acks = 0
        latencies: list[float] = []

        for replica in self._replicas:
            try:
                gen = replica.put(key, value)
                replica_latency = 0.0
                try:
                    while True:
                        delay = next(gen)
                        replica_latency += delay
                        yield delay
                except StopIteration:
                    pass

                latencies.append(replica_latency)
                acks += 1

            except Exception:
                self._replica_timeouts += 1
                continue

        # Check if we met the consistency requirement
        if acks >= required:
            self._write_successes += 1
            if latencies:
                # Latency is time until we got required acks
                sorted_latencies = sorted(latencies)
                self._write_latencies.append(sorted_latencies[required - 1])
            return True
        else:
            self._write_failures += 1
            return False

    def delete(self, key: str) -> Generator[float, None, bool]:
        """Delete a key from all replicas.

        Requires write consistency for deletion.

        Args:
            key: The key to delete.

        Yields:
            Delete latency delays.

        Returns:
            True if delete met consistency requirement.
        """
        required = self._required_responses(self._write_consistency)
        acks = 0
        existed = False

        for replica in self._replicas:
            try:
                gen = replica.delete(key)
                result = None
                try:
                    while True:
                        delay = next(gen)
                        yield delay
                except StopIteration as e:
                    result = e.value

                acks += 1
                if result:
                    existed = True

            except Exception:
                self._replica_timeouts += 1
                continue

        return acks >= required and existed

    def get_replica_status(self) -> list[dict]:
        """Get status of all replicas.

        Returns:
            List of dictionaries with replica status.
        """
        status = []
        for i, replica in enumerate(self._replicas):
            info = {
                'index': i,
                'name': replica.name,
                'size': replica.size if hasattr(replica, 'size') else 0,
            }
            if hasattr(replica, 'stats'):
                info['reads'] = replica.stats.reads
                info['writes'] = replica.stats.writes
            status.append(info)
        return status

    def handle_event(self, event: Event) -> None:
        """ReplicatedStore can handle events for store operations."""
        pass
