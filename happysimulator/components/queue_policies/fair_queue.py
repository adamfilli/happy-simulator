"""Fair Queue implementation.

Provides per-flow fair queuing that prevents any single flow from
monopolizing queue resources.

Example:
    from happysimulator.components.queue_policies import FairQueue

    queue = FairQueue(
        get_flow_id=lambda event: event.context.get("client_id", "default"),
        max_flows=100,
        per_flow_capacity=10,
    )
"""

from collections import deque, OrderedDict
from dataclasses import dataclass
from typing import TypeVar, Optional, Callable

from happysimulator.components.queue_policy import QueuePolicy

T = TypeVar("T")


@dataclass(frozen=True)
class FairQueueStats:
    """Statistics tracked by FairQueue."""

    enqueued: int = 0
    dequeued: int = 0
    rejected_flow_capacity: int = 0  # Rejected because flow queue full
    rejected_max_flows: int = 0  # Rejected because too many flows
    flows_created: int = 0
    flows_removed: int = 0


class FairQueue(QueuePolicy[T]):
    """Per-flow fair queuing.

    Maintains separate queues for each flow and dequeues in round-robin
    fashion, ensuring fair bandwidth allocation across flows.

    This prevents a single aggressive flow from monopolizing the queue
    and starving other flows.

    Attributes:
        get_flow_id: Function to extract flow ID from items.
        max_flows: Maximum number of concurrent flows.
        per_flow_capacity: Maximum items per flow queue.
    """

    def __init__(
        self,
        get_flow_id: Callable[[T], str],
        max_flows: int | None = None,
        per_flow_capacity: int | None = None,
    ):
        """Initialize the fair queue.

        Args:
            get_flow_id: Function that extracts flow ID from an item.
            max_flows: Maximum number of concurrent flows (None = unlimited).
            per_flow_capacity: Maximum items per flow (None = unlimited).

        Raises:
            ValueError: If parameters are invalid.
        """
        if max_flows is not None and max_flows < 1:
            raise ValueError(f"max_flows must be >= 1 or None, got {max_flows}")
        if per_flow_capacity is not None and per_flow_capacity < 1:
            raise ValueError(f"per_flow_capacity must be >= 1 or None, got {per_flow_capacity}")

        self._get_flow_id = get_flow_id
        self._max_flows = max_flows
        self._per_flow_capacity = per_flow_capacity if per_flow_capacity else float("inf")

        # OrderedDict maintains round-robin order
        # Each flow maps to its deque of items
        self._flows: OrderedDict[str, deque[T]] = OrderedDict()
        self._total_items = 0

        # Statistics
        self._enqueued = 0
        self._dequeued = 0
        self._rejected_flow_capacity = 0
        self._rejected_max_flows = 0
        self._flows_created = 0
        self._flows_removed = 0

    @property
    def stats(self) -> FairQueueStats:
        """Return a frozen snapshot of queue statistics."""
        return FairQueueStats(
            enqueued=self._enqueued,
            dequeued=self._dequeued,
            rejected_flow_capacity=self._rejected_flow_capacity,
            rejected_max_flows=self._rejected_max_flows,
            flows_created=self._flows_created,
            flows_removed=self._flows_removed,
        )

    @property
    def capacity(self) -> float:
        """Total capacity (flows * per_flow_capacity)."""
        if self._max_flows is None:
            return float("inf")
        return self._max_flows * self._per_flow_capacity

    @property
    def flow_count(self) -> int:
        """Number of active flows."""
        return len(self._flows)

    @property
    def max_flows(self) -> int | None:
        """Maximum number of concurrent flows."""
        return self._max_flows

    @property
    def per_flow_capacity(self) -> float:
        """Maximum items per flow."""
        return self._per_flow_capacity

    def get_flow_depth(self, flow_id: str) -> int:
        """Get current queue depth for a specific flow."""
        if flow_id in self._flows:
            return len(self._flows[flow_id])
        return 0

    def push(self, item: T) -> bool:
        """Add item to its flow's queue.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if rejected.
        """
        flow_id = self._get_flow_id(item)

        # Check if flow exists
        if flow_id not in self._flows:
            # Check flow limit
            if self._max_flows is not None and len(self._flows) >= self._max_flows:
                self._rejected_max_flows += 1
                return False

            # Create new flow queue
            self._flows[flow_id] = deque()
            self._flows_created += 1

        flow_queue = self._flows[flow_id]

        # Check per-flow capacity
        if len(flow_queue) >= self._per_flow_capacity:
            self._rejected_flow_capacity += 1
            return False

        flow_queue.append(item)
        self._total_items += 1
        self._enqueued += 1
        return True

    def pop(self) -> Optional[T]:
        """Remove and return the next item using round-robin.

        Returns:
            The next item, or None if empty.
        """
        if not self._flows:
            return None

        # Get the first flow (round-robin order)
        flow_id, flow_queue = next(iter(self._flows.items()))

        if not flow_queue:
            # Empty flow, remove it and try again
            self._remove_flow(flow_id)
            return self.pop()

        item = flow_queue.popleft()
        self._total_items -= 1
        self._dequeued += 1

        # Move this flow to the end (round-robin)
        self._flows.move_to_end(flow_id)

        # Clean up empty flows
        if not flow_queue:
            self._remove_flow(flow_id)

        return item

    def _remove_flow(self, flow_id: str) -> None:
        """Remove a flow from the queue."""
        if flow_id in self._flows:
            del self._flows[flow_id]
            self._flows_removed += 1

    def peek(self) -> Optional[T]:
        """Return the next item without removing it."""
        if not self._flows:
            return None

        for flow_queue in self._flows.values():
            if flow_queue:
                return flow_queue[0]
        return None

    def is_empty(self) -> bool:
        """Return True if no items in queue."""
        return self._total_items == 0

    def __len__(self) -> int:
        """Return total number of items across all flows."""
        return self._total_items
