"""Weighted Fair Queue implementation.

Provides weighted per-flow fair queuing where flows can have different
priorities based on weights.

Example:
    from happysimulator.components.queue_policies import WeightedFairQueue

    def get_weight(flow_id: str) -> int:
        # Premium users get 3x the bandwidth
        return 3 if flow_id.startswith("premium_") else 1

    queue = WeightedFairQueue(
        get_flow_id=lambda event: event.context.get("client_id", "default"),
        get_weight=get_weight,
    )
"""

from collections import deque, OrderedDict
from dataclasses import dataclass
from typing import TypeVar, Optional, Callable

from happysimulator.components.queue_policy import QueuePolicy

T = TypeVar("T")


@dataclass(frozen=True)
class WeightedFairQueueStats:
    """Statistics tracked by WeightedFairQueue."""

    enqueued: int = 0
    dequeued: int = 0
    rejected_capacity: int = 0
    flows_created: int = 0
    flows_removed: int = 0


@dataclass
class _FlowState:
    """State tracked for each flow."""

    queue: deque
    weight: int
    credits: int  # Current credits for weighted round-robin


class WeightedFairQueue(QueuePolicy[T]):
    """Weighted fair queuing with priority classes.

    Similar to FairQueue but flows are assigned weights that determine
    how many items they can dequeue per round-robin cycle.

    A flow with weight 3 gets to dequeue 3 items for every 1 item
    dequeued from a flow with weight 1.

    This implements a deficit round-robin (DRR) style algorithm where
    each flow accumulates credits based on its weight.

    Attributes:
        get_flow_id: Function to extract flow ID from items.
        get_weight: Function to get weight for a flow ID.
        capacity: Total queue capacity.
    """

    def __init__(
        self,
        get_flow_id: Callable[[T], str],
        get_weight: Callable[[str], int],
        capacity: int | None = None,
        per_flow_capacity: int | None = None,
    ):
        """Initialize the weighted fair queue.

        Args:
            get_flow_id: Function that extracts flow ID from an item.
            get_weight: Function that returns weight for a flow ID.
                       Higher weight = more bandwidth share.
            capacity: Total queue capacity (None = unlimited).
            per_flow_capacity: Maximum items per flow (None = unlimited).

        Raises:
            ValueError: If parameters are invalid.
        """
        if capacity is not None and capacity < 1:
            raise ValueError(f"capacity must be >= 1 or None, got {capacity}")
        if per_flow_capacity is not None and per_flow_capacity < 1:
            raise ValueError(f"per_flow_capacity must be >= 1 or None, got {per_flow_capacity}")

        self._get_flow_id = get_flow_id
        self._get_weight = get_weight
        self._capacity = float("inf") if capacity is None else capacity
        self._per_flow_capacity = float("inf") if per_flow_capacity is None else per_flow_capacity

        # Flow state tracking
        self._flows: OrderedDict[str, _FlowState] = OrderedDict()
        self._total_items = 0

        # Statistics
        self._enqueued = 0
        self._dequeued = 0
        self._rejected_capacity = 0
        self._flows_created = 0
        self._flows_removed = 0

    @property
    def stats(self) -> WeightedFairQueueStats:
        """Return a frozen snapshot of queue statistics."""
        return WeightedFairQueueStats(
            enqueued=self._enqueued,
            dequeued=self._dequeued,
            rejected_capacity=self._rejected_capacity,
            flows_created=self._flows_created,
            flows_removed=self._flows_removed,
        )

    @property
    def capacity(self) -> float:
        """Total queue capacity."""
        return self._capacity

    @property
    def flow_count(self) -> int:
        """Number of active flows."""
        return len(self._flows)

    def get_flow_depth(self, flow_id: str) -> int:
        """Get current queue depth for a specific flow."""
        if flow_id in self._flows:
            return len(self._flows[flow_id].queue)
        return 0

    def get_flow_weight(self, flow_id: str) -> int:
        """Get weight for a specific flow."""
        if flow_id in self._flows:
            return self._flows[flow_id].weight
        return self._get_weight(flow_id)

    def push(self, item: T) -> bool:
        """Add item to its flow's queue.

        Args:
            item: The item to enqueue.

        Returns:
            True if accepted, False if rejected.
        """
        # Check total capacity
        if self._total_items >= self._capacity:
            self._rejected_capacity += 1
            return False

        flow_id = self._get_flow_id(item)

        # Create flow if needed
        if flow_id not in self._flows:
            weight = self._get_weight(flow_id)
            if weight < 1:
                weight = 1  # Minimum weight is 1
            self._flows[flow_id] = _FlowState(
                queue=deque(),
                weight=weight,
                credits=weight,  # Start with full credits
            )
            self._flows_created += 1

        flow_state = self._flows[flow_id]

        # Check per-flow capacity
        if len(flow_state.queue) >= self._per_flow_capacity:
            self._rejected_capacity += 1
            return False

        flow_state.queue.append(item)
        self._total_items += 1
        self._enqueued += 1
        return True

    def pop(self) -> Optional[T]:
        """Remove and return the next item using weighted round-robin.

        Returns:
            The next item, or None if empty.
        """
        if not self._flows:
            return None

        # Find a flow with items and credits
        attempts = 0
        max_attempts = len(self._flows) * 2  # Prevent infinite loop

        while attempts < max_attempts:
            attempts += 1

            # Get the first flow
            flow_id = next(iter(self._flows))
            flow_state = self._flows[flow_id]

            if not flow_state.queue:
                # Empty flow, remove it
                self._remove_flow(flow_id)
                if not self._flows:
                    return None
                continue

            if flow_state.credits > 0:
                # Dequeue from this flow
                item = flow_state.queue.popleft()
                self._total_items -= 1
                flow_state.credits -= 1
                self._dequeued += 1

                # If flow exhausted credits, move to end and reset
                if flow_state.credits <= 0:
                    self._flows.move_to_end(flow_id)
                    flow_state.credits = flow_state.weight

                # Clean up empty flows
                if not flow_state.queue:
                    self._remove_flow(flow_id)

                return item

            # Flow has no credits, reset and move to end
            flow_state.credits = flow_state.weight
            self._flows.move_to_end(flow_id)

        return None

    def _remove_flow(self, flow_id: str) -> None:
        """Remove a flow from the queue."""
        if flow_id in self._flows:
            del self._flows[flow_id]
            self._flows_removed += 1

    def peek(self) -> Optional[T]:
        """Return the next item without removing it."""
        if not self._flows:
            return None

        for flow_state in self._flows.values():
            if flow_state.queue:
                return flow_state.queue[0]
        return None

    def is_empty(self) -> bool:
        """Return True if no items in queue."""
        return self._total_items == 0

    def __len__(self) -> int:
        """Return total number of items across all flows."""
        return self._total_items
