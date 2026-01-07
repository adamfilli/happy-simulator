"""
Abstract base class for entities with queuing and concurrency control.

Provides the infrastructure for:
- Queue management with pluggable policies
- Concurrency-limited worker pool
- Automatic worker lifecycle (spawn on arrival, retire when idle)

Subclasses implement only the business logic via `process_item()`.

Example:
    class MyServer(QueuedEntity):
        def __init__(self, name: str, latency: float):
            super().__init__(name, concurrency=2, queue_capacity=100)
            self.latency = latency
        
        def process_item(self, item):
            yield self.latency
            return []
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generator, Optional, Union

from .entity import Entity, SimYield, SimReturn
from .queue_policy import QueuePolicy, FIFOQueue
from ..events.event import Event


@dataclass
class QueuedEntityStats:
    """
    Statistics tracked by QueuedEntity.
    
    Attributes:
        items_accepted: Total items successfully enqueued.
        items_dropped: Total items rejected due to queue full.
        items_completed: Total items that finished processing.
    """
    items_accepted: int = 0
    items_dropped: int = 0
    items_completed: int = 0


class QueuedEntity(Entity):
    """
    Abstract base class for entities with queuing and concurrency control.
    
    This class handles the infrastructure of queue management and worker
    lifecycle, allowing subclasses to focus purely on business logic.
    
    Subclasses must implement:
        - `process_item(item)`: Generator that yields delays while processing.
    
    Optionally override:
        - `on_item_accepted(item)`: Called when item enters queue.
        - `on_item_dropped(item)`: Called when item is rejected.
        - `on_item_completed(item)`: Called when processing finishes.
        - `create_queue()`: Override to use a different queue policy.
        - `extract_item(event)`: Override to queue something other than the event.
    
    Attributes:
        concurrency_limit: Maximum number of parallel workers.
        stats: QueuedEntityStats tracking accepted/dropped/completed counts.
    
    Note:
        The worker loop yields control back to the simulation after each delay,
        ensuring proper global event ordering. Subclasses should NOT assume they
        know the current simulation time inside process_item - use event context
        (e.g., item.time, item.context["created_at"]) for timing information.
    """
    
    def __init__(
        self,
        name: str,
        concurrency: int = 1,
        queue_capacity: float = float('inf'),
        queue: Optional[QueuePolicy] = None
    ):
        """
        Initialize a queued entity.
        
        Args:
            name: Entity name.
            concurrency: Maximum concurrent workers (parallel processing slots).
            queue_capacity: Maximum queue size (ignored if custom queue provided).
            queue: Optional custom queue policy. If None, uses FIFOQueue.
        """
        super().__init__(name)
        self.concurrency_limit = concurrency
        self._queue: QueuePolicy = queue or self.create_queue(queue_capacity)
        self._active_workers = 0
        self.stats = QueuedEntityStats()
    
    def create_queue(self, capacity: float) -> QueuePolicy:
        """
        Factory method to create the queue. Override for custom policies.
        
        Args:
            capacity: Maximum queue capacity.
        
        Returns:
            A QueuePolicy instance.
        """
        return FIFOQueue(capacity)
    
    # --- Public Properties ---
    
    @property
    def queue(self) -> QueuePolicy:
        """The underlying queue. Useful for inspection or custom operations."""
        return self._queue
    
    @property
    def queue_depth(self) -> int:
        """Number of items waiting in queue (not yet processing)."""
        return len(self._queue)
    
    @property
    def active_workers(self) -> int:
        """Number of workers currently processing items."""
        return self._active_workers
    
    @property
    def in_flight(self) -> int:
        """Total items in system (queued + being processed)."""
        return self.queue_depth + self._active_workers
    
    @property
    def is_idle(self) -> bool:
        """True if no items queued and no active workers."""
        return self._queue.is_empty() and self._active_workers == 0
    
    # --- Convenience accessors mirroring stats ---
    
    @property
    def requests_completed(self) -> int:
        """Alias for stats.items_completed (for compatibility)."""
        return self.stats.items_completed
    
    @property
    def requests_dropped(self) -> int:
        """Alias for stats.items_dropped (for compatibility)."""
        return self.stats.items_dropped
    
    # --- Event Handling ---
    
    def handle_event(self, event: Event) -> Union[Generator[SimYield, None, SimReturn], list[Event], None]:
        """
        Handle incoming event by queuing and potentially spawning a worker.
        
        The event (or its payload) is enqueued. If a worker slot is available,
        a new worker process is started to drain the queue.
        
        Args:
            event: The incoming event to process.
        
        Returns:
            A worker generator if a new worker was spawned, else None.
        """
        # Extract the item to queue (subclasses can override extract_item)
        item = self.extract_item(event)
        
        # Attempt to enqueue
        accepted = self._queue.push(item)
        
        if not accepted:
            self.stats.items_dropped += 1
            self.on_item_dropped(item)
            return self.on_drop_reaction(item)
        
        self.stats.items_accepted += 1
        self.on_item_accepted(item)
        
        # Spawn worker if capacity available
        if self._active_workers < self.concurrency_limit:
            self._active_workers += 1
            return self._worker_loop()
        
        # Item queued, will be processed when a worker becomes free
        return None
    
    def extract_item(self, event: Event):
        """
        Extract the item to queue from the incoming event.
        
        Override to queue something other than the event itself
        (e.g., event.payload, a Request object, etc.).
        
        Args:
            event: The incoming event.
        
        Returns:
            The item to enqueue.
        """
        return event
    
    # --- Worker Process ---
    
    def _worker_loop(self) -> Generator[SimYield, None, SimReturn]:
        """
        Internal worker loop that drains the queue.
        
        Continuously processes items until queue is empty, then retires.
        Each yield returns control to the simulation loop, ensuring proper
        global event ordering - other events may be processed between yields.
        
        Important:
            This generator does NOT track simulation time internally.
            Each yield suspends execution and returns control to the main
            simulation loop, which may process other events before resuming.
            This maintains the discrete-event simulation invariant that
            events are processed in strict global time order.
        
        Yields:
            Delays from the subclass's process_item implementation.
        """
        try:
            while not self._queue.is_empty():
                item = self._queue.pop()
                if item is None:
                    # Safety check (shouldn't happen in single-threaded sim)
                    break
                
                # Delegate to subclass implementation
                process_gen = self.process_item(item)
                
                # If process_item returns a generator, exhaust it
                if hasattr(process_gen, '__next__'):
                    try:
                        while True:
                            yielded = next(process_gen)
                            yield yielded
                    except StopIteration as e:
                        # Capture any returned events from the generator
                        result = e.value
                        if result:
                            # Emit side-effect events from processing
                            if isinstance(result, list):
                                for evt in result:
                                    yield (0, [evt], None)
                            else:
                                yield (0, [result], None)
                
                self.stats.items_completed += 1
                self.on_item_completed(item)
        finally:
            # Worker retires when queue is empty
            self._active_workers -= 1
    
    # --- Abstract Method (subclass must implement) ---
    
    @abstractmethod
    def process_item(self, item) -> Generator[SimYield, None, SimReturn]:
        """
        Process a single item from the queue.
        
        This is where subclasses define their business logic.
        Yield delays to simulate processing time. The simulation will
        suspend this generator and may process other events before resuming.
        
        Important:
            Do NOT assume you know the current simulation time. If you need
            timing information, use the item's context (e.g., item.time or
            item.context["created_at"] if item is an Event). Returned events
            are scheduled at current simulation time (after all yields complete).
        
        Args:
            item: The item to process (as returned by extract_item).
        
        Yields:
            float: Delay in seconds to simulate processing.
            tuple: (delay, side_effect_events, None) for mid-process events.
        
        Returns:
            Optional list of Events to schedule upon completion. These events
            will be scheduled at the simulation time when the generator finishes.
        
        Example:
            def process_item(self, request):
                yield 0.1  # 100ms processing - yields to simulation
                # Forward to sink; scheduled at current sim time after yield
                return Event(time=request.time, target=self.sink, context=request.context)
        """
        pass
    
    # --- Hooks (optional overrides) ---
    
    def on_item_accepted(self, item) -> None:
        """Called when an item is successfully enqueued. Override for custom logic."""
        ...  # Override in subclass
    
    def on_item_dropped(self, item) -> None:
        """Called when an item is rejected due to queue full. Override for custom logic."""
        ...  # Override in subclass
    
    def on_item_completed(self, item) -> None:
        """Called when an item finishes processing. Override for custom logic."""
        ...  # Override in subclass
    
    def on_drop_reaction(self, item) -> Union[list[Event], Event, None]:  # noqa: ARG002
        """
        Return events to schedule when an item is dropped.
        
        Override to emit rejection events, backpressure signals, etc.
        Default returns None (no reaction).
        
        Args:
            item: The dropped item.
        
        Returns:
            Events to schedule, or None.
        """
        del item  # Unused in base implementation
        return None
