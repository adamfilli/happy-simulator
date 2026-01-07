from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Generator, Optional, Union, List

from .entity import Entity, SimYield, SimReturn
from .queue_policy import QueuePolicy, FIFOQueue
from ..events.event import Event, ProcessContinuation
from ..utils.instant import Instant


@dataclass
class QueuedItem:
    """Wrapper to track when an item was enqueued."""
    payload: any
    enqueued_at: Instant


@dataclass
class QueuedEntityStats:
    """Statistics tracked by QueuedEntity."""
    items_accepted: int = 0
    items_dropped: int = 0
    items_completed: int = 0
    total_queue_time_seconds: float = 0.0


@dataclass
class QueuePollEvent(Event):
    """Internal event: poll the queue and start work if possible."""


class QueuedEntity(Entity):
    """
    Abstract base class for entities with queuing and concurrency control.
    
    Implements the "Worker Loop" pattern:
    1. Items are pushed to a passive QueuePolicy.
    2. A 'Poll' event checks for available worker slots.
    3. If slots exist, 'ProcessContinuation' events are spawned immediately.
    """
    
    def __init__(
        self,
        name: str,
        concurrency: int = 1,
        queue_capacity: float = float('inf'),
        queue: Optional[QueuePolicy] = None
    ):
        super().__init__(name)
        self.concurrency_limit = concurrency
        self._queue: QueuePolicy = queue or self.create_queue(queue_capacity)
        self._active_workers = 0
        self._poll_scheduled = False
        self.stats = QueuedEntityStats()

        # Cached event types for performance
        self._poll_type = f"sys.poll::{self.name}"
        self._worker_type = f"sys.worker::{self.name}"
    
    def create_queue(self, capacity: float) -> QueuePolicy:
        return FIFOQueue(capacity)
    
    # --- Properties ---
    
    @property
    def queue(self) -> QueuePolicy:
        return self._queue
    
    @property
    def queue_depth(self) -> int:
        return len(self._queue)
    
    @property
    def active_workers(self) -> int:
        return self._active_workers
    
    @property
    def is_idle(self) -> bool:
        return self._queue.is_empty() and self._active_workers == 0

    # --- Event Handling ---
    
    def handle_event(self, event: Event) -> Union[Generator, List[Event], None]:
        # 1. Handle Internal Control Events
        if event.event_type == self._poll_type:
            return self._handle_queue_poll(event)

        # 2. Handle External Items (Arrivals)
        item = self.extract_item(event)
        
        # Wrap with enqueue timestamp for queue-time tracking
        queued_item = QueuedItem(payload=item, enqueued_at=event.time)
        
        if not self._queue.push(queued_item):
            self.stats.items_dropped += 1
            self.on_item_dropped(item)
            return self.on_drop_reaction(item)
        
        self.stats.items_accepted += 1
        self.on_item_accepted(item)

        # 3. Schedule Poll (if not already pending)
        return self._ensure_poll_scheduled(event.time)
    
    def extract_item(self, event: Event):
        """Override to extract specific payloads from the event."""
        return event

    # --- Internal Logic ---

    def _ensure_poll_scheduled(self, time: Instant) -> List[Event]:
        if self._poll_scheduled:
            return []

        self._poll_scheduled = True
        return [
            QueuePollEvent(
                time=time,
                event_type=self._poll_type,
                daemon=True,
                target=self,
            )
        ]

    def _handle_queue_poll(self, event: Event) -> List[Event]:
        """
        Polls the queue and spawns workers directly.
        """
        self._poll_scheduled = False

        free_slots = self.concurrency_limit - self._active_workers
        if free_slots <= 0 or self._queue.is_empty():
            return []

        new_processes = []
        count = min(free_slots, len(self._queue))
        
        for _ in range(count):
            queued_item: QueuedItem = self._queue.pop()
            if queued_item is None:
                break

            # Track queue time
            queue_wait = (event.time - queued_item.enqueued_at).to_seconds()
            self.stats.total_queue_time_seconds += queue_wait

            self._active_workers += 1
            
            # Create the generator for this specific item
            gen = self._process_wrapper(queued_item.payload, dequeue_time=event.time)
            
            # Schedule to start NOW (queue delay already accounted for by simulation time)
            proc = ProcessContinuation(
                time=event.time,
                event_type=self._worker_type,
                target=self,
                process=gen,
                daemon=True,
                context={
                    "enqueued_at": queued_item.enqueued_at,
                    "dequeued_at": event.time,
                    "queue_wait_s": queue_wait,
                }
            )
            new_processes.append(proc)

        return new_processes

    def _process_wrapper(self, item, dequeue_time: Instant) -> Generator[SimYield, None, SimReturn]:
        """
        Wraps the user's business logic to handle lifecycle (stats, polling).
        
        Yields:
            Delay values to simulate processing time.
        
        Returns:
            List of events to schedule after completion (via StopIteration.value).
        """
        result: SimReturn = None

        try:
            user_gen = self.process_item(item)
            
            if hasattr(user_gen, '__next__'):
                # It's a generator — forward all yields
                try:
                    while True:
                        yielded = next(user_gen)
                        yield yielded
                except StopIteration as e:
                    result = e.value
            else:
                # User returned a value directly (not a generator)
                result = user_gen

            self.stats.items_completed += 1
            self.on_item_completed(item)
            
        except Exception:
            raise
        finally:
            self._active_workers -= 1

        # Prepare output events
        out_events = []
        if result:
            if isinstance(result, list):
                out_events.extend(result)
            elif isinstance(result, Event):
                out_events.append(result)

        # CRITICAL: Schedule another poll to drain the queue
        # Use Instant.Epoch as placeholder — ProcessContinuation handler should use current sim time
        if not self._queue.is_empty() or self._active_workers < self.concurrency_limit:
            # The poll will be scheduled at the time the ProcessContinuation completes
            # This requires the simulation to pass completion time, or we yield a "schedule poll" marker
            pass  # See note below

        return out_events
    
    # --- Hooks ---
    
    @abstractmethod
    def process_item(self, item) -> Generator[SimYield, None, SimReturn]:
        """Implement business logic here."""
        pass
    
    def on_item_accepted(self, item) -> None: ...
    def on_item_dropped(self, item) -> None: ...
    def on_item_completed(self, item) -> None: ...
    def on_drop_reaction(self, item) -> Union[List[Event], Event, None]: return None