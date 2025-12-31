import uuid
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Dict, Generator, Optional, Union

from happysimulator.entities.entity import Entity
from happysimulator.utils.instant import Instant

_global_event_counter = count()

@dataclass
class Event:
    time: Instant
    event_type: str
    entity: Entity = field(repr=False)
    
    # Unified Payload: Can hold a Generator (for processes) or arbitrary data
    payload: Optional[Union[Generator, Any]] = field(default=None, repr=False, compare=False)

    # Context & Tracing
    context: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    # Sorting Internals
    _sort_index: int = field(default_factory=_global_event_counter.__next__, init=False, repr=False)
    _id: uuid.UUID = field(default_factory=uuid.uuid4, init=False, repr=False)

    def __post_init__(self):
        if not self.context:
            self.context = {
                "id": str(self._id),
                "created_at": self.time,
                "stack": [],
                "metadata": {}
            }

    @classmethod
    def create_continuation(cls, time: Instant, entity: Entity, generator: Generator, parent_context: Dict = None) -> "Event":
        """
        Helper to create a 'Resume' event that carries over the parent's context.
        """
        # Shallow copy context to preserve the Trace ID through the flow
        ctx = parent_context.copy() if parent_context else {}
        
        return cls(
            time=time,
            event_type="__RESUME__", # internal marker
            entity=entity,
            payload=generator,
            context=ctx
        )

    def is_continuation(self) -> bool:
        """Check if this event is actually a paused process."""
        return isinstance(self.payload, Generator)

    def __lt__(self, other: "Event") -> bool:
        """
        1. Time (Primary)
        2. Insert Order (Secondary - guarantees FIFO for simultaneous events)
        """
        if self.time != other.time:
            return self.time < other.time
        return self._sort_index < other._sort_index

    def __hash__(self):
        return hash(self._id)

    def __eq__(self, other):
        if not isinstance(other, Event):
            return NotImplemented
        return self._id == other._id

    def add_context(self, key: str, value: Any):
        self.context.setdefault("metadata", {})[key] = value

    def get_context(self, key: str) -> Any:
        return self.context.get("metadata", {}).get(key)