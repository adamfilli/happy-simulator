"""Lightweight entity adapters for function-based event handling.

CallbackEntity wraps a plain function as an Entity, bridging the gap between
function-based actions and the entity-based event system. NullEntity is a
singleton that silently discards all events.
"""

from typing import Callable, Union

from happysimulator.core.entity import Entity
from happysimulator.core.event import Event


class CallbackEntity(Entity):
    """Entity that delegates handle_event to a callback function.

    Use this to schedule a function call as a target-based event without
    creating a full Entity subclass. For one-shot convenience, prefer
    ``Event.once()``.

    Args:
        name: Identifier for logging and debugging.
        fn: Function called with the event; returns events or None.
    """

    def __init__(
        self,
        name: str,
        fn: Callable[[Event], Union[list[Event], Event, None]],
    ):
        super().__init__(name)
        self._fn = fn

    def handle_event(self, event: Event) -> list[Event] | Event | None:
        return self._fn(event)


class NullEntity(Entity):
    """Entity that accepts and discards all events. Singleton.

    Useful for fire-and-forget events or test placeholders where
    no processing is needed.
    """

    _instance: "NullEntity | None" = None

    def __new__(cls) -> "NullEntity":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if not hasattr(self, "_initialized"):
            super().__init__("NullEntity")
            self._initialized = True

    def handle_event(self, event: Event) -> None:
        return None
