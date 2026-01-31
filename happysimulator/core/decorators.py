"""Decorators for creating simulation-compatible classes.

The @simulatable decorator transforms any class into a simulation participant
by injecting the clock infrastructure. This allows users to create entities
without inheriting from Entity.

ADVANCED USE CASE: This decorator is intended for scenarios where inheritance
from Entity is not possible (e.g., adapting third-party classes). For most
use cases, inheriting from Entity is recommended as it provides full IDE
autocompletion and type checker support.

Note: Static type checkers (Pylance, mypy) cannot see the attributes added
by this decorator (`now`, `set_clock`). You may see linter warnings like
"Instance of 'X' has no 'now' member". The code works correctly at runtime.

Example:
    @simulatable
    class MyServer:
        def __init__(self, name: str):
            self.name = name
            self.requests_handled = 0

        def handle_event(self, event: Event) -> list[Event] | None:
            self.requests_handled += 1
            print(f"Handled request at {self.now}")  # Works at runtime
            return None

    # Use in simulation
    server = MyServer("my-server")
    sim = Simulation(entities=[server], ...)
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from happysimulator.core.clock import Clock
    from happysimulator.core.temporal import Instant

logger = logging.getLogger(__name__)

T = TypeVar("T")


def simulatable(cls: type[T]) -> type[T]:
    """Class decorator that adds simulation clock infrastructure.

    Transforms a plain class into one that satisfies the Simulatable protocol
    by injecting:
    - _clock: Private attribute to store the simulation clock
    - set_clock(clock): Method called by Simulation during initialization
    - now: Property to access current simulation time

    The decorated class must have:
    - name: str attribute (set in __init__)
    - handle_event(event): Method to process events

    Args:
        cls: The class to decorate.

    Returns:
        The decorated class with clock infrastructure added.

    Example:
        @simulatable
        class MyClient:
            def __init__(self, name: str):
                self.name = name

            def handle_event(self, event):
                print(f"Processing at {self.now}")
                return None
    """
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._clock: Clock | None = None

    cls.__init__ = new_init

    def set_clock(self, clock: "Clock") -> None:
        """Receive the simulation clock. Called by Simulation during setup."""
        self._clock = clock
        logger.debug("[%s] Clock injected", getattr(self, "name", type(self).__name__))

    cls.set_clock = set_clock

    @property
    def now(self) -> "Instant":
        """Current simulation time from the injected clock.

        Raises:
            RuntimeError: If accessed before clock injection (outside simulation).
        """
        if self._clock is None:
            name = getattr(self, "name", type(self).__name__)
            raise RuntimeError(
                f"Entity {name} is not attached to a simulation (Clock is None). "
                "Ensure this entity is registered with Simulation(entities=[...])."
            )
        return self._clock.now

    cls.now = now

    # Add default has_capacity if not present
    if not hasattr(cls, "has_capacity"):
        cls.has_capacity = lambda self: True

    return cls
