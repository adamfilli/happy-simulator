"""SimFuture — yield on events, not just delays.

SimFuture enables generators to pause until an external condition is met,
rather than only pausing for fixed time delays. When a generator yields a
SimFuture, the process parks until another entity calls future.resolve(value).

This unlocks natural request-response modeling, resource acquisition, lock
waiting, timeout races (any_of), and quorum waits (all_of).

Example::

    class Client(Entity):
        def handle_event(self, event):
            future = SimFuture()
            # Send request with the future so the server can resolve it
            yield 0.0, [Event(
                time=self.now, event_type="Request", target=self.server,
                context={"reply_future": future},
            )]
            response = yield future  # Park until server resolves
            # response is the value passed to future.resolve(value)

    class Server(Entity):
        def handle_event(self, event):
            yield 0.1  # Processing time
            event.context["reply_future"].resolve({"status": "ok"})
"""

from __future__ import annotations

import contextvars
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from happysimulator.core.clock import Clock
    from happysimulator.core.event import ProcessContinuation
    from happysimulator.core.event_heap import EventHeap

logger = logging.getLogger(__name__)

# Active simulation context — set by Simulation.run(), used by resolve()
# Uses contextvars for async/thread safety instead of module-level globals.
_active_heap_var: contextvars.ContextVar[EventHeap | None] = contextvars.ContextVar(
    "_active_heap", default=None
)
_active_clock_var: contextvars.ContextVar[Clock | None] = contextvars.ContextVar(
    "_active_clock", default=None
)


def _set_active_context(heap: EventHeap, clock: Clock) -> None:
    """Set the active simulation context. Called by Simulation.run()."""
    _active_heap_var.set(heap)
    _active_clock_var.set(clock)


def _clear_active_context() -> None:
    """Clear the active simulation context. Called when Simulation.run() exits."""
    _active_heap_var.set(None)
    _active_clock_var.set(None)


def _get_active_heap() -> EventHeap | None:
    """Return the active event heap, or None if no simulation is running."""
    return _active_heap_var.get()


class SimFuture:
    """A future that generators can yield to park until resolved.

    When a generator yields a SimFuture, the ProcessContinuation parks
    instead of scheduling a time-based resume. The generator stays suspended
    until some other entity calls resolve(value), at which point a new
    continuation is scheduled to resume the generator with the resolved
    value via gen.send(value).

    Each SimFuture can only be yielded by one generator. For broadcast
    patterns, create separate futures per consumer.

    Attributes:
        is_resolved: True if the future has been resolved.
        value: The resolved value (raises RuntimeError if not yet resolved).
    """

    __slots__ = (
        "_resolved", "_value",
        "_parked_process", "_parked_event_type", "_parked_daemon",
        "_parked_target", "_parked_on_complete", "_parked_context",
        "_settle_callbacks",
    )

    def __init__(self) -> None:
        self._resolved: bool = False
        self._value: Any = None

        # Parked continuation state (set by ProcessContinuation when it yields this)
        self._parked_process = None
        self._parked_event_type: str | None = None
        self._parked_daemon: bool = False
        self._parked_target = None
        self._parked_on_complete = None
        self._parked_context = None

        # Callbacks fired when the future resolves
        self._settle_callbacks: list[Callable[[SimFuture], None]] = []

    @property
    def is_resolved(self) -> bool:
        """Whether this future has been resolved."""
        return self._resolved

    @property
    def value(self) -> Any:
        """The resolved value.

        Raises:
            RuntimeError: If the future hasn't been resolved yet.
        """
        if not self._resolved:
            raise RuntimeError("SimFuture has not been resolved yet")
        return self._value

    def _park(self, continuation: ProcessContinuation) -> None:
        """Store continuation state so we can resume when resolved.

        Called by ProcessContinuation.invoke() when the generator yields
        a SimFuture. If the future is already resolved, resumes immediately.

        Args:
            continuation: The ProcessContinuation that yielded this future.

        Raises:
            RuntimeError: If another generator is already parked on this future.
        """
        if self._parked_process is not None and not self._resolved:
            raise RuntimeError(
                "SimFuture already has a parked process. "
                "Each SimFuture can only be yielded by one generator."
            )

        self._parked_process = continuation.process
        self._parked_event_type = continuation.event_type
        self._parked_daemon = continuation.daemon
        self._parked_target = continuation.target
        self._parked_on_complete = continuation.on_complete
        self._parked_context = continuation.context

        if self._resolved:
            self._resume()

    def resolve(self, value: Any = None) -> None:
        """Resolve the future with a value, resuming the parked generator.

        The parked generator will be resumed at the current simulation time
        with the value sent via gen.send(value). If no generator is parked
        yet, the value is stored and the generator will resume immediately
        when it yields this future.

        Resolving an already-resolved future is a no-op.

        Args:
            value: The value to send into the generator.
        """
        if self._resolved:
            return
        self._resolved = True
        self._value = value
        if self._parked_process is not None:
            self._resume()
        self._fire_callbacks()

    def _add_settle_callback(self, fn: Callable[[SimFuture], None]) -> None:
        """Register a callback to fire when this future resolves.

        If the future is already resolved, the callback fires immediately.
        Used internally by any_of/all_of combinators.
        """
        if self._resolved:
            fn(self)
        else:
            self._settle_callbacks.append(fn)

    def _fire_callbacks(self) -> None:
        """Fire all registered settle callbacks."""
        callbacks = list(self._settle_callbacks)
        self._settle_callbacks.clear()
        for cb in callbacks:
            cb(self)

    def _resume(self) -> None:
        """Create and schedule a ProcessContinuation to resume the generator."""
        from happysimulator.core.event import ProcessContinuation

        heap = _active_heap_var.get()
        clock = _active_clock_var.get()
        if heap is None or clock is None:
            raise RuntimeError(
                "SimFuture.resolve() called outside a running simulation. "
                "Futures can only be resolved during Simulation.run()."
            )

        continuation = ProcessContinuation(
            time=clock.now,
            event_type=self._parked_event_type,
            daemon=self._parked_daemon,
            target=self._parked_target,
            process=self._parked_process,
            on_complete=self._parked_on_complete,
            context=self._parked_context,
        )
        continuation._send_value = self._value

        heap.push(continuation)

        # Clear parked state
        self._parked_process = None

    def __repr__(self) -> str:
        if self._resolved:
            return f"SimFuture(resolved={self._value!r})"
        elif self._parked_process is not None:
            return "SimFuture(parked)"
        else:
            return "SimFuture(pending)"


def any_of(*futures: SimFuture) -> SimFuture:
    """Return a future that resolves when ANY input future resolves.

    The composite future resolves with a tuple ``(index, value)`` where
    ``index`` is the position of the first future to resolve and ``value``
    is its resolved value.

    This enables timeout races and first-to-complete patterns::

        timeout = SimFuture()
        # Schedule a timeout event that resolves timeout future
        yield 0.0, [Event.once(time=self.now + 5.0, event_type="Timeout",
                               fn=lambda e: timeout.resolve("timeout"))]

        response = SimFuture()
        # Send request with response future
        yield 0.0, [Event(time=self.now, event_type="Request", target=server,
                          context={"reply_future": response})]

        idx, value = yield any_of(timeout, response)
        if idx == 0:
            print("Timed out!")
        else:
            print(f"Got response: {value}")

    Args:
        *futures: Two or more SimFuture instances to race.

    Returns:
        A new SimFuture that resolves when any input resolves.
    """
    if len(futures) < 2:
        raise ValueError("any_of() requires at least 2 futures")

    composite = SimFuture()

    for i, f in enumerate(futures):
        f._add_settle_callback(
            lambda sf, idx=i: composite.resolve((idx, sf._value))
        )

    return composite


def all_of(*futures: SimFuture) -> SimFuture:
    """Return a future that resolves when ALL input futures resolve.

    The composite future resolves with a list of values in the same
    order as the input futures.

    This enables quorum waits and barrier patterns::

        ack1, ack2, ack3 = SimFuture(), SimFuture(), SimFuture()
        # Send requests to three replicas
        yield 0.0, [
            Event(time=self.now, event_type="Write", target=replica1,
                  context={"ack": ack1}),
            Event(time=self.now, event_type="Write", target=replica2,
                  context={"ack": ack2}),
            Event(time=self.now, event_type="Write", target=replica3,
                  context={"ack": ack3}),
        ]
        results = yield all_of(ack1, ack2, ack3)
        # results = [value1, value2, value3]

    Args:
        *futures: Two or more SimFuture instances to wait on.

    Returns:
        A new SimFuture that resolves when all inputs resolve.
    """
    if len(futures) < 2:
        raise ValueError("all_of() requires at least 2 futures")

    composite = SimFuture()
    results: list[Any] = [None] * len(futures)
    remaining = len(futures)

    def on_settle(settled: SimFuture, idx: int = 0) -> None:
        nonlocal remaining
        if composite._resolved:
            return
        results[idx] = settled._value
        remaining -= 1
        if remaining == 0:
            composite.resolve(list(results))

    for i, f in enumerate(futures):
        f._add_settle_callback(lambda sf, i=i: on_settle(sf, i))

    return composite
