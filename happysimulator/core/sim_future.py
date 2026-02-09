"""SimFuture — yield on events, not just delays.

SimFuture enables generators to pause until an external condition is met,
rather than only pausing for fixed time delays. When a generator yields a
SimFuture, the process parks until another entity calls future.resolve(value)
or future.fail(exception).

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

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from happysimulator.core.clock import Clock
    from happysimulator.core.event import ProcessContinuation
    from happysimulator.core.event_heap import EventHeap

logger = logging.getLogger(__name__)

# Active simulation context — set by Simulation.run(), used by resolve/fail
_active_heap: EventHeap | None = None
_active_clock: Clock | None = None


def _set_active_context(heap: EventHeap, clock: Clock) -> None:
    """Set the active simulation context. Called by Simulation.run()."""
    global _active_heap, _active_clock
    _active_heap = heap
    _active_clock = clock


def _clear_active_context() -> None:
    """Clear the active simulation context. Called when Simulation.run() exits."""
    global _active_heap, _active_clock
    _active_heap = None
    _active_clock = None


class SimFuture:
    """A future that generators can yield to park until resolved.

    When a generator yields a SimFuture, the ProcessContinuation parks
    instead of scheduling a time-based resume. The generator stays suspended
    until some other entity calls resolve(value) or fail(exception), at which
    point a new continuation is scheduled to resume the generator with the
    resolved value via gen.send(value).

    Each SimFuture can only be yielded by one generator. For broadcast
    patterns, create separate futures per consumer.

    Attributes:
        is_settled: True if the future has been resolved or failed.
        value: The resolved value (raises RuntimeError if not yet resolved).
    """

    __slots__ = (
        "_resolved", "_failed", "_value", "_exception",
        "_parked_process", "_parked_event_type", "_parked_daemon",
        "_parked_target", "_parked_on_complete", "_parked_context",
        "_settle_callbacks",
    )

    def __init__(self) -> None:
        self._resolved: bool = False
        self._failed: bool = False
        self._value: Any = None
        self._exception: BaseException | None = None

        # Parked continuation state (set by ProcessContinuation when it yields this)
        self._parked_process = None
        self._parked_event_type: str | None = None
        self._parked_daemon: bool = False
        self._parked_target = None
        self._parked_on_complete = None
        self._parked_context = None

        # Callbacks fired when the future settles
        self._settle_callbacks: list[Callable[[SimFuture], None]] = []

    @property
    def is_settled(self) -> bool:
        """Whether this future has been resolved or failed."""
        return self._resolved or self._failed

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
        a SimFuture. If the future is already settled, resumes immediately.

        Args:
            continuation: The ProcessContinuation that yielded this future.

        Raises:
            RuntimeError: If another generator is already parked on this future.
        """
        if self._parked_process is not None and not self.is_settled:
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

        if self.is_settled:
            self._resume()

    def resolve(self, value: Any = None) -> None:
        """Resolve the future with a value, resuming the parked generator.

        The parked generator will be resumed at the current simulation time
        with the value sent via gen.send(value). If no generator is parked
        yet, the value is stored and the generator will resume immediately
        when it yields this future.

        Resolving an already-settled future is a no-op.

        Args:
            value: The value to send into the generator.
        """
        if self.is_settled:
            return
        self._resolved = True
        self._value = value
        if self._parked_process is not None:
            self._resume()
        self._fire_callbacks()

    def fail(self, exception: BaseException) -> None:
        """Fail the future, throwing an exception into the parked generator.

        The parked generator will be resumed at the current simulation time
        with the exception thrown via gen.throw(). If no generator is parked
        yet, the exception is stored and will be thrown when a generator
        yields this future.

        Failing an already-settled future is a no-op.

        Args:
            exception: The exception to throw into the generator.
        """
        if self.is_settled:
            return
        self._failed = True
        self._exception = exception
        if self._parked_process is not None:
            self._resume()
        self._fire_callbacks()

    def _add_settle_callback(self, fn: Callable[[SimFuture], None]) -> None:
        """Register a callback to fire when this future settles.

        If the future is already settled, the callback fires immediately.
        Used internally by any_of/all_of combinators.
        """
        if self.is_settled:
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

        heap = _active_heap
        clock = _active_clock
        if heap is None or clock is None:
            raise RuntimeError(
                "SimFuture.resolve()/fail() called outside a running simulation. "
                "Futures can only be settled during Simulation.run()."
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

        if self._resolved:
            continuation._send_value = self._value
        else:
            continuation._throw_exception = self._exception

        heap.push(continuation)

        # Clear parked state
        self._parked_process = None

    def __repr__(self) -> str:
        if self._resolved:
            return f"SimFuture(resolved={self._value!r})"
        elif self._failed:
            return f"SimFuture(failed={self._exception!r})"
        elif self._parked_process is not None:
            return "SimFuture(parked)"
        else:
            return "SimFuture(pending)"


def any_of(*futures: SimFuture) -> SimFuture:
    """Return a future that resolves when ANY input future settles.

    The composite future resolves with a tuple ``(index, value)`` where
    ``index`` is the position of the first future to settle and ``value``
    is its resolved value.

    If the first future to settle fails, the composite future also fails.

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
        A new SimFuture that settles when any input settles.
    """
    if len(futures) < 2:
        raise ValueError("any_of() requires at least 2 futures")

    composite = SimFuture()

    def on_settle(settled: SimFuture, idx: int = 0) -> None:
        if composite.is_settled:
            return
        if settled._resolved:
            composite.resolve((idx, settled._value))
        else:
            composite.fail(settled._exception)

    for i, f in enumerate(futures):
        f._add_settle_callback(lambda sf, i=i: on_settle(sf, i))

    return composite


def all_of(*futures: SimFuture) -> SimFuture:
    """Return a future that resolves when ALL input futures resolve.

    The composite future resolves with a list of values in the same
    order as the input futures. If any input future fails, the composite
    future fails immediately with that exception.

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
        A new SimFuture that settles when all inputs settle.
    """
    if len(futures) < 2:
        raise ValueError("all_of() requires at least 2 futures")

    composite = SimFuture()
    results: list[Any] = [None] * len(futures)
    remaining = len(futures)

    def on_settle(settled: SimFuture, idx: int = 0) -> None:
        nonlocal remaining
        if composite.is_settled:
            return
        if settled._failed:
            composite.fail(settled._exception)
            return
        results[idx] = settled._value
        remaining -= 1
        if remaining == 0:
            composite.resolve(list(results))

    for i, f in enumerate(futures):
        f._add_settle_callback(lambda sf, i=i: on_settle(sf, i))

    return composite
